import torch.nn.functional as F
import random
import requests, json
import time
import torchvision.transforms as transforms
from scipy import ndimage
import cv2
import matplotlib.pyplot as plt
import copy
import torch
from torch.autograd import Variable
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop
import os
import torchvision.models as torch_models
import argparse
import sys
import dill
import logging
import importlib.util
# Ensure the repository root is on sys.path so imports like `utils.func_utils`
# work when this file is executed directly (e.g. from repo/attacks/TEA).
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _repo_root not in sys.path:
    # Put repo root first so `import utils` prefers the package we create
    # instead of a local `utils.py` when TEA.py is executed directly.
    sys.path.insert(0, _repo_root)

import utils.func_utils as utils


torch_available = False
try:
    import torch
    from PIL import Image

    torch_available = True
except ImportError:
    from PIL import Image

class TEA:
    def __init__(self, model, mean, std, args, num_classes=None):
        self.model = model
        self.mean = mean
        self.std = std
        self.args = args
        self.num_classes = num_classes
        self.device = args.get("device", 'cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = args.get("batch_size", 1)
        self.q_budget_global = args.get("q_budget_global", 20)
        self.q_budget_patch = args.get("q_budget_patch", 20)
        self.initial_step_factor = args.get("initial_step_factor", 0.0001)
        self.step_size_factor = args.get("step_size_factor", 0.005)
        self.momentum = args.get("momentum", 0.9)
        self.use_cgba_refinement = bool(args.get("use_cgba_refinement", args.get("use-cgba-refinement", True)))
        self.refinement_attack_method = args.get("refinement_attack_method", args.get("refinement-attack-method", "CGBA_H"))
        self.refinement_dim_reduc_factor = int(args.get("refinement_dim_reduc_factor", args.get("refinement-dim-reduc-factor", 4)))
        self.refinement_iteration = int(args.get("refinement_iteration", args.get("refinement-iteration", 93)))
        self.refinement_initial_query = int(args.get("refinement_initial_query", args.get("refinement-initial-query", 30)))
        self.refinement_tol = float(args.get("refinement_tol", args.get("refinement-tol", 0.0001)))
        self.refinement_sigma = float(args.get("refinement_sigma", args.get("refinement-sigma", 0.0002)))
        self.targeted = bool(args.get("targeted", False))
        self.target_label = args.get("target_label", args.get("target-label", None))
        if self.targeted and self.target_label is None:
            raise ValueError("TEA targeted=True requires `target_label` (or `--target-label`).")
        if self.target_label is not None:
            self.target_label = int(self.target_label)

    def inv_tf(self, x, mean, std):
        for i in range(len(mean)):
            x[i] = np.multiply(x[i], std[i], dtype=np.float32)
            x[i] = np.add(x[i], mean[i], dtype=np.float32)
        x = np.swapaxes(x, 0, 2)
        x = np.swapaxes(x, 0, 1)
        return x

    def is_adversarial(self, image, model, tar_lbl):
        predict_label = torch.argmax(model.forward(Variable(image, requires_grad=True)).data).item()
        is_adv = predict_label == tar_lbl
        return 1 if is_adv else -1

    def create_gaussian_weight(self, patch_height, patch_width, sigma=None, device='cpu'):
        if sigma is None:
            sigma = min(patch_height, patch_width) / 2.0
        y = torch.arange(patch_height, device=device).float() - (patch_height - 1) / 2.0
        x = torch.arange(patch_width, device=device).float() - (patch_width - 1) / 2.0
        y = y.view(-1, 1)
        x = x.view(1, -1)
        gauss = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        gauss = gauss / gauss.max()
        return gauss.unsqueeze(0).unsqueeze(0)

    def extract_edge_data(self, image_tensor, low_threshold=50, high_threshold=150):
        image_np = image_tensor.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
        image_np = np.clip(image_np, 0, 1)
        gray_image = np.dot(image_np[..., :3], [0.2989, 0.5870, 0.1140])
        sx = ndimage.sobel(gray_image, axis=0, mode='reflect')
        sy = ndimage.sobel(gray_image, axis=1, mode='reflect')
        grad_magnitude = np.hypot(sx, sy)
        grad_magnitude = (grad_magnitude / (grad_magnitude.max() + 1e-8)) * 255
        grad_magnitude = grad_magnitude.astype(np.uint8)
        edge_mask = np.zeros_like(grad_magnitude)
        edge_mask[(grad_magnitude >= low_threshold) & (grad_magnitude <= high_threshold)] = 255
        return edge_mask, gray_image

    def create_soft_edge_mask(self, edge_mask, blur_size=5, intensity=0.8):
        soft_mask = cv2.GaussianBlur(edge_mask.astype(np.float32), (blur_size, blur_size), 0)
        soft_mask = soft_mask / soft_mask.max()
        soft_mask = (soft_mask * intensity).astype(np.float32)
        return torch.tensor(soft_mask, dtype=torch.float32)

    def denormalize_image(self, image_tensor, mean, std):
        mean = torch.tensor(mean).view(3, 1, 1)
        std = torch.tensor(std).view(3, 1, 1)
        denorm_image = image_tensor.detach().cpu() * std + mean
        denorm_image = torch.clamp(denorm_image, 0, 1)
        return denorm_image.squeeze(0).permute(1, 2, 0).numpy()

    def show_image_with_label(self, img_tensor, src_img, mean, std, total_calls, title=None, src_lbl=None, tar_lbl=None,
                              is_adv=None, caption=None):
        img_np = self.denormalize_image(img_tensor, mean, std)
        src_np = self.denormalize_image(src_img, mean, std)
        dist = np.linalg.norm(img_np - src_np)
        fig, ax = plt.subplots()
        ax.imshow(img_np)
        ax.axis("off")
        if is_adv == 1:
            lbl = tar_lbl
        else:
            lbl = src_lbl
        fig.text(0.5, 0.065, f"Label = {lbl}", ha='center', va='top', fontsize=12)
        if total_calls != 0:
            fig.text(0.5, -0.055, f"Total queries used = {total_calls}", ha='center', va='top', fontsize=10)
        if title:
            fig.suptitle(title, fontsize=14)
        if title != "Source":
            if caption:
                fig.text(0.5, 0.005, f"ℓ² to source = {caption:.4f}", ha='center', va='top', fontsize=10)
            else:
                fig.text(0.5, 0.005, f"ℓ² to source = {dist:.4f}", ha='center', va='top', fontsize=10)
        #plt.show()

    def preprocess_image(self, image, mean, std, device=None):
        if device is None:
            device = self.device
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        return transform(image).to(device)

    def global_search(self, x_0, x_random, soft_edge_mask_y, model, tar_lbl, initial_step_factor=0.0001, momentum=0.9, max_calls=20):
        num_calls = 0
        direction = x_0 - x_random
        velocity = torch.zeros_like(direction)
        current_point = x_random.clone()
        last_adv_point = current_point.clone()
        initial_distance = torch.norm(direction).item()
        step_size = initial_distance * initial_step_factor
        non_edge_mask_y = (1 - soft_edge_mask_y)
        while num_calls < max_calls:
            velocity = momentum * velocity + (1 - momentum) * direction
            step = step_size * velocity
            non_edge_mask_y = non_edge_mask_y.to(current_point.device)
            next_point = current_point + step * non_edge_mask_y
            num_calls += 1
            if self.is_adversarial(next_point, model, tar_lbl) == 1:
                last_adv_point = next_point.clone()
                current_point = next_point
                step_size *= 1.1
            else:
                step_size *= 0.5
                break
            if torch.norm(step).cpu().numpy() < 0.0001:
                break
        return last_adv_point, num_calls

    def patch_search(self, x_0, x_random, soft_edge_mask_y, model, tar_lbl, mean, std, min_patch_size=16, max_patch_size=64, step_size_factor=0.005, momentum=0.9, max_calls=20, iterations_to_show=3):
        total_calls = 0
        best_x_random = x_random.clone()
        _, _, H, W = x_random.shape
        patch_boundary_mask = torch.zeros((H, W), device=x_random.device, dtype=torch.float32)
        x0_inv = self.inv_tf(copy.deepcopy(x_0.cpu()[0].squeeze()), mean, std)
        overall_break = False
        last_query = 0
        rarity_break = 0
        recorded = []
        record_count = 0
        while True:
            diff_map = torch.abs(x_0 - best_x_random).mean(dim=1, keepdim=True)
            diff_map = F.avg_pool2d(diff_map, kernel_size=3, stride=1, padding=1)
            diff_flat = diff_map.view(-1)
            high_diff_indices = torch.argsort(diff_flat, descending=True)[:H * W // 5]
            center_idx = high_diff_indices[random.randint(0, len(high_diff_indices) - 1)]
            i_center = (center_idx // W).item()
            j_center = (center_idx % W).item()
            patch_size = random.randint(min_patch_size, max_patch_size)
            i_start = max(0, i_center - patch_size // 2)
            j_start = max(0, j_center - patch_size // 2)
            i_end = min(H, i_start + patch_size)
            j_end = min(W, j_start + patch_size)
            bw = 4
            patch_boundary_mask[i_start:min(i_start + bw, H), j_start:j_end] = 1
            patch_boundary_mask[max(i_end - bw, 0):i_end, j_start:j_end] = 1
            patch_boundary_mask[i_start:i_end, j_start:min(j_start + bw, W)] = 1
            patch_boundary_mask[i_start:i_end, max(j_end - bw, 0):j_end] = 1
            pre_patch = best_x_random.clone()
            pre_patch_inv = self.inv_tf(copy.deepcopy(pre_patch.cpu()[0].squeeze()), mean, std)
            pre_norm = torch.norm(x0_inv - pre_patch_inv).item()
            patch_updated = False
            for _iteration in range(max_calls):
                if total_calls > 25 + last_query or rarity_break > 5000:
                    overall_break = True
                    break
                local_direction = x_0[:, :, i_start:i_end, j_start:j_end] - best_x_random[:, :, i_start:i_end, j_start:j_end]
                momentum_patch = momentum * torch.zeros_like(local_direction) + (1 - momentum) * local_direction
                step_size = torch.norm(x_0 - best_x_random).item() * step_size_factor
                patch_h, patch_w = i_end - i_start, j_end - j_start
                gaussian_weight = self.create_gaussian_weight(patch_h, patch_w, sigma=patch_size / 4.0, device=x_random.device)
                mask_patch = soft_edge_mask_y[:, :, i_start:i_end, j_start:j_end].to(gaussian_weight.device)
                update_weight = gaussian_weight * (1 - 0.9 * mask_patch)
                step = step_size * momentum_patch * update_weight
                next_patch = best_x_random[:, :, i_start:i_end, j_start:j_end] + step
                temp_x_random = best_x_random.clone()
                temp_x_random[:, :, i_start:i_end, j_start:j_end] = next_patch
                new_distance = torch.norm(x_0 - temp_x_random).item()
                if new_distance >= 0.999 * torch.norm(x_0 - best_x_random).item():
                    rarity_break += 1
                    break
                total_calls += 1
                if self.is_adversarial(temp_x_random, model, tar_lbl) == 1:
                    patch_updated = True
                    rarity_break = 0
                    best_x_random[:, :, i_start:i_end, j_start:j_end] = next_patch.clone()
                    last_query = total_calls
                else:
                    break
                if overall_break:
                    break
            if patch_updated and record_count < iterations_to_show:
                post_patch_inv = self.inv_tf(copy.deepcopy(best_x_random.cpu()[0].squeeze()), mean, std)
                post_norm = torch.norm(x0_inv - post_patch_inv).item()
                before_np = self.denormalize_image(pre_patch, mean, std)
                after_np = self.denormalize_image(best_x_random, mean, std)
                mask_map = torch.zeros((H, W), dtype=torch.bool)
                mask_map[i_start:i_end, j_start:j_end] = True
                highlight = before_np.copy()
                highlight[~mask_map.numpy()] = (highlight[~mask_map.numpy()] * 0.5).astype(highlight.dtype)
                recorded.append((before_np, highlight, after_np, pre_norm, post_norm))
                record_count += 1
            if overall_break:
                break
        if recorded:
            rows = len(recorded)
            fig, axes = plt.subplots(rows, 3, figsize=(12, 4 * rows))
            for i, (bef, hl, aft, _pre_d, post_d) in enumerate(recorded):
                row_axes = axes[i] if rows > 1 else axes
                for j, img in enumerate((bef, hl, aft)):
                    ax = row_axes[j]
                    ax.imshow(img)
                    ax.axis('off')
                    if i == 0:
                        ax.set_title(("Before", "Selected Patch", "After")[j])
                    if j == 2:
                        ax.text(
                            0.5, -0.025,
                            f"ℓ² to source  = {post_d:.4f}",
                            ha='center', va='top',
                            transform=ax.transAxes,
                            fontsize=10
                        )
            plt.tight_layout()
            #plt.show()
        return best_x_random, total_calls, patch_boundary_mask

    def initialize_attack(self, pair_id, src_img, tar_img, mean, std, model_arch='ViT', force_target_label=None):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        np.random.seed(42)
        device = self.device
        try:
            is_module = isinstance(model_arch, torch.nn.Module)
        except Exception:
            is_module = False

        # Accept plain nn.Module, TorchScript/ScriptModule, or any callable proxy
        if is_module:
            net = model_arch
        elif hasattr(model_arch, '__call__') or hasattr(model_arch, 'forward'):
            # model_arch may be a torch.jit.ScriptModule or a proxy that implements __call__
            net = model_arch
            # Ensure there's a .forward attribute so code below can call net.forward(...)
            try:
                if not hasattr(net, 'forward') and callable(net):
                    net.forward = net.__call__
            except Exception:
                pass
        else:
            if model_arch == 'resnet50':
                net = torch_models.resnet50(pretrained=True)
            elif model_arch == 'resnet101':
                net = torch_models.resnet101(pretrained=True)
            elif model_arch == 'vgg16':
                net = torch_models.vgg16(pretrained=True)
            elif model_arch == 'ViT':
                import timm
                net = timm.create_model('vit_base_patch16_224', pretrained=True)
            else:
                raise ValueError(f"Unsupported model architecture: {model_arch}")

        net = net.to(device)
        net.eval()
        model = net
        x_0 = self.preprocess_image(src_img, mean, std, device).unsqueeze(0)
        x_t = self.preprocess_image(tar_img, mean, std, device).unsqueeze(0)
        orig_label = torch.argmax(net.forward(Variable(x_0, requires_grad=True)).data).item()
        tar_pred_label = torch.argmax(net.forward(Variable(x_t, requires_grad=True)).data).item()

        if force_target_label is None:
            tar_label_for_attack = tar_pred_label
        else:
            tar_label_for_attack = int(force_target_label)

        if orig_label == tar_label_for_attack:
            print(
                f"Pair {pair_id}: Source already matches the attack target label ({tar_label_for_attack}). Skipping attack."
            )
            return
        src_img = x_0
        tar_img = x_t
        src_lbl = torch.argmax(model.forward(src_img)).item()
        tar_lbl = tar_label_for_attack
        url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
        response = requests.get(url)
        idx_to_label = {int(k): v[1] for k, v in json.loads(response.content).items()}
        src_lbl_name = idx_to_label[src_lbl]
        tar_lbl_name = idx_to_label.get(tar_lbl, str(tar_lbl))
        src_np = self.denormalize_image(src_img, mean, std)
        tar_np = self.denormalize_image(tar_img, mean, std)
        dist = np.linalg.norm(tar_np - src_np)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(src_np)
        axes[0].axis('off')
        axes[0].set_title(f"Source\nLabel = {src_lbl_name}", fontsize=12)
        axes[1].imshow(tar_np)
        axes[1].axis('off')
        axes[1].set_title(f"Target\nLabel = {tar_lbl_name}", fontsize=12)
        axes[1].text(
            0.5, -0.025,
            f"ℓ² to source = {dist:.4f}",
            ha='center', va='top',
            transform=axes[1].transAxes,
            fontsize=10
        )
        plt.suptitle(f"Pair {pair_id}", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        #plt.show()
        return_arguments = [model, mean, std, src_img, tar_img, tar_lbl, src_lbl_name, tar_lbl_name]
        return return_arguments

    def edge_mask_initialization(self, mean, std, src_img, tar_img):
        x_inv = self.inv_tf(copy.deepcopy(src_img.cpu()[0, :, :, :].squeeze()), mean, std)
        x_adv = tar_img
        x_adv_inv = self.inv_tf(copy.deepcopy(x_adv.cpu()[0, :, :, :].squeeze()), mean, std)
        _norm = torch.norm(x_inv - x_adv_inv)
        edge_mask_adv, _gray_adv = self.extract_edge_data(x_adv)
        soft_edge_mask_adv = self.create_soft_edge_mask(edge_mask_adv)
        orig_np = self.denormalize_image(x_adv, mean, std)
        edge_np = edge_mask_adv
        soft_np = soft_edge_mask_adv.squeeze().cpu().numpy()
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(orig_np)
        axes[0].axis('off')
        axes[0].set_title('Target')
        axes[1].imshow(edge_np, cmap='gray')
        axes[1].axis('off')
        axes[1].set_title('Edge Mask')
        axes[2].imshow(soft_np, cmap='gray')
        axes[2].axis('off')
        axes[2].set_title('Soft Edge Mask')
        plt.tight_layout()
        #plt.show()
        return soft_edge_mask_adv

    def global_edge_informed_search(
            self,
            soft_edge_mask_adv,
            model, mean, std, src_img,
            tar_img, tar_lbl,
            src_lbl_name, tar_lbl_name,
            initial_step_factor,
            q_budget_global,
            momentum,
            iterations_to_show=3
            ):

        total_calls = 0
        x_inv = self.inv_tf(copy.deepcopy(src_img.cpu()[0, :, :, :].squeeze()), mean, std)
        x_adv = tar_img
        x_adv_inv = self.inv_tf(copy.deepcopy(x_adv.cpu()[0, :, :, :].squeeze()), mean, std)
        norm = torch.norm(x_inv - x_adv_inv)
        previous_norm = norm
        continuous = 0
        recorded = []
        for i in range(100):
            x_adv, num_calls = self.global_search(
                src_img,
                x_adv,
                soft_edge_mask_adv,
                model,
                tar_lbl,
                initial_step_factor=initial_step_factor,
                max_calls=q_budget_global,
                momentum=momentum
            )
            total_calls += num_calls
            if i < iterations_to_show:
                img_np = self.denormalize_image(x_adv, mean, std)
                recorded.append((img_np, norm.item()))
            x_adv_inv = self.inv_tf(copy.deepcopy(x_adv.cpu()[0, :, :, :].squeeze()), mean, std)
            norm = torch.norm(x_inv - x_adv_inv)
            if norm == previous_norm:
                continuous += 1
                if continuous > 1:
                    num_calls = 0
                    adv = x_adv
                    cln = src_img
                    while True:
                        mid = (cln + adv) / 2.0
                        num_calls += 1
                        total_calls += 1
                        if self.is_adversarial(mid, model, tar_lbl) == 1:
                            adv = mid
                        else:
                            cln = mid
                        if torch.norm(adv - cln).cpu().numpy() < 0.0001 or num_calls >= 25:
                            break
                    x_adv = adv
                    break
            else:
                continuous = 0
            previous_norm = norm
        if recorded:
            fig, axes = plt.subplots(1, len(recorded), figsize=(4 * len(recorded), 4))
            for idx, (img, dist) in enumerate(recorded):
                ax = axes[idx] if len(recorded) > 1 else axes
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(f"Iteration {idx + 1}")
                ax.text(
                    0.5, -0.025,
                    f"ℓ² to source  = {dist:.4f}",
                    ha='center', va='top',
                    transform=ax.transAxes,
                    fontsize=10
                )
            #plt.tight_layout()
            #plt.show()
        print("\n")
        return_arguments = [src_img, x_adv, soft_edge_mask_adv, total_calls, tar_lbl, src_lbl_name, tar_lbl_name, model]
        return return_arguments

    def patch_based_edge_informed_search(
            self,
            src_img,
            x_adv,
            soft_edge_mask_adv,
            total_calls,
            tar_lbl,
            src_lbl_name,
            tar_lbl_name,
            model,
            mean,
            std,
            model_arch="ViT",
            file_a=None,
            file_b=None,
            iterations_to_show=3,
            q_budget_patch=20,
            step_size_factor=0.005,
            momentum=0.9
        ):
        if isinstance(model_arch, str):
            arch_name = model_arch
        else:
            try:
                if hasattr(model_arch, 'module'):
                    arch_name = model_arch.module.__class__.__name__
                else:
                    arch_name = model_arch.__class__.__name__
            except Exception:
                arch_name = 'model'

        model_folder = arch_name.capitalize() if isinstance(arch_name, str) and arch_name.lower().startswith('resnet') else arch_name
        soft_edge_mask_adv = soft_edge_mask_adv.unsqueeze(0).unsqueeze(0)
        os.makedirs(model_folder, exist_ok=True)
        x_inv = self.inv_tf(copy.deepcopy(src_img.cpu()[0, :, :, :].squeeze()), mean, std)
        x_adv, patch_calls, _patch_boundary_mask = self.patch_search(
            src_img,
            x_adv,
            soft_edge_mask_adv,
            model,
            tar_lbl,
            mean,
            std,
            iterations_to_show=iterations_to_show,
            max_calls=q_budget_patch,
            step_size_factor=step_size_factor,
            momentum=momentum
        )
        total_calls += patch_calls
        x_adv_inv = self.inv_tf(x_adv.cpu().detach()[0, :, :, :].clone(), mean, std)
        norm = torch.norm(x_inv - x_adv_inv)
        print(f'Norm after patch refinement: {norm}, queries used: {total_calls}')
        save_dir = os.path.join('Tensors', model_folder)
        os.makedirs(save_dir, exist_ok=True)
        timestamp = int(time.time() * 1000)
        if isinstance(file_a, str):
            base_a = os.path.basename(file_a)
            base_a = base_a.replace('.JPEG', '.pt').replace('.JPEG', '.pt').replace('.jpg', '.pt').replace('.jpeg', '.pt')
        else:
            base_a = f"src_{timestamp}.pt"

        if isinstance(file_b, str):
            base_b = os.path.basename(file_b)
            base_b = base_b.replace('.JPEG', '.pt').replace('.jpg', '.pt').replace('.jpeg', '.pt')
        else:
            base_b = f"adv_{timestamp}.pt"

        src_tensor_path = os.path.join(save_dir, base_a)
        adv_tensor_path = os.path.join(save_dir, base_b)
        torch.save(src_img.cpu(), src_tensor_path)
        torch.save(x_adv.cpu(), adv_tensor_path)
        return x_adv, total_calls, norm

    def refine_with_cgba(
            self,
            src_img,
            tea_adv_img,
            model,
            mean,
            std,
            tar_lbl,
            total_calls=0,
            attack_method='CGBA_H',
            dim_reduc_factor=4,
            iteration=93,
            initial_query=30,
            tol=0.0001,
            sigma=0.0002,
            verbose_control='Yes'
        ):
        if attack_method not in ('CGBA', 'CGBA_H'):
            raise ValueError("attack_method must be either 'CGBA' or 'CGBA_H'")

        from CGBA.proposed_attack import Proposed_attack
        from CGBA.utils import valid_bounds

        if src_img.ndim == 3:
            src_img = src_img.unsqueeze(0)
        if tea_adv_img.ndim == 3:
            tea_adv_img = tea_adv_img.unsqueeze(0)

        src_img = src_img.to(self.device)
        tea_adv_img = tea_adv_img.to(self.device)

        src_np = self.denormalize_image(src_img, mean, std)
        src_pil = Image.fromarray(np.clip(src_np * 255.0, 0, 255).astype(np.uint8))
        lb_np, ub_np = valid_bounds(src_pil, delta=255)

        norm_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        lb = norm_tf(Image.fromarray(lb_np)).unsqueeze(0).to(self.device)
        ub = norm_tf(Image.fromarray(ub_np)).unsqueeze(0).to(self.device)

        attack = Proposed_attack(
            model=model,
            src_img=src_img,
            mean=mean,
            std=std,
            lb=lb,
            ub=ub,
            tar_img=tea_adv_img,
            dim_reduc_factor=dim_reduc_factor,
            attack_method=attack_method,
            iteration=iteration,
            initial_query=initial_query,
            tol=tol,
            sigma=sigma,
            verbose_control=verbose_control
        )

        refined_adv, refinement_queries_curve, _ = attack.Attack()
        refinement_queries = int(refinement_queries_curve[-1]) if len(refinement_queries_curve) > 0 else 0
        combined_queries = int(total_calls) + refinement_queries

        adv_pred = torch.argmax(model.forward(Variable(refined_adv, requires_grad=False)).data).item()
        src_denorm = self.denormalize_image(src_img, mean, std)
        adv_denorm = self.denormalize_image(refined_adv, mean, std)
        l2 = float(np.linalg.norm(adv_denorm - src_denorm))

        try:
            denorm_img = utils.remove_normalization(refined_adv.clone().squeeze(0).cpu(), mean, std)
            denorm_np = denorm_img.cpu().detach().numpy()
            denorm_np = np.clip(denorm_np, 0.0, 1.0)
        except Exception:
            denorm_np = refined_adv.squeeze(0).cpu().detach().numpy()

        try:
            src_denorm = utils.remove_normalization(src_img.clone().squeeze(0).cpu(), mean, std)
            src_denorm_np = np.clip(src_denorm.cpu().detach().numpy(), 0.0, 1.0)
            linf = float(np.max(np.abs(denorm_np - src_denorm_np)))
        except Exception:
            linf = float(np.max(np.abs(np.clip(adv_denorm, 0.0, 1.0) - np.clip(src_denorm, 0.0, 1.0))))

        return {
            'attack_name': f'TEA+{attack_method}',
            'perturbed_image': np.asarray([denorm_np]),
            'perturbed_label': np.asarray([adv_pred]),
            'total_queries': [combined_queries],
            'l2': [l2],
            'success': int(adv_pred == int(tar_lbl)),
            'max_perturbation': linf
        }

    def tensor_to_pil(self, tensor):
        """
        Convert a torch tensor image [C,H,W] in [0,1] to PIL Image
        """
        tensor = tensor.clone().detach().cpu()
        if tensor.ndim == 4:
            tensor = tensor.squeeze(0)
        tensor = tensor.permute(1, 2, 0)
        tensor = tensor.numpy()
        tensor = np.clip(tensor * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(tensor)

    def perturb(self, images, labels):
        perturbed_images = []
        perturbed_labels = []
        total_queries = []
        l2s = []
        linf_values = []
        success_list = []
        attack_name = 'TEA'

        num_images = len(images)
        target_indices = None
        if self.targeted:
            target_indices = []
            for i in range(num_images):
                try:
                    lbl_val = int(labels[i].item())
                except Exception:
                    continue
                if lbl_val == self.target_label:
                    target_indices.append(i)
            if len(target_indices) == 0:
                raise ValueError(
                    f"TEA targeted=True requested target_label={self.target_label}, but none of the provided images have that label. "
                    "Increase `total_images` or ensure your batch contains examples of the target class."
                )

            def _model_predicts_target_label(img_tensor, desired_label: int) -> bool:
                try:
                    pil_img = self.tensor_to_pil(img_tensor)
                    x = self.preprocess_image(pil_img, self.mean, self.std, self.device).unsqueeze(0)
                    pred = torch.argmax(self.model.forward(Variable(x, requires_grad=False)).data).item()
                    return int(pred) == int(desired_label)
                except Exception:
                    return False

        for src_idx in range(num_images):
            src_img = images[src_idx].unsqueeze(0)

            tar_img = None
            if self.targeted:
                start_pos = src_idx % len(target_indices)
                chosen = None
                for k in range(len(target_indices)):
                    cand = target_indices[(start_pos + k) % len(target_indices)]
                    if cand == src_idx and len(target_indices) > 1:
                        continue
                    if _model_predicts_target_label(images[cand], self.target_label):
                        chosen = cand
                        break
                if chosen is None:
                    chosen = target_indices[start_pos]
                    if chosen == src_idx and len(target_indices) > 1:
                        chosen = target_indices[(start_pos + 1) % len(target_indices)]
                    print(
                        f"Source {src_idx}: warning: no target-class image was predicted as target_label={self.target_label}; "
                        f"falling back to index {chosen}."
                    )
                tar_img = images[chosen].unsqueeze(0)
                print(f"Source {src_idx}: targeted=True, using target index {chosen} (label={self.target_label}).")
            else:
                target_found = False
                for offset in range(1, num_images):
                    cand = (src_idx + offset) % num_images
                    try:
                        src_lbl_val = int(labels[src_idx].item())
                        cand_lbl_val = int(labels[cand].item())
                    except Exception:
                        src_lbl_val = None
                        cand_lbl_val = None

                    if src_lbl_val is None or cand_lbl_val is None:
                        continue

                    if cand_lbl_val != src_lbl_val:
                        tar_img = images[cand].unsqueeze(0)
                        target_found = True
                        print(f"Source {src_idx}: using target index {cand} (different class).")
                        break

                if not target_found:
                    print(f"Source {src_idx}: no different-class target found; skipping source.")
                    continue

            pair_id = src_idx + 1
            returned = self.initialize_attack(
                pair_id=pair_id + 1,
                src_img=self.tensor_to_pil(src_img),
                tar_img=self.tensor_to_pil(tar_img),
                mean=self.mean,
                std=self.std,
                model_arch=self.model,
                force_target_label=(self.target_label if self.targeted else None),
            )
            if returned is None:
                continue

            tea_model, tea_mean, tea_std, src_img_t, tar_img_t, tar_lbl, src_lbl_name, tar_lbl_name = returned

            soft_edge_mask = self.edge_mask_initialization(
                tea_mean, tea_std, src_img_t, tar_img_t
            )

            global_result = self.global_edge_informed_search(
                soft_edge_mask,
                tea_model,
                tea_mean,
                tea_std,
                src_img_t,
                tar_img_t,
                tar_lbl,
                src_lbl_name,
                tar_lbl_name,
                self.initial_step_factor,
                self.q_budget_global,
                momentum=self.momentum
            )
            adv_img, queries, l2 = self.patch_based_edge_informed_search(
                *global_result,
                model_arch=self.model,
                file_a=src_img,
                file_b=tar_img,
                iterations_to_show=3,
                mean=tea_mean,
                std=tea_std,
                q_budget_patch=self.q_budget_patch,
                step_size_factor=self.step_size_factor,
                momentum=self.momentum
            )

            if self.use_cgba_refinement:
                try:
                    refinement_result = self.refine_with_cgba(
                        src_img=src_img_t,
                        tea_adv_img=adv_img,
                        model=tea_model,
                        mean=tea_mean,
                        std=tea_std,
                        tar_lbl=tar_lbl,
                        total_calls=queries,
                        attack_method=self.refinement_attack_method,
                        dim_reduc_factor=self.refinement_dim_reduc_factor,
                        iteration=self.refinement_iteration,
                        initial_query=self.refinement_initial_query,
                        tol=self.refinement_tol,
                        sigma=self.refinement_sigma
                    )
                    attack_name = refinement_result.get('attack_name', attack_name)
                    denorm_np = refinement_result['perturbed_image'][0]
                    adv_pred = int(refinement_result['perturbed_label'][0])
                    queries = int(refinement_result['total_queries'][0])
                    l2 = float(refinement_result['l2'][0])
                    success_list.append(int(refinement_result['success']))
                except Exception as refinement_error:
                    print(
                        f"Source {src_idx}: refinement failed with {type(refinement_error).__name__}: {repr(refinement_error)}. "
                        "Falling back to TEA patch result."
                    )
                    adv_pred = torch.argmax(
                        tea_model(adv_img.clone())
                    ).item()
                    success_list.append(int(adv_pred == tar_lbl))
                    try:
                        denorm_img = utils.remove_normalization(adv_img.clone().squeeze(0).cpu(), self.mean, self.std)
                        denorm_np = denorm_img.cpu().detach().numpy()
                        denorm_np = np.clip(denorm_np, 0.0, 1.0)
                    except Exception:
                        denorm_np = adv_img.squeeze(0).cpu().numpy()
            else:
                adv_pred = torch.argmax(
                    tea_model(adv_img.clone())
                ).item()
                success_list.append(int(adv_pred == tar_lbl))
                try:
                    denorm_img = utils.remove_normalization(adv_img.clone().squeeze(0).cpu(), self.mean, self.std)
                    denorm_np = denorm_img.cpu().detach().numpy()
                    denorm_np = np.clip(denorm_np, 0.0, 1.0)
                except Exception:
                    denorm_np = adv_img.squeeze(0).cpu().numpy()

            perturbed_images.append(denorm_np)
            perturbed_labels.append(adv_pred)
            total_queries.append(queries)
            l2s.append(float(l2))
            try:
                src_raw_np = np.clip(src_img.squeeze(0).detach().cpu().numpy(), 0.0, 1.0)
                adv_raw_np = np.clip(np.asarray(denorm_np), 0.0, 1.0)
                linf_values.append(float(np.max(np.abs(adv_raw_np - src_raw_np))))
            except Exception:
                pass

        print("TEA attack completed on all pairs.")
        print("L2 distances:", l2s)
        print("Total queries:", total_queries)
        total_success = sum(success_list)
        maxp_val = float(max(linf_values)) if len(linf_values) > 0 else 0.0
        return {
            'attack_name': attack_name,
            'perturbed_image': np.asarray(perturbed_images),
            'perturbed_label': np.asarray(perturbed_labels),
            'total_queries': total_queries,
            'l2': l2s,
            'success': total_success,
            'max_perturbation': maxp_val
        }


def execute_attack(model, images, labels, args, mean, std, num_classes):
    """
    TEA attack wrapper fully compatible with pipeline
    """
    tea = TEA(model=model, mean=mean, std=std, args=args, num_classes=num_classes)
    return tea.perturb(images, labels)


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()

        # attack params
        parser.add_argument('--eps', type=float, help='Perturbation budget')
        parser.add_argument('--targeted', action='store_true')
        parser.add_argument('--batch-size', type=int)

        # general params
        parser.add_argument('--model', type=str, required=True)
        parser.add_argument('--dataset', type=str, required=True)
        parser.add_argument('--total-images', type=int, required=True)
        parser.add_argument('--log', action='store_true')
        parser.add_argument('--attack-config', type=str, default='')
        parser.add_argument('--results-path', type=str, required=True)
        parser.add_argument('--q-budget-global', type=int, default=200)
        parser.add_argument('--q-budget-patch', type=int, default=200)
        parser.add_argument('--initial-step-factor', type=float, default=0.0001)
        parser.add_argument('--step-size-factor', type=float, default=0.005)
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--target-label', type=int, default=1)
        parser.add_argument('--use-cgba-refinement', dest='use_cgba_refinement', action='store_true')
        #parser.add_argument('--no-cgba-refinement', dest='use_cgba_refinement', action='store_false')
        #parser.set_defaults(use_cgba_refinement=True)
        parser.add_argument('--refinement-attack-method', type=str, default='CGBA_H')
        parser.add_argument('--refinement-dim-reduc-factor', type=int, default=4)
        parser.add_argument('--refinement-iteration', type=int, default=93)
        parser.add_argument('--refinement-initial-query', type=int, default=30)
        parser.add_argument('--refinement-tol', type=float, default=0.0001)
        parser.add_argument('--refinement-sigma', type=float, default=0.0002)
        
        parsed_args, unknown = parser.parse_known_args()
        ini_args = vars(parsed_args)
        ini_args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

        args = utils.prepare_attack_arguments(ini_args, ini_args.get('attack_config', ''), 'tea')

        if not os.path.isdir(args['results_path']):
            os.makedirs(args['results_path'])

        logging.basicConfig(filename=os.path.join(args['results_path'], 'TEA.log'), format='%(asctime)s | %(levelname)s | %(message)s', datefmt= '%m-%d-%Y %H:%M:%S', level=logging.INFO)

        ini_stdout = sys.stdout
        sys.stdout = utils.StdoutToLogging()

        print("Loading TEA Attack")

        with open(args['model'], "rb") as file:
            data = dill.loads(file.read())
            model = data['model']
            classified_labels = data.get('classified_labels', None)

        model = model.to(args['device'])
        model.eval()
        print("Model loaded")
        spec = importlib.util.spec_from_file_location("dataset", args['dataset'])
        dataset = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dataset)

        testloader, _ = dataset.dataLoader()

        images, labels = utils.get_images_labels_from_dataLoader(
            testloader, args['device'], args['total_images']
        )
        images, labels, classified_labels, _, orig_indices = utils.filter_correctly_classified(
            images, labels, classified_labels
        )
        if images is None or (torch.is_tensor(images) and images.shape[0] == 0):
            print("No correctly classified samples found; skipping TEA.")
            empty_results = {
                'attack_name': 'TEA',
                'perturbed_image': [],
                'perturbed_label': [],
                'total_queries': [],
                'l2': [],
                'success': 0,
                'max_perturbation': 0.0,
                'orig_index': []
            }
            utils.save_statistics_of_attack(empty_results, labels, classified_labels, args['results_path'])
            utils.save_images_attack(empty_results, labels, os.path.join(args['results_path'], 'perturbed_images'))
            sys.stdout = ini_stdout
            raise SystemExit(0)

        print("Dataset loaded")

        attack_results = execute_attack(
            model,
            images,
            labels,
            args,
            dataset.MEAN,
            dataset.STD,
            dataset.NUM_CLASSES
        )

        attack_results['orig_index'] = orig_indices

        print(f"TEA attack finished, perturbed {len(attack_results['perturbed_image'])} image batches")


        utils.save_statistics_of_attack(attack_results, labels, classified_labels, args['results_path'])
        utils.save_images_attack(attack_results, labels, os.path.join(args['results_path'], 'perturbed_images'))

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        print(f"TEA attack failed: {type(e).__name__}: {repr(e)}")

    finally:
        try:
            sys.stdout = ini_stdout
        except Exception:
            pass
