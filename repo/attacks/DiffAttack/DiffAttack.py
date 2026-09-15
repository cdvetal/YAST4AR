import sys
import os
import argparse
import logging
import dill
import importlib
import importlib.util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Ensure the repository root is on sys.path so imports like `utils.func_utils`
# work when this file is executed directly (e.g. from repo/attacks/TEA).
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _repo_root not in sys.path:
    # Put repo root first so `import utils` prefers the package we create
    # instead of a local `utils.py` when TEA.py is executed directly.
    sys.path.insert(0, _repo_root)

import utils.func_utils as utils

class DiffAttack:
    def __init__(self, model, args, mean, std):
        self.model = model
        self.args = args
        self.mean = mean
        self.std = std

    def _autocast_ctx(self, device: str, enabled: bool):
        if not enabled:
            from contextlib import nullcontext
            return nullcontext()
        if not isinstance(device, str) or not device.startswith("cuda"):
            from contextlib import nullcontext
            return nullcontext()
        # torch.autocast is available in torch>=1.10 (and works well in 2.x)
        return torch.autocast(device_type="cuda", dtype=torch.float16)


    def _infer_dataset_id(self, dataset_path: str) -> str:
        """Best-effort dataset identifier for checkpoint naming."""
        if not dataset_path:
            return "unknown"
        # Typical loader path: repo/datasets/cifar-10/datasetLoader.py
        parent = os.path.basename(os.path.dirname(os.path.abspath(dataset_path)))
        if parent:
            return parent
        base = os.path.splitext(os.path.basename(dataset_path))[0]
        return base or "unknown"


    def _ae_checkpoint_path(self, dataset_id: str) -> str:
        base_dir = os.path.join(os.path.dirname(__file__), "weights", dataset_id)
        os.makedirs(base_dir, exist_ok=True)
        return os.path.join(base_dir, "ae_checkpoint.pth.tar")


    def _save_ae_checkpoint(
        self,
        checkpoint_path: str,
        model_clean: nn.Module,
        model_adv: nn.Module,
        *,
        ae_input_size: int,
        mean,
        std,
    ):
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        save_dict = {
            "state_dict_clean": model_clean.state_dict(),
            "state_dict_adv": model_adv.state_dict(),
            "ae_input_size": int(ae_input_size),
            "mean": list(mean),
            "std": list(std),
        }
        tmp_path = checkpoint_path + ".tmp"
        torch.save(save_dict, tmp_path)
        os.replace(tmp_path, checkpoint_path)


    @torch.no_grad()
    def _to_01_from_norm(self, x_norm: torch.Tensor, mean, std) -> torch.Tensor:
        x_01 = utils.remove_normalization(x_norm, mean, std)
        return torch.clamp(x_01, 0.0, 1.0)


    def _resize_bchw(self, x: torch.Tensor, size: int) -> torch.Tensor:
        if x.shape[-1] == size and x.shape[-2] == size:
            return x
        return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)


    def _normalize_bchw(self, x_01: torch.Tensor, mean, std) -> torch.Tensor:
        mean_t = torch.tensor(mean, device=x_01.device, dtype=x_01.dtype).view(1, -1, 1, 1)
        std_t = torch.tensor(std, device=x_01.device, dtype=x_01.dtype).view(1, -1, 1, 1)
        return (x_01 - mean_t) / std_t


    def _decode_ae_to_01(self, x_ae: torch.Tensor) -> torch.Tensor:
        # AE output is tanh in [-1,1]
        return torch.clamp(x_ae * 0.5 + 0.5, 0.0, 1.0)


    def _margin_losses(
        self,
        logits: torch.Tensor,
        true_labels: torch.Tensor,
        *,
        target_label: int = -1,
        margin: float = 5.0,
    ) -> torch.Tensor:
        """Vectorized version of main.py's adv_loss, returning per-sample losses."""
        if logits.ndim != 2:
            raise ValueError("logits must be 2D (N, C)")
        n, c = logits.shape

        if target_label is None:
            target_label = -1

        if target_label < 0:
            y = true_labels.view(-1).long()
            if y.numel() == 1 and n > 1:
                y = y.repeat(n)
            one_hot = torch.zeros((n, c), device=logits.device, dtype=torch.bool)
            one_hot.scatter_(1, y.view(-1, 1), True)
            true_log = logits[torch.arange(n, device=logits.device), y]
            max_other = logits.masked_fill(one_hot, float("-inf")).max(dim=1).values
            diff = true_log - max_other
            # relu(diff + m) - m
            return F.relu(diff + margin) - margin
        else:
            t = torch.full((n,), int(target_label), device=logits.device, dtype=torch.long)
            one_hot = torch.zeros((n, c), device=logits.device, dtype=torch.bool)
            one_hot.scatter_(1, t.view(-1, 1), True)
            target_log = logits[torch.arange(n, device=logits.device), t]
            max_other = logits.masked_fill(one_hot, float("-inf")).max(dim=1).values
            diff = max_other - target_log
            return F.relu(diff + margin) - margin


    def _is_success(self, preds: torch.Tensor, true_label: int, target_label: int = -1) -> torch.Tensor:
        if target_label is not None and target_label >= 0:
            return preds == int(target_label)
        return preds != int(true_label)


    def _pgd_linf_on_normalized(
        self,
        model: nn.Module,
        x_norm: torch.Tensor,
        y: torch.Tensor,
        *,
        eps: float,
        alpha: float,
        steps: int,
        mean,
        std,
        batch_size: int | None = None,
        amp: bool = False,
    ):
        """Simple PGD($\ell_\infty$) in *pixel* budget eps, operating on normalized tensors."""
        if steps <= 0 or eps <= 0:
            return x_norm.detach()

        if batch_size is not None and int(batch_size) <= 0:
            batch_size = None

        # Chunk PGD to avoid OOM (PGD needs backward through the classifier).
        if batch_size is not None and x_norm.shape[0] > int(batch_size):
            outs = []
            start = 0
            bs = int(batch_size)
            while start < x_norm.shape[0]:
                end = min(start + bs, x_norm.shape[0])
                try:
                    outs.append(
                        self._pgd_linf_on_normalized(
                            model,
                            x_norm[start:end],
                            y[start:end],
                            eps=eps,
                            alpha=alpha,
                            steps=steps,
                            mean=mean,
                            std=std,
                            batch_size=None,
                            amp=amp,
                        )
                    )
                    start = end
                except torch.cuda.OutOfMemoryError:
                    if not torch.cuda.is_available() or bs <= 1:
                        raise
                    torch.cuda.empty_cache()
                    bs = max(1, bs // 2)
            return torch.cat(outs, dim=0).detach()

        device = x_norm.device
        std_t = torch.tensor(std, device=device, dtype=x_norm.dtype).view(1, -1, 1, 1)
        eps_norm = eps / std_t
        alpha_norm = alpha / std_t

        delta = torch.empty_like(x_norm).uniform_(-1.0, 1.0) * eps_norm
        delta = delta.detach()

        for _ in range(steps):
            delta.requires_grad_(True)
            with self._autocast_ctx(str(device), bool(amp)):
                logits = model(x_norm + delta)
                loss = F.cross_entropy(logits, y)

            grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]
            delta = (delta + alpha_norm * grad.sign()).detach()
            # clamp supports tensor bounds inconsistently across torch versions; do it manually.
            delta = torch.max(torch.min(delta, eps_norm), -eps_norm).detach()

            # Project back into valid pixel bounds [0,1]
            x_adv_01 = self._to_01_from_norm(x_norm + delta, mean, std)
            x_adv_norm = utils.normalize_image(x_adv_01, mean, std)
            delta = (x_adv_norm - x_norm).detach()

        return (x_norm + delta).detach()

    def _adv_loss_train(
        self,
        y_01: torch.Tensor,          # (N,C,H,W) in [0,1]
        labels: torch.Tensor,        # (N,)
        surrogate_models: list[nn.Module],
        *,
        targeted: bool = False,
        margin: float = 5.0,
        random_margin: bool = False,
    ):
        """
        Equivalent to original adv_loss_train but:
        - vectorized
        - single model
        - works inside new pipeline
        """
        total_loss = 0

        for surrogate in surrogate_models:

            logits = surrogate(y_01)
            n, c = logits.shape

            if random_margin:
                margin_val = torch.randint(
                    low=0,
                    high=int(margin),
                    size=(1,),
                    device=logits.device
                ).item()
            else:
                margin_val = margin

            labels = labels.view(-1).long()

            one_hot = torch.zeros((n, c), device=logits.device, dtype=torch.bool)

            if not targeted:
                # untargeted: true_logit - max_other
                one_hot.scatter_(1, labels.view(-1, 1), True)
                true_log = logits[torch.arange(n, device=logits.device), labels]
                max_other = logits.masked_fill(one_hot, float("-inf")).max(dim=1).values
                diff = true_log - max_other
            else:
                # targeted: max_other - true_logit
                one_hot.scatter_(1, labels.view(-1, 1), True)
                true_log = logits[torch.arange(n, device=logits.device), labels]
                max_other = logits.masked_fill(one_hot, float("-inf")).max(dim=1).values
                diff = max_other - true_log

            loss = F.relu(diff + margin_val) - margin_val
            total_loss += loss.mean()
        
        total_loss /= len(surrogate_models)
        return total_loss
    
    def _train_autoencoders(
        self,
        *,
        surrogate_models: list[nn.Module],
        train_loader,
        mean,
        std,
        device: str,
        ae_device: str | None = None,
        checkpoint_path: str,
        epochs: int = 1,
        lr: float = 1e-4,
        batch_size: int = 64,
        ae_input_size: int = 224,
        pgd_eps: float = 8 / 255,
        pgd_alpha: float = 2 / 255,
        pgd_steps: int = 3,
        pgd_batch_size: int | None = None,
        train_fraction: float = 1.0,
        amp: bool = True,
    ):
        """Trains the two AEs (clean + adv) and saves a combined checkpoint.

        Notes:
        - This is a lightweight adaptation to your pipeline: it uses your *target model* to
        generate adversarial samples for training the "adv" AE.
        - Images are resized to `ae_input_size` so we can reuse the upstream AE architecture.
        """
        from autoencoder import Autoencoder

        if epochs <= 0:
            raise ValueError("epochs must be >= 1")

        # Support quick smoke tests by training on only a subset of the dataset.
        # Accepted forms:
        # - fraction in (0,1]: 0.1 => 10%, 0.01 => 1%
        # - percentage in (1,100]: 10 => 10%, 1 => 1%
        frac_in = float(train_fraction)
        if frac_in <= 0:
            raise ValueError("train_fraction must be > 0")
        if frac_in > 1.0:
            if frac_in <= 100.0:
                frac = frac_in / 100.0
            else:
                raise ValueError("train_fraction > 100 is invalid")
        else:
            frac = frac_in

        # DataLoader coming from your datasetLoader uses batch_size=1; rebuild with a configurable batch.
        try:
            base_dataset = train_loader.dataset
            if frac < 1.0:
                total = len(base_dataset)
                keep = max(1, int(total * frac))
                perm = torch.randperm(total)[:keep].tolist()
                base_dataset = torch.utils.data.Subset(base_dataset, perm)
                print(f"[DiffAttack] AE training subset: {keep}/{total} samples ({frac*100:.2f}%).")

            train_loader = torch.utils.data.DataLoader(
                base_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
            )
        except Exception:
            # Fallback: keep original loader
            pass

        model_device = device
        if ae_device is None:
            ae_device = model_device

        model_clean = Autoencoder().to(ae_device)
        model_adv = Autoencoder().to(ae_device)

        optimizer_clean = torch.optim.Adam(model_clean.parameters(), lr=lr, weight_decay=1e-5)
        optimizer_adv = torch.optim.Adam(model_adv.parameters(), lr=lr, weight_decay=1e-5)
        mse = nn.MSELoss()

        #use the first surrogate model for PGD sample generation
        surrogate_model = surrogate_models[0]  
        surrogate_model = surrogate_model.to(model_device)
        surrogate_model.eval()
        # PGD needs gradients w.r.t input, but we never need parameter gradients.
        # Freezing parameters saves a lot of memory during PGD backward.
        prev_req_grads = []
        try:
            for p in surrogate_model.parameters():
                prev_req_grads.append(p.requires_grad)
                p.requires_grad_(False)
        except Exception:
            prev_req_grads = None

        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda", enabled=(amp and str(ae_device).startswith("cuda")))
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=(amp and str(ae_device).startswith("cuda")))

        # Persist an initial checkpoint immediately so a partial/aborted run still leaves
        # a valid file for subsequent runs.
        self._save_ae_checkpoint(
            checkpoint_path,
            model_clean,
            model_adv,
            ae_input_size=ae_input_size,
            mean=mean,
            std=std,
        )

        try:
            for epoch in range(epochs):
                model_clean.train()
                model_adv.train()

                running = 0.0
                seen = 0

                for batch_norm, labels in train_loader:
                    batch_norm = batch_norm.to(model_device)
                    labels = labels.to(model_device).long()

                    # Generate adversarial examples in normalized space (same space model expects)
                    # Use chunked PGD to avoid OOM on modest GPUs.
                    batch_adv_norm = self._pgd_linf_on_normalized(
                        surrogate_model,
                        batch_norm,
                        labels,
                        eps=pgd_eps,
                        alpha=pgd_alpha,
                        steps=pgd_steps,
                        mean=mean,
                        std=std,
                        batch_size=pgd_batch_size,
                        amp=bool(amp),
                    )

                    # Convert to pixel space and then to AE space [-1,1]; resize for AE architecture.
                    batch_01 = self._to_01_from_norm(batch_norm, mean, std)
                    batch_adv_01 = self._to_01_from_norm(batch_adv_norm, mean, std)

                    batch_01 = self._resize_bchw(batch_01, ae_input_size)
                    batch_adv_01 = self._resize_bchw(batch_adv_01, ae_input_size)

                    batch_ae = batch_01 * 2.0 - 1.0
                    batch_adv_ae = batch_adv_01 * 2.0 - 1.0

                    batch_ae = batch_ae.to(ae_device)
                    batch_adv_ae = batch_adv_ae.to(ae_device)

                    with self._autocast_ctx(str(ae_device), bool(amp)):
                        output_clean, *z_clean = model_clean(batch_ae)
                        output_adv, *z_adv = model_adv(batch_adv_ae)

                    # Cross-decode (swap semantics/visual streams as in upstream code)
                    # z_* layout from Autoencoder: vis0, vis, z2_vis, z3_vis, z4_vis, sem0, sem, z2_sem, z3_sem, z4_sem
                    zc_vis = z_clean[0:5]
                    zc_sem = z_clean[5:10]
                    za_vis = z_adv[0:5]
                    za_sem = z_adv[5:10]

                    with self._autocast_ctx(str(ae_device), bool(amp)):
                        out_inter1 = model_adv.decode(*zc_vis, *za_sem)
                        out_inter2 = model_clean.decode(*za_vis, *zc_sem)

                        loss_recon = mse(output_clean, batch_ae) + mse(output_adv, batch_adv_ae)
                        loss_cross = mse(out_inter1, batch_ae) + mse(out_inter2, batch_adv_ae)

                        # Convert cross outputs from [-1,1] → [0,1] for classifier
                        out_inter1_01 = torch.clamp(out_inter1 * 0.5 + 0.5, 0.0, 1.0)
                        out_inter2_01 = torch.clamp(out_inter2 * 0.5 + 0.5, 0.0, 1.0)

                        # Resize back to classifier input size if needed
                        if out_inter1_01.shape[-1] != batch_norm.shape[-1]:
                            out_inter1_01 = F.interpolate(
                                out_inter1_01,
                                size=batch_norm.shape[-2:],
                                mode="bilinear",
                                align_corners=False
                            )
                            out_inter2_01 = F.interpolate(
                                out_inter2_01,
                                size=batch_norm.shape[-2:],
                                mode="bilinear",
                                align_corners=False
                            )

                        # Compute adversarial semantic alignment losses
                        loss4 = self._adv_loss_train(
                            out_inter1_01,
                            labels,
                            surrogate_models,
                            targeted=False,
                            margin=float(self.args.get("innermargin", 5.0)),
                            random_margin=False,
                        )

                        loss5 = self._adv_loss_train(
                            out_inter2_01,
                            labels,
                            surrogate_models,
                            targeted=True,
                            margin=float(self.args.get("innermargin", 5.0)),
                            random_margin=False,
                        )

                        loss = loss_recon + loss_cross + loss4 + loss5
                    
                    optimizer_clean.zero_grad(set_to_none=True)
                    optimizer_adv.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer_clean)
                    scaler.step(optimizer_adv)
                    scaler.update()

                    # Avoid empty_cache every step (very slow); only do it if we are on CUDA.
                    if torch.cuda.is_available() and (seen % max(batch_size * 20, 1) == 0):
                        torch.cuda.empty_cache()

                    running += float(loss.detach().item()) * batch_norm.size(0)
                    seen += batch_norm.size(0)

                avg = running / max(seen, 1)
                print(f"[DiffAttack] AE training epoch {epoch+1}/{epochs} | avg loss: {avg:.6f}")

                self._save_ae_checkpoint(
                    checkpoint_path,
                    model_clean,
                    model_adv,
                    ae_input_size=ae_input_size,
                    mean=mean,
                    std=std,
                )
        except Exception:
            # Best effort: keep latest available weights for future non-training runs.
            self._save_ae_checkpoint(
                checkpoint_path,
                model_clean,
                model_adv,
                ae_input_size=ae_input_size,
                mean=mean,
                std=std,
            )
            raise

        # Restore requires_grad flags if we changed them.
        if prev_req_grads is not None:
            try:
                for p, rg in zip(surrogate_model.parameters(), prev_req_grads):
                    p.requires_grad_(rg)
            except Exception:
                pass
        return model_clean.eval(), model_adv.eval()


    def _load_or_train_autoencoders(self, surrogate_models, train_loader, mean, std, args: dict):
        dataset_id = self._infer_dataset_id(args.get("dataset", ""))
        ckpt_path = self._ae_checkpoint_path(dataset_id)

        ae_device = args.get("ae_device", args.get("device"))

        if bool(args.get("train_autoencoders", False)):
            print(f"[DiffAttack] Training AEs (dataset={dataset_id}) -> {ckpt_path}")
            return self._train_autoencoders(
                surrogate_models=surrogate_models,
                train_loader=train_loader,
                mean=mean,
                std=std,
                device=args["device"],
                ae_device=ae_device,
                checkpoint_path=ckpt_path,
                epochs=int(args.get("ae_epochs", 1)),
                lr=float(args.get("ae_lr", 1e-4)),
                batch_size=int(args.get("ae_batch_size", 64)),
                ae_input_size=int(args.get("ae_input_size", 224)),
                pgd_eps=float(args.get("ae_pgd_eps", 8 / 255)),
                pgd_alpha=float(args.get("ae_pgd_alpha", 2 / 255)),
                pgd_steps=int(args.get("ae_pgd_steps", 3)),
                pgd_batch_size=(None if args.get("ae_pgd_batch_size") is None else int(args.get("ae_pgd_batch_size"))),
                train_fraction=float(args.get("ae_train_fraction", 1.0)),
                amp=bool(args.get("amp", True)),
            )

        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"[DiffAttack] Missing AE checkpoint for dataset '{dataset_id}'. "
                f"Set DiffAttack.train_autoencoders=true in configs/attacks_config.yaml to train it, "
                f"or run DiffAttack once with --train-autoencoders. Expected: {ckpt_path}"
            )

        from autoencoder import Autoencoder
        print(f"[DiffAttack] Loading existing AEs (dataset={dataset_id}) <- {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=ae_device)
        model_clean = Autoencoder().to(ae_device).eval()
        model_adv = Autoencoder().to(ae_device).eval()
        model_clean.load_state_dict(ckpt["state_dict_clean"])
        model_adv.load_state_dict(ckpt["state_dict_adv"])
        return model_clean, model_adv

    def perturb(self, model, images, labels, args, mean, std):
        if images is None or labels is None:
            raise ValueError("images and labels are required")
        if images.ndim != 4:
            raise ValueError("images must be a 4D tensor (N,C,H,W)")

        device = args.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        ae_device = args.get("ae_device", device)
        model = model.to(device).eval()

        #surrogate_model = utils.load_surrogate_from_ckpt(args.get('surrogate_model', 'DenseNet121')) if args.get('surrogate_model') else None
        #surrogate_model.eval()
        surrogate_models_arg = args.get('surrogate_models', '')
        if isinstance(surrogate_models_arg, list):
            surrogate_models_arg = ' '.join(surrogate_models_arg)
        surrogate_models_names = [s.strip() for s in str(surrogate_models_arg).split(',') if s.strip()]
        surrogate_models = [utils.load_surrogate_from_ckpt(name) for name in surrogate_models_names]

        # Load/ensure the autoencoders.
        if args.get("train_loader") is None:
            raise ValueError("DiffAttack requires args['train_loader'] to train/load AEs")
        model_clean, model_adv = self._load_or_train_autoencoders(surrogate_models, args.get("train_loader"), mean, std, args)
        model_adv.eval()

        q_budget = int(args.get("q_budget", args.get("max_queries", 250)))
        npop = int(args.get("npop", 10))
        pop_chunk = int(args.get("pop_chunk", min(npop, 2)))
        sigma = float(args.get("sigma", 0.1))
        sigma_f = float(args.get("sigma_f", 0.1))
        lr = float(args.get("lr", 0.01))
        linf_eps = float(args.get("eps", args.get("linf_eps", 8 / 255)))
        inner_margin = float(args.get("innermargin", 5.0))
        target_label = int(args.get("target_label", -1))
        targeted = bool(args.get("targeted", False))
        if not targeted:
            target_label = -1

        ae_input_size = int(args.get("ae_input_size", 224))
        amp = bool(args.get("amp", True))

        n = int(images.shape[0])
        adv_images = images.clone().detach()
        adv_pred_labels = torch.empty((n,), device=device, dtype=torch.long)
        queries = [0 for _ in range(n)]
        l2s = [0.0 for _ in range(n)]

        # Precompute original pixel images (0..1)
        with torch.no_grad():
            orig_01_all = self._to_01_from_norm(x_norm=images.to(device), mean=mean, std=std)

        for i in range(n):
            x_norm = images[i : i + 1].to(device)
            y = labels[i : i + 1].to(device).long()
            y_int = int(y.item())

            x_01 = orig_01_all[i : i + 1]
            h0, w0 = int(x_01.shape[-2]), int(x_01.shape[-1])

            # AE operates on resized image and [-1,1] space
            x_01_ae_m = self._resize_bchw(x_01, ae_input_size)
            x_01_ae = x_01_ae_m.to(ae_device)
            x_ae_ae = x_01_ae * 2.0 - 1.0

            # Clamp bounds in AE pixel-space (on ae_device)
            lower = torch.clamp(x_01_ae - linf_eps, 0.0, 1.0)
            upper = torch.clamp(x_01_ae + linf_eps, 0.0, 1.0)

            # Baseline prediction
            with torch.no_grad():
                base_pred = model(x_norm).argmax(dim=1)
            adv_pred_labels[i] = base_pred

            # If already success (only relevant for untargeted if model misclassifies)
            if bool(self._is_success(base_pred, y_int, target_label).item()):
                queries[i] = 0
                l2s[i] = 0.0
                continue

            #Starting point with the clean AE, since we consider only open-set
            with torch.no_grad():
                with self._autocast_ctx(str(ae_device), bool(amp)):
                    _, z_vis0, z_vis, z2_vis, z3_vis, z4_vis, _, _, _, _, _ = model_clean(x_ae_ae)

            mu = (sigma * torch.randn_like(x_ae_ae)).detach()
            used = 0
            best_adv_norm = x_norm.detach()
            best_pred = base_pred.detach()
            best_l2 = float("inf")

            while used + npop <= q_budget:
                mu_z = torch.randn((npop, 3, ae_input_size, ae_input_size), device=ae_device, dtype=x_ae_ae.dtype)
                rewards = torch.empty((npop,), device=device, dtype=torch.float32)

                found = False
                # Evaluate population in chunks to reduce peak memory.
                for start in range(0, npop, pop_chunk):
                    end = min(start + pop_chunk, npop)
                    chunk = end - start
                    mu_z_c = mu_z[start:end]
                    modify = mu.repeat(chunk, 1, 1, 1) + sigma_f * mu_z_c
                    batch_perturb = x_ae_ae.repeat(chunk, 1, 1, 1) + modify

                    with torch.no_grad():
                        with self._autocast_ctx(str(ae_device), bool(amp)):
                            out = model_adv(batch_perturb)
                            z_p_sem0, z_p_sem, z2_p_sem, z3_p_sem, z4_p_sem = out[6], out[7], out[8], out[9], out[10]

                            decoded_ae = model_adv.decode(
                                z_vis0.repeat(chunk, 1, 1, 1),
                                z_vis.repeat(chunk, 1, 1, 1),
                                z2_vis.repeat(chunk, 1, 1, 1),
                                z3_vis.repeat(chunk, 1, 1, 1),
                                z4_vis.repeat(chunk, 1, 1, 1),
                                z_p_sem0,
                                z_p_sem,
                                z2_p_sem,
                                z3_p_sem,
                                z4_p_sem,
                            )

                            cand_01_ae = self._decode_ae_to_01(decoded_ae)
                            if linf_eps > 0:
                                cand_01_ae = torch.max(torch.min(cand_01_ae, upper), lower)

                            cand_01 = self._resize_bchw(cand_01_ae, h0) if (h0 != ae_input_size or w0 != ae_input_size) else cand_01_ae
                            if cand_01.shape[-1] != w0:
                                cand_01 = F.interpolate(cand_01, size=(h0, w0), mode="bilinear", align_corners=False)

                            cand_01_m = cand_01.to(device)
                            cand_norm = self._normalize_bchw(cand_01_m, mean, std)
                            logits = model(cand_norm)
                            preds = logits.argmax(dim=1)

                    # Compute reward on fp32 logits
                    losses_c = self._margin_losses(logits.float(), y.repeat(chunk), target_label=target_label, margin=inner_margin)
                    rewards[start:end] = (-losses_c).detach()

                    success_mask = self._is_success(preds, y_int, target_label)
                    if success_mask.any():
                        succ_idx = torch.where(success_mask)[0]
                        succ_idx_cand = succ_idx.to(cand_01.device)
                        cand_01_succ = cand_01.index_select(0, succ_idx_cand)
                        diffs = (cand_01_succ.to(device) - x_01.to(device).repeat(len(succ_idx), 1, 1, 1)).flatten(start_dim=1)
                        succ_l2 = torch.norm(diffs, dim=1)
                        min_pos = int(torch.argmin(succ_l2).item())
                        chosen_local = int(succ_idx[min_pos].item())

                        chosen_l2 = float(succ_l2[min_pos].item())
                        if chosen_l2 < best_l2:
                            best_l2 = chosen_l2
                            best_adv_norm = cand_norm[chosen_local : chosen_local + 1].detach()
                            best_pred = preds[chosen_local : chosen_local + 1].detach()
                        found = True
                        break

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                used += npop
                if found:
                    break

                # NES-style update
                reward_std = torch.std(rewards) + 1e-10
                a = (rewards - torch.mean(rewards)) / reward_std
                a_ae = a.to(ae_device)
                mu_update = (a_ae.view(npop, 1, 1, 1) * mu_z).sum(dim=0, keepdim=True)
                mu = (mu + (lr / (npop * sigma_f)) * mu_update).detach()

            # Save per-sample outputs
            adv_images[i : i + 1] = best_adv_norm
            adv_pred_labels[i] = best_pred
            queries[i] = int(used)
            l2s[i] = 0.0 if best_l2 == float("inf") else float(best_l2)

            # Free up cached activations
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return adv_images.detach(), adv_pred_labels.detach().cpu().numpy(), queries, l2s

def execute_attack(model, images, labels, args, mean, std):
    attack = DiffAttack(model, args, mean, std)
    adversarial_images, adversarial_labels, all_queries, l2s = attack.perturb(model, images, labels, args, mean, std)
    # Derive basic success/max-perturbation for statistics.
    adv_pred_t = torch.tensor(adversarial_labels)
    if bool(args.get('targeted', False)) and int(args.get('target_label', -1)) >= 0:
        total_success = int((adv_pred_t == int(args.get('target_label'))).sum().item())
    else:
        total_success = int((adv_pred_t != labels.detach().cpu()).sum().item())

    with torch.no_grad():
        adv_01 = attack._to_01_from_norm(adversarial_images.to(args['device']), mean, std)
        orig_01 =attack._to_01_from_norm(images.to(args['device']), mean, std)
        maxp_val = float((adv_01 - orig_01).abs().flatten(start_dim=1).max(dim=1).values.max().item())

    # Denormalize adversarial images for saving and statistics
    try:
        if isinstance(adversarial_images, torch.Tensor):
            denorm_imgs = utils.remove_normalization(adversarial_images.clone().cpu(), mean, std)
            perturbed_images = denorm_imgs.cpu().numpy()
        else:
            adv_np = np.asarray(adversarial_images)
            try:
                adv_tensor = torch.from_numpy(adv_np)
            except Exception:
                adv_tensor = torch.tensor(adv_np)
            denorm_imgs = utils.remove_normalization(adv_tensor.clone().cpu(), mean, std)
            perturbed_images = denorm_imgs.cpu().numpy()
    except Exception:
        perturbed_images = np.asarray(adversarial_images)
    
    perturbed_labels = np.asarray(adversarial_labels)
    
    return {
        'attack_name': 'DiffAttack',
        'perturbed_image': perturbed_images,
        'perturbed_label': perturbed_labels,
        'total_queries': all_queries,
        'l2': l2s,
        'success': total_success,
        'max_perturbation': maxp_val
    }


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', type=str)
        parser.add_argument(
            '--surrogate_models', '--surrogate-models',
            dest='surrogate_models',
            type=str,
            nargs='+',
        )
        parser.add_argument('--dataset', type=str)
        parser.add_argument('--total-images', type=int)
        parser.add_argument('--results-path', type=str)
        parser.add_argument('--log', action='store_true')
        parser.add_argument('--q-budget', type=int, default=250)

        # Optional: train the two AEs on the current dataset's train split.
        parser.add_argument('--train-autoencoders', action='store_true', dest='train_autoencoders')
        parser.add_argument('--ae-epochs', type=int, default=None)
        parser.add_argument('--ae-batch-size', type=int, default=None)
        parser.add_argument('--ae-lr', type=float, default=None)
        parser.add_argument('--ae-input-size', type=int, default=None)
        parser.add_argument('--ae-pgd-eps', type=float, default=None)
        parser.add_argument('--ae-pgd-alpha', type=float, default=None)
        parser.add_argument('--ae-pgd-steps', type=int, default=None)
        parser.add_argument('--ae-pgd-batch-size', type=int, default=None)
        parser.add_argument('--ae-train-fraction', type=float, default=None)

        # DiffAttack black-box parameters
        parser.add_argument('--targeted', action='store_true')
        parser.add_argument('--target-label', type=int, default=None)
        parser.add_argument('--npop', type=int, default=None)
        parser.add_argument('--pop-chunk', type=int, default=None)
        parser.add_argument('--sigma', type=float, default=None)
        parser.add_argument('--sigma-f', type=float, default=None, dest='sigma_f')
        parser.add_argument('--lr', type=float, default=None)
        parser.add_argument('--eps', type=float, default=None)
        parser.add_argument('--innermargin', type=float, default=None)

        parser.add_argument('--ae-device', type=str, default=None)

        parsed_args, unknown = parser.parse_known_args()
        ini_args = vars(parsed_args)
        ini_args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

        args = utils.prepare_attack_arguments(ini_args, ini_args.get('attack_config', ''), 'diffattack')


        if not os.path.isdir(args['results_path']):
            os.makedirs(args['results_path'])

        logging.basicConfig(filename=os.path.join(args['results_path'], 'DiffAttack.log'), format='%(asctime)s | %(levelname)s | %(message)s', datefmt= '%m-%d-%Y %H:%M:%S', level=logging.INFO)

        ini_stdout = sys.stdout
        sys.stdout = utils.StdoutToLogging()

        print("Loading model")
        # load target model
        model = None
        classified_labels = None
        if args.get('model'):
            with open(args['model'], 'rb') as f:
                data = dill.loads(f.read())
                model = data['model']
                classified_labels = data.get('classified_labels', None)

        if model is None:
            raise ValueError('Target model not provided or could not be loaded')

        model = model.to(args['device'])
        model.eval()

        # load dataset
        spec = importlib.util.spec_from_file_location('dataset', args['dataset'])
        dataset = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dataset)
        testloader, trainloader = dataset.dataLoader()

        print("loading images")
        images, labels = utils.get_images_labels_from_dataLoader(testloader, args['device'], args['total_images'])
        images, labels, classified_labels, _, orig_indices = utils.filter_correctly_classified(
            images, labels, classified_labels
        )
        if images is None or (torch.is_tensor(images) and images.shape[0] == 0):
            print("No correctly classified samples found; skipping DiffAttack.")
            diff_attack_dict = {
                'attack_name': 'DiffAttack',
                'perturbed_image': [],
                'perturbed_label': [],
                'total_queries': [],
                'l2': [],
                'success': 0,
                'max_perturbation': 0.0,
                'orig_index': []
            }
            utils.save_statistics_of_attack(diff_attack_dict, labels, classified_labels, args['results_path'])
            utils.save_images_attack(diff_attack_dict, labels, os.path.join(args['results_path'], 'perturbed_images'))
            sys.stdout = ini_stdout
            raise SystemExit(0)
        print("Normalizing images")
        images = utils.normalize_image(images, dataset.MEAN, dataset.STD)

        # Provide train loader to execute_attack so it can optionally train AEs.
        args['train_loader'] = trainloader

        diff_attack_dict = execute_attack(
            model,
            images,
            labels,
            args,
            dataset.MEAN,
            dataset.STD
        )

        diff_attack_dict['orig_index'] = orig_indices

        utils.save_statistics_of_attack(diff_attack_dict, labels, classified_labels, args['results_path'])
        utils.save_images_attack(diff_attack_dict, labels, os.path.join(args['results_path'], 'perturbed_images'))
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Attack execution failed: {e}")
        raise Exception(f"Attack execution failed: {e}")
    finally:
        try:
            sys.stdout = ini_stdout
        except Exception:
            pass
