import sys
import os

# get repo path for imports
abs_path = os.path.abspath(os.path.realpath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.dirname(abs_path)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import argparse
import logging
import dill
import importlib

from attacks.Attack import Attack
import utils.func_utils as utils


class DDN(Attack):
    """
    DDN attack: decoupling the direction and norm of the perturbation to achieve a small L2 norm in few steps.

    Parameters
    ----------
    steps : int
        Number of steps for the optimization.
    gamma : float, optional
        Factor by which the norm will be modified. new_norm = norm * (1 + or - gamma).
    init_norm : float, optional
        Initial value for the norm.
    quantize : bool, optional
        If True, the returned adversarials will have quantized values to the specified number of levels.
    levels : int, optional
        Number of levels to use for quantization (e.g. 256 for 8 bit images).
    max_norm : float or None, optional
        If specified, the norms of the perturbations will not be greater than this value which might lower success rate.
    device : torch.device, optional
        Device on which to perform the attack.
    callback : object, optional
        Visdom callback to display various metrics.

    """

    def __init__(self, steps: int, gamma: float = 0.05, init_norm: float = 1., quantize: bool = True, levels: int = 256,
                 max_norm: Optional[float] = None, device: torch.device = torch.device('cpu'), callback = None) -> None:

        super().__init__()

        self.steps = steps
        self.gamma = gamma
        self.init_norm = init_norm

        self.quantize = quantize
        self.levels = levels
        self.max_norm = max_norm

        self.device = device
        self.callback = callback

    def perturb(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor, mean, std,
                targeted: bool = False) -> torch.Tensor:

        """
        Performs the attack of the model for the inputs and labels.

        Parameters
        ----------
        model : nn.Module
            Model to attack.
        inputs : torch.Tensor
            Batch of samples to attack. Values should be in the [0, 1] range. Normalized
        labels : torch.Tensor
            Labels of the samples to attack if untargeted, else labels of targets.
        targeted : bool, optional
            Whether to perform a targeted attack or not.
        mean:
        std:

        Returns
        -------
        torch.Tensor
            Batch of samples modified to be adversarial to the model.

        """

        def predict(model, adv, mean, std):
            image = torch.clone(adv)
            if image.dim() == 3:
                for i in range(image.size(0)):
                    image[i, :, :] = (image[i, :, :] - mean[i]) / std[i]
            else:
                for i in range(image.size(1)):
                    image[:, i, :, :] = (image[:, i, :, :] - mean[i]) / std[i]
            return model(image)

        if inputs.min() < 0 or inputs.max() > 1: raise ValueError('Input values should be in the [0, 1] range.')

        batch_size = inputs.shape[0]
        multiplier = 1 if targeted else -1
        delta = torch.zeros_like(inputs, requires_grad=True)
        norm = torch.full((batch_size,), self.init_norm, device=self.device, dtype=torch.float)
        worst_norm = torch.max(inputs, 1 - inputs).view(batch_size, -1).norm(p=2, dim=1)

        # Setup optimizers
        optimizer = optim.SGD([delta], lr=1)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.steps, eta_min=0.01)

        best_l2 = worst_norm.clone()
        best_delta = torch.zeros_like(inputs)
        adv_found = torch.zeros(inputs.size(0), dtype=torch.bool, device=self.device)

        for i in range(self.steps):

            l2 = delta.data.view(batch_size, -1).norm(p=2, dim=1)
            adv = inputs + delta
            logits = predict(model, adv, mean, std)
            pred_labels = logits.argmax(1)
            ce_loss = F.cross_entropy(logits, labels, reduction='sum')
            loss = multiplier * ce_loss

            is_adv = (pred_labels == labels) if targeted else (pred_labels != labels)
            is_smaller = l2 < best_l2
            is_both = is_adv * is_smaller
            adv_found[is_both] = True
            best_l2[is_both] = l2[is_both]
            best_delta[is_both] = delta.data[is_both]

            optimizer.zero_grad()
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])

            if self.callback:
                cosine = F.cosine_similarity(-delta.grad.view(batch_size, -1),
                                             delta.data.view(batch_size, -1), dim=1).mean().item()
                self.callback.scalar('ce', i, ce_loss.item() / batch_size)
                self.callback.scalars(
                    ['max_norm', 'l2', 'best_l2'], i,
                    [norm.mean().item(), l2.mean().item(),
                     best_l2[adv_found].mean().item() if adv_found.any() else norm.mean().item()]
                )
                self.callback.scalars(['cosine', 'lr', 'success'], i,
                                      [cosine, optimizer.param_groups[0]['lr'], adv_found.float().mean().item()])

            optimizer.step()
            scheduler.step()

            norm.mul_(1 - (2 * is_adv.float() - 1) * self.gamma)
            norm = torch.min(norm, worst_norm)

            delta.data.mul_((norm / delta.data.view(batch_size, -1).norm(2, 1)).view(-1, 1, 1, 1))
            delta.data.add_(inputs)
            if self.quantize:
                delta.data.mul_(self.levels - 1).round_().div_(self.levels - 1)
            delta.data.clamp_(0, 1).sub_(inputs)

        if self.max_norm:
            best_delta.renorm_(p=2, dim=0, maxnorm=self.max_norm)
            if self.quantize:
                best_delta.mul_(self.levels - 1).round_().div_(self.levels - 1)

        return inputs + best_delta


def execute_attack(model, images, args, mean, std):
    total_success = 0

    all_original_images, all_labels, all_adversarial_images, all_adversarial_labels, all_queries, all_l2s = [], [], [], [], [], []

    if 'batch_size' in args and args['batch_size'] > 0:
        b_size = args['batch_size']
    else:
        print("ERROR: Batch size not define")
        exit(1)

    ddnattacker = DDN(steps=args["steps"], device=args['device'], gamma=args["gamma"], init_norm=args["init_norm"],
                    quantize=args["quantize"], levels=args["levels"])

    for i in range(args['total_images'] // b_size):
        images_batch = images[i * b_size:b_size * (i + 1)]
        labels_batch = labels[i * b_size:b_size * (i + 1)]

        perturbed_image = ddnattacker.perturb(model.to(args['device']), inputs=images_batch.to(args['device']),
                                                labels=labels_batch.to(args['device']), targeted=args["targeted"],
                                                mean=mean, std=std)

        pred_ddn = model(perturbed_image).argmax(dim=1).cpu()
        success = (pred_ddn != labels_batch.cpu()).float().mean().item() * 100


        l2_norms = []
        for ii in range(len(perturbed_image)):
            l2_norms.append(utils.calculate_l2_norm(perturbed_image.cpu().detach().numpy()[ii],
                                                images_batch.cpu().detach().numpy()[ii]))

        all_l2s.extend(l2_norms)
        all_original_images.extend(images_batch.cpu().detach().numpy())
        perturbed_image = perturbed_image.cpu().detach().numpy()
        all_adversarial_images.extend(perturbed_image)
        all_labels.extend(labels_batch.cpu())
        all_adversarial_labels.extend(pred_ddn.cpu())
        all_queries.extend([args["steps"]] * len(images_batch.cpu()))

        success = sum((labels_batch.cpu() != pred_ddn.cpu()).float() == 1.)
        total_success += success.item()

        if 'log' in args: print("Images done: {}/{}; Success: {:.2f}%".format(b_size * (i + 1), args['total_images'], success.item() / b_size * 100))

    if 'log' in args: print('DDN done: Success: {:.2f}%'.format(total_success / len(all_l2s) * 100))

    return all_adversarial_images, all_adversarial_labels, all_queries, all_l2s, total_success


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, help='')
    parser.add_argument('--targeted', action='store_true')
    parser.add_argument('--gamma', type=float, help='')
    parser.add_argument('--init-norm', type=float, help='')
    parser.add_argument('--quantize', action='store_true')
    parser.add_argument('--levels', type=int, help='')
    parser.add_argument('--batch-size', type=int, help='')

    parser.add_argument('--model', type=str, default = 'Path for binary file containing the model')
    parser.add_argument('--dataset', type=str, default = 'Path for dataset loader file')
    parser.add_argument('--total-images', type=int, help='')
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--attack-config', type=str, default = '', help='Config file to be passed in instead of arguments')
    parser.add_argument('--results-path', type=str, default = '', help="Path to store results")

    ini_args = vars(parser.parse_args())

    ini_args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    # initial argument verifications
    args = utils.prepare_attack_arguments(ini_args, ini_args['attack_config'], 'ddn')


    try:
        # setup logger
        if not os.path.isdir(args['results_path']):
            os.makedirs(args['results_path'])

        logging.basicConfig(filename=os.path.join(args['results_path'], 'DDN.log'), format='%(asctime)s | %(levelname)s | %(message)s', datefmt= '%m-%d-%Y %H:%M:%S', level=logging.INFO)

        ini_stdout = sys.stdout
        sys.stdout = utils.StdoutToLogging()
        
        print("Loading DDN")
        

        # load model
        with open(args['model'], "rb") as file:
            serialized_data = file.read()
            data = dill.loads(serialized_data)
            #model_name = data['model_name']
            model = data['model']
            classified_labels = data['classified_labels']

        print("Model loaded")

        # load dataset
        spec = importlib.util.spec_from_file_location("dataset", args['dataset'])
        dataset = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dataset)
        testloader, trainloader = dataset.dataLoader()

        images, labels = utils.get_images_labels_from_dataLoader(testloader, args['device'], args['total_images'])
        
        print("Dataset loaded")

        print("Running DDN on {}".format(args['device']))
        
        adversarial_images, adversarial_labels, all_queries, l2s, total_success = execute_attack(model, images, args, dataset.MEAN, dataset.STD)
        
        
        ddn_dict = {
            'attack_name': 'DDN',
            'perturbed_image': adversarial_images,
            'perturbed_label': adversarial_labels,
            'total_queries': all_queries,
            'l2': l2s,
            'success': total_success
        }

        print("Saving Results")

        utils.save_statistics_of_attack(ddn_dict, labels, classified_labels, args['results_path'])

        utils.save_images_attack(ddn_dict, labels, os.path.join(args['results_path'], 'perturbed_images'))

    finally:
        sys.stdout = ini_stdout
