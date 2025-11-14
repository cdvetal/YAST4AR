import sys
import os

# get repo path for imports
abs_path = os.path.abspath(os.path.realpath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.dirname(abs_path)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

import argparse
import torch
import torchvision
from torchvision import models, transforms
from torch.utils import data
import math

from models.test.ResNet import ResNet18
from attacks.DDN.DDN import DDN
import utils.static_vars as static
import utils.func_utils as utils
from utils.func_utils import calculate_l2_norm


def main(args):
    if "json-config" in args:
        args = utils.prepare_attack_arguments(args, args["json-config"], "DDN")
        steps = args["steps"]
        targeted = args["targeted"]
        gamma = args["gamma"]
        init_norm = args["init-norm"]
        quantize = args["quantize"]
        levels = args["levels"]
        path_output_folder = args["path-output-folder"]

    else:
        steps = args["steps"] if "steps" in args else 1000
        targeted = args["targeted"] if "targeted" in args else True
        gamma = args["gamma"] if "gamma" in args else 0.05
        init_norm = args["init-norm"] if "init-norm" in args else 1.0
        quantize = args["quantize"]if "quantize" in args else True
        levels = args["levels"] if "levels" in args else 256
        path_output_folder = args["path-output-folder"] if "path-output-folder" in args else None

    total_images =  args["total-images"] if "total-images" in args else 32
    batch_size = args["batch-size"] if "batch-size" in args else 16


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ResNet18()
    model.to(device)

    mean = static.CIFAR_MEAN
    std = static.CIFAR_STD

    testset = torchvision.datasets.CIFAR10(
        root= static.PATH_DATASET + '\\', train=False, download=True,
        transform=transforms.ToTensor())

    testloader = data.DataLoader(testset, shuffle=False)

    images, labels = utils.get_images_labels_from_dataLoader(testloader, device, total_images = total_images)


    ddnattacker = DDN(steps=steps, device=device, gamma=gamma, init_norm=init_norm,
                        quantize=quantize, levels=levels)

    
    total_success = 0

    all_original_images = []
    all_labels = []
    all_adversarial_images = []
    all_adversarial_labels = []
    all_queries = []
    all_l2s = []

    print("Executing DDN attack...")
    print("Device: " + device)
    print(steps, targeted, gamma, init_norm, quantize, levels, path_output_folder, total_images, batch_size)


    for i in range(math.ceil(total_images / batch_size)):
        images_batch = images[i * batch_size:batch_size * (i + 1)]
        labels_batch = labels[i * batch_size:batch_size * (i + 1)]

        perturbed_image = ddnattacker.perturb(model.to(device), inputs=images_batch.to(device),
                                                labels=labels_batch.to(device), targeted=targeted,
                                                mean=mean, std=std)

        pred_ddn = model(perturbed_image).argmax(dim=1).cpu()
        success = (pred_ddn != labels_batch.cpu()).float().mean().item() * 100

        if "log" in args:
            print('DDN done: Success: {:.2f}%'.format(success))

        l2_norms = []
        for ii in range(len(perturbed_image)):
            l2_norms.append(calculate_l2_norm(perturbed_image.cpu().detach().numpy()[ii],
                                                images_batch.cpu().detach().numpy()[ii]))

        all_l2s.extend(l2_norms)
        all_original_images.extend(images_batch.cpu().detach().numpy())
        perturbed_image = perturbed_image.cpu().detach().numpy()
        all_adversarial_images.extend(perturbed_image)
        all_labels.extend(labels_batch.cpu())
        all_adversarial_labels.extend(pred_ddn.cpu())
        all_queries.extend([steps] * len(images_batch.cpu()))

        success = sum((labels_batch.cpu() != pred_ddn.cpu()).float() == 1.)
        total_success += success.item()

    ddn_dict = {'attack_name': 'DDN (W)',
                'perturbed_image': all_adversarial_images,
                'perturbed_label': all_adversarial_labels,
                'total_queries': all_queries,
                'l2': all_l2s,
                'success': total_success
                }

    print("attack_name:", ddn_dict["attack_name"])
    print("total_queries:", ddn_dict["total_queries"])
    print("l2:", ddn_dict["l2"])
    print("success:", ddn_dict["success"])

    
    del ddnattacker
    del ddn_dict


if __name__ == '__main__':
    # get all possible parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--total-images', type=int)
    parser.add_argument('--batch-size', type=int, help='batch size for bandits')
    parser.add_argument('--json-config', type=str, help='a config file to be passed in instead of arguments')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--targeted', type=bool)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--init-norm', type=float)
    parser.add_argument('--quantize', type=bool)
    parser.add_argument('--levels', type=int)
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--path-output-folder', type=str, help="Path to output folder")
    args = vars(parser.parse_args())
    args = {k: v for k, v in args.items() if v is not None}
    main(args)
