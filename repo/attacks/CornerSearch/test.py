import sys
import os

# get repo path for imports
abs_path = os.path.abspath(os.path.realpath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.dirname(abs_path)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

import random

import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch import argmax

from attacks.CornerSearch.CornerSearch import CSattack
from utils.func_utils import normalize_image, calculate_l2_norm
from models.test.ResNet import ResNet18
import utils.static_vars as static

if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset_name = 'cifar'
    net = ResNet18()
    net_name = 'ResNet18'
    net.to(device)
    net = torch.nn.DataParallel(net)
    net.eval()

    print('==> Preparing data..')

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    testset = torchvision.datasets.CIFAR10(
        root = static.PATH_DATASET + '\\', train=False, download=False,
        transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=10, shuffle=False, num_workers=1)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    images = torch.tensor([], device=device)
    labels = torch.tensor([], device=device, dtype=torch.int64)
    num_corr_labels = 0
    for i, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        images_batch = inputs
        labels_batch = targets
        inputs_normalized = images_batch.clone()
        #output = net(apply_normalization(inputs_normalized, dataset_name))
        #predicted_labels = argmax(output, dim=1)
        #correctly_classified_labels = torch.where(predicted_labels == labels_batch)
        #num_corr_labels += len(correctly_classified_labels[0])
        #images = torch.cat((images, images_batch[correctly_classified_labels]), 0)
        #labels = torch.cat((labels, labels_batch[correctly_classified_labels]), 0)

    mean = static.CIFAR_MEAN
    std = static.CIFAR_STD

    args = {'type_attack': "L0+Linf",
            'n_iter': 1000,
            'n_max': 100,
            'kappa': -1,
            'epsilon': -1,
            'sparsity': 10,
            'size_incr': 1,
            'log': True}
    args.update({'device': device})

    attack = CSattack(args)

    images_batch = images[0:10]
    labels_batch = labels[0:10]
    images_batch_np = np.transpose(images_batch.cpu().detach().numpy(), (0, 2, 3, 1))


    adv, pixels_changed, fl_success, total_queries = attack.perturb(x_nat=images_batch_np, y_nat=labels_batch.cpu().numpy(), \
                                                                    n_classes=10, model=net, mean=mean, std=std)

    images_perturbed = adv.copy()
    images_perturbed = torch.from_numpy(np.transpose(images_perturbed, (0, 3, 1, 2)))
    images_perturbed_normalized = normalize_image(images_perturbed, mean, std)

    l2_norms = []
    for ii in range(len(images_perturbed)):
        l2_norms.append(calculate_l2_norm(images_perturbed.cpu().detach().numpy()[ii],
                                          images_batch.cpu().detach().numpy()[ii]))

    print("L2: {}".format(l2_norms))

    pred = net(images_perturbed_normalized).argmax(dim=1).cpu()
           
    print('Original labels: {}\nPerturbed labels: {}\nfl_success: {}\npixels_changed: {}\ntotal queries: {}'.format(labels_batch, pred, fl_success, pixels_changed, total_queries))

    #print('attack successful: {:.2f}%'.format((1.0 - np.mean(fl_success))*100.0))
    #print('Robust accuracy at {} pixels: {:.2f}%'.format(self.k, np.sum(corr_pred) / x_nat.shape[0] * 100.0))
    #print('Maximum perturbation size: {:.5f}'.format(np.amax(np.abs(adv - x_nat))))
    #print("="*100)
