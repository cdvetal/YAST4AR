import sys
import os

# get repo path for imports
abs_path = os.path.abspath(os.path.realpath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.dirname(abs_path)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

from attacks.BanditsPrior.BanditsPrior import BanditsPrior
from models.test.ResNet import ResNet18
from utils.func_utils import normalize_image, calculate_l2_norm
import utils.static_vars as static
import torch
from torch.utils import data
from torchvision import models, transforms
import torchvision


def main():
    max_queries = 200
    fd_eta = 0.01
    image_lr = 0.5
    online_lr = 0.1
    mode = "l2"  # [linf|l2]
    exploration = 0.01
    tile_size = 50
    epsilon = 5.0
    batch_size = 10
    log_progress = True
    nes = False
    tiling = False
    gradient_iters = 1
    path_output_folder = "outputs\\BanditsPriorResults"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = ResNet18()
    net.to(device)

    mean = static.CIFAR_MEAN
    std = static.CIFAR_STD


    testset = torchvision.datasets.CIFAR10(
        root= static.PATH_DATASET + '\\', train=False, download=True,
        transform=transforms.ToTensor())

    testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    images, labels = next(iter(testloader))
    images_batch = images.to(device)
    labels_batch = labels.to(device)

    dataset_size = images[0].shape[1]

    print("="*100)
 
    
    banditsprior_attack = BanditsPrior(images_batch.cpu(), labels_batch.cpu(), max_queries, fd_eta, image_lr, online_lr,
                                       mode, exploration, tile_size, epsilon, batch_size, log_progress, nes, tiling,
                                       gradient_iters, net, dataset_size, mean, std, device)

    ncc, average_queries, success_rate, images_orig, images_adv, all_queries, correctly_classified, success = banditsprior_attack.perturb()
    dic = { 'average_queries': ncc, 
            'num_correctly_classified': average_queries, 
            'success_rate': success_rate, 
            'images_orig': images_orig, 
            'images_adv': images_adv, 
            'all_queries': all_queries, 
            'correctly_classified': correctly_classified, 
            'success': success
            }

    print("Function Deprecated. Results not saved")

    images_adv = torch.from_numpy(images_adv)
    images_adv = images_adv.to(device)
    pred_bandits = net(normalize_image(images_adv, mean, std)).argmax(dim=1).cpu()
    success = (pred_bandits.cpu() != labels_batch.cpu()).float().mean().item() * 100

    l2_norms = []
    for ii in range(len(images_adv)):
        l2_norms.append(calculate_l2_norm(images_adv.cpu().detach().numpy()[ii],
                                          images_batch.cpu().detach().numpy()[ii]))

    print('Original labels: {}\nAdversarial labels: {}\nL2: {}'.format(
            labels_batch.cpu(), pred_bandits, l2_norms))



if __name__ == '__main__':
    main()
