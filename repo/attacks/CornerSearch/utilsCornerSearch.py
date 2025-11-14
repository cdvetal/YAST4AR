

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np

from utils.func_utils import normalize_image

def get_logits(model, x_nat, mean, std, device):
    x = x_nat.copy()
    x = torch.from_numpy(x).permute(0, 3, 1, 2).float()
    x = normalize_image(x, mean, std)
    with torch.no_grad():
        output = model(x.to(device))

    return output.cpu().numpy()

def get_predictions(model, x_nat, y_nat, mean, std, device):
    x = x_nat.copy()
    x = torch.from_numpy(x).permute(0, 3, 1, 2).float()
    x = normalize_image(x, mean, std)
    y = torch.from_numpy(y_nat)
    with torch.no_grad():
        output = model(x.to(device))

    return (output.cpu().max(dim=-1)[1] == y).numpy()


