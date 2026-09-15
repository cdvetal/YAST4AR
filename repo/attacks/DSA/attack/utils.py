import os
import pickle
from typing import Tuple, TypeVar, Dict, Any, Callable, List

import torch
import torch.nn as nn

from models.base import Model
from utils.criterion import Criterion, Misclassification

T = TypeVar("T", bound=torch.Tensor)


def calc_dist(origin_input: torch.Tensor, target_input: torch.Tensor, norm: str, keepdim: bool = False) -> torch.Tensor:
    ''' Calculating the distance between original input and target input in some norms

    Args:
        origin_input (torch.Tensor): original input
        target_input (torch.Tensor): target input
        norm (Any): order of the norm ['l2', 'linf']
        keepdim (bool): keep dimensions of output or not
    '''
    if norm not in ['l2', 'linf']:
        raise NotImplementedError('Norm must in np.inf, 2')

    batch_size = origin_input.shape[0]
    diff = (origin_input - target_input).reshape(batch_size, -1)

    if norm == 'l2':
        dist = torch.norm(diff, p=2, dim=1, keepdim=keepdim)
    elif norm == 'linf':
        dist = torch.norm(diff, p=float('inf'), dim=1, keepdim=keepdim)
    return dist


@torch.no_grad()
def make_decision(model: nn.Module, origin_input: torch.Tensor, origin_label: torch.Tensor,
                  target_label: torch.Tensor, clip_min: float, clip_max: float, targeted: bool) -> Tuple[
    float, torch.Tensor]:
    ''' Make a decision.
        Decision function output 1 on the desired side of the boundary, 0 otherwise.
    
    Args:
        model (nn.Module): black-box model
        origin_input (torch.Tensor): original input
        origin_label (torch.Tensor): original label
        target_label (torch.Tensor): target label   (in targeted attacks, target_label is not working)
        clip_min (float): minimal number of image
        clip_max (float): maximal number of image
        targeted (bool): targeted attacks or not
    '''
    clipped_input = torch.clamp(origin_input, clip_min, clip_max)
    if hasattr(model, 'detecting') is True and model.detecting is True:
        logit, result = model(clipped_input)
    else:
        logit = model(clipped_input)
    pred = torch.argmax(logit, dim=1)

    if hasattr(model, 'detecting') is True and model.detecting is True:
        if targeted is True:
            return (pred == target_label).float(), pred, result
        return (pred != origin_label).float(), pred, result
    else:
        if targeted is True:
            return (pred == target_label).float(), pred
        return (pred != origin_label).float(), pred


def verify_input_bounds(inputs: T, model: Model) -> None:
    # verify that input to the attack lies within model's input bounds
    assert int(inputs.min().item()) >= int(model.bounds.lower)
    assert int(inputs.max().item()) <= int(model.bounds.upper)


def get_criterion(criterion) -> Criterion:
    if isinstance(criterion, Criterion):
        return criterion
    else:
        return Misclassification(criterion)


def raise_if_kwargs(kwargs: Dict[str, Any]) -> None:
    if kwargs:
        raise TypeError(
            f"attack got an unexpected keyword argument '{next(iter(kwargs.keys()))}'"
        )


def extend_dim(x: T, k: int) -> T:
    shape = x.shape + (1,) * (k - x.ndim)
    return x.reshape(shape)


def get_init_advs(dataset):
    def wrapper(N, is_adversarial: Callable) -> T:
        return dataset.sample(N, is_adversarial)

    return wrapper


def get_random_start(inputs, epsilon, p=2):
    if p == "inf":
        return inputs + torch.zeros_like(inputs).uniform_(-epsilon, epsilon)
    elif p == 2:
        from utils.distance import flatten
        batch_size, n = flatten(inputs).shape
        x = torch.randn(batch_size, n + 1)
        r = torch.norm(x, dim=-1, keepdim=True)
        s = x / r
        b = s[:, :n].reshape(inputs.shape)
        return inputs + epsilon * b.to(inputs.device)
    else:
        raise NotImplementedError(f"random start do not support p = {p}")


atleast_kd = extend_dim
