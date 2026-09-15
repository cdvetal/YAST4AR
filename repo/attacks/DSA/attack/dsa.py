from typing import Any, Optional, Union, Literal

import numpy as np
import torch
from models.base import Model, ModelList
from utils.distance import get_distance, l2, flatten
from utils.result import Result

from .mi_fgsm import MIFGSM
from .base import get_is_adversarial, get_random_start, MinimizationAttack, T
from .utils import raise_if_kwargs, get_criterion


class DSA(MinimizationAttack):
    distance = l2
    wba = MIFGSM()  # white box attack

    def __init__(self,
                 local_models: Model | ModelList | None = None,
                 momentum: float = 1,
                 constraint: Union[Literal["linf"], Literal["l2"]] = "l2",
                 epsilon: float = 1.75,
                 wba_steps: int = 100,
                 bba_steps: int = 100,
                 wba_iters: int = 10,  # white box attack inner steps
                 random_starting: bool = True,
                 budget: int = 4000,
                 **kwargs
                 ) -> None:
        self.momentum = momentum
        self.constraint = constraint
        self.epsilon = epsilon
        self.gamma = 1.0  # hsja gamma
        self.wba_steps = wba_steps
        self.distance = get_distance(constraint)
        self.alpha = epsilon / wba_iters
        self.budget = budget
        # `local_models` can be:
        # - ModelList: used as-is
        # - single Model: wrapped into a list
        # - list/tuple of Models: used as a flat list
        if isinstance(local_models, ModelList):
            self.local_models = local_models
        elif isinstance(local_models, (list, tuple)):
            self.local_models = list(local_models)
        else:
            self.local_models = [local_models]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.random_starting = random_starting
        if self.local_models[0] is None:
            raise ValueError('DSA requires at least one surrogate model (local_models)')
        self.wba.__init__(constraint=self.constraint, epsilon=epsilon, local_models=self.local_models[0])  # reinit
        self.input_transform = AdaptiveChoose([self.random_transform])
        self.model_generator = AdaptiveChoose(self.local_models)

    def run(self,
            model: Model,
            inputs: T,
            criterion: Any,
            *,
            starting_points: Optional[T] = None,
            **kwargs: Any
            ):
        raise_if_kwargs(kwargs)
        originals = inputs.clone()
        criterion = get_criterion(criterion)

        self.is_adversarial = is_adversarial = get_is_adversarial(criterion, model)
        self.result = Result(inputs.shape[0], query_budget=self.budget)
        inputs.requires_grad = True

        ######################
        # local model attack #
        ######################
        x_adv = starting_points.clone()
        cur_dist = self.distance(originals, x_adv)
        candidate_distance = cur_dist
        self.result.update(torch.zeros(len(cur_dist)) == 1, is_adversarial.query, cur_dist)
        for i in range(1, self.wba_steps + 1):
            # candidate distance record
            inputs = originals.clone()
            inputs = self.input_transform(inputs) if self.random_starting else inputs
            inputs = inputs.requires_grad_()

            for _ in range(len(self.local_models)):
                # gen various epsilon
                dist_range = torch.normal(self.epsilon, cur_dist.max().item() - self.epsilon, size=(100,))
                dist_range = dist_range[(dist_range > self.epsilon / 2) & (dist_range < cur_dist.max().item())]
                eps = dist_range[0]
                local_model = self.model_generator.choose_one()

                # reinit white box attack
                self.wba.__init__(constraint=self.constraint, epsilon=eps.item(), local_models=local_model)
                is_adversarial_local = get_is_adversarial(criterion, local_model)

                # gen candidates from local white box attack
                inputs = self.wba.run(local_model, originals, criterion=criterion, starting_points=inputs)

                if is_adversarial_local(inputs).any():
                    # detach and biased inputs
                    candidate_distance = self.distance(originals, inputs)
                    # query target model
                    is_adv_mask = is_adversarial(inputs)
                    # update hist biased grad
                    if bool(is_adv_mask.any().item()):
                        self.model_generator.update_ratio()
                        self.input_transform.update_ratio()
                else:
                    is_adv_mask = torch.zeros_like(cur_dist, dtype=torch.bool)
                update_cond = (candidate_distance < cur_dist) & is_adv_mask
                x_adv[update_cond] = inputs[update_cond].detach()
                cur_dist = self.distance(originals, x_adv)

                #  update result
                self.result.update(cur_dist < self.epsilon, is_adversarial.query, cur_dist)
                if bool(is_adv_mask.any().item()) or self.result.finished:
                    break
            print(f"Dispersed Sampling @step = {i}", self.result)
            if self.result.finished:
                return x_adv

        return x_adv

    def random_transform(self, x: T):
        return get_random_start(x, 0.5 * self.epsilon, self.distance.p)


class AdaptiveChoose:

    def __init__(self, trans_fn, ratio_base=10):
        self.trans_fn = trans_fn
        self.ratio = ratio_base * np.ones(len(trans_fn))
        self.index = 0

    def __call__(self, *args, **kwargs):
        trans_fn = self.choose_one()
        return trans_fn(*args, **kwargs)

    def choose_one(self):
        probability = (self.ratio / self.ratio.sum()).cumsum()
        r = np.random.rand()
        self.index = np.nonzero(probability > r)[0][0]
        return self.trans_fn[self.index]

    def update_ratio(self):
        self.ratio[self.index] += 1
