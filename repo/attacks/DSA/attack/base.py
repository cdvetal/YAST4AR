from abc import abstractmethod, ABC
from typing import Callable, TypeVar, Any, Optional

import numpy as np
import torch
from torch import nn

from models.base import Model, ModelList
from utils.criterion import Criterion, Misclassification
from utils.distance import Distance, flatten
from utils.result import Result
from .utils import get_criterion, raise_if_kwargs

T = TypeVar("T", bound=torch.Tensor)


class Optimizer(ABC):
    @abstractmethod
    def __call__(self, gradient):
        ...


class GDOptimizer(Optimizer):
    def __init__(self, stepsize: float, norm_fn: Callable):
        self.stepsize = stepsize
        self.norm_fn = norm_fn

    def __call__(
            self,
            gradient: T,
    ) -> T:
        return self.stepsize * self.norm_fn(gradient)


class GDMOptimizer(Optimizer):
    def __init__(self, stepsize, momentum, norm_fn, init_grad=0):
        self.stepsize = stepsize
        self.momentum = momentum
        self.global_grad = init_grad
        self.norm_fn = norm_fn

    def __call__(self, gradient):
        self.global_grad = self.global_grad * self.momentum + gradient
        # self.global_grad = self.norm_fn(self.global_grad)
        return self.stepsize * self.norm_fn(self.global_grad)


class BaseGradientDescent(ABC):
    local_models: Model
    optimizer: Optimizer
    result: Result

    def __init__(
            self,
            *,
            abs_stepsize: Optional[float] = None,
            steps: int = 10,
            random_start: bool = False,
            epsilon: float = 16 / 255,
            **kwargs,
    ):
        self.abs_stepsize = abs_stepsize
        self.steps = steps
        self.random_start = random_start
        self.epsilon = epsilon

    def get_loss_fn(
            self, model: Model, labels: T
    ) -> Callable[[T], T]:
        # can be overridden by users
        loss_fn = nn.CrossEntropyLoss()

        def _labels_for_inputs(_labels: T, _inputs: T) -> T:
            """Return class labels aligned with `_inputs` batch size.

            Upstream code sometimes passes a batch of labels already shaped (B,).
            Repeating those by B again creates B^2 targets and breaks CE loss.
            """
            if not isinstance(_labels, torch.Tensor):
                return _labels
            if _labels.ndim == 0 or _labels.numel() == 1:
                return _labels.reshape(1).repeat(_inputs.shape[0])
            # Already a batch of labels
            if _labels.shape[0] == _inputs.shape[0]:
                return _labels
            # Best-effort: if labels is longer (e.g. original batch) take prefix
            if _labels.shape[0] > _inputs.shape[0]:
                return _labels[: _inputs.shape[0]]
            raise ValueError(
                f"Label batch size ({_labels.shape[0]}) does not match input batch size ({_inputs.shape[0]})."
            )

        def loss_cal(inputs: T) -> T:
            logits = model(inputs)
            # return torch.nn.functional.cross_entropy(logits, labels).sum()
            return loss_fn(logits, _labels_for_inputs(labels, inputs))

        def models_loss_cal(inputs: T) -> T:
            loss = model(inputs, _labels_for_inputs(labels, inputs))
            return loss

        if isinstance(model, ModelList):
            return models_loss_cal
        else:
            return loss_cal

    def get_optimizer(self, stepsize: float) -> Optimizer:
        return GDOptimizer(stepsize, norm_fn=self.distance.normalize)

    def value_and_grad(
            # can be overridden by users
            self,
            loss_fn,
            x: T,
    ) -> T:
        loss = loss_fn(x)
        return torch.autograd.grad(loss, x)[0]

    @property
    @abstractmethod
    def distance(self) -> Distance:
        ...

    def run(
            self,
            target_model: Model,
            inputs: T,
            criterion: Criterion,
            *,
            starting_points,
            **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        x0 = inputs.clone()
        criterion_ = get_criterion(criterion)
        self.is_adversarial = is_adversarial = get_is_adversarial(criterion, target_model)  # result criterion
        self.result = Result(inputs.shape[0])
        # the model which generate adversarial examples
        # if local model is None -> white box attack, else -> transfer attack
        models = target_model if self.local_models is None else self.local_models
        # verify_input_bounds(x0, models)
        epsilon = self.epsilon

        # perform a gradient ascent (targeted attack) or descent (untargeted attack)
        if isinstance(criterion_, Misclassification):
            gradient_step_sign = 1.0
            classes = criterion_.labels
        elif hasattr(criterion_, "target_classes"):
            gradient_step_sign = -1.0
            classes = criterion_.target_classes  # type: ignore
        else:
            raise ValueError("unsupported criterion")

        loss_fn = self.get_loss_fn(models, classes)

        stepsize = self.abs_stepsize

        self.optimizer = self.get_optimizer(stepsize)
        x = inputs if starting_points is None else starting_points
        if self.random_start:
            x = self.get_random_start(x, epsilon)
            x = torch.clip(x, *models.bounds)

        x = x.detach().requires_grad_()

        for i in range(self.steps):
            gradients = self.value_and_grad(loss_fn, x)
            gradients = self.grad_normalize(gradients)
            x = x + gradient_step_sign * self.optimizer(gradients)
            x = self.distance.clip_perturbation(x0, x, epsilon)
            x = torch.clip(x, *models.bounds)
        self.result.update(is_adversarial(x), is_adversarial.query, self.distance(x0, x))
        return x

    def get_random_start(self, x0: T, epsilon: float) -> T:
        ...

    def __call__(  # type: ignore
            self,
            model: Model,
            inputs: T,
            criterion: Any,
            starting_points: Optional[T],
            **kwargs: Any,
    ):
        # verify_input_bounds(inputs, model)
        criterion = get_criterion(criterion)
        # run the actual attack
        inputs_clone = inputs.cpu().numpy()

        # run the actual attack
        xp = self.run(model, inputs, criterion, starting_points=starting_points, **kwargs)
        xp = xp.detach().cpu()

        # make sure inputs do not change when attack running
        assert (inputs_clone == inputs.detach().cpu().numpy()).all()

        return xp

    def __repr__(self):
        return self.__class__.__name__

    def grad_normalize(self, gradients):
        return gradients / torch.mean(torch.abs(gradients), dim=(1, 2, 3), keepdim=True)
        # return self.distance.normalize(gradients)


class MinimizationAttack:
    """Minimization attacks try to find adversarials with minimal perturbation sizes"""
    optimizer: Optimizer

    @abstractmethod
    def run(
            self,
            model,
            inputs: T,
            criterion: Any,
            *,
            starting_points: Optional[T] = None,
            **kwargs: Any,
    ) -> T:
        """Runs the attack and returns perturbed inputs.

        The size of the perturbations should be as small as possible such that
        the perturbed inputs are still adversarial. In general, this is not
        guaranteed and the caller has to verify this.
        """
        ...

    @property
    @abstractmethod
    def distance(self) -> Distance:
        ...

    def __repr__(self):
        return self.__class__.__name__

    def __call__(  # type: ignore
            self,
            model: Model,
            inputs: T,
            criterion: Any,
            starting_points: Optional[T],
            **kwargs: Any,
    ):
        # verify_input_bounds(inputs, model)
        criterion = get_criterion(criterion)
        # run the actual attack
        inputs_clone = inputs.cpu().numpy()

        # run the actual attack
        xp = self.run(model, inputs, criterion, starting_points=starting_points, **kwargs)
        xp = xp.detach().cpu()

        # check inputs do not change when attack running
        assert (inputs_clone == inputs.detach().cpu().numpy()).all()
        print("Done!")
        return xp


class IsAdversarial:
    def __init__(self, criterion: Criterion, model):
        self.model = model
        self.criterion = criterion
        self.query = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def __call__(self, perturbed: T, query_increase=True) -> T:
        self.query += 1 if query_increase else 0
        logit = self.model(perturbed)
        outputs = torch.softmax(logit, dim=-1)
        result = self.criterion(outputs)
        return result


def get_is_adversarial(criterion: Criterion, model):
    """
    args:
        criterion: Misclassification (untargeted) or TargetedMisclassification (targeted)
        model: target model
        return: a function to check whether the perturbed image is adversarial
    """
    if isinstance(model, Model):
        is_adversarial = IsAdversarial(criterion, model)
    else:
        raise KeyError
    return is_adversarial


def get_random_start(inputs, epsilon, p=2):
    if p == torch.inf:
        return inputs + torch.zeros_like(inputs).uniform_(-epsilon, epsilon)
    elif p == 2:
        batch_size, n = flatten(inputs).shape
        x = torch.randn(batch_size, n + 1)
        r = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        s = x / r
        b = s[:, :n].reshape(inputs.shape)
        return inputs + epsilon * b.to(inputs.device)
    else:
        raise NotImplementedError(f"random start do not support p = {p}")
