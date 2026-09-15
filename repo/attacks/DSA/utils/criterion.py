from abc import ABC, abstractmethod
from typing import TypeVar

import torch

T = TypeVar("T", bound=torch.Tensor)


class Criterion(ABC):
    """Abstract base class to implement new criteria."""

    @abstractmethod
    def __repr__(self) -> str:
        ...

    @abstractmethod
    def __call__(self, outputs: T) -> T:
        """Returns a boolean tensor indicating which perturbed inputs are adversarial.

        Args:
            perturbed: Tensor with perturbed inputs ``(batch, ...)``.
            outputs: Tensor with model outputs for the perturbed inputs ``(batch, ...)``.

        Returns:
            A boolean tensor indicating which perturbed inputs are adversarial ``(batch,)``.
        """
        ...


class Misclassification(Criterion):
    """Considers those perturbed inputs adversarial whose predicted class
    differs from the label.

    Args:
        labels: Tensor with labels of the unperturbed inputs ``(batch,)``.
    """

    def __init__(self, labels: T):
        super().__init__()
        self.labels: T = labels

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.labels!r})"

    def __call__(self, outputs: T) -> T:
        classes = outputs.argmax(dim=-1)
        assert classes.shape == self.labels.shape, \
            f"Error: shape is not equal,classes:{classes.shape} while labels:{self.labels.shape}"
        is_adv = classes != self.labels
        return is_adv


class TargetedMisclassification(Criterion):
    """Considers those perturbed inputs adversarial whose predicted class
    matches the target class.

    Args:
        target_classes: Tensor with target classes ``(batch,)``.
    """

    def __init__(self, target_classes: T):
        super().__init__()
        self.target_classes: T = target_classes

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.target_classes!r})"

    def __call__(self, outputs: T) -> T:
        classes = outputs.argmax(dim=-1)
        assert classes.shape == self.target_classes.shape, \
            f"Error: shape is not equal,classes:{classes.shape} while labels:{self.target_classes.shape}"
        is_adv = classes == self.target_classes
        return is_adv
