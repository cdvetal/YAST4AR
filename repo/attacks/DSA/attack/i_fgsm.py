from typing import Union
from typing_extensions import Literal
from models.base import Model, ModelList
from utils.distance import linf, get_distance
from .base import BaseGradientDescent


class IFGSM(BaseGradientDescent):
    distance = linf
    random_start = False

    def __init__(self,
                 *,
                 local_models: Model = None,
                 tensorboard: Union[Literal[False], None, str] = False,
                 constraint: Union[Literal["linf"], Literal["l2"]] = "l2",
                 epsilon: float = 4.6,
                 steps=10,
                 alpha: float = None,
                 **kwargs,
                 ) -> None:
        self.local_models = local_models
        self.tensorboard = tensorboard
        self.constraint = constraint
        self.epsilon = epsilon
        self.steps = steps
        self.distance = get_distance(constraint)
        self.alpha = epsilon / steps if alpha is None else alpha
        super().__init__(abs_stepsize=self.alpha, steps=steps, random_start=self.random_start, epsilon=self.epsilon)


class FGSM(IFGSM):
    distance = linf

    def __init__(self, **kwargs):
        super().__init__(steps=1, **kwargs)
