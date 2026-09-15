from .base import GDMOptimizer
from .i_fgsm import IFGSM


class MIFGSM(IFGSM):

    def __init__(self,
                 *,
                 momentum=1,
                 init_grad=0,
                 **kwargs,
                 ) -> None:
        self.momentum = momentum
        self.init_grad = init_grad
        super().__init__(**kwargs)

    def get_optimizer(self, stepsize: float):
        return GDMOptimizer(stepsize, self.momentum, self.distance.normalize, self.init_grad)
