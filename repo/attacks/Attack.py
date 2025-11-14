import abc


class Attack(abc.ABC):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abc.abstractmethod
    def perturb(self, **kwargs):
        raise NotImplementedError
