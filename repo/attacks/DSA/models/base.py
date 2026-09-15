import ssl
from typing import TypeVar, Any, Union, Tuple, NamedTuple, Callable

import torch
from torch import nn

EPS = 1e-8
T = TypeVar("T", bound=torch.Tensor)


def get_device(device: Any = None) -> Any:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str):
        return torch.device(device)
    return device


class Bounds(NamedTuple):
    lower: float
    upper: float


class Model(nn.Module):
    def __init__(self, model: nn.Module, bounds: Union[Bounds, Tuple[float, float]] = (0, 1),
                 device: Any = None) -> None:
        super().__init__()
        self.model = model.to(device)
        self.bounds = Bounds(*bounds)
        # dummy=ep.torch.zeros(0,device=device)
        self.preprocess = model.preprocess if hasattr(model, "preprocess") else None
        # Some user models (and DataParallel wrappers) don't define a `.name`.
        self.name = getattr(self.model, 'name', self.model.__class__.__name__)

    def forward(self, inputs: T) -> torch.Tensor:
        """Passes inputs through the model and returns the model's output"""
        _inputs = self.preprocess(inputs) if self.preprocess else inputs
        output = self.model(_inputs)
        return output


class ModelList:
    bounds = Bounds(0, 1)

    def __init__(
            self,
            model_dir,
            model_names,
            dataset_name: str = "ImageNet",
            loss_fn: Callable = None,
            device=None
    ):

        ssl._create_default_https_context = ssl._create_unverified_context
        self.model_dir = torch.hub.get_dir()
        self.dataset_name = dataset_name
        self.model_names = model_names if isinstance(model_names, list) else [model_names]
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # `pretrainedmodels` is only required when using ModelList.
        import pretrainedmodels
        self.available_models = pretrainedmodels.model_names
        self.model_list = []
        self.loss_fn = torch.nn.CrossEntropyLoss() if loss_fn is None else loss_fn
        self.get_models()

    def get_models(self):
        from models.utils import get_model
        for name in self.model_names:
            # print(name)
            model = get_model(name, self.dataset_name).to("cpu")
            self.model_list.append(model)

    def __len__(self):
        return len(self.model_list)

    def __getitem__(self, item):
        m = self.model_list[item]
        # adapt to slicing
        if isinstance(m, list):
            return [x.to(self.device).eval() for x in m]
        else:
            return m.to(self.device).eval()

    def __delitem__(self, key):
        del self.model_list[key]
        del self.model_names[key]

    def __call__(self, inputs: T, label: T, *args, **kwargs):
        loss = 0
        for model in self:
            # mean gradient equal to mean loss
            logits = model(inputs)
            loss += self.loss_fn(logits, label)
        loss /= len(self)
        return loss

    def __repr__(self):
        return self.__class__.__name__ + ": " + str(self)

    def __str__(self):
        return "-".join(sorted(self.model_names))


