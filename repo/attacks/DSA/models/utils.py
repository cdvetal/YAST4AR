import math

import pretrainedmodels
import timm

from munch import munchify
from pretrainedmodels.utils import ToSpaceBGR, ToRange255
from timm import create_model
from timm.data import IMAGENET_DEFAULT_MEAN
from torchvision.transforms import transforms

from .base import Model


def get_pretrained_model(model_name: str, num_classes=1000, pretrained='imagenet') -> Model:
    if model_name not in pretrainedmodels.model_names:
        # use timm to laod pretrained models
        model = create_model(model_name, pretrained=(pretrained is not None))
        transform = timm.data.create_transform(**timm.data.resolve_data_config(model.pretrained_cfg))
        transform.transforms.pop(2)
    else:
        # use the lib `pretrainedmodels` to load models
        model = pretrainedmodels.__dict__[model_name](num_classes, pretrained)
        transform = TransformImage(model)
    model.preprocess = transform
    model.name = model_name
    return Model(model)


def get_model(model_str: str, dataset_str: str = "ImageNet"):
    if dataset_str == "ImageNet":
        model = get_pretrained_model(model_str, 1000, "imagenet")
    else:
        raise NotImplementedError
    model.eval()
    return model


class TransformImage(object):
    """
    this class is a clone from pretrainedmodels library
    and the to_tensor transform operation has been changed
    """

    def __init__(self, opts, scale=0.875, random_crop=False,
                 random_hflip=False, random_vflip=False,
                 preserve_aspect_ratio=True):
        if type(opts) == dict:
            opts = munchify(opts)
        self.input_size = opts.input_size
        self.input_space = opts.input_space
        self.input_range = opts.input_range
        self.mean = opts.mean
        self.std = opts.std

        # https://github.com/tensorflow/models/blob/master/research/inception/inception/image_processing.py#L294
        self.scale = scale
        self.random_crop = random_crop
        self.random_hflip = random_hflip
        self.random_vflip = random_vflip

        tfs = []
        if preserve_aspect_ratio:
            tfs.append(transforms.Resize(int(math.floor(max(self.input_size) / self.scale))))
        else:
            height = int(self.input_size[1] / self.scale)
            width = int(self.input_size[2] / self.scale)
            tfs.append(transforms.Resize((height, width)))

        if random_crop:
            tfs.append(transforms.RandomCrop(max(self.input_size)))
        else:
            tfs.append(transforms.CenterCrop(max(self.input_size)))

        if random_hflip:
            tfs.append(transforms.RandomHorizontalFlip())

        if random_vflip:
            tfs.append(transforms.RandomVerticalFlip())

        # tfs.append(transforms.ToTensor())  # lcy modify
        tfs.append(ToSpaceBGR(self.input_space == 'BGR'))
        tfs.append(ToRange255(max(self.input_range) == 255))
        tfs.append(transforms.Normalize(mean=self.mean, std=self.std))

        self.tf = transforms.Compose(tfs)

    def __call__(self, img):
        tensor = self.tf(img)
        return tensor
