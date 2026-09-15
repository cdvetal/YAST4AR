from yacs.config import CfgNode as CN

cfg = CN()
cfg.dataset = CN()
# dataset root dir
cfg.dataset.root_dir = r"your_imagenet_root_dir" # e.g., "/path/to/imagenet"

cfg.dataset.name = 'ImageNet'
if cfg.dataset.name in ["ImageNet"]:
    cfg.dataset.shape = [3, 299, 299]
    cfg.dataset.num_classes = 1000
else:
    raise NotImplementedError

cfg.model = CN()
cfg.model.root_path = None  # the model download path, which can be set to None.

# set model names according to the options provided by the `pretrainedmodels` and `timm` libraries
target_model = "vgg16"
substitute_models = ["inceptionresnetv2", "densenet121", "inceptionv3"]
# model names: first one is the target model, the rest are the substitute models
cfg.model.names = [target_model] + substitute_models

cfg.attack = CN()
cfg.attack.targeted = False
# attack name, which can be selected from ["IFGSM", "MIFGSM", "DSA"]
cfg.attack.name = "DSA"
cfg.attack.budget = 4000  # query budget
cfg.attack.steps = int(1e4)  # attack steps
cfg.attack.constraint = "linf"  # "l2" or "linf"
cfg.attack.eps = 16 / 255  # perturbation threshold


def get_cfg_defaults():
    """ Get a yacs CfgNode object with default values for my_project.
    """
    return cfg.clone()


def update_cfg(cfg, cfg_file):
    """ Updating the configuration
    """
    cfg.merge_from_file(cfg_file)
    return cfg.clone()
