import torch
import torchvision.datasets
from torchvision import transforms
from yacs.config import CfgNode

from attack import get_attack
from config.attack_config import get_cfg_defaults
from models.base import ModelList
from utils.criterion import Misclassification, TargetedMisclassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
to_pil = torchvision.transforms.ToPILImage()


def load_images():
    # load dataset
    dataset = torchvision.datasets.ImageNet(
        cfg.dataset.root_dir,
        "val",
        transform=transforms.Compose([transforms.ToTensor(), transforms.Resize([299, 299])])
    )

    # load images
    # set the index of original image and target image
    original_index, target_index = 757, 254
    original_image, original_label = dataset[original_index]
    target_image, target_label = dataset[target_index]
    assert original_label != target_label, "original label should not be equal to target label"
    # move to device
    original_image = original_image.to(device).unsqueeze(0)
    target_image = target_image.to(device).unsqueeze(0)
    original_label, target_label = torch.tensor([original_label]).to(device), torch.tensor([target_label]).to(device)
    return original_image, original_label, target_image, target_label


def attack(cfg: CfgNode):
    """ the main function to run attack
    :param cfg: config Node
    """
    # load images
    original_image, original_label, target_image, target_label = load_images()

    # load models
    models = ModelList(cfg.model.root_path, cfg.model.names, cfg.dataset.name)
    # set target model and substitute models
    target_model = models[0]
    del models[0]
    local_models = models

    # load attack method
    _attack = get_attack(cfg.attack.name, local_models=local_models, constraint=cfg.attack.constraint)
    print("\n###### ", _attack, "is Running ######")

    # set targeted or untargeted mode
    criterion = Misclassification(original_label)

    # use another image as the starting point for query-based attacks
    if cfg.attack.name in ["DSA"]:
        starting_points = target_image
    else:
        starting_points = None

    # attack
    adv_image = _attack(target_model, original_image, criterion=criterion, starting_points=starting_points)

    # show the adversarial image
    pil_img = to_pil(adv_image[0])
    pil_img.show()


if __name__ == "__main__":
    # load config
    cfg = get_cfg_defaults()
    # set model download path
    if cfg.model.root_path is not None:
        torch.hub.set_dir(cfg.model.root_path)

    # attack process
    attack(cfg)

    print("\n\n" + "#" * 20 + " Main Program Exit " + "#" * 20 + "\n\n")
