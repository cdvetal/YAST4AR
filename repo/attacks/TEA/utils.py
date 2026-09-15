import os
import torch
from torchvision import transforms
from PIL import Image, UnidentifiedImageError

VALID_MODELS = ("resnet50", "resnet101", "vgg16", "ViT")

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def get_model_choice(valid_models=VALID_MODELS):
    model = input(f"Enter model ({', '.join(valid_models)}): ").strip()
    if model not in valid_models:
        raise ValueError(f"Invalid model selection. Choose from {', '.join(valid_models)}.")
    return model

def get_transform(resize=256, crop=224):
    return transforms.Compose([transforms.Resize(resize),transforms.CenterCrop(crop)])

def sort_key(filename):
    name, _ = os.path.splitext(filename)
    number = int(name[:-1])
    letter = name[-1]
    return number, letter

def get_sorted_image_pairs(folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith('.jpeg')]
    return sorted(files, key=sort_key)

def load_and_transform_image(path, transform):
    try:
        return transform(Image.open(path).convert("RGB"))
    except UnidentifiedImageError:
        print(f"Warning: Failed to load image {path}. Skipping.")
        return None

def validate_pair_id(pair_id, num_files):
    num_pairs = num_files // 2
    if not (1 <= pair_id <= num_pairs):
        raise ValueError(f"pair_id must be between 1 and {num_pairs}")

def get_pair_files(files, pair_id):
    start = 2 * (pair_id - 1)
    return files[start], files[start + 1]
