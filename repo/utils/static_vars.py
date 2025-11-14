import os
from torchvision import transforms

ROOT_PATH = os.path.abspath(os.curdir)
PATH_MODEL = os.path.join(ROOT_PATH,"models")
PATH_DATASET = os.path.join(ROOT_PATH,"datasets")
PATH_ATTACKS = os.path.join(ROOT_PATH,"attacks")
PATH_RESULTS = os.path.join(ROOT_PATH,"outputs")
PATH_TEMP = os.path.join(ROOT_PATH,"temp")


CIFAR_SIZE = 32
CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]
CIFAR_TRANSFORM = transforms.Compose([
    transforms.ToTensor()])

CIFAR_TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])


IMAGENET_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGENET_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()])    


MNIST_SIZE = 28
MNIST_MEAN = [0.5]
MNIST_STD = [1.0]
MNIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor()])


INCEPTION_SIZE = 299
INCEPTION_TRANSFORM = transforms.Compose([
    transforms.Resize(342),
    transforms.CenterCrop(299),
    transforms.ToTensor()])
