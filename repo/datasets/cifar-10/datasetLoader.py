import torch
import torchvision
from torchvision import transforms
import pathlib

global DATASET_PATH, DATASET_NAME, IMAGE_SIZE, MEAN, STD, TRANSFORM, TRAIN_TRANSFORM, NUM_CLASSES

DATASET_PATH = pathlib.Path(__file__).resolve().parent
DATASET_NAME = 'cifar10'
IMAGE_SIZE = 32
MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2023, 0.1994, 0.2010]
TRANSFORM = transforms.Compose([
    transforms.ToTensor()])
NUM_CLASSES = 10

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)])


def dataLoader():

    testset = torchvision.datasets.CIFAR10(
        root = DATASET_PATH, train=False, download=True,
        transform = TRANSFORM)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size = 1, shuffle=False, num_workers=1)

    trainset = torchvision.datasets.CIFAR10(
        root = DATASET_PATH, train=True, download=True,
        transform = TRAIN_TRANSFORM)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size = 1, shuffle=False, num_workers=1)
    
    return testloader, trainloader