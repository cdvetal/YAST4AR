import sys
import os

# get repo path for imports
abs_path = os.path.abspath(os.path.realpath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.dirname(abs_path)))
if parent_path not in sys.path:
    sys.path.append(parent_path)
    
from datetime import datetime

import torch
import importlib

import utils.static_vars as static
import utils.func_utils as utils


if __name__ == '__main__':
    overshoot = 0.02
    max_iter = 50

    total_images = 20
    log = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'



    dataset_loader_path = 'datasets\\cifar-10\\datasetLoader.py'
    model_path = 'models\\test\\MobileNet.py'
    model_name = 'MobileNet'
    checkpoint_path = 'models\\test\\checkpoints\\ckpt_MobileNet.pth'
    args = ""

    # Setup dataset
    spec = importlib.util.spec_from_file_location("dataset", dataset_loader_path)
    dataset = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dataset)
    testloader, trainloader = dataset.dataLoader()

    images, labels = utils.get_images_labels_from_dataLoader(testloader, device, total_images)
    normalized_images = utils.normalize_image(images, dataset.MEAN, dataset.STD)

    ################################################################################################

    # Setup model
    model = utils.load_model(model_path, model_name, device, args = args, checkpoint = checkpoint_path)

    labels = model(normalized_images).argmax(dim=1).cpu().detach().numpy()

    ################################################################################################

    # create results folder
    folder_name = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    results_path = os.path.join(static.PATH_RESULTS, str(folder_name) + '_' + model_name)
    if not os.path.isdir(results_path):
        os.makedirs(results_path)

    ################################################################################################

    # create binary file
    model_path = utils.generate_model_labels_binary(model, labels)


    # Execute attack
    command = 'attacks\\DeepFool\\DeepFool.py'
    parameters = ['--overshoot', str(overshoot), '--max-iter', str(max_iter), '--total-images', str(total_images),
                    '--model', model_path, '--dataset', dataset_loader_path, '--results-path', os.path.join(results_path, 'attacks', 'DeepFool')]

    if log:
        parameters.append('--log')

    utils.execute_command(command, parameters)

    ################################################################################################

    # delete bin file
    if os.path.exists(model_path):
        os.remove(model_path)
