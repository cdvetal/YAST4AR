import torch
import numpy as np

from datetime import datetime
import os
from PIL import Image
import subprocess

import dill
import importlib
import json
import shutil
import yaml

import utils.func_utils as utils 
import utils.static_vars as static

REPO_DIR = os.path.dirname(os.path.abspath(__file__))

class Pipeline:
    def __init__(self, args, dataset, model, config_path, results_path):
        self.batch_size = args["batch_size"]
        self.total_images = args["total_images"]
        self.log = args["log"]
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.dataset_name = next(iter(dataset))
        self.dataset_loader_path = dataset[self.dataset_name]['dataset_loader_path']

        self.set_model(model)

        self.config_path = config_path
        self.load_config(config_path)

        self.results_path = results_path        

        self.original_data = {}
        self.n_batches = self.total_images // self.batch_size
        self.results = {}
        self.classified_labels = []
        self.num_samples = self.n_batches * self.batch_size

    def set_model(self, model):
        try:
            self.model_name = next(iter(model))
            v = model[self.model_name]
            self.model = utils.load_model(v['model_path'], v['method_name'], self.device, args = v['args'], checkpoint = v['checkpoint_path'])
            print("Model {} loaded successfully".format(self.model_name))
        
        except Exception as e:
            print("ERROR: Exception loading model: {}".format(e))
            exit(0)

    def verify_paths(self, config_attacks):       
        for attack_name, attack_values in config_attacks.items():
            attack_values['path'] = os.path.join(REPO_DIR, attack_values['path'].replace('\\\\', os.path.sep))
        return config_attacks

    def load_config(self, config_path):
        with open(config_path, "r") as stream:
            try:
                self.config = yaml.safe_load(stream)['ATTACK']
                self.config = self.verify_paths(self.config)
            except yaml.YAMLError as exc:
                print("ERROR: Exception loading config file: {}".format(exc))
                exit(0)

        print("Attack config arguments loaded successfully")

    def execute(self):    

        # Setup dataset
        spec = importlib.util.spec_from_file_location("dataset", self.dataset_loader_path)
        dataset = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dataset)
        testloader, trainloader = dataset.dataLoader()

        # Keep the full dataset tensor on CPU to avoid GPU OOM on large runs.
        images, labels = utils.get_images_labels_from_dataLoader(testloader, 'cpu', self.total_images)
        original_labels = []
        with torch.no_grad():
            for i in range(0, images.size(0), self.batch_size):
                images_batch = images[i:i + self.batch_size]
                normalized_batch = utils.normalize_image(images_batch, dataset.MEAN, dataset.STD)
                preds = self.model(normalized_batch.to(self.device)).argmax(dim=1).cpu()
                original_labels.append(preds)

        original_labels = torch.cat(original_labels, dim=0).detach().numpy()
        print("Dataset {} loaded successfully".format(self.dataset_name))

        ###########################################################################################

        attacks_path = os.path.join(self.results_path, 'attacks')

        if not os.path.exists(attacks_path):
            os.makedirs(attacks_path)

        # Generate binary file with model and classified labels
        model_path = utils.generate_model_labels_binary(self.model_name, self.model, original_labels)
        print("Binary file generated")

        # save configuration file in output directory
        shutil.copy(self.config_path, self.results_path)
        print("All results will be saved in this folder: {}".format(self.results_path))

        ###########################################################################################

        print("Loading complete")

        print('Running pipeline for model: {} with dataset: {}'.format(self.model_name, self.dataset_name))


        default_params = ['--total-images', str(self.total_images), '--model', model_path, '--dataset', self.dataset_loader_path]

        if self.log:
            default_params.append('--log')

        num_attacks = 0
        for attack_name, args in self.config.items():
            if not args['run']:
                continue

            print('Executing {} attack...'.format(attack_name))

            file_path = args['path']
            
            args['results_path'] = os.path.join(attacks_path, attack_name)
            del args['run']
            del args['path']

            process_args = self.prepare_arguments(default_params, args)
            utils.execute_command(file_path, process_args)
            num_attacks += 1

            print("Finished running {} attack".format(attack_name))


        print("Finished running all attacks")

        if num_attacks == 0:
            print("No results to be saved.") 
            return
    

        utils.save_original_images(os.path.join(self.results_path,"original"), images)

        ############################################################################################

        results = utils.load_attack_results(attacks_path, dataset.MEAN, dataset.STD)

        utils.generate_plots(results, images, labels, os.path.join(self.results_path, "plots"), self.total_images, images_per_window = self.batch_size)

        utils.save_correct_classification(self.results_path, attacks_path)

        utils.calculate_robustness_score(self.results_path, attacks_path)


        # delete bin file
        if os.path.exists(model_path):
            os.remove(model_path)




    def prepare_arguments(self, default, params):
        arguments = default.copy()
        for key, value in params.items():
            if isinstance(value, bool):
                if value:
                    arguments.append('--{}'.format(key.replace("_", "-")))
                continue
            arguments.append('--{}'.format(key.replace("_", "-")))
            arguments.append(str(value))

        return arguments
    

    def find_default_parameters(self, params, attack_name):
        for name, params in params.items():
            if (name.lower() == attack_name.lower()):
                return params
        return None


    def save_original_images(self, folder_path, original_images):
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)

        for j in range(len(original_images)):
            im = Image.fromarray(
                (np.transpose(original_images[j].cpu().detach().numpy(), (1, 2, 0)) * 255).astype(np.uint8))
            filename_save = 'original_{}.jpeg'.format(str(j))
            save_path = os.path.join(folder_path, filename_save)
            im.save(save_path)


    



        
