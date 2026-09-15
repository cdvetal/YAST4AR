import sys
import os

repo_path = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
if repo_path not in sys.path:
    sys.path.append(repo_path)

import torch
import numpy as np

from datetime import datetime
from PIL import Image

import importlib
import shutil
import yaml
import time

from celery import Celery
from celery.result import AsyncResult
import base64
import zipfile

import utils.func_utils as utils 


REPO_DIR = os.path.dirname(os.path.abspath(__file__))
TIMEOUT = 60000
SOFT_TIMEOUT = 50000
MAX_RETRIES = 3

class Pipeline:
    def __init__(self, args, dataset, model, config_path, results_path, celery_config, app=None, timeouts={}):
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

        if app is None:
            self.app = Celery(celery_config['service_name'], broker=celery_config['broker_url'], backend='rpc://')
            self.app.conf.task_acks_late = False
        else:
            self.app = app

        # set timeouts
        global TIMEOUT, SOFT_TIMEOUT, MAX_RETRIES
        if 'TIMEOUT' in timeouts:
            TIMEOUT = timeouts['TIMEOUT']

        if 'SOFT_TIMEOUT' in timeouts:
            SOFT_TIMEOUT = timeouts['SOFT_TIMEOUT']

        if 'MAX_RETRIES' in timeouts:
            MAX_RETRIES = timeouts['MAX_RETRIES']
        


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


    def queue_attack(self, model_path, attack_name, attack_args):
        attack_args = attack_args.copy()

        with open(model_path, "rb") as f:
            content = f.read()
            encoded_content = base64.b64encode(content).decode('utf-8')

        # prepare args
        attack_args['--model'] = os.path.basename(attack_args['--model'])

        repo_index = attack_args['--dataset'].index(REPO_DIR) + len(REPO_DIR)
        attack_args['--dataset'] = attack_args['--dataset'][repo_index+1:]

        job = self.app.send_task('tasks.queue_attack', kwargs={'model_file_name': os.path.basename(model_path), \
                                                                'model_file_content': encoded_content, 'attack_name': attack_name, \
                                                                'attack_args': attack_args, 'delete_results': True})
        return job

    def save_results_attack(self, folder, content):
        if not os.path.exists(folder):
            os.makedirs(folder)

        temp_file = os.path.join(folder, 'temp.zip')
        
        # write ziped file
        with open(temp_file, 'wb') as f:
            f.write(base64.b64decode(content))


        with zipfile.ZipFile(temp_file, 'r') as zip_ref:
            zip_ref.extractall(folder)

        os.remove(temp_file)
        

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


        default_params = {
            '--total-images': str(self.total_images), 
            '--model': model_path, 
            '--dataset': self.dataset_loader_path
        }

        if self.log:
            default_params['--log'] = True


        jobs = []
        jobs_args = {} # used to save the arguments of each job in case of retry

        for attack_name, args in self.config.items():
            if not args['run']:
                continue

            print('Adding attack {} to queue...'.format(attack_name))


            del args['run']
            del args['path']

            
            process_args = self.prepare_arguments(default_params, args)
            jobs_args[attack_name] = {
                'model_path': model_path,
                'attack_name': attack_name,
                'process_args': process_args
            }

            job = {
                'attack': attack_name,
                'job': self.queue_attack(model_path, attack_name, process_args),
                'retry': 0
            }
            jobs.append(job)
            
            
            
        # Wait for attacks to finish and retry failed tasks
        counter = 0
        outputs = 0
        while counter < len(jobs):
            task = jobs[counter]

            task_id = task['job'].id
            result = AsyncResult(task_id, app=self.app)
            try:
                res = result.get(timeout=TIMEOUT)
                print("Task finished with success")
                self.save_results_attack(os.path.join(attacks_path, task['attack']), res)
                outputs += 1

            except (TimeoutError, Exception):
                self.app.control.revoke(task_id, terminate=True)
                print("Task failed")

                # retry task
                if task['retry'] < MAX_RETRIES:
                    new_job = {
                        'attack': task['attack'],
                        'job': self.queue_attack(jobs_args[task['attack']]['model_path'], jobs_args[task['attack']]['attack_name'], jobs_args[task['attack']]['process_args']),
                        'retry': task['retry'] + 1
                    }
                    jobs.append(new_job)
                    print("Retrying {} attack for the {}º time".format(new_job['attack'], new_job['retry']))
            
            counter += 1



        print("Finished running all attacks")

        if outputs == 0:
            print("No results to be saved.") 
            return
        
    
        utils.save_original_images(os.path.join(self.results_path,"original"), images)

        ############################################################################################

        results = utils.load_attack_results(attacks_path, dataset.MEAN, dataset.STD)

        utils.generate_plots(results, images, labels, os.path.join(self.results_path, "plots"), self.total_images, images_per_window = self.batch_size)

        utils.save_correct_classification(self.results_path, attacks_path)

        (
            model_dataset_name,
            dic_results,
            perfect_score,
            num_misclassified_images,
            clean_acc_by_attack,
            robust_acc_by_attack,
            linf_avg_by_attack,
            l2_avg_by_attack,
            queries_avg_by_attack,
            clean_acc_str_by_attack,
            robust_acc_str_by_attack,
            linf_str_by_attack,
            l2_str_by_attack,
            queries_str_by_attack,
        ) = utils.calculate_robustness_score(self.results_path, attacks_path)

        # update global log
        utils.update_log_robustness(
            model_dataset_name,
            dic_results,
            num_misclassified_images,
            perfect_score,
            REPO_DIR,
            clean_acc_by_attack,
            robust_acc_by_attack,
            linf_avg_by_attack,
            l2_avg_by_attack,
            queries_avg_by_attack,
            clean_acc_str_by_attack,
            robust_acc_str_by_attack,
            linf_str_by_attack,
            l2_str_by_attack,
            queries_str_by_attack,
        )

        # update user log
        utils.update_log_robustness(
            model_dataset_name,
            dic_results,
            num_misclassified_images,
            perfect_score,
            os.path.abspath(self.results_path),
            clean_acc_by_attack,
            robust_acc_by_attack,
            linf_avg_by_attack,
            l2_avg_by_attack,
            queries_avg_by_attack,
            clean_acc_str_by_attack,
            robust_acc_str_by_attack,
            linf_str_by_attack,
            l2_str_by_attack,
            queries_str_by_attack,
        )

        # delete bin file
        if os.path.exists(model_path):
            os.remove(model_path)




    def prepare_arguments(self, default, params):
        arguments = default.copy()
        for key, value in params.items():
            if isinstance(value, bool):
                if value:
                    arguments['--{}'.format(key.replace("_", "-"))] = True
                continue
            arguments['--{}'.format(key.replace("_", "-"))] = str(value)

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


    



        
