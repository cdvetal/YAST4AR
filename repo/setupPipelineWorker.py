import sys

import argparse
import os
import logging
import yaml

from pipeline import Pipeline
import utils.func_utils as utils

REPO_DIR = os.path.dirname(os.path.abspath(__file__))

def verify_paths(config_dic):
    for dataset_name, dataset_values in config_dic['DATASET'].items():
        dataset_values['dataset_loader_path'] = os.path.join(REPO_DIR, dataset_values['dataset_loader_path']).replace('\\', os.path.sep)

    for model_name, model_values in config_dic['MODEL'].items():
        if model_values.get('model_path'):
            model_values['model_path'] = os.path.join(REPO_DIR, model_values['model_path']).replace('\\', os.path.sep)
        if model_values.get('checkpoint_path'):
            model_values['checkpoint_path'] = os.path.join(REPO_DIR, model_values['checkpoint_path']).replace('\\', os.path.sep)

    return config_dic

def main(args):

    config_dic = {}

    with open(args['config'], "r") as stream:
        try:
            config_dic = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            exit(0)

    config_dic = verify_paths(config_dic)

    counter = 1

    for dataset_name, dataset_values in config_dic['DATASET'].items():
        if not dataset_values['run']:
                continue
        
        if not dataset_values['dataset_loader_path'] or not os.path.exists(dataset_values['dataset_loader_path']):
            print("ERROR: Config file not properly configured for dataset " + dataset_name)
            continue

        for model_name, model_values in config_dic['MODEL'].items():
            if not model_values['run']:
                continue

            checkpoint_path = model_values.get('checkpoint_path', '')
            ext = os.path.splitext(str(checkpoint_path).lower())[1]
            is_script_checkpoint = ext in ('.pt', '.ts', '.jit')

            if is_script_checkpoint:
                if not checkpoint_path or not os.path.exists(checkpoint_path):
                    basename = os.path.basename(checkpoint_path)
                    candidates = []
                    for root, dirs, files in os.walk(REPO_DIR):
                        if basename in files:
                            candidates.append(os.path.join(root, basename))
                    if candidates:
                        model_values['checkpoint_path'] = os.path.abspath(candidates[0])
                        checkpoint_path = model_values['checkpoint_path']
                        print(f"Found checkpoint for model {model_name} at {checkpoint_path}; using it.")
                    else:
                        print("ERROR: Config file not properly configured for model " + model_name)
                        continue
            elif not model_values.get('model_path') or not os.path.exists(model_values['model_path']) or not model_values.get('method_name'):
                print("ERROR: Config file not properly configured for model " + model_name)
                continue            

            ##############################################################################################

            print("Starting pipeline for dataset {} and model {}".format(dataset_name, model_name))
            try:
                # create results folder
                results_path = os.path.join(args['output_path'], str(counter) + '_' + model_name + '_' + dataset_name)
                if not os.path.isdir(results_path):
                    os.makedirs(results_path)
                
                # setup logger
                ini_stdout = sys.stdout
                logging.basicConfig(filename=os.path.join(results_path, 'pipeline.log'), format='%(asctime)s | %(levelname)s | %(message)s', datefmt= '%m-%d-%Y %H:%M:%S', level=logging.INFO)
                sys.stdout = utils.StdoutToLogging()


                model = {model_name: model_values}
                dataset = {dataset_name: dataset_values}

                pipeline = Pipeline(args, dataset, model, args['config_attacks'], results_path)
                pipeline.execute()

                logging.shutdown()
                for handler in logging.root.handlers[:]:
                    logging.root.removeHandler(handler)

                counter += 1

            finally:
                sys.stdout = ini_stdout


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--config', type=str, default='config.config', help='Config file with datasets and models information')
    parser.add_argument('--config-attacks', type=str, required=True, help='Config file for attacks to be passed in instead of arguments')
    parser.add_argument('--batch-size', type=int, default=5, help='Number of images in each batch')
    parser.add_argument('--total-images', type=int, default=20, help='Total number of images to use')
    parser.add_argument('--output-path', type=str, required=True, help='Path to folder where to store results')

    args = vars(parser.parse_args())

    main(args)

