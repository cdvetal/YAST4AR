import sys

import argparse
import os
from datetime import datetime
import logging
import yaml

import utils.static_vars as static
from pipeline import Pipeline
import utils.func_utils as utils


def main(args):

    config_dic = {}

    with open(args['config'], "r") as stream:
        try:
            config_dic = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            exit(0)
    
    for dataset_name, dataset_values in config_dic['DATASET'].items():
        if not dataset_values['run']:
                continue
        if not dataset_values['dataset_loader_path'] or not os.path.exists(dataset_values['dataset_loader_path']):
            print("ERROR: Config file not properly configured for dataset " + dataset_name)
            continue

        for model_name, model_values in config_dic['MODEL'].items():
            if not model_values['run']:
                continue
            if not model_values['model_path'] or not os.path.exists(model_values['model_path']) or not model_values['method_name']:
                print("ERROR: Config file not properly configured for model " + model_name)
                continue
            print("Starting pipeline for dataset {} and model {}".format(dataset_name, model_name))
            try:
                # create results folder
                folder_name = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
                results_path = os.path.join(static.PATH_RESULTS, str(folder_name) + '_' + model_name)
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

            finally:
                sys.stdout = ini_stdout


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--config', type=str, default='config.config', help='Config file with datasets and models information')
    parser.add_argument('--config-attacks', type=str, required=True, help='Config file for attacks to be passed in instead of arguments')
    parser.add_argument('--batch-size', type=int, default=5, help='Number of images in each batch')
    parser.add_argument('--total-images', type=int, default=20, help='Total number of images to use')

    args = vars(parser.parse_args())

    main(args)
