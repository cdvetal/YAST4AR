import pickle
import sys
import os

repo_path = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
if repo_path not in sys.path:
    sys.path.append(repo_path)

import argparse
import logging
import yaml

from pipelineDistributed import Pipeline
import utils.func_utils as utils

  
REPO_DIR = os.path.dirname(os.path.abspath(__file__))


def verify_paths(config_dic):
    for dataset_name, dataset_values in config_dic['DATASET'].items():
        dataset_values['dataset_loader_path'] = os.path.join(REPO_DIR, dataset_values['dataset_loader_path']).replace('\\', os.path.sep)

    for model_name, model_values in config_dic['MODEL'].items():
        model_values['model_path'] = os.path.join(REPO_DIR, model_values['model_path']).replace('\\', os.path.sep)
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
            
            model_values['model_path'] = os.path.join(REPO_DIR, model_values['model_path']).replace('\\', os.path.sep)
            if not model_values['model_path'] or not os.path.exists(model_values['model_path']) or not model_values['method_name']:
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
                celery_config = {
                    "service_name": 'YASTservice',
                    "broker_url": f"redis://{args['master_ip']}:{args['port']}/0"
                }
                
                if 'app' in args:
                    pipeline = Pipeline(args, dataset, model, args['config_attacks'], results_path, celery_config, args['app'], timeouts= args['timeouts'])
                else:
                    pipeline = Pipeline(args, dataset, model, args['config_attacks'], results_path, celery_config, timeouts= args['timeouts'])
                pipeline.execute()

                logging.shutdown()
                for handler in logging.root.handlers[:]:
                    logging.root.removeHandler(handler)

                counter += 1

            finally:
                sys.stdout = ini_stdout


def execute_setup(app, config, config_attacks, output_path, batch_size=5, total_images=20, log=True, master_ip='', port=6379, timeouts={}):
    args = {
        'log': log,
        'config': config,
        'config_attacks': config_attacks,
        'batch_size': batch_size,
        'total_images': total_images,
        'output_path': output_path,
        'master_ip': master_ip,
        'port': port,
        'app': app,
        'timeouts': timeouts
    }
    main(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--config', type=str, default='config.config', help='Config file with datasets and models information')
    parser.add_argument('--config-attacks', type=str, required=True, help='Config file for attacks to be passed in instead of arguments')
    parser.add_argument('--batch-size', type=int, default=5, help='Number of images in each batch')
    parser.add_argument('--total-images', type=int, default=20, help='Total number of images to use')
    parser.add_argument('--output-path', type=str, required=True, help='Path to folder where to store results')

    parser.add_argument('--master-ip', type=str, help='IP of master machine')
    parser.add_argument('--port', type=int, default=6379, help='IP of master machine')
    parser.add_argument('--load-app', type=str, help='Load Celery app from binary file')
    parser.add_argument('--timeout', type=int, default=3600, help='Timeouts for each attack')
    parser.add_argument('--soft-timeout', type=int, default=3300, help='Soft timeout for each attack')
    parser.add_argument('--max-retries', type=int, default=3, help='Max retries for each attack')
    args = vars(parser.parse_args())
    
    args['timeouts'] = {
                        'TIMEOUT': args['timeout'],
                        'SOFT_TIMEOUT': args['soft_timeout'],
                        'MAX_RETRIES': args['max_retries']
                        }

    # if load_app is define load binary file
    if args['load_app']:
        with open(args['load_app'], 'rb') as f:
            args['app'] = pickle.load(f)

    main(args)

