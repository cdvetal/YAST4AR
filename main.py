import os
import pickle
import random
import string
import subprocess
import time
import socket

from tasks import app
from celery.result import AsyncResult

import yaml
import base64
import shutil
from datetime import datetime
import hashlib
import math

from repo.setupPipelineDistributed import execute_setup

import threading
import signal
from collections import defaultdict

IP_ADDRESS = '127.0.0.1'
PORT = 6379
REPO_DIR = os.path.join(os.path.dirname(__file__), 'repo')
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
TEMP_DIR = os.path.join(os.path.dirname(__file__), 'temp')
MODELS_DIR = os.path.join(REPO_DIR, 'models')

HEARTBEAT = 5
TIMEOUT = 36000
SOFT_TIMEOUT = 33000
MAX_RETRIES = 3
CHUNK_SIZE = 1024*1024*32

# list of workers, structur: {worker_name: '', worker_status: 'live/offline'}
workers = []
num_live_workers = 0
threads_pipelines = []

stop_event = threading.Event()

def signal_handler(sig, frame):
    print('Exiting program, please wait...')
    stop_event.set()


signal.signal(signal.SIGINT, signal_handler)


def get_ip_address():
    global IP_ADDRESS
    broker_host = os.environ.get('REDIS_HOST') or os.environ.get('BROKER_HOST')
    broker_port = os.environ.get('REDIS_PORT', str(PORT))
    if broker_host:
        IP_ADDRESS = broker_host
        app.conf.broker_url = f'redis://{IP_ADDRESS}:{broker_port}/0'
        print(app.conf.broker_url)
        return

    try:
        fqdn = socket.getfqdn()
        IP_ADDRESS = socket.gethostbyname(fqdn)
        app.conf.broker_url = f'redis://{IP_ADDRESS}:6379/0'
        print(f'redis://{IP_ADDRESS}:6379/0')
    except:
        print("Could not get own server ip")
        exit(1)


def get_task_stats(config_file, config):
    with open(config_file, 'r') as f:
        file_c = yaml.safe_load(f)
    
    with open(config, 'r') as f:
        file_conf = yaml.safe_load(f)

    # Count number of run: true entries
    count = 0
    attacks = []
    for name, attack in file_c['ATTACK'].items():
        if attack.get('run', True):
            count += 1
            attacks.append(name)

    
    n_datasets = 0
    datasets = []
    for name, dt in file_conf['DATASET'].items():
        if dt.get('run', True):
            n_datasets += 1
            datasets.append(name)

    n_models = 0
    models = []
    for name, model in file_conf['MODEL'].items():
        if model.get('run', True):
            n_models += 1
            models.append(name)

    return attacks, datasets, models, count * n_datasets * n_models


def count_attacks(config_file, config):
    with open(config_file, 'r') as f:
        file_c = yaml.safe_load(f)
    
    with open(config, 'r') as f:
        file_conf = yaml.safe_load(f)

    # Count number of run: true entries
    count = 0
    for attack in file_c['ATTACK'].values():
        if attack.get('run', True):
            count += 1

    
    n_datasets = 0
    for dt in file_conf['DATASET'].values():
        if dt.get('run', True):
            n_datasets += 1

    n_models = 0
    for model in file_conf['MODEL'].values():
        if model.get('run', True):
            n_models += 1

    return count * n_datasets * n_models


def zip_folder(folder_path):
    shutil.make_archive(folder_path, 'zip', folder_path)
    return os.path.basename(folder_path + '.zip')


def execute_shell_command(exe, exe_file, args):
    command = exe + ' ' + exe_file

    for key, value in args.items():
        if isinstance(value, bool):
            command += ' ' + str(key)
        else:
            command += ' ' + str(key) + ' ' + str(value)

    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process.wait()
    if process.returncode != 0:
        error_output = process.stderr.read().decode('utf-8')
        error = ("An error occurred while running the command: {}\nReturn code: {}\nError message: {}".format(command, process.returncode, error_output))
        print(error)
        raise Exception(error)


def wait_for_tasks_to_finish():
    if len(threads_pipelines) == 0:
        return
    
    print("Waiting for {} tasks to finish".format(len(threads_pipelines)))
    while len(threads_pipelines) > 0:
        time.sleep(HEARTBEAT)

    print("All tasks finished")


def queue_repo(worker_name, zip_repo, empty_folders = ['outputs', 'temp'], chunk_size = CHUNK_SIZE, log=False):
    if log:
        file_size = os.path.getsize(zip_repo)
        print("Sending repo to {} worker. Repo size: {} MB".format(worker_name, file_size//1024//1024))

    # read zip and broadcast chunks
    sha1 = hashlib.sha1()
    first = True
    counter = 0
    
    with open(zip_repo, "rb") as f:
        while not stop_event.is_set():
            fails = 0
            chunk = f.read(chunk_size)
            if not chunk:
                break

            sha1.update(chunk)
            retries = 0

            encoded_repo = base64.b64encode(chunk).decode('utf-8')
            while not stop_event.is_set():
                job = app.send_task('tasks.receive_repo', args=[worker_name, encoded_repo], kwargs={'first': first}, \
                                                            routing_key=worker_name)

                task_id = job.id
                result = AsyncResult(task_id, app=app)
                try:
                    response = result.get(timeout=60)
                except (TimeoutError, Exception):
                    retries += 1
                    if retries > MAX_RETRIES:
                        print("{}: Repo update failed with timeout".format(worker_name))
                        return

                if response == True:
                    break

                # task catched by wrong worker
                app.control.revoke(task_id, terminate=True)
                time.sleep(1)
                fails += 1
                if fails > max(60, num_live_workers*2):
                    print("{}: Repo update failed with timeout".format(worker_name))
                    return

            counter += 1
            if log:
                print("{}/{} chunks".format(counter, math.ceil(file_size/CHUNK_SIZE)))
                    
            if first:
                first = False
            time.sleep(1)

    # send task to verify hash and unzip repo
    fails = 0
    while not stop_event.is_set():
        job = app.send_task('tasks.update_repo', args=[worker_name, chunk_size, sha1.hexdigest()], \
                            kwargs={'empty_folders': empty_folders}, routing_key=worker_name)
        task_id = job.id
        result = AsyncResult(task_id, app=app)
        try:
            response = result.get(timeout=60)
        except (TimeoutError, Exception):
            retries += 1
            if retries > MAX_RETRIES:
                print("{}: Repo update failed with timeout".format(worker_name))
                return
            
        fails += 1
        if fails > max(60, num_live_workers*2):
            print("{}: Repo update failed with timeout".format(worker_name))
            return

        if response != None:
            break

        # task catched by wrong worker
        app.control.revoke(task_id, terminate=True)
        time.sleep(1)
    
    if response == True:
        print("{}: Repo updated successfully".format(worker_name))
    else:
        print("{}: Repo update failed".format(worker_name))

    
def queue_models(worker_name, zip_model, chunk_size = CHUNK_SIZE, log=False):
    if log:
        file_size = os.path.getsize(zip_model)
        print("Sending repo to {} worker. Repo size: {} MB".format(worker_name, file_size//1024//1024))

    # read zip and broadcast chunks
    sha1 = hashlib.sha1()
    first = True
    counter = 0
    
    with open(zip_model, "rb") as f:
        while not stop_event.is_set():
            fails = 0
            chunk = f.read(chunk_size)
            if not chunk:
                break

            sha1.update(chunk)
            retries = 0

            encoded_repo = base64.b64encode(chunk).decode('utf-8')
            while not stop_event.is_set():
                job = app.send_task('tasks.receive_models', args=[worker_name, encoded_repo], kwargs={'first': first}, \
                                                            routing_key=worker_name)

                task_id = job.id
                result = AsyncResult(task_id, app=app)
                try:
                    response = result.get(timeout=60)
                except (TimeoutError, Exception):
                    retries += 1
                    if retries > MAX_RETRIES:
                        print("{}: Models update failed with timeout".format(worker_name))
                        return

                if response == True:
                    break

                # task catched by wrong worker
                app.control.revoke(task_id, terminate=True)
                time.sleep(1)
                fails += 1
                if fails > max(60, num_live_workers*2):
                    print("{}: Models update failed with timeout".format(worker_name))
                    return

            counter += 1
            if log:
                print("{}/{} chunks".format(counter, math.ceil(file_size/CHUNK_SIZE)))
                    
            if first:
                first = False
            time.sleep(1)

    # send task to verify hash and unzip repo
    fails = 0
    while not stop_event.is_set():
        job = app.send_task('tasks.update_models', args=[worker_name, chunk_size, sha1.hexdigest()], routing_key=worker_name)
        task_id = job.id
        result = AsyncResult(task_id, app=app)
        try:
            response = result.get(timeout=60)
        except (TimeoutError, Exception):
            retries += 1
            if retries > MAX_RETRIES:
                print("{}: Models update failed with timeout".format(worker_name))
                return
            
        fails += 1
        if fails > max(60, num_live_workers*2):
            print("{}: Models update failed with timeout".format(worker_name))
            return

        if response != None:
            break

        # task catched by wrong worker
        app.control.revoke(task_id, terminate=True)
        time.sleep(1)
    
    if response == True:
        print("{}: Repo updated successfully".format(worker_name))
    else:
        print("{}: Repo update failed".format(worker_name))


def send_repo_to_worker():
    live_workers = [{'worker_name': w['worker_name']} for w in workers if w['status'] == 'live']
    
    print("Select workers to update repo:")
    print("0  - All workers")
    for i, w in enumerate(live_workers):
        print("{}  - {}".format(i+1, w['worker_name']))

    print("-1 - Cancel")
    option = int(input("Select option: "))
    if option == -1:
        return
    
    wait_for_tasks_to_finish()

    print("Creating repo zip...")
    
    start = time.time()
    # zip repo
    zip_repo = zip_folder(REPO_DIR)
    zip_time = time.time()

    
    print("Repo zipped in {} seconds".format(zip_time - start))
    print("Broadcasting repo to workers...")


    # broadcast zip repo to selected workers
    if option == 0:
        for live_w in live_workers:
            queue_repo(live_w['worker_name'], zip_repo, log=True)
    else:
        queue_repo(live_workers[option-1]['worker_name'], zip_repo, log=True)
      

    # delete zip repo
    os.remove(zip_repo)

    broadcast_time = time.time()
    print("Repo broadcasted in {} seconds".format(broadcast_time - zip_time)) 

############################################################################################


def attack_model(config, config_attacks, batch_size, total_images, log):
    config_attacks_dic = {'path': config_attacks,
                        'name': os.path.basename(config_attacks),
                        'content': None
                        }
    config_dic = {'path': config,
                    'name': os.path.basename(config),
                    'content': None
                }
    
    with open(config_attacks_dic['path'], 'r') as f:
        config_attacks_dic['content'] = f.read()

    with open(config_dic['path'], 'r') as f:
        config_dic['content'] = f.read()

    command_args = {
        '--config': config_dic['name'],
        '--config-attacks': config_attacks_dic['name'],
        '--batch-size': batch_size,
        '--total-images': total_images,
        '--log': log
    }

    n_attacks = count_attacks(config_attacks, config)
    calculated_timeout = TIMEOUT * (n_attacks + 1)
    task = app.send_task('tasks.attack_model', args=[command_args, config_attacks_dic['name'], config_attacks_dic['content'], config_dic['name'], \
                                                            config_dic['content']], kwargs={'delete_results': False})

    for i in range(MAX_RETRIES):
        try:
            task_id = task.id
            result = AsyncResult(task_id, app=app)

            zip_name, content = result.get(timeout=calculated_timeout)
            print("Task finished with success")
            break
        except (TimeoutError, Exception):
            app.control.revoke(task_id, terminate=True)
            task = app.send_task('tasks.attack_model', args=[command_args, config_attacks_dic['name'], config_attacks_dic['content'], \
                                                                config_dic['name'], config_dic['content']], kwargs={'delete_results': True})
            print("Task failed")
            print("Retrying task for the {} time".format(i))
    
    with open(os.path.join(OUTPUTS_DIR, zip_name), 'wb') as f:
        f.write(base64.b64decode(content))


    return zip_name


def attack_model_by_attack(config, config_attacks, batch_size, total_images, log):

    # serialize celery app
    serialized_app = pickle.dumps(app)

    while not stop_event.is_set():
        file_name = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        temp_path = os.path.join(TEMP_DIR, file_name)
        if not os.path.isfile(temp_path):
            break

    # Write the serialized app to a file
    with open(temp_path, 'wb') as f:
        f.write(serialized_app)


    folder_name = datetime.now().strftime("%d_%m_%Y_%H_%M_%S_%f")
    results_path = os.path.join(OUTPUTS_DIR, str(folder_name))

    os.makedirs(results_path)

    # copy configs files
    shutil.copyfile(config, os.path.join(results_path, os.path.basename(config)))
    shutil.copyfile(config_attacks, os.path.join(results_path, os.path.basename(config_attacks)))

    # run pipeline
    #execute_setup(app = app, config = config, config_attacks = config_attacks, output_path = results_path, \
    #                batch_size = batch_size, total_images = total_images, log = log, master_ip = IP_ADDRESS, \
    #                port = PORT, timeouts = timeouts)
    
    # create subprocess to run pipeline
    args = {
        '--log': log,
        '--config': config,
        '--config-attacks': config_attacks,
        '--batch-size': batch_size,
        '--total-images': total_images,
        '--output-path': results_path,
        '--master-ip': IP_ADDRESS,
        '--port': PORT,
        '--load-app': temp_path,
        '--timeout': TIMEOUT,
        '--soft-timeout': SOFT_TIMEOUT,
        '--max-retries': MAX_RETRIES
    }

    execute_shell_command('python3', os.path.join(REPO_DIR, 'setupPipelineDistributed.py'), args)


    # zip results
    zip_name = zip_folder(results_path)

    # delete results folder
    shutil.rmtree(results_path)
    os.remove(temp_path)

    return zip_name


############################################################################################


def sincronize_repo():
    
    print("NOTE: It's recommended to do this action without any models beign tested.")
    conf = input("This operation may take a long time to complete. Do you want to continue? (y/n) ")
    if conf != 'y':
        print("Operation aborted")
        return

    print("Starting synchronization...")

    send_repo_to_worker()


def sincronize_models():
    conf = input("Do you wish to broadcast the models to all workers? (y/n) ")
    if conf != 'y':
        print("Operation aborted")
        return
    
    live_workers = [{'worker_name': w['worker_name']} for w in workers if w['status'] == 'live']
    

    wait_for_tasks_to_finish()

    print("Creating repo zip...")
    
    start = time.time()
    # zip models
    zip_models = zip_folder(MODELS_DIR)
    zip_models = os.path.join('repo', zip_models)
    zip_time = time.time()

    
    print("Repo zipped in {} seconds".format(zip_time - start))
    print("Broadcasting models to workers...")


    # broadcast models to workers
    for live_w in live_workers:
        queue_models(live_w['worker_name'], zip_models, log=True)
      

    # delete zip repo
    os.remove(zip_models)

    broadcast_time = time.time()
    print("Repo broadcasted in {} seconds".format(broadcast_time - zip_time))     
    

def add_pipeline():
    global threads_pipelines

    config = input("Select the configuration file: (Default: configs/config.yaml)\n") or 'configs/config.yaml'

    config_attacks = input("Select the attacks configuration file: (Default: configs/attacks_config.yaml)\n") or 'configs/attacks_config.yaml'

    try:
        pipeline_mode = int(input("Select the pipeline mode: (Default: 0) \n\t0 - Distribute attacks to workers \n\t1 - Execute all attacks in a single worker\n") or 0)
    except ValueError:
        pipeline_mode = 0

    try:
        batch_size = int(input("Select the batch size: (Default: 5)\n") or 5)
    except ValueError:
        batch_size = 5
    
    try:
        total_images = int(input("Select the total number of images to be tested: (Default: 10)\n") or 10)
    except ValueError:
        total_images = 10
    
    log = input("Do you want to log the results? (y/n) (Default: y)\n") or "y"
    log = True if log.lower() == "y" else False


    if pipeline_mode == 0:
        pipe_t = threading.Thread(target=attack_model_by_attack, args=[config, config_attacks, batch_size, total_images, log])
    else:
        pipe_t = threading.Thread(target=attack_model, args=[config, config_attacks, batch_size, total_images, log])


    pipe_t.start()
    
    attacks, datasets, models, count = get_task_stats(config_attacks, config)

    threads_pipelines.append({"attacks": attacks, "datasets": datasets, "models": models, "count": count, "thread": pipe_t})

    print("Pipeline added to the queue")


def list_running_jobs():
    print("Jobs in progress:")
    if len(threads_pipelines) == 0:
        print("No jobs in progress")
        return
    
    print("Attacks;\t\tDatasets;\t\tModels;\t\tCount")
    for job in threads_pipelines:
        print("{};\t{};\t{};\t{}".format(job['attacks'], job['datasets'], job['models'], job['count']))
    


menu = \
"""List of available commands:
    lw, list_workers -l | -all              : list all workers
    s_repo, sincronize_repo                 : sincronize repo with workers
    s_models, sincronize_models             : sincronize models with workers
    lj, list_jobs                           : list jobs in progress
    add, add_pipeline                       : add a pipeline to the queue
    h, help                                 : show this menu

"""
def command_line():
    while not stop_event.is_set():
        try:
            cmd = input(">> ")
            parts = cmd.split(' ')

            if parts[0] in ('h', 'help'):
                print(menu)

            elif parts[0] in ('lw', 'list_workers'):
                if len(parts) != 1:
                    if parts[1] == '-l':
                        live_workers = [w['worker_name'] for w in workers if w['status'] == 'live']
                        print("Workers:\n", '\n'.join(live_workers))

                    elif parts[1] == '-all':
                        print("Workers:")
                        for w in sorted(workers, key=lambda x: x['status'], reverse=False):
                            print(f"{w['worker_name']} : {w['status']}")
                else:
                    live_workers = [w['worker_name'] for w in workers if w['status'] == 'live']
                    print("Workers:\n", '\n'.join(live_workers))

            elif parts[0] in ('s_repo', 'sincronize_repo'):
                sincronize_repo()

            elif parts[0] in ('s_models', 'sincronize_models'):
                sincronize_models()

            elif parts[0] in ('add', 'add_pipeline'):
                add_pipeline()
                    
            elif parts[0] in ('lj', 'list_jobs'):
                list_running_jobs()
            elif parts[0] == '':
                pass

            else:
                print("Command not found")
        
        except Exception as e:
            pass

    
def shutdown_worker(worker_name):
    app.send_task('tasks.shutdown', routing_key=worker_name)


def remove_repeated_workers(workers):
    seen_names = set()
    new_workers = []
    for worker in workers:
        name = worker['worker_name']
        if name not in seen_names:
            new_workers.append(worker)
            seen_names.add(name)

    return new_workers


def thread_heartbeat_workers():
    global workers, num_live_workers

    # keep a list of workers and their status
    while not stop_event.is_set():
        try:
            replies = app.control.broadcast(command='get_workers', reply=True, timeout=10.0)
            # get live workers
            live_workers = []
            for reply in replies:
                live_workers.append({'worker_name': list(reply.keys())[0], 'status': 'live'})
            
            # check if there are no repeated workers
            worker_counts = defaultdict(int)
            for worker in live_workers:
                worker_counts[worker['worker_name']] += 1

            duplicated_workers = [worker_name for worker_name, count in worker_counts.items() if count > 1]
            if len(duplicated_workers) > 0:
                print("Warning: duplicate workers found: {}".format(duplicated_workers))
                print("Workers with the same name will be shutdown, please give different names to each worker.")
                for worker_name in duplicated_workers:
                    shutdown_worker(worker_name)
                
                live_workers = remove_repeated_workers(live_workers)

            num_live_workers = len(live_workers)
            # update workers array
            updated_workers = []
            for worker in workers:
                if any(worker['worker_name'] == w['worker_name'] for w in live_workers):
                    worker['status'] = 'live'
                    updated_workers.append(worker)
                else:
                    worker['status'] = 'offline'
                    updated_workers.append(worker)

            # add any new workers that were not in the original array
            for worker in live_workers:
                if worker['worker_name'] not in [w['worker_name'] for w in workers]:
                    updated_workers.append(worker)
            workers = updated_workers

            time.sleep(HEARTBEAT)
        except Exception as e:
            print(e)
            time.sleep(HEARTBEAT)


def update_tasks_thread():
    global threads_pipelines
    while not stop_event.is_set():
        for t in threads_pipelines:
            if not t["thread"].is_alive():
                threads_pipelines.remove(t)

        time.sleep(HEARTBEAT)


def start_timer(interval, task_func):
    timer = threading.Timer(interval, task_wrapper, args=[interval, task_func])
    timer.start()


def task_wrapper(interval, task_func):
    if not stop_event.is_set():
        task_func()
        start_timer(interval, task_func)

def remove_old_content(folders=['temp'], days=14):
    try:
        cutoff = time.time() - (days * 24*60*60)

        # remove old content
        for folder in folders:
            files_folders = os.listdir(folder)
            for path in files_folders:
                p = os.path.join(folder, path)
                if os.path.getmtime(p) < cutoff:
                    if os.path.isfile(p):
                        os.remove(p)
                    else:
                        shutil.rmtree(p)
    except Exception as e:
        pass
    

def check_folders():
    if not os.path.isdir('temp'):
        os.mkdir('temp')
    if not os.path.isdir('outputs'):
        os.mkdir('outputs')

if __name__ == '__main__':
    get_ip_address()
    check_folders()

    # start threads
    # command line thread
    cmd_t = threading.Thread(target=command_line, args=[])
    cmd_t.start()

    # heartbeat thread
    hb_t = threading.Thread(target=thread_heartbeat_workers, args=[])
    hb_t.start()

    # tasks update thread
    tasks_t = threading.Thread(target=update_tasks_thread, args=[])
    tasks_t.start()

    # deletes old content
    start_timer(5, remove_old_content)
     
    hb_t.join()
    tasks_t.join()
    print("Press enter to quit...")
    cmd_t.join()