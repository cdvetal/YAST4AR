from celery import Celery
from celery.worker.control import Panel
from celery.exceptions import Retry
import socket
import os

import shutil
import base64
import subprocess
from datetime import datetime
import random
import string
import hashlib
import zipfile


WORKING_FOLDER = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(WORKING_FOLDER, 'repo')
OUTPUTS_DIR = os.path.join(WORKING_FOLDER, 'outputs')
TEMP_DIR = os.path.join(WORKING_FOLDER, 'temp')
ATTACKS_DIR = os.path.join(REPO_DIR, 'attacks')
MODELS_DIR = os.path.join(REPO_DIR, 'models')

MASTER_IP = os.environ.get('REDIS_HOST') or os.environ.get('BROKER_HOST') or '127.0.0.1'
REDIS_PORT = os.environ.get('REDIS_PORT', '6379')
MY_IP_ADDRESS = '127.0.0.1'
HOSTNAME = 'worker_not_defined'


TIMEOUT = 60000
SOFT_TIMEOUT = 50000
MAX_RETRIES = 3
MAX_CURRENT_TASKS = 1


def get_ip_address():
    global MY_IP_ADDRESS
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        fqdn = socket.getfqdn()
        MY_IP_ADDRESS = socket.gethostbyname(fqdn)
    except:
        print("Could not get own server ip")
        exit(1)


get_ip_address()

app = Celery('YASTservice', broker=f'redis://{MASTER_IP}:{REDIS_PORT}/0', backend='rpc://')
app.conf.task_acks_late = False
app.conf.task_time_limit = TIMEOUT
app.conf.task_soft_time_limit = SOFT_TIMEOUT
app.conf.worker_concurrency = MAX_CURRENT_TASKS
app.conf.worker_prefetch_multiplier = MAX_CURRENT_TASKS


##################################################################################


def save_file_outputs(folder_path, content):
    if os.path.exists(folder_path):
        os.remove(folder_path)
    
    with open(folder_path, 'w') as f:
        f.write(content)


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

def zip_folder(folder_path):
    shutil.make_archive(folder_path, 'zip', folder_path)
    return os.path.basename(folder_path + '.zip')


def prepare_to_execute_attack(model_file_name, model_file_content, attack_name):
    # generate temp folder
    while True:
        folder_name = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        folder_path = os.path.join(TEMP_DIR, folder_name)
        if not os.path.exists(folder_path):
            break
    create_folder(None, folder_name, TEMP_DIR)

    # write model file inside folder
    save_file(None, model_file_name, folder_path, model_file_content)

    # create folder for results of attack
    create_folder(None, attack_name, folder_path)

    return folder_path


def execute_attack(attack_name, attack_args, folder_results):
    attack_path = os.path.join(ATTACKS_DIR, attack_name, attack_name + '.py')


    command = 'python3' + ' ' + attack_path
    for key, value in attack_args.items():
        if isinstance(value, bool):
            command += ' ' + str(key)
        else:
            command += ' ' + str(key) + ' ' + str(value)

    command += ' --results-path ' + folder_results

    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process.wait()

    if process.returncode != 0:
        error_output = process.stderr.read().decode('utf-8') if process.stderr else ''
        raise RuntimeError(
            "Attack command failed with return code {}: {}\n{}".format(
                process.returncode,
                command,
                error_output,
            )
        )


##################################################################################


@app.task(bind=True)
def shutdown(self):
    app.control.shutdown()


@app.task()
@Panel.register
def get_workers(panel):
    global HOSTNAME
    HOSTNAME = panel['hostname']
    return True


@app.task(bind=True, retry_kwargs={'max_retries': 5, 'countdown': 30})
def receive_repo(self, worker_name, content_chunk, first=False, filename='repo.zip'):
    try:
        if self.request.hostname != worker_name:
            return False

        # write chunk
        if first:
            with open(filename, 'wb') as f:
                f.write(base64.b64decode(content_chunk))
        else:
            with open(filename, 'ab') as f:
                f.write(base64.b64decode(content_chunk))

        return True
    except Exception as e:
        # If an exception is raised, retry the task
        raise Retry(exc=e)
    
@app.task(bind=True, retry_kwargs={'max_retries': 5, 'countdown': 30})
def receive_models(self, worker_name, content_chunk, first=False, filename='models.zip'):
    try:
        if self.request.hostname != worker_name:
            return False

        # write chunk
        if first:
            with open(os.path.join(TEMP_DIR, filename), 'wb') as f:
                f.write(base64.b64decode(content_chunk))
        else:
            with open(os.path.join(TEMP_DIR, filename), 'ab') as f:
                f.write(base64.b64decode(content_chunk))

        return True
    except Exception as e:
        # If an exception is raised, retry the task
        raise Retry(exc=e)

 
@app.task(bind=True)
def update_repo(self, worker_name, chunk_size, hash, empty_folders=['outputs', 'temp'], filename='repo.zip', repo_folder='repo'):
    try:
        if self.request.hostname != worker_name:
            return None

        # check hash
        sha1 = hashlib.sha1()
        with open(filename, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                sha1.update(chunk)

        # different hash, remove file and notify master
        if sha1.hexdigest() != hash:
            #os.remove(filename)
            return False

        # right hash, update repo
        # remove repo and temp folders
        if os.path.exists(repo_folder):
            shutil.rmtree(repo_folder)
        os.makedirs(repo_folder)

        for folder in empty_folders:
            if os.path.exists(folder):
                shutil.rmtree(folder)
            os.makedirs(folder)


        # unzip repo
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(WORKING_FOLDER, repo_folder))

        # remove zip file
        os.remove(filename)

        return True

    except Exception as e:
        # If an exception is raised, retry the task
        raise Retry(exc=e)
    
@app.task(bind=True)
def update_models(self, worker_name, chunk_size, hash):
    try:
        if self.request.hostname != worker_name:
            return None
        
        filename = 'models.zip'
        folder = os.path.join('repo', 'models')

        # check hash
        sha1 = hashlib.sha1()
        with open(os.path.join(TEMP_DIR, filename), "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                sha1.update(chunk)
        

        # different hash, remove file and notify master
        if sha1.hexdigest() != hash:
            os.remove(os.path.join(TEMP_DIR, filename))
            return False
        
        # right hash, update repo
        # remove repo and temp folders
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

        # unzip repo
        with zipfile.ZipFile(os.path.join(TEMP_DIR, filename), 'r') as zip_ref:
            zip_ref.extractall(folder)

        # remove zip file
        os.remove(os.path.join(TEMP_DIR, filename))

        return True

    except Exception as e:
        # If an exception is raised, retry the task
        raise Retry(exc=e)


@app.task()
@Panel.register
def save_file(panel, filename, filepath, content, remove_existing=False):
    filedir = os.path.join(filepath, filename)
    if remove_existing:
        if os.path.exists(filedir):
            os.remove(filedir)
    elif os.path.exists(filedir):
        return

    with open(filedir, 'wb') as f:
        f.write(base64.b64decode(content))


@app.task()
@Panel.register
def create_folder(panel, foldername, folder_path, remove_existing=False):
    #foldername = args['foldername']
    #folder_path = args['folder_path']
    #remove_existing = args['remove_existing']

    folder = os.path.join(folder_path, foldername)
    if remove_existing and os.path.exists(folder):
        shutil.rmtree(folder)
    elif os.path.exists(folder):
        return

    os.makedirs(folder, exist_ok=True)

# =============================================================================



@app.task #(bind=True); add self,
def attack_model(command_args, attack_config_name, attack_config_content, config_name, config_content, delete_results=True):
    # create folder for results
    folder_name = datetime.now().strftime("%d_%m_%Y_%H_%M_%S_%f")
    results_path = os.path.join(OUTPUTS_DIR, str(folder_name))

    os.makedirs(results_path)


    # save configs file
    save_file_outputs(os.path.join(results_path, attack_config_name), attack_config_content)
    save_file_outputs(os.path.join(results_path, config_name), config_content)

    command_args['--config-attacks'] = os.path.join(results_path, attack_config_name)
    command_args['--config'] = os.path.join(results_path, config_name)
    command_args['--output-path'] = results_path

    # run pipeline
    execute_shell_command('python3', os.path.join(REPO_DIR, 'setupPipelineWorker.py'), command_args)

    # zip results
    zip_name = zip_folder(results_path)

    # delete results folder
    shutil.rmtree(results_path)

    # return zip file
    path_file = os.path.join(OUTPUTS_DIR, zip_name)
    with open(path_file, 'rb') as f:
        content = f.read()
        encoded_content = base64.b64encode(content).decode('utf-8')
    
    if delete_results:
        os.remove(path_file)
    
    return zip_name, encoded_content


@app.task(bind=True, max_retries=MAX_RETRIES, default_retry_delay=TIMEOUT)
def queue_attack(self, model_file_name, model_file_content, attack_name, attack_args, delete_results=True):
    try:
        ## NEEDS:
        # binary file
        # empty folder for results
        folder_path = prepare_to_execute_attack(model_file_name, model_file_content, attack_name)

        # update args
        attack_args['--model'] = os.path.join(folder_path, model_file_name)
        attack_args['--dataset'] = os.path.join(REPO_DIR, attack_args['--dataset'])

        # execute attack (store results in temp folder)
        execute_attack(attack_name, attack_args, os.path.join(folder_path, attack_name))

        # zip results
        zip_folder(os.path.join(folder_path, attack_name))

        # return zip file
        with open(os.path.join(folder_path, attack_name) + '.zip', 'rb') as f:
            content = f.read()
            encoded_content = base64.b64encode(content).decode('utf-8')

        # delete temp folder
        if delete_results:
            shutil.rmtree(folder_path)

        return encoded_content
    except Exception as exc:
        # overrides the default delay to retry after 1 minute
        raise self.retry(exc=exc, countdown=60)
    