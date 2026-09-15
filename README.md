# YAST

**Yet Another Security Toolbox**

## What is YAST?

YAST is a Python-based toolbox designed to help users test their machine learning models against adversarial samples in the image classification landscape. YAST is implemented using a distributed computing architecture, leveraging Celery as a task queue and Redis as message broker for efficient processing of many models and adversarial attacks.

## Who is it for?

YAST is intended for any user who wants to evaluate the robustness of their machine learning models against various adversarial attacks.

## Why should anyone use it?

YAST offers a capable and robust pipeline batch system that allows extensive adversarial robustness tests to be performed with minimal code. Its use of Celery enables efficient processing of many models and adversarial attacks, making it possible to test a wide variety of models against various adversarial attacks in a reasonable time frame.

With YAST, users can easily test the robustness of their models and gain insights into which models perform better under different types of adversarial attacks. Moreover, the automatic functionalities of YAST make it easy to use, even for users with limited experience in adversarial testing.

<br/>

---

<br/>

## Setup

In this section, we will describe the necessary setup to run YAST in a master-slave architecture. The following commands will be for Ubuntu machines, for other operating systems commands may vary.

Before starting it's important to mention that in order to properly clone the repository, the user must have "git lfs" for the handling of binary files. After cloning the repository you may use the following command to check the binary files:

```bash
    git lfs ls-files
```

It's recommended to use a virtual environment. In order to create a virtual environment use the following commands:

```bash
    # Create a new virtual environment
    python -m venv "venv_name"

    # Activate virtual environment
    source "venv_name"/bin/activate

    # Deactivate virtual environment
    deactivate
```

1. Start by installing Celery in all machine.

```bash
    # [ALL MACHINES]
    sudo apt-get install celery -y
```

2. On the Master machine install Redis server.

```bash
    # [MASTER MACHINE]
    sudo apt-get install redis-server -y
```

3. Depending on the network the redis configuration file may need to be changed.

```bash
    # [MASTER MACHINE]
    # edit the config file
    sudo nano /etc/redis/redis.conf

    # in the NETWORK section change the bind to the master IP or 0.0.0.0 to listen to all interfaces
    bind 0.0.0.0
```

4. After that make sure the redis-server service is running.

```bash
    # get status of redis-server
    sudo systemctl status redis-server

    # restart redis-server
    sudo systemctl restart redis-server
```

5. Now navigate to the project directory and install all the requirements.

```bash
    # [ALL MACHINES]
    pip install -r requirements.txt
```

**Note:** The python version used for development and testing: **3.8.10**.

<br/>

---

<br/>

## How to run

1. Start the master machine.

```bash
    python main.py
```

2. Start the workers.

```bash
    python worker.py {masterip} {workername}
```

**Note:** Workers must have different names.

## Docker Compose

You can run the master, a worker, and Redis with Docker Compose.

1. Make sure Docker and Docker Compose are installed, and that GPU support is enabled on the host with the NVIDIA Container Toolkit.
2. Build and start the stack.

```bash
    docker compose up --build
```

The Compose file starts:

- `redis` as the message broker
- `master` running `main.py`
- `worker` running `worker.py`

The image is built with the full `repo/` tree, including `repo/models/test/checkpoints`, so the checkpoints used by the project are copied into the container image.

By default the worker name is `alice`. You can change it by editing `WORKER_NAME` in `docker-compose.yml`.

<br/>

**List of available commands:**

```
    - list_workers -l | -all              -> list live | all workers
    - sincronize_repo                     -> sincronize repo with workers
    - sincronize_models                   -> sincronize models with workers
    - list_jobs                           -> list jobs in progress
    - add_pipeline                        -> add a pipeline to the queue
    - h, help                             -> shows the list of available commands
```

To exit the program, use control + C and wait until the program exits.

<br/>

---

<br/>

## List of attacks

- Bandits and Prior (Black-box)
- CornerSearch (Black-box)
- DDN (White-box)
- DeepFool (White-box)
- GeoDA (Black-box)
- LogBarrier (White-box)
- SimBA (Black-box)
- Square Attack (Black-box)
- UAP (White-box)

**Note:** In the future, others will be added.

<br/>

## List of datasets

- CIFAR10

**Note:** In the future, others will be added.

<br/>

## List of models

- densenet
- googlenet
- mobilenet
- resnet
- senet
- vgg

**Note:** Others may be added.

<br/>

---

<br/>

## Configuration files

To test a model 2 configuration files are needed. By default, the program will get the 2 configuration files inside the _'configs'_ folder. The paths can be specified, and the configuration files edited accordingly to the necessities.

### General configuration file:

By default, it's the _'config.yaml'_ configuration file. This file contains the models and datasets information necessary to run the program.

**Structure:**

```yaml
    DATASET:
        dataset_name:
            run: [true|false] whether to run the dataset or not
            dataset_loader_path: path_to_dataset_loader


    MODEL:
        model_name:
            run: [true|false] whether to run the model or not
            model_path: path_to_model
            method_name: method_to_load_inside_the_file
            checkpoint_path: checkpoint_to_load [optional]
            args: arguments to load the method [optional]
```

**Note:** Multiple datasets and models can be defined, and the ones with the run flag as true will be tested.

<br/>

### Attack configuration file:

By default, it's the _'attacks_config.yaml'_ configuration file. This file contains the attacks parameters to execute on the model and dataset.

**Structure:**

```yaml
    ATTACK:
        attack_name:
            run: [true|false] whether to run the attack or not
            path: path_to_attack_file
            arg1: attack_argument_value
            arg2: attack_argument_value2
            ...
```

**Note:** Multiple attacks can be defined in the file. Use the default _'attacks_config.yaml'_ file as a guideline for the attacks arguments.

<br/>

---

<br/>

## How to add datasets

To add a different dataset, you must do the following steps:

### 1. DatasetLoader file:

Create a file where you define the following global variables:

- **DATASET_PATH** (path of the current file)
- **DATASET_NAME** (name of the dataset)
- **IMAGE_SIZE** (size of the images in the dataset)
- **MEAN** (mean of the dataset by channel)
- **STD** (standard deviation of the dataset by channel)
- **TRANSFORM** (transforms compose for test dataset)
- **TRAIN_TRANSFORM** (transforms compose for train dataset)
- **NUM_CLASSES** (number of classes of the dataset)

There also should exist a **_'dataLoader()'_** function that returns the train and tests loaders (dtype = torch.utils.data.dataloader.DataLoader).

You may use the datasetLoader.py file as a guide, available in **_'datasets\cifar-10\datasetLoader.py'_**.

### 2. Add dataset to configuration file:

Add the dataset information to the configuration file, by default the file is on the root of the project and is called **_'config.yaml'_**.

Use the following structure:

```yaml
dataset_name:
  run: [true|false]
  dataset_loader_path: path to the previously created file
```

<br/>

---

<br/>

## How to add models

To add a new model to be tested, you need to include some information in the **_'config.yaml'_** file. \
Add the information relative to your model using the following structure:

```yaml
model_name:
    run: [true|false]
    model_path: path to the file containing the model
    method_name: name of the class or method of the model
    checkpoint_path: [optional] path of model checkpoint
    args:
        - arg_1
        - arg_2
        [optional] arguments to be passed to the method/class
```

<br/>

---

<br/>

## How to add attacks

### 1. Create file

You can add an attack from any directory, but it's recommended that you create a folder with the name of the attack, inside include a file attack\*name.py and this folder should be inside the folder \*\*\*'attacks/'\_\*\* visible from the root of the project.

Then inside the created folder, create the file attack_name.py where attack_name is the name of your attack.

### 2. Prepare file

The file should be ready to be executed as an independent that receives the following parameters:

- model : path for a binary file that contains the model
- dataset : path for datasetLoader file
- total-images : number of images to be tested
- log : [true|false] flag for the logging messages
- results-path : path to store the results

**Note:** add all the necessary parameters to run your attack as well. The parameters above are just the default.

<br/>

**File behavior:**

The file should load the dataset from the datasetLoader, extract the model and the classified labels of the images from the binary file. \
Then execute the attack.
Store the results following the correct structure (results structure available in a section below), all results should go inside the attack_name folder.
The attack should also have a logger that writes to the attack_name.log file.

**Note:** There are functions available in the **_'utils\func_utils.py'_** file that can be used to store the results with the proper structure.

<br/>

### 3. Add attack to the attack configuration file

To add the attack, you should modify the **_'attacks_config.yaml'_** file and add the necessary parameters to run the attack. Use the following structure:

```yaml
attack_name:
    run: [true|false]
    path: path to the attack file
    param1: ...
    param2: ...
    ...
```

<br/>

---

<br/>

## Results structure

In this section it's presented the structure of the zip generated after executing the attacks that contains all the results.

```
zip_results
├── 1_model_dataset
│     ├── attacks
│     │     ├── attack_name
│     │     │     ├── perturbed_images
│     │     │     │     ├── l2_attackName_imageId.jpeg
│     │     │     │     └── ...
│     │     │     ├── attack_name.log
│     │     │     ├── labels.csv
│     │     │     └── sums.csv
│     │     └── ...
│     ├── original
│     │     ├── original_id.jpeg
│     │     └── ...
│     ├── plots
│     │     ├── plot_n.png
│     │     └── ...
│     ├── pipeline.log
│     ├── config.yaml
│     ├── results_by_attack.csv
│     └── robustness_score.csv
├── 2_...
├── config.yaml
├── attacks_config.yaml
└── robustness_score.csv
```

Where the **_attack_name_** and **_imageId_** should be changed to the name of the attack and id of the image in the dataset, respectively.

Clarification on the purpose of some folder and files:

- **_zip_results [folder]_**: folder where all results will be stored.
- **_1_model_dataset [folder]_**: folder where the results of one pipeline will be stored. Depending on the configuration files multiple files can be generated, the inital number represents a counter, the model and dataset are the names of the models and datasets used.
- **_attacks [folder]_**: folder where the results of individual attacks will be stored.
- **_perturbed_images [folder]_**: the folder contains all the perturbed images of the attack.
- **_config.yaml [file]_**: configuration file used to execute the pipeline, contains the parameters for the attacks used.
- **_attacks_config.yaml [file]_**: configuration file used to prepare the pipeline, contains parameters to load the models and datasets.
- **_l2_attackName_imageId.jpeg [file]_**: perturbed image, name follows the structure (l2_attackName_imageId).
- **_attack_name.log [file]_**: log file of the attack.
- **_labels.csv [file]_**: Stores values for each image tested (Columns: Ground truth label, Original classified label, Perturbed image classified label, L2, Queries).
- **_sums.csv [file]_**: Stores the total sum of some values (Columns: Number of images, Original images correctly classified by the model, Adversarial images correctly classified by the model).
- **_original [folder]_**: folder where the original images tested are stored.
- **_original_id [file]_**: original image, follows the structure (original_imageId).
- **_plots [folder]_**: folder where the plots generated are stored.
- **_plot_n.png [file]_**: plot generated after executing all the attack in a pipeline.
- **_pipeline.log [file]_**: log file of the pipeline.
- **_results_by_attack.csv [file]_**: Summarizes the information of all the attacks executed (Columns: Attack name, Adversarial images correctly classified by the model)
- **_robustness_score.csv [file]_**: File containing the robustness score given to the model.
# Thesis-Project
