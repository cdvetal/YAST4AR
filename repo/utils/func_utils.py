import os
import torch
import sys

from matplotlib import pyplot as plt
from PIL import Image
import glob
import yaml
import csv
import logging
import importlib
import subprocess

from datetime import datetime
import dill

import pandas as pd
import numpy as np

import utils.static_vars as static


class StdoutToLogging:
    def write(self, message):
        message = message.rstrip()
        if message.strip():
            if message.startswith("ERROR:"):
                message = message.replace("ERROR:", "", 1).strip()
                logging.error(message)
            elif message.startswith("WARNING:"):
                message = message.replace("WARNING:", "", 1).strip()
                logging.warning(message)
            else:
                logging.info(message)




def execute_command(command, parameters):
    print([sys.executable, command] + parameters)

    process = subprocess.Popen([sys.executable, command] + parameters, shell=False, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=os.getcwd())

    while True:
        line = process.stdout.readline()
        if not line:
            break
        sys.stdout.write(line.decode("utf-8"))
        #sys.stdout.flush()

    process.wait()
    line = process.stderr.readline()
    while line:
        sys.stdout.write("ERROR:" + line.decode("utf-8"))
        line = process.stderr.readline()


def generate_model_labels_binary(model_name, model, labels = []):
    file_name = "model_labels_" + datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + ".bin"
    path = os.path.join(static.PATH_TEMP, file_name)

    data = { 'model_name': model_name, 
            'model': model,
            'classified_labels': labels}
    
    serialized_data = dill.dumps(data)

    with open(path, "wb") as file:
        file.write(serialized_data)
        
    
    return path


def load_model(model_path, model_method_name, device, args, checkpoint = ''):

    # load model
    spec = importlib.util.spec_from_file_location('model', model_path)
    model_lib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_lib)

    # get function
    function = getattr(model_lib, model_method_name)
    if args:
        model = function(*args)
    else:
        model = function()

    model.to(device)
    model = torch.nn.DataParallel(model)
    if checkpoint:
        if torch.cuda.is_available():
            map_loc = None
        else:
            map_loc = torch.device('cpu')
        print("Loading {} checkpoint...".format(model_method_name))
        ckpt_model = torch.load(os.path.join(static.ROOT_PATH, checkpoint), map_location=map_loc)
        model.load_state_dict(ckpt_model['net'])
        model.eval()
        model.to(device)
    
    return model


## Normalization functions

def normalize_image(image, mean, std):
    imgs_tensor = image.clone()
    if imgs_tensor.dim() == 3:
        for i in range(imgs_tensor.size(0)):
            imgs_tensor[i, :, :] = (imgs_tensor[i, :, :] - mean[i]) / std[i]
    else:
        for i in range(imgs_tensor.size(1)):
            imgs_tensor[:, i, :, :] = (imgs_tensor[:, i, :, :] - mean[i]) / std[i]

    return imgs_tensor


def normalize_image_numpy(image, mean, std):
    if image.ndim == 3:
        for i in range(image.shape[0]):
            image[i, :, :] = (image[i, :, :] - mean[i]) / std[i]
    elif image.ndim == 4:
        for i in range(image.shape[1]):
            image[:, i, :, :] = (image[:, i, :, :] - mean[i]) / std[i]

    return image


def remove_normalization(image, mean, std):
    imgs_trans = image.clone()
    if len(image.size()) == 3:
        for i in range(image.size(0)):
            imgs_trans[i, :, :] = imgs_trans[i, :, :] * std[i] + mean[i]
    else:
        for i in range(image.size(1)):
            imgs_trans[:, i, :, :] = imgs_trans[:, i, :, :] * std[i] + mean[i]
    return imgs_trans


def calculate_l2_norm(im1, im2):
    return np.sqrt(np.sum((im1 - im2) ** 2))


################################################################################################
## Square attack axiliary functions

def dense_to_onehot(y_test, n_cls):
    y_test_onehot = np.zeros([len(y_test), n_cls], dtype=bool)
    y_test_onehot[np.arange(len(y_test)), y_test] = True
    return y_test_onehot


def random_classes_except_current(y_test, n_cls):
    y_test_new = np.zeros_like(y_test)
    for i_img in range(y_test.shape[0]):
        lst_classes = list(range(n_cls))
        lst_classes.remove(y_test[i_img])
        y_test_new[i_img] = np.random.choice(lst_classes)
    return y_test_new


################################################################################################

# prepare images and labels to be used by attacks
def get_images_labels_from_dataLoader(dataLoader, device, total_images = None):
    images = torch.tensor([], device=device)
    labels = torch.tensor([], device=device)
    
    for i, (inputs, targets) in enumerate(dataLoader):
        if total_images != None and i >= total_images:
            break
        images = torch.cat((images, inputs.to(device)), 0)
        labels = torch.cat((labels, targets.to(device)), 0).long()
        
    return images.to(device), labels.to(device)


# prepares arguments for attack
def prepare_attack_arguments(current_args, config_path, attack_name):
    # loads default arguments for attack
    default_args = {}
    if os.path.isfile(config_path):
        with open(config_path, "r") as stream:
            try:
                params = yaml.safe_load(stream)['ATTACK']
            except yaml.YAMLError as exc:
                print("ERROR: Exception loading config file: {}".format(exc))
                exit(0)

        for name, values in params.items():
            if (name.lower() == attack_name.lower()):
                default_args = values
                default_args['attack_name'] = name
                break
    
    # overwrites default argument if passed argument is available
    for arg_name, value in current_args.items():
        if value is not None:
            default_args[arg_name] = value

    # adds device argument
    if not 'device' in default_args or default_args['device'] is None:
        default_args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return default_args


################################################################################################

## Functions related to storage of the attack results

def save_statistics_of_attack(results_dict, original_labels, classified_labels, save_path, label_file_name = "labels.csv", sums_file_name = "sums.csv"):

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    correctly_classified_adversarial_images = 0
    initial_correct_classified_images = 0

    # csv results per image
    with open(os.path.join(save_path, label_file_name), 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Ground Truth label', 'Original classified label', 'Perturbed image classified label', 'L2', 'Queries'])

        for iter in range(len(results_dict['l2'])):
            writer.writerow([original_labels[iter].item(), classified_labels[iter], results_dict['perturbed_label'][iter].item(), results_dict['l2'][iter], results_dict['total_queries'][iter]])

            if original_labels[iter].item() == classified_labels[iter] == results_dict['perturbed_label'][iter].item():
                correctly_classified_adversarial_images += 1

            if original_labels[iter].item() == classified_labels[iter]:
                initial_correct_classified_images += 1

    # csv result sum
    with open(os.path.join(save_path, sums_file_name), 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Number of images', 'Original images correctly classified by the model', 'Adversarial images correctly classified by the model'])
        writer.writerow([len(original_labels), initial_correct_classified_images, correctly_classified_adversarial_images])


def save_images_attack(results_dict, original_labels, save_path):

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    for i in range(len(results_dict['perturbed_image'])):
        if results_dict['perturbed_label'][i] != original_labels[i].cpu():

            p_image = np.transpose(results_dict['perturbed_image'][i], (1, 2, 0))
            im = Image.fromarray((p_image * 255).astype(np.uint8))
            attack_name_new = results_dict['attack_name'].split(' ')[0]
            filename_save = '{}_{}_{}.jpeg'.format(str(results_dict['l2'][i]), attack_name_new, str(i))
            im.save(os.path.join(save_path, filename_save))


def save_original_images(folder_path, original_images):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    for j in range(len(original_images)):
        im = Image.fromarray(
            (np.transpose(original_images[j].cpu().detach().numpy(), (1, 2, 0)) * 255).astype(np.uint8))
        filename_save = 'original_{}.jpeg'.format(str(j))
        save_path = os.path.join(folder_path, filename_save)
        im.save(save_path)


def load_folder_images(folder_path, image_type='jpeg'):
    image_dict = {}
    for filename in glob.glob(os.path.join(folder_path, "*." + image_type)):
        im=Image.open(filename)
        image_dict[filename] = np.array(im, dtype=float)

    return image_dict


def load_attack_results(folder_path, dataset_mean, dataset_std, file_name = 'labels.csv'):
    if not os.path.isdir(folder_path):
        raise Exception("Folder path does not exist: {}".format(folder_path))
    
    results = {}
    
    for folder_name in os.listdir(folder_path):
        if not os.path.isfile(os.path.join(folder_path, folder_name, file_name)):
            continue

        labels = pd.read_csv(os.path.join(folder_path, folder_name, file_name), header=0, \
                            names=['true_label', 'classified_label', 'perturbed_label', 'l2', 'queries'], \
                            usecols=['true_label', 'classified_label', 'perturbed_label', 'l2', 'queries'])

        dict = load_folder_images(os.path.join(folder_path, folder_name, 'perturbed_images'))
    
        perturb_images = [[] for _ in range(len(labels['true_label']))]
        for name, image in dict.items():
            id = os.path.basename(name).split("_")[-1].split(".")[0]
            image = normalize_image_numpy(np.transpose(image, (2, 0, 1)), dataset_mean, dataset_std)
            image *= 1.0/image.max()
            perturb_images[int(id)] = np.clip(image, 0.0, 1.0)
        
        results[folder_name] = {
            'attack_name': folder_name,
            'perturbed_image': perturb_images,
            'perturbed_label': labels['perturbed_label'],
            'true_label': labels['true_label'],
            'classified_label': labels['classified_label'],
            'total_queries': labels['queries'],
            'l2': labels['l2']
        }

    return results


def generate_plots(results, original_images, correct_labels, save_path, num_samples, images_per_window = 5, img_size = 32):

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    curr_image = 0
    num_iterations = num_samples // images_per_window

    for i in range(num_iterations):
        fig, axes = plt.subplots(images_per_window, len(results.keys()) + 1, figsize=(img_size, img_size))
        axes[0, 0].set_title('Original')
        plot_iterator = 0

        for j in range(curr_image, curr_image + images_per_window, 1):
            axes[plot_iterator, 0].imshow(np.transpose(original_images[j].cpu().detach().numpy(), (1, 2, 0)))
            axes[plot_iterator, 0].axis('off')
            text_plot = 'Label: {}'.format(correct_labels[j])
            axes[plot_iterator, 0].annotate(xy=(0, -15), text=text_plot, xycoords='axes pixels')
            plot_iterator += 1

        plot_counter = 1
        for value in results.values():

            axes[0, plot_counter].set_title(value['attack_name'])
            
            plot_iterator = 0

            for j in range(curr_image, curr_image + images_per_window, 1):
                if correct_labels[j] == value['classified_label'][j] and value['perturbed_label'][j] != value['classified_label'][j]:
                    value['perturbed_image'][j] = np.clip(value['perturbed_image'][j], 0, 1.0)
                    axes[plot_iterator, plot_counter].imshow(np.transpose(value['perturbed_image'][j], (1, 2, 0)))
                    axes[plot_iterator, plot_counter].axis('off')
                    text_plot = 'Label: {}\nL2: {:.4f}\nQueries: {}'.format(value['perturbed_label'][j], value['l2'][j],
                                                                            value['total_queries'][j])
                    axes[plot_iterator, plot_counter].annotate(xy=(0, -50), text=text_plot, xycoords='axes pixels')                 

                else:
                    axes[plot_iterator, plot_counter].axis('off')

                plot_iterator += 1
            plot_counter += 1

        curr_image += images_per_window

        
        plt.subplots_adjust(hspace=2)
        filename_save = 'plot_{}.png'.format(str(i))
        fig.savefig(os.path.join(save_path, filename_save))
        plt.close()


def save_correct_classification(save_path, attacks_path, file_name = 'results_by_attack.csv', file_attack_info = 'labels.csv'):
    counters = {}
    for forder_name in os.listdir(attacks_path):
        counter = 0
        path = os.path.join(attacks_path, forder_name, file_attack_info)
        if os.path.isfile(path):
            data = pd.read_csv(path, names=['truth','classified', 'perturbed', 'l2', 'queries'])
            for i, row in data.iterrows():
                if row['truth'] == row['classified'] == row['perturbed']:
                    counter += 1

        counters[forder_name] = counter
    
    # csv result correclty classified by attack
    with open(os.path.join(save_path, file_name), 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Attack name', 'Adversarial images correctly classified by the model'])

        for name, counter in counters.items():
            writer.writerow([name, counter])

    
def calculate_robustness_score(save_path, attacks_path, file_name = 'robustness_score.csv', file_attack_info = 'labels.csv'):
    attacks = []
    scores = []
    perfect_score = 0
    num_misclassified_images = 0

    for forder_name in os.listdir(attacks_path):
        path = os.path.join(attacks_path, forder_name, file_attack_info)
        if os.path.isfile(path):
            data = pd.read_csv(path, header=0, names=['true_label', 'classified_label', 'perturbed_label', 'l2', 'queries'])
            
            model_robustness_score = 0
            for i in range(len(data)):
                label_value = 1
                if data['true_label'][i] != data['classified_label'][i]:
                    label_value = 0

                misclassified_value = 0
                if data['classified_label'][i] != data['perturbed_label'][i]:
                    misclassified_value = 1
                    num_misclassified_images += 1

                model_robustness_score += label_value * (1 - (1 / (1 + data['l2'][i]) * 0.5 + 0.5 * misclassified_value))
                perfect_score += 1

        attacks.append(forder_name)
        scores.append(model_robustness_score)

    print('Model Robustness Score: {} out of {}'.format(model_robustness_score, perfect_score))

    # get model and dataset names
    model_dataset_name = os.path.basename(os.path.dirname(attacks_path))
    model_dataset_name = model_dataset_name.split("_", 1)[1]


    with open(os.path.join(save_path, file_name), 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model_dataset used'] + attacks + ['Model Robustness Score', 'Max Robustness Score'])
        writer.writerow([model_dataset_name] + scores + [sum(scores), perfect_score])

    return model_dataset_name, {key: value for key, value in zip(attacks, scores)}, perfect_score, num_misclassified_images


def check_csv_headers(csv_file, expected_headers):
    # Check if the CSV file exists
    if not os.path.isfile(csv_file):
        return

    # Read the headers from the CSV file
    data_rows = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)
        for row in reader:
            data_rows.append(row)

    # Check if the headers match the expected headers
    if headers != expected_headers:

        # Identify extra headers that are not in the expected headers
        extra_headers = set(headers) - set(expected_headers)

        # Remove extra headers from the headers list
        headers = [header for header in headers if header not in extra_headers]

        # Remove corresponding columns from the data rows
        for row in data_rows:
            row[:] = [value for header, value in zip(headers, row) if header not in extra_headers]

        # Add missing headers/columns to the CSV file
        missing_headers = set(expected_headers) - set(headers)

        if missing_headers:
            new_headers = headers + list(missing_headers)

            # Add empty values for missing headers in the existing data rows
            for row in data_rows:
                while len(row) < len(new_headers):
                    row.append('')

            with open(csv_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(new_headers)
                writer.writerows(data_rows)




def update_log_robustness(dataset_model_name, dic_results, num_misclassified_images, max_possible, save_path, file_name = 'Robustness_log.csv'):
    log = os.path.join(os.path.dirname(save_path), file_name)

    # get list of all attacks
    attacks = []
    attacks_path = os.path.join(save_path, 'attacks')
    for forder_name in os.listdir(attacks_path):
        if os.path.isdir(os.path.join(attacks_path, forder_name)) and forder_name != '__pycache__':
            attacks.append(forder_name)

    if not os.path.isfile(log):
        # create csv file
        with open(log, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Model_dataset used', 'id'] + attacks + ['Model Robustness Score', 'Max Robustness Score', 'Images Correctly Classified'])

    check_csv_headers(log, ['Model_dataset used', 'id'] + attacks + ['Model Robustness Score', 'Max Robustness Score', 'Images Correctly Classified'])
    
    # calculate id
    with open(log, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)
        last_id = rows[-1][1]
        try: 
            last_id = int(last_id)
        except:
            last_id = 0


    # append to csv file
    with open(log, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([dataset_model_name, str(last_id + 1)] + prepare_attacks_values(dic_results, attacks) + [str(sum(dic_results.values())), str(max_possible), str(max_possible - num_misclassified_images)])


    

def prepare_attacks_values(dic_results, list_all_attacks):
    values = []
    for attack in list_all_attacks:
        if attack in dic_results.keys():
            values.append(dic_results[attack])
        else:
            values.append(None)

    return values