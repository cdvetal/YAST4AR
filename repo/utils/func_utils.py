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

import traceback

from datetime import datetime
import dill

import pandas as pd
import numpy as np
import pickle
from . import static_vars as static


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

    if process.returncode != 0:
        raise RuntimeError(
            "Command failed with return code {}: {} {}".format(
                process.returncode,
                command,
                " ".join(parameters),
            )
        )


def generate_model_labels_binary(model_name, model, labels = []):
    # Backwards-compatible signature: if first arg is a model instance, shift parameters
    if not isinstance(model_name, str) and (hasattr(model_name, '__call__') or isinstance(model_name, torch.nn.Module)):
        # called as generate_model_labels_binary(model, labels)
        model_obj = model_name
        labels = model
        model_name = getattr(model_obj, 'name', 'model')
        model = model_obj

    file_name = "model_labels_" + datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    path = os.path.join(static.PATH_TEMP, file_name + ".bin")

    os.makedirs(static.PATH_TEMP, exist_ok=True)

    # If model is a TorchScript/ScriptModule, avoid pickling the module directly
    # (may reference foreign classes). Save it to a temp file and wrap it in a
    # lightweight module-compatible proxy.
    class ScriptedModelProxy(torch.nn.Module):
        def __init__(self, ts_path, name=None, device='cpu'):
            super().__init__()
            self._ts_path = ts_path
            self.name = name
            self._module = None
            self._device = device

        def _ensure_loaded(self):
            if self._module is None:
                self._module = torch.jit.load(self._ts_path, map_location=self._device)
                try:
                    self._module.eval()
                except Exception:
                    pass

        def forward(self, *args, **kwargs):
            self._ensure_loaded()
            return self._module(*args, **kwargs)

        def to(self, device):
            self._device = device
            if self._module is not None:
                try:
                    self._module.to(device)
                except Exception:
                    pass
            return self

        def eval(self):
            if self._module is not None:
                try:
                    self._module.eval()
                except Exception:
                    pass
            return self

    # Keep compatibility with pickles that resolve through this synthetic module.
    try:
        ScriptedModelProxy.__module__ = 'repo_utils.func_utils'
    except Exception:
        pass

    # Try to save scripted model if possible
    use_proxy = False
    proxy = None
    try:
        # attempt to save model as TorchScript to avoid pickling framework classes
        ts_path = os.path.join(static.PATH_TEMP, file_name + ".ts")
        # if model is already a scripted module, torch.jit.save will work; if not, this will raise
        torch.jit.save(model, ts_path)
        proxy = ScriptedModelProxy(ts_path, name=model_name)
        use_proxy = True
    except Exception:
        use_proxy = False

    data = { 'model_name': model_name,
            'model': proxy if use_proxy else model,
            'classified_labels': labels}

    serialized_data = dill.dumps(data)

    with open(path, "wb") as file:
        file.write(serialized_data)

    return path


def load_model(model_path, model_method_name, device, args, checkpoint = ''):
    loaded_from_torchscript = False

    def _to_abs_path(path):
        if not path:
            return ''
        if os.path.isabs(path):
            return path
        return os.path.join(static.ROOT_PATH, path)

    def _is_state_dict(candidate):
        return isinstance(candidate, dict) and len(candidate) > 0 and all(torch.is_tensor(v) for v in candidate.values())

    def _extract_state_dict(checkpoint_obj):
        if _is_state_dict(checkpoint_obj):
            return checkpoint_obj

        if isinstance(checkpoint_obj, dict):
            for key in ['net', 'state_dict', 'model_state_dict', 'weights', 'params']:
                value = checkpoint_obj.get(key)
                if _is_state_dict(value):
                    return value

            model_entry = checkpoint_obj.get('model')
            if _is_state_dict(model_entry):
                return model_entry

        return None

    def _load_state_dict_with_fallbacks(base_model, state_dict):
        if state_dict is None:
            raise RuntimeError('No valid state_dict found in checkpoint.')

        direct_error = None
        stripped_error = None

        try:
            base_model.load_state_dict(state_dict)
            return
        except Exception as e:
            direct_error = e

        stripped_state = {}
        for k, v in state_dict.items():
            nk = k.replace('module.', '', 1) if k.startswith('module.') else k
            stripped_state[nk] = v

        try:
            base_model.load_state_dict(stripped_state)
            return
        except Exception as e:
            stripped_error = e

        raise RuntimeError(
            'Failed to load checkpoint state_dict into model. '
            'Direct load error: {} | Stripped-prefix load error: {}'.format(direct_error, stripped_error)
        )

    def _build_model_from_factory(path, method_name, method_args):
        if not path or not method_name:
            return None

        spec = importlib.util.spec_from_file_location('model', path)
        model_lib = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_lib)

        function = getattr(model_lib, method_name)
        if method_args:
            return function(*method_args)
        return function()

    def _load_checkpoint(path, map_loc, try_weights_only_first=False):
        weights_only_error = None

        if try_weights_only_first:
            try:
                return torch.load(path, map_location=map_loc, weights_only=True)
            except TypeError:
                # Older torch versions may not support weights_only.
                pass
            except Exception as e:
                weights_only_error = e

        try:
            return torch.load(path, map_location=map_loc, pickle_module=pickle, encoding='latin1')
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "Checkpoint requires external module '{}' during deserialization. "
                "This usually means the .pt stores a full pickled model from another framework. "
                "Either install that framework in this environment, or re-export the checkpoint as state_dict "
                "(for example {{'state_dict': model.state_dict()}}) from the original training code.".format(str(e).split("'")[1] if "'" in str(e) else str(e))
            ) from e
        except Exception as e:
            if weights_only_error is not None:
                raise RuntimeError(
                    'Failed to load checkpoint with both weights_only and regular torch.load. '
                    'weights_only error: {} | regular load error: {}'.format(weights_only_error, e)
                ) from e
            raise

    def _try_load_torchscript(path, map_loc):
        try:
            return torch.jit.load(path, map_location=map_loc)
        except Exception as e:
            raise Exception("MIAU", str(e))

    try:
        model = _build_model_from_factory(model_path, model_method_name, args)

        if checkpoint:
            if torch.cuda.is_available():
                map_loc = None
            else:
                map_loc = torch.device('cpu')

            checkpoint_path = _to_abs_path(checkpoint)
            # If the provided path doesn't exist, try to find the file by basename in the project.
            if not os.path.exists(checkpoint_path):
                basename = os.path.basename(checkpoint_path)
                print(f"Warning: checkpoint {checkpoint_path} not found — searching for '{basename}' in project...")
                candidates = []
                for root, dirs, files in os.walk(static.ROOT_PATH):
                    if basename in files:
                        candidates.append(os.path.join(root, basename))
                if candidates:
                    old = checkpoint_path
                    checkpoint_path = os.path.abspath(candidates[0])
                    print(f"Found checkpoint at {checkpoint_path}; using this path instead of {old}")

            print("Loading {} checkpoint...".format(model_method_name if model_method_name else 'custom'))

            _, ext = os.path.splitext(checkpoint_path.lower())
            ckpt_model = None

            # First try TorchScript for architecture-agnostic inference.
            if ext in ['.pt', '.jit', '.ts']:
                scripted_model = _try_load_torchscript(checkpoint_path, map_loc)
                if scripted_model is not None:
                    model = scripted_model
                    loaded_from_torchscript = True

            if not loaded_from_torchscript:
                ckpt_model = _load_checkpoint(checkpoint_path, map_loc, try_weights_only_first=(ext == '.pt'))

            # For unknown architectures (.pt), allow full serialized model loading.
            if ext == '.pt' and not loaded_from_torchscript:
                if isinstance(ckpt_model, torch.nn.Module):
                    model = ckpt_model
                elif isinstance(ckpt_model, dict) and isinstance(ckpt_model.get('model'), torch.nn.Module):
                    model = ckpt_model['model']
                else:
                    state_dict = _extract_state_dict(ckpt_model)
                    if model is None:
                        raise RuntimeError(
                            'Checkpoint {} appears to contain only weights/state_dict. '
                            'Please provide model_path and method_name, or save a full model object in the .pt file.'.format(checkpoint_path)
                        )
                    _load_state_dict_with_fallbacks(model, state_dict)

            elif not loaded_from_torchscript:
                # Backward-compatible .pth/.pth.tar flow with additional fallback keys.
                state_dict = _extract_state_dict(ckpt_model)
                if model is None:
                    raise RuntimeError(
                        'Checkpoint {} requires a model definition (model_path + method_name).'.format(checkpoint_path)
                    )
                _load_state_dict_with_fallbacks(model, state_dict)

            if model is None:
                raise RuntimeError('Model could not be created. Please check model_path/method_name or checkpoint format.')

            model.to(device)
            model.eval()
            model.to(device)

            return model
    except Exception as e:
        try:
            os.makedirs(static.PATH_TEMP, exist_ok=True)
            tb_path = os.path.join(static.PATH_TEMP, 'loader_traceback.log')
            with open(tb_path, 'a') as f:
                f.write('---- {} ----\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                f.write('model_path: {}\n'.format(model_path))
                f.write('method_name: {}\n'.format(model_method_name))
                f.write('checkpoint: {}\n'.format(checkpoint))
                f.write('Exception: {}\n'.format(repr(e)))
                f.write(traceback.format_exc())
                f.write('\n')
            print('ERROR: Loader crashed; full traceback written to {}'.format(tb_path))
        except Exception:
            # best-effort; don't mask original exception
            pass
        raise


def _normalize_path(p, base_dir):
    if p is None:
        return None
    p = str(p).strip()
    p = p.replace('\\', os.sep).replace('/', os.sep)
    return os.path.abspath(os.path.join(base_dir, p)) if not os.path.isabs(p) else os.path.abspath(p)


def load_surrogate_from_ckpt(model_class='DenseNet121'):
    # project root for this repository (two levels up from repo/utils)
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(this_dir, '..', '..'))

    # load config.yaml to get correct paths
    config_path = os.path.join(project_root, 'configs', 'config.yaml')
    model_path = None
    checkpoint_path = None
    method_name = model_class
    try:
        if os.path.isfile(config_path):
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
            models_cfg = cfg.get('MODEL') or {}
            for k, v in models_cfg.items():
                if k.lower() == model_class.lower():
                    model_path = _normalize_path(v.get('model_path'), project_root)
                    checkpoint_path = _normalize_path(v.get('checkpoint_path'), project_root)
                    method_name = v.get('method_name') or method_name
                    break
    except Exception:
        pass

    # fallback sensible defaults (relative to project root)
    if model_path is None:
        model_path = os.path.join(project_root, 'models', 'test', f'{model_class}.py')
    if checkpoint_path is None:
        checkpoint_path = os.path.join(project_root, 'models', 'test', 'checkpoints', f'ckpt_{model_class}.pth')

    # try alternative base (some projects store models under repo/)
    alt_model_path = os.path.join(project_root, 'repo', os.path.relpath(model_path, project_root))
    alt_checkpoint_path = os.path.join(project_root, 'repo', os.path.relpath(checkpoint_path, project_root))

    if not os.path.isfile(model_path) and os.path.isfile(alt_model_path):
        model_path = alt_model_path
    if not os.path.isfile(checkpoint_path) and os.path.isfile(alt_checkpoint_path):
        checkpoint_path = alt_checkpoint_path

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # save into project temp folder to avoid cluttering project root
    temp_dir = os.path.join(project_root, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    out_path = os.path.join(temp_dir, f'surrogate_{model_class}.bin')

    spec = importlib.util.spec_from_file_location('surrogate_model_module', model_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    model_factory = getattr(m, method_name)
    model = model_factory()

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state = None
    if isinstance(ckpt, dict):
        for key in ('net', 'state_dict', 'model'):
            if key in ckpt:
                state = ckpt[key]
                break
    if state is None:
        state = ckpt

    try:
        model.load_state_dict(state)
    except Exception:
        new_state = {}
        if isinstance(state, dict):
            for k, v in state.items():
                nk = k.replace('module.', '') if k.startswith('module.') else k
                new_state[nk] = v
        try:
            model.load_state_dict(new_state)
        except Exception as e:
            raise RuntimeError(f'Failed loading checkpoint into model: {e}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    model.to(device)

    with open(out_path, 'wb') as f:
        dill.dump({'model': model, 'classified_labels': []}, f)

    print('Saved surrogate to', os.path.abspath(out_path))
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


def filter_correctly_classified(images, labels, classified_labels):
    if classified_labels is None:
        return images, labels, classified_labels, None, None

    try:
        cls_tensor = torch.as_tensor(classified_labels).view(-1)
    except Exception:
        return images, labels, classified_labels, None, None

    if cls_tensor.numel() == 0:
        return images, labels, classified_labels, None, None

    try:
        lbl_tensor = labels if torch.is_tensor(labels) else torch.as_tensor(labels)
        lbl_tensor = lbl_tensor.view(-1)
    except Exception:
        return images, labels, classified_labels, None, None

    n = min(lbl_tensor.numel(), cls_tensor.numel())
    if n == 0:
        return images, labels, classified_labels, None, None

    if lbl_tensor.numel() != cls_tensor.numel():
        print("Warning: classified_labels length does not match labels length; truncating to shortest.")

    lbl_slice = lbl_tensor[:n]
    cls_slice = cls_tensor[:n].to(lbl_slice.device)
    mask = cls_slice == lbl_slice
    try:
        orig_indices = torch.arange(n, device=mask.device)[mask].detach().cpu().numpy().tolist()
    except Exception:
        orig_indices = None

    if torch.is_tensor(images):
        try:
            mask_for_images = mask
            if images.device != mask.device:
                mask_for_images = mask.to(images.device)
            images = images[:n][mask_for_images]
        except Exception:
            pass

    try:
        labels = lbl_slice[mask]
    except Exception:
        labels = lbl_slice

    try:
        classified_labels = cls_slice[mask]
    except Exception:
        classified_labels = cls_slice

    return images, labels, classified_labels, mask, orig_indices


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
    # determine how many rows we can safely write (guard against mismatched lengths)
    l2_list = results_dict.get('l2', [])
    perturbed_list = results_dict.get('perturbed_label', [])
    queries_list = results_dict.get('total_queries', [])
    maxp = results_dict.get('max_perturbation', '')
    orig_indices = results_dict.get('orig_index', None)
    if orig_indices is not None and not isinstance(orig_indices, (list, tuple, np.ndarray)):
        try:
            orig_indices = list(orig_indices)
        except Exception:
            orig_indices = None

    # lengths (treat pandas/series as having len)
    num_l2 = len(l2_list) if hasattr(l2_list, '__len__') else 0
    num_orig = len(original_labels) if original_labels is not None else 0
    num_classified = len(classified_labels) if (classified_labels is not None and hasattr(classified_labels, '__len__')) else num_orig

    rows = min(num_l2, num_orig, num_classified)
    if orig_indices is not None:
        try:
            rows = min(rows, len(orig_indices))
        except Exception:
            orig_indices = None

    with open(os.path.join(save_path, label_file_name), 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        headers = ['Ground Truth label', 'Original classified label', 'Perturbed image classified label', 'MaxPerturbation', 'L2', 'Queries']
        if orig_indices is not None:
            headers.append('orig_index')
        writer.writerow(headers)

        for i in range(rows):
            # get original label
            try:
                orig_lbl = int(original_labels[i].item())
            except Exception:
                try:
                    orig_lbl = int(original_labels[i])
                except Exception:
                    orig_lbl = None

            # get classified label (fallback to orig if missing)
            try:
                cls_val = classified_labels[i]
                if hasattr(cls_val, 'item'):
                    cls_lbl = int(cls_val.item())
                else:
                    cls_lbl = int(cls_val)
            except Exception:
                cls_lbl = orig_lbl

            # get perturbed label
            try:
                pert_val = perturbed_list[i]
                if hasattr(pert_val, 'item'):
                    pert_lbl = int(pert_val.item())
                else:
                    pert_lbl = int(pert_val)
            except Exception:
                pert_lbl = None

            # get queries and l2
            try:
                l2_val = l2_list[i]
            except Exception:
                l2_val = ''
            try:
                q_val = queries_list[i]
            except Exception:
                q_val = ''

            row = [orig_lbl, cls_lbl, pert_lbl, maxp, l2_val, q_val]
            if orig_indices is not None:
                try:
                    row.append(int(orig_indices[i]))
                except Exception:
                    row.append(orig_indices[i])
            writer.writerow(row)

            if orig_lbl is not None and cls_lbl is not None and pert_lbl is not None:
                if orig_lbl == cls_lbl == pert_lbl:
                    correctly_classified_adversarial_images += 1

            if orig_lbl is not None and cls_lbl is not None and orig_lbl == cls_lbl:
                initial_correct_classified_images += 1

    # csv result sum
    with open(os.path.join(save_path, sums_file_name), 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Number of images', 'Original images correctly classified by the model', 'Adversarial images correctly classified by the model'])
        writer.writerow([len(original_labels), initial_correct_classified_images, correctly_classified_adversarial_images])


def save_images_attack(results_dict, original_labels, save_path):

    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    perturbed_images = results_dict.get('perturbed_image', [])
    perturbed_labels = results_dict.get('perturbed_label', [])
    orig_indices = results_dict.get('orig_index', None)
    if orig_indices is not None and not isinstance(orig_indices, (list, tuple, np.ndarray)):
        try:
            orig_indices = list(orig_indices)
        except Exception:
            orig_indices = None

    # Determine safe number of items to iterate
    try:
        num_perturbed = len(perturbed_images)
    except Exception:
        num_perturbed = 0

    try:
        num_original = len(original_labels)
    except Exception:
        num_original = 0

    num = min(num_perturbed, num_original)

    for i in range(num):
        # get original label as int
        try:
            orig_lbl = original_labels[i].cpu().item()
        except Exception:
            try:
                orig_lbl = int(original_labels[i])
            except Exception:
                orig_lbl = None

        # get perturbed label as int
        try:
            pert_val = perturbed_labels[i]
            if hasattr(pert_val, 'item'):
                pert_lbl = int(pert_val.item())
            else:
                pert_lbl = int(pert_val)
        except Exception:
            pert_lbl = None

        # If labels differ (or pert_lbl missing), save the perturbed image
        save_image = False
        if orig_lbl is None:
            save_image = True
        elif pert_lbl is None:
            save_image = True
        else:
            save_image = (pert_lbl != int(orig_lbl))

        if save_image:
            p_image = np.transpose(perturbed_images[i], (1, 2, 0))
            im = Image.fromarray((p_image * 255).astype(np.uint8))
            attack_name_new = results_dict.get('attack_name', '').split(' ')[0]
            l2_val = results_dict.get('l2', [])
            try:
                l2_str = str(l2_val[i])
            except Exception:
                l2_str = ''
            try:
                orig_idx = int(orig_indices[i]) if orig_indices is not None else int(i)
            except Exception:
                orig_idx = i
            filename_save = '{}_{}_{}.jpeg'.format(l2_str, attack_name_new, str(orig_idx))
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


def _read_labels_csv(csv_path):
    default_names = ['true_label', 'classified_label', 'perturbed_label', 'max_perturbation', 'l2', 'queries']
    column_map = {
        'Ground Truth label': 'true_label',
        'Original classified label': 'classified_label',
        'Perturbed image classified label': 'perturbed_label',
        'MaxPerturbation': 'max_perturbation',
        'L2': 'l2',
        'Queries': 'queries',
        'orig_index': 'orig_index',
        'Original index': 'orig_index',
    }

    try:
        labels = pd.read_csv(csv_path)
        labels = labels.rename(columns=column_map)
        required = set(default_names)
        if not required.issubset(set(labels.columns)):
            labels = pd.read_csv(csv_path, header=0, names=default_names)
        labels = labels.rename(columns=column_map)
        return labels
    except Exception:
        labels = pd.read_csv(csv_path, header=0, names=default_names)
        return labels.rename(columns=column_map)


def load_attack_results(folder_path, dataset_mean, dataset_std, file_name = 'labels.csv'):
    if not os.path.isdir(folder_path):
        raise Exception("Folder path does not exist: {}".format(folder_path))
    
    results = {}
    
    for folder_name in os.listdir(folder_path):
        if not os.path.isfile(os.path.join(folder_path, folder_name, file_name)):
            continue

        labels = _read_labels_csv(os.path.join(folder_path, folder_name, file_name))

        orig_indices = None
        if 'orig_index' in labels.columns:
            try:
                orig_indices = labels['orig_index'].tolist()
                orig_indices = [int(x) for x in orig_indices if pd.notna(x)]
            except Exception:
                orig_indices = None

        dict = load_folder_images(os.path.join(folder_path, folder_name, 'perturbed_images'))
    
        if orig_indices is not None and len(orig_indices) > 0:
            size = max(orig_indices) + 1
            perturb_images = [None for _ in range(size)]
        else:
            perturb_images = [[] for _ in range(len(labels['true_label']))]
        for name, image in dict.items():
            id = os.path.basename(name).split("_")[-1].split(".")[0]
            image = normalize_image_numpy(np.transpose(image, (2, 0, 1)), dataset_mean, dataset_std)
            image *= 1.0/image.max()
            try:
                perturb_images[int(id)] = np.clip(image, 0.0, 1.0)
            except Exception:
                pass

        if orig_indices is not None and len(orig_indices) > 0:
            size = len(perturb_images)
            true_label = [None for _ in range(size)]
            classified_label = [None for _ in range(size)]
            perturbed_label = [None for _ in range(size)]
            l2_list = [None for _ in range(size)]
            queries_list = [None for _ in range(size)]
            maxp_list = [None for _ in range(size)]

            for idx, row in labels.iterrows():
                try:
                    oi = int(row.get('orig_index'))
                except Exception:
                    continue
                if oi < 0 or oi >= size:
                    continue
                true_label[oi] = row.get('true_label')
                classified_label[oi] = row.get('classified_label')
                perturbed_label[oi] = row.get('perturbed_label')
                l2_list[oi] = row.get('l2')
                queries_list[oi] = row.get('queries')
                maxp_list[oi] = row.get('max_perturbation')
        else:
            true_label = labels.get('true_label')
            classified_label = labels.get('classified_label')
            perturbed_label = labels.get('perturbed_label')
            l2_list = labels.get('l2')
            queries_list = labels.get('queries')
            maxp_list = labels.get('max_perturbation')

        results[folder_name] = {
            'attack_name': folder_name,
            'perturbed_image': perturb_images,
            'perturbed_label': perturbed_label,
            'true_label': true_label,
            'classified_label': classified_label,
            'total_queries': queries_list,
            'l2': l2_list,
            'max_perturbation': maxp_list
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
                # Helper to safely get indexed values from lists, numpy arrays or pandas Series
                def _get(container, idx):
                    try:
                        if container is None:
                            return None
                        if hasattr(container, 'iloc'):
                            if idx < 0 or idx >= len(container):
                                return None
                            return container.iloc[idx]
                        if isinstance(container, (list, tuple, np.ndarray)):
                            if idx < 0 or idx >= len(container):
                                return None
                            return container[idx]
                        return container[idx]
                    except Exception:
                        try:
                            return container.get(idx, None)
                        except Exception:
                            return None

                cls_lbl = _get(value.get('classified_label', None), j)
                pert_lbl = _get(value.get('perturbed_label', None), j)
                pert_img = _get(value.get('perturbed_image', None), j)
                l2_val = _get(value.get('l2', None), j)
                q_val = _get(value.get('total_queries', None), j)

                if cls_lbl is not None and pert_lbl is not None and pert_img is not None and correct_labels[j] == cls_lbl and pert_lbl != cls_lbl:
                    try:
                        img_to_show = np.clip(pert_img, 0, 1.0)
                        axes[plot_iterator, plot_counter].imshow(np.transpose(img_to_show, (1, 2, 0)))
                        axes[plot_iterator, plot_counter].axis('off')
                    except Exception:
                        axes[plot_iterator, plot_counter].axis('off')

                    # include max perturbation (may be scalar or per-sample)
                    maxp_raw = value.get('max_perturbation', '')
                    maxp_display = ''
                    try:
                        if hasattr(maxp_raw, 'iloc'):
                            raw_val = maxp_raw.iloc[j]
                        elif isinstance(maxp_raw, (list, tuple, np.ndarray)):
                            raw_val = maxp_raw[j]
                        else:
                            raw_val = maxp_raw

                        if raw_val is None or (isinstance(raw_val, float) and np.isnan(raw_val)):
                            maxp_display = ''
                        else:
                            maxp_display = f"{float(raw_val):.4f}"
                    except Exception:
                        try:
                            maxp_display = str(maxp_raw)
                        except Exception:
                            maxp_display = ''

                    text_plot = 'Label: {}\nL2: {}\nQueries: {}\nMaxPerturb: {}'.format(
                        str(pert_lbl), (f"{float(l2_val):.4f}" if (l2_val is not None) else ''), (str(q_val) if q_val is not None else ''), maxp_display)

                    ax = axes[plot_iterator, plot_counter]
                    ax.text(0.5, -0.18, text_plot, transform=ax.transAxes, ha='center', va='top', fontsize=8,
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
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
            data = _read_labels_csv(path)
            for i, row in data.iterrows():
                if row.get('true_label') == row.get('classified_label') == row.get('perturbed_label'):
                    counter += 1

        counters[forder_name] = counter
    
    # csv result correclty classified by attack
    with open(os.path.join(save_path, file_name), 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Attack name', 'Adversarial images correctly classified by the model'])

        for name, counter in counters.items():
            writer.writerow([name, counter])

def mean_std(series):
    series = pd.to_numeric(series, errors='coerce')
    mean_val = series.mean(skipna=True)
    std_val = series.std(skipna=True)
    if pd.isna(mean_val):
        return None, None
    return float(mean_val), (None if pd.isna(std_val) else float(std_val))

def fmt_mean_std(mean_val, std_val, decimals=4):
    if mean_val is None:
        return None
    if std_val is None:
        return '{:.{prec}f}'.format(mean_val, prec=decimals)
    return '{:.{prec}f} ± {:.{prec}f}'.format(mean_val, std_val, prec=decimals)

    
def calculate_robustness_score(save_path, attacks_path, file_name = 'robustness_score.csv', file_attack_info = 'labels.csv'):
    attacks = []
    scores = []
    perfect_score = 0
    num_misclassified_images = 0
    clean_acc_by_attack = {}
    robust_acc_by_attack = {}
    linf_avg_by_attack = {}
    l2_avg_by_attack = {}
    queries_avg_by_attack = {}
    clean_acc_str_by_attack = {}
    robust_acc_str_by_attack = {}
    linf_str_by_attack = {}
    l2_str_by_attack = {}
    queries_str_by_attack = {}

    for forder_name in os.listdir(attacks_path):
        path = os.path.join(attacks_path, forder_name, file_attack_info)
        if os.path.isfile(path):
            data = _read_labels_csv(path)
            
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

            total_samples = max(1, len(data))
            clean_acc = float((data['true_label'] == data['classified_label']).sum()) / total_samples
            robust_acc = float((data['true_label'] == data['perturbed_label']).sum()) / total_samples
            clean_acc_by_attack[forder_name] = clean_acc
            robust_acc_by_attack[forder_name] = robust_acc

            linf_mean, linf_std = mean_std(data['max_perturbation'])
            l2_mean, l2_std = mean_std(data['l2'])
            queries_mean, queries_std = mean_std(data['queries'])
            clean_acc_str_by_attack[forder_name] = fmt_mean_std(clean_acc, None, decimals=4)
            robust_acc_str_by_attack[forder_name] = fmt_mean_std(robust_acc, None, decimals=4)
            linf_avg_by_attack[forder_name] = linf_mean
            l2_avg_by_attack[forder_name] = l2_mean
            queries_avg_by_attack[forder_name] = queries_mean
            linf_str_by_attack[forder_name] = fmt_mean_std(linf_mean, linf_std, decimals=4)
            l2_str_by_attack[forder_name] = fmt_mean_std(l2_mean, l2_std, decimals=4)
            queries_str_by_attack[forder_name] = fmt_mean_std(queries_mean, queries_std, decimals=4)

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

    return (
        model_dataset_name,
        {key: value for key, value in zip(attacks, scores)},
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
    )


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




def update_log_robustness(
    dataset_model_name,
    dic_results,
    num_misclassified_images,
    max_possible,
    save_path,
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
    file_name = 'Robustness_log.csv',
):
    log = os.path.join(os.path.dirname(save_path), file_name)
    log_xlsx = os.path.splitext(log)[0] + '.xlsx'
    log_tex = os.path.splitext(log)[0] + '.tex'

    # get list of all attacks
    attacks = []
    attacks_path = os.path.join(save_path, 'attacks')
    for forder_name in os.listdir(attacks_path):
        if os.path.isdir(os.path.join(attacks_path, forder_name)) and forder_name != '__pycache__':
            attacks.append(forder_name)

    clean_acc_headers = ['Clean Accuracy {}'.format(a) for a in attacks]
    robust_acc_headers = ['Robust Accuracy {}'.format(a) for a in attacks]
    linf_avg_headers = ['Avg L_inf {}'.format(a) for a in attacks]
    l2_avg_headers = ['Avg L2 {}'.format(a) for a in attacks]
    queries_avg_headers = ['Avg Queries {}'.format(a) for a in attacks]
    clean_acc_str_headers = ['Clean Accuracy (mean ± std) {}'.format(a) for a in attacks]
    robust_acc_str_headers = ['Robust Accuracy (mean ± std) {}'.format(a) for a in attacks]
    linf_str_headers = ['Avg L_inf (mean ± std) {}'.format(a) for a in attacks]
    l2_str_headers = ['Avg L2 (mean ± std) {}'.format(a) for a in attacks]
    queries_str_headers = ['Avg Queries (mean ± std) {}'.format(a) for a in attacks]

    def aggregate_mean_std(values):
        numeric = [v for v in values if v is not None]
        if not numeric:
            return None, None
        series = pd.Series(numeric, dtype='float')
        return float(series.mean()), float(series.std()) if len(series) > 1 else 0.0

    agg_clean_mean, agg_clean_std = aggregate_mean_std(prepare_attacks_values(clean_acc_by_attack, attacks))
    agg_robust_mean, agg_robust_std = aggregate_mean_std(prepare_attacks_values(robust_acc_by_attack, attacks))
    agg_linf_mean, agg_linf_std = aggregate_mean_std(prepare_attacks_values(linf_avg_by_attack, attacks))
    agg_l2_mean, agg_l2_std = aggregate_mean_std(prepare_attacks_values(l2_avg_by_attack, attacks))
    agg_queries_mean, agg_queries_std = aggregate_mean_std(prepare_attacks_values(queries_avg_by_attack, attacks))

    if not os.path.isfile(log):
        # create csv file
        with open(log, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                ['Model_dataset used', 'id']
                + attacks
                + clean_acc_headers
                + robust_acc_headers
                + linf_avg_headers
                + l2_avg_headers
                + queries_avg_headers
                #+ clean_acc_str_headers
                #+ robust_acc_str_headers
                #+ linf_str_headers
                #+ l2_str_headers
                #+ queries_str_headers
                + ['All Attacks Clean Accuracy (mean ± std)']
                + ['All Attacks Robust Accuracy (mean ± std)']
                + ['All Attacks Avg L_inf (mean ± std)']
                + ['All Attacks Avg L2 (mean ± std)']
                + ['All Attacks Avg Queries (mean ± std)']
                + ['Model Robustness Score', 'Max Robustness Score', 'Images Correctly Classified']
            )

    check_csv_headers(
        log,
        ['Model_dataset used', 'id']
        + attacks
        + clean_acc_headers
        + robust_acc_headers
        + linf_avg_headers
        + l2_avg_headers
        + queries_avg_headers
        #+ clean_acc_str_headers
        #+ robust_acc_str_headers
        #+ linf_str_headers
        #+ l2_str_headers
        #+ queries_str_headers
        + ['All Attacks Clean Accuracy (mean ± std)']
        + ['All Attacks Robust Accuracy (mean ± std)']
        + ['All Attacks Avg L_inf (mean ± std)']
        + ['All Attacks Avg L2 (mean ± std)']
        + ['All Attacks Avg Queries (mean ± std)']
        + ['Model Robustness Score', 'Max Robustness Score', 'Images Correctly Classified'],
    )
    
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
        writer.writerow(
            [dataset_model_name, str(last_id + 1)]
            + prepare_attacks_values(dic_results, attacks)
            + prepare_attacks_values(clean_acc_by_attack, attacks)
            + prepare_attacks_values(robust_acc_by_attack, attacks)
            + prepare_attacks_values(linf_avg_by_attack, attacks)
            + prepare_attacks_values(l2_avg_by_attack, attacks)
            + prepare_attacks_values(queries_avg_by_attack, attacks)
            + [fmt_mean_std(agg_clean_mean, agg_clean_std, decimals=4)]
            + [fmt_mean_std(agg_robust_mean, agg_robust_std, decimals=4)]
            + [fmt_mean_std(agg_linf_mean, agg_linf_std, decimals=4)]
            + [fmt_mean_std(agg_l2_mean, agg_l2_std, decimals=4)]
            + [fmt_mean_std(agg_queries_mean, agg_queries_std, decimals=4)]
            + [str(sum(dic_results.values())), str(max_possible), str(max_possible - num_misclassified_images)]
        )

    # export to Excel and LaTeX
    try:
        df = pd.read_csv(log)
        df.to_excel(log_xlsx, index=False)
        with open(log_tex, 'w', encoding='UTF8') as f:
            f.write(df.to_latex(index=False))
    except Exception as e:
        print('WARNING: Failed to export robustness log to Excel/LaTeX: {}'.format(e))


    

def prepare_attacks_values(dic_results, list_all_attacks):
    values = []
    for attack in list_all_attacks:
        if attack in dic_results.keys():
            values.append(dic_results[attack])
        else:
            values.append(None)

    return values