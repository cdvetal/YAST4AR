import sys
import os
import argparse
import logging
import dill
import importlib
import importlib.util
import numpy as np
import torch
import torch.nn as nn

import algorithm.attack as attack
# Ensure the repository root is on sys.path so imports like `utils.func_utils`
# work when this file is executed directly (e.g. from repo/attacks/TEA).
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _repo_root not in sys.path:
    # Put repo root first so `import utils` prefers the package we create
    # instead of a local `utils.py` when TEA.py is executed directly.
    sys.path.insert(0, _repo_root)

import utils.func_utils as utils

def run_on_images(target_model, surrogate_model, images, labels, device='cpu', q_budget=250, max_images=None):
    import time
    adv_images = []
    adv_labels = []
    queries = []
    l2s = []
    total_success = 0

    # ensure model in eval and on device
    target_model.to(device)
    target_model.eval()

    n = images.shape[0]
    if max_images is not None:
        n = min(n, max_images)

    lossF = torch.nn.CrossEntropyLoss().to(device)

    for i in range(n):
        x = images[i:i+1].to(device)
        y = labels[i:i+1].to(device)

        try:
            print("Trying SQBA attack")
            alg = attack.SQBA(device, model=target_model, sub_model=surrogate_model, lossF=lossF, q_budgets=[q_budget], stop=True)
            ladv, lquery, iter0, iter1 = alg.untarget(x, y)
        except Exception as e :
            # fallback: no adversarial found
            import traceback
            traceback.print_exc()
            print(f"SQBA attack failed on sample {i}: {e}")
            ladv = x.clone()
            lquery = 0
            raise Exception(f"SQBA attack failed on sample {i}: {e}")

        # ensure tensor
        if isinstance(ladv, np.ndarray):
            ladv_t = torch.from_numpy(ladv).float().to(device)
        else:
            ladv_t = ladv.to(device) if hasattr(ladv, 'to') else torch.tensor(ladv, device=device)

        with torch.no_grad():
            pred = target_model(ladv_t).argmax(dim=1).cpu().detach()

        # compute l2
        l2_val = float((torch.norm(torch.abs(ladv - x)) / torch.norm(x)))
        print("l2:", l2_val)
        adv_images.append(ladv_t.cpu().detach().numpy()[0])
        adv_labels.append(int(pred.item()))
        queries.append(int(lquery))
        l2s.append(float(l2_val))

        if int(pred.item()) != int(labels[i].cpu().item()):
            total_success += 1

    return adv_images, adv_labels, queries, l2s, total_success



def execute_attack(model, surrogate_model, images, labels, args, mean, std, num_classes=None):
    device = args.get('device', 'cpu')
    q_budget = args.get('q_budget', 250)
    max_images = None

    print("Normalizing images")
    images_norm = utils.normalize_image(images.clone(), mean, std).to(device)
    print("Launching attack with SQBA")
    
    adv_images, adv_labels, queries, l2s, total_success = run_on_images(
        target_model=model, surrogate_model=surrogate_model, images=images_norm, labels=labels,
        device=device, q_budget=q_budget, max_images=max_images)

    # convert outputs
    # denormalize adversarial images (SQBA returns images in normalized space)
    try:
        if isinstance(adv_images, torch.Tensor):
            denorm_imgs = utils.remove_normalization(adv_images.clone().cpu(), mean, std)
            perturbed_images = denorm_imgs.cpu().numpy()
        else:
            adv_np = np.asarray(adv_images)
            try:
                adv_tensor = torch.from_numpy(adv_np)
            except Exception:
                adv_tensor = torch.tensor(adv_np)
            denorm_imgs = utils.remove_normalization(adv_tensor.clone().cpu(), mean, std)
            perturbed_images = denorm_imgs.cpu().numpy()
    except Exception:
        perturbed_images = np.asarray(adv_images)

    perturbed_labels = np.asarray(adv_labels)

    return perturbed_images, perturbed_labels, queries, l2s, total_success


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', type=str)
        parser.add_argument('--surrogate-model', type=str)
        parser.add_argument('--dataset', type=str)
        parser.add_argument('--total-images', type=int)
        parser.add_argument('--batch-size', type=int)
        parser.add_argument('--results-path', type=str)
        parser.add_argument('--log', action='store_true')
        parser.add_argument('--q-budget', type=int, default=250)

        parsed_args, unknown = parser.parse_known_args()
        ini_args = vars(parsed_args)
        ini_args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

        args = utils.prepare_attack_arguments(ini_args, ini_args.get('attack_config', ''), 'sqba')


        if not os.path.isdir(args['results_path']):
            os.makedirs(args['results_path'])

        logging.basicConfig(filename=os.path.join(args['results_path'], 'SQBA.log'), format='%(asctime)s | %(levelname)s | %(message)s', datefmt= '%m-%d-%Y %H:%M:%S', level=logging.INFO)

        ini_stdout = sys.stdout
        sys.stdout = utils.StdoutToLogging()

        print("Loading model")
        # load target model
        model = None
        classified_labels = None
        if args.get('model'):
            with open(args['model'], 'rb') as f:
                data = dill.loads(f.read())
                model = data['model']
                classified_labels = data.get('classified_labels', None)

        if model is None:
            raise ValueError('Target model not provided or could not be loaded')

        model = model.to(args['device'])
        model.eval()

        # load surrogate model if provided
        surrogate = utils.load_surrogate_from_ckpt(args.get('surrogate_model', 'DenseNet121')) if args.get('surrogate_model') else None

        # load dataset
        spec = importlib.util.spec_from_file_location('dataset', args['dataset'])
        dataset = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dataset)
        testloader, trainloader = dataset.dataLoader()

        print("loading images")
        batch_size = args.get('batch_size')
        images, labels = utils.get_images_labels_from_dataLoader(
            testloader, args['device'], args.get('total_images')
        )
        images, labels, classified_labels, _, orig_indices = utils.filter_correctly_classified(
            images, labels, classified_labels
        )
        if images is None or (torch.is_tensor(images) and images.shape[0] == 0):
            print("No correctly classified samples found; skipping SQBA.")
            sqba_dict = {
                'attack_name': 'SQBA',
                'perturbed_image': [],
                'perturbed_label': [],
                'total_queries': [],
                'l2': [],
                'success': 0,
                'max_perturbation': 0.0,
                'orig_index': []
            }
            utils.save_statistics_of_attack(sqba_dict, labels, classified_labels, args['results_path'])
            utils.save_images_attack(sqba_dict, labels, os.path.join(args['results_path'], 'perturbed_images'))
            sys.stdout = ini_stdout
            raise SystemExit(0)

        all_adv_images = []
        all_adv_labels = []
        all_queries = []
        all_l2s = []
        total_success = 0
        maxp_val = ''

        step = batch_size or images.size(0)
        for start in range(0, images.size(0), step):
            end = start + step
            batch_images = images[start:end]
            batch_labels = labels[start:end]

            adv_images, adv_labels, queries, l2s, success = execute_attack(
                model, surrogate, batch_images, batch_labels, args, dataset.MEAN, dataset.STD)

            all_adv_images.extend(list(np.asarray(adv_images)))
            all_adv_labels.extend(list(np.asarray(adv_labels)))
            all_queries.extend(list(queries))
            all_l2s.extend(list(l2s))
            total_success += success

            batch_maxp = ''
            try:
                adv = np.asarray(adv_images)
                orig = batch_images.detach().cpu().numpy()
                if adv.shape == orig.shape:
                    diff = np.abs(adv - orig)
                    linf_per = diff.reshape(diff.shape[0], -1).max(axis=1)
                    batch_maxp = float(linf_per.max())
                else:
                    if len(l2s) > 0:
                        batch_maxp = float(max(l2s))
            except Exception as e:
                print("Could not compute max perturbation for batch", str(e))
                try:
                    if len(l2s) > 0:
                        batch_maxp = float(max(l2s))
                except Exception:
                    batch_maxp = ''

            if batch_maxp != '':
                if maxp_val == '':
                    maxp_val = batch_maxp
                else:
                    maxp_val = max(float(maxp_val), float(batch_maxp))

        sqba_dict = {
            'attack_name': 'SQBA',
            'perturbed_image': all_adv_images,
            'perturbed_label': np.asarray(all_adv_labels),
            'total_queries': all_queries,
            'l2': all_l2s,
            'success': total_success,
            'max_perturbation': maxp_val,
            'orig_index': orig_indices
        }

        utils.save_statistics_of_attack(sqba_dict, labels, classified_labels, args['results_path'])
        utils.save_images_attack(sqba_dict, labels, os.path.join(args['results_path'], 'perturbed_images'))
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Attack execution failed: {e}")
        raise Exception(f"Attack execution failed: {e}")
    finally:
        try:
            sys.stdout = ini_stdout
        except Exception:
            pass
