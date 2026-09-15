import sys
import os

# get repo path for imports
abs_path = os.path.abspath(os.path.realpath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.dirname(abs_path)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

import argparse
import logging
import dill
import importlib

import torch
import numpy as np

from attacks.Attack import Attack
import utils.func_utils as utils
from attacks.AutoAttack.autoattack import AutoAttack


def execute_attack(model, images, args, mean, std):
    device = args.get('device', 'cpu')
    batch_size = args.get('batch_size', 250)

    # Ensure images are in pixel space [0,1]
    images = images.clone().to(device)

    # Prepare labels

    labels = args['classified_labels']
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
    labels = labels.to(device)

    # Define wrapper model that normalizes internally
    class NormalizedModel(torch.nn.Module):
        def __init__(self, model, mean, std):
            super().__init__()
            self.model = model
            self.mean = torch.tensor(mean).view(1, -1, 1, 1).to(device)
            self.std = torch.tensor(std).view(1, -1, 1, 1).to(device)

        def forward(self, x):
            x = (x - self.mean) / self.std
            return self.model(x)

    wrapped_model = NormalizedModel(model, mean, std)

    # AutoAttack 
    aa = AutoAttack(
        wrapped_model,
        norm=args.get('norm', 'Linf'),
        eps=args.get('eps', 8.0 / 255.0),  # default fixed (~0.03)
        seed=args.get('seed', None),
        verbose=('log' in args),
        version=args.get('version', 'standard'),
        attacks_to_run=args.get('attacks_to_run', []),
        device=device
    )

    #debug: just to force square attack to see if queries are being recorded properly
    # try:
    #     aa.attacks_to_run = ['square']
    #     # allow overriding n_queries via CLI
    #     n_q = args.get('square_n_queries', None)
    #     if n_q is not None and hasattr(aa, 'square'):
    #         aa.square.n_queries = int(n_q)
    #     logging.info(f"Forcing only Square attack. n_queries={getattr(aa.square, 'n_queries', 'unknown')}")
    # except Exception as e:
    #     logging.info(f"Failed to force only-square: {e}")

    # Run attack
    max_perturbation, all_queries, x_adv, y_adv = aa.run_standard_evaluation(
        images,
        labels,
        bs=batch_size,
        return_labels=True
    )
    print("queries: ", all_queries)
    # Compute L2 
    diff = (x_adv - images).view(x_adv.size(0), -1)
    l2_tensor = diff.norm(p=2, dim=1)

    all_l2s = l2_tensor.detach().cpu().numpy().tolist()

    # Convert outputs
    perturbed_images = x_adv.detach().cpu().numpy()
    perturbed_labels = y_adv.detach().cpu().numpy()
    original_labels = labels.detach().cpu().numpy()

    total_success = int(np.sum(perturbed_labels != original_labels))
    #all_queries = [0] * len(all_l2s)

    return max_perturbation, perturbed_images, perturbed_labels, all_queries, all_l2s, total_success


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--model', type=str, default='Path for binary file containing the model')
    parser.add_argument('--dataset', type=str, default='Path for dataset loader file')
    parser.add_argument('--total-images', type=int)
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--attack-config', type=str, default='')
    parser.add_argument('--results-path', type=str, default='')
    parser.add_argument('--norm', type=str, default='Linf')
    parser.add_argument('--eps', type=float, default=0.3)
    parser.add_argument('--version', type=str, default='standard')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--attack-name', type=str, default='autoattack')

    ini_args = vars(parser.parse_args())
    ini_args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    args = utils.prepare_attack_arguments(
        ini_args,
        ini_args['attack_config'],
        'autoattack'
    )

    try:
        if not os.path.isdir(args['results_path']):
            os.makedirs(args['results_path'])

        logging.basicConfig(
            filename=os.path.join(args['results_path'], 'AutoAttack.log'),
            format='%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%m-%d-%Y %H:%M:%S',
            level=logging.INFO
        )

        ini_stdout = sys.stdout
        sys.stdout = utils.StdoutToLogging()

        print("Loading AutoAttack")

        # Load model
        with open(args['model'], "rb") as file:
            serialized_data = file.read()
            data = dill.loads(serialized_data)

            model = data['model']
            classified_labels = data['classified_labels']

            # Ensure model on correct device
            model = model.to(args['device'])
            model.eval()

        # FIX: convert classified_labels properly
        if isinstance(classified_labels, np.ndarray):
            classified_labels = torch.from_numpy(classified_labels)

        classified_labels = classified_labels.to(args['device'])

        print("Model loaded")

        # Load dataset
        spec = importlib.util.spec_from_file_location("dataset", args['dataset'])
        dataset = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dataset)

        testloader, trainloader = dataset.dataLoader()

        images, labels = utils.get_images_labels_from_dataLoader(
            testloader,
            args['device'],
            args['total_images']
        )

        images, labels, classified_labels, _, orig_indices = utils.filter_correctly_classified(
            images, labels, classified_labels
        )
        if images is None or (torch.is_tensor(images) and images.shape[0] == 0):
            print("No correctly classified samples found; skipping AutoAttack.")
            empty_results = {
                'attack_name': 'AutoAttack',
                'perturbed_image': [],
                'perturbed_label': [],
                'total_queries': [],
                'l2': [],
                'success': 0,
                'max_perturbation': 0.0,
                'orig_index': []
            }
            utils.save_statistics_of_attack(
                empty_results,
                labels,
                classified_labels,
                args['results_path']
            )
            utils.save_images_attack(
                empty_results,
                labels,
                os.path.join(args['results_path'], 'perturbed_images')
            )
            sys.stdout = ini_stdout
            raise SystemExit(0)

        images = images.to(args['device'])

        print("Dataset loaded")
        print(f"Running AutoAttack on {args['device']}")

        args['classified_labels'] = classified_labels

        max_perturbation, adversarial_images, adversarial_labels, all_queries, l2s, total_success = execute_attack(
            model,
            images,
            args,
            dataset.MEAN,
            dataset.STD
        )

        # Normalize/convert queries into a plain Python list of ints so CSV/plots show simple numbers
        queries_list = None
        try:
            if isinstance(all_queries, torch.Tensor):
                # convert torch tensor -> cpu numpy -> ints -> list
                queries_list = all_queries.detach().cpu().numpy().astype(int).tolist()
            elif isinstance(all_queries, np.ndarray):
                queries_list = all_queries.astype(int).tolist()
            elif isinstance(all_queries, list):
                queries_list = []
                for q in all_queries:
                    if isinstance(q, torch.Tensor):
                        queries_list.append(int(q.detach().cpu().item()))
                    else:
                        queries_list.append(int(q))
            else:
                # last resort
                queries_list = [int(x) for x in list(all_queries)]
        except Exception:
            # fallback to zeros if conversion fails
            queries_list = [0] * len(l2s)

        # Helpful log when everything is zero
        if sum(queries_list) == 0:
            logging.info('All query counts are zero. This is expected if the attacks that report queries (e.g. Square) were not run, or samples were already misclassified by earlier attacks.')

        # ensure max_perturbation is a plain Python number (not a torch tensor)
        try:
            if isinstance(max_perturbation, torch.Tensor):
                maxp_val = float(max_perturbation.detach().cpu().item())
            elif isinstance(max_perturbation, np.ndarray):
                # handle 0-d numpy arrays
                try:
                    maxp_val = float(max_perturbation.item())
                except Exception:
                    maxp_val = float(np.asarray(max_perturbation).ravel()[0])
            else:
                maxp_val = float(max_perturbation)
        except Exception:
            maxp_val = max_perturbation

        aa_dict = {
            'attack_name': 'AutoAttack',
            'perturbed_image': adversarial_images,
            'perturbed_label': adversarial_labels,
            'total_queries': queries_list,
            'l2': l2s,
            'success': total_success,
            'max_perturbation': maxp_val,
            'orig_index': orig_indices
        }

        print("Saving Results")

        utils.save_statistics_of_attack(
            aa_dict,
            labels,
            classified_labels.detach().cpu().numpy(),
            args['results_path']
        )

        utils.save_images_attack(
            aa_dict,
            labels,
            os.path.join(args['results_path'], 'perturbed_images')
        )

    finally:
        sys.stdout = ini_stdout
