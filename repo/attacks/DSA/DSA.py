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

# NOTE ABOUT IMPORTS
# ------------------
# This DSA package has its own `utils/` and `models/` packages that the DSA
# implementation imports as `utils.*` and `models.*`.
# The thesis repo also has a top-level `utils/` package.
# If we put the repo root first on `sys.path`, it will shadow DSA's own
# `utils/` and break imports like `from utils.distance import ...`.
#
# So: keep this folder first on sys.path (default when executing this script)
# and load the repo's `utils/func_utils.py` explicitly via importlib.

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def _load_repo_func_utils():
    """Load the repo's utils/func_utils.py without polluting sys.path.

    We create a small synthetic package ("repo_utils") so relative imports inside
    repo files (e.g. `from . import static_vars`) work.
    """
    import types

    utils_dir = os.path.join(_repo_root, 'utils')
    func_utils_path = os.path.join(utils_dir, 'func_utils.py')
    static_vars_path = os.path.join(utils_dir, 'static_vars.py')

    if not os.path.isfile(func_utils_path):
        raise ImportError(f'Could not find repo func_utils at: {func_utils_path}')
    if not os.path.isfile(static_vars_path):
        raise ImportError(f'Could not find repo static_vars at: {static_vars_path}')

    pkg_name = 'repo_utils'
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [utils_dir]
        sys.modules[pkg_name] = pkg

    def _load(fullname: str, path: str):
        spec = importlib.util.spec_from_file_location(fullname, path)
        if spec is None or spec.loader is None:
            raise ImportError(f'Could not load module {fullname} from: {path}')
        module = importlib.util.module_from_spec(spec)
        sys.modules[fullname] = module
        spec.loader.exec_module(module)
        return module

    # Load dependency first so func_utils' relative import resolves.
    _load(f'{pkg_name}.static_vars', static_vars_path)
    func_utils = _load(f'{pkg_name}.func_utils', func_utils_path)

    # Compatibility alias for dill payloads that were serialized with
    # `utils.func_utils` as module path.
    try:
        if 'utils.func_utils' not in sys.modules:
            sys.modules['utils.func_utils'] = func_utils
    except Exception:
        pass

    return func_utils


repo_utils = _load_repo_func_utils()


#to_pil = torchvision.transforms.ToPILImage()

def execute_attack(model, surrogate_models, images, labels, args, mean, std):
    from attack.dsa import DSA as DSAAttack
    from models.base import Model as DsaModel
    from utils.criterion import Misclassification, TargetedMisclassification

    device = args.get('device', 'cpu')

    # Ensure tensors are on the right device
    labels = labels.to(device)
    images = images.to(device)

    # Convert normalized inputs back to [0,1] for the attack.
    images_raw = repo_utils.remove_normalization(images.detach().clone(), mean, std)
    images_raw = torch.clamp(images_raw, 0.0, 1.0)

    mean_t = torch.tensor(mean, device=device, dtype=images_raw.dtype).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=device, dtype=images_raw.dtype).view(1, -1, 1, 1)

    class NormalizeWrapper(nn.Module):
        def __init__(self, base_model: nn.Module):
            super().__init__()
            self.base_model = base_model

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x_norm = (x - mean_t) / std_t
            return self.base_model(x_norm)

    # Wrap target model and surrogate models so they accept raw [0,1] inputs.
    target_wrapped = DsaModel(NormalizeWrapper(model).to(device).eval(), bounds=(0.0, 1.0), device=device)
    local_models_wrapped = [
        DsaModel(NormalizeWrapper(m).to(device).eval(), bounds=(0.0, 1.0), device=device)
        for m in (surrogate_models or [])
    ]
    if len(local_models_wrapped) == 0:
        raise ValueError('DSA requires at least one surrogate model')

    # Criterion
    targeted = bool(args.get('targeted', False))
    if targeted:
        print("Running attack in targeted mode")
        # If provided, use a single target class for *all* samples.
        # Otherwise, fall back to a simple deterministic target different from the true label.
        with torch.no_grad():
            n_classes = int(model(images[:1]).shape[-1])
        chosen_target = args.get('target_label', None)
        if chosen_target is not None:
            chosen_target = int(chosen_target)
            if chosen_target < 0 or chosen_target >= n_classes:
                raise ValueError(f"target_label must be in [0, {n_classes - 1}], got {chosen_target}")
            target_labels = torch.full_like(labels, fill_value=chosen_target)
        else:
            target_labels = (labels + 1) % n_classes
        print("Target_labels: ", target_labels)
        criterion = TargetedMisclassification(target_labels)
    else:
        print("Running attack in untargeted mode")
        criterion = Misclassification(labels)

    # Starting points: use other images in the batch (rolled) as a simple
    # adversarial starting point heuristic for query-based attacks.
    starting_points = torch.rand_like(images_raw)
    starting_points = torch.clamp(starting_points, 0.0, 1.0)

    attack = DSAAttack(
        local_models=local_models_wrapped,
        constraint=str(args.get('constraint', 'linf')),
        epsilon=float(args.get('eps', 16 / 255)),
        budget=int(args.get('q_budget', 1000)),
    )

    adv_raw = attack(target_wrapped, images_raw, criterion=criterion, starting_points=starting_points)
    if not isinstance(adv_raw, torch.Tensor):
        adv_raw = torch.tensor(np.asarray(adv_raw))
    adv_raw = adv_raw.to(device)

    # Re-normalize for the rest of the repo's pipeline.
    adv_norm = repo_utils.normalize_image(adv_raw.clone(), mean, std).to(device)

    with torch.no_grad():
        pred = model(adv_norm).argmax(dim=1)

    success_mask = (pred != labels) if not targeted else (pred == target_labels)
    total_success = int(success_mask.sum().item())

    # L2 in raw pixel space
    l2s = torch.linalg.vector_norm((adv_raw - images_raw).flatten(1), ord=2, dim=1).detach().cpu().tolist()

    # DSA counts "queries" as number of forward calls; we expose per-image queries.
    #basically, it uses the whole batch in each iteration, instead of using only one sample.
    try:
        all_queries = attack.result.query.tolist()
    except Exception:
        all_queries = [0 for _ in range(int(images_raw.shape[0]))]

    return adv_norm.detach().cpu(), pred.detach().cpu().numpy(), all_queries, l2s, total_success


def execute_attack_batched(model, surrogate_models, images, labels, args, mean, std):
    batch_size = args.get('batch_size', None)
    if batch_size is None:
        return execute_attack(model, surrogate_models, images, labels, args, mean, std)

    try:
        batch_size = int(batch_size)
    except Exception:
        batch_size = -1

    if batch_size <= 0 or images.shape[0] <= batch_size:
        return execute_attack(model, surrogate_models, images, labels, args, mean, std)

    adv_chunks = []
    pred_chunks = []
    query_chunks = []
    l2_chunks = []
    total_success = 0

    for start in range(0, images.shape[0], batch_size):
        end = min(start + batch_size, images.shape[0])
        images_batch = images[start:end]
        labels_batch = labels[start:end]

        adv_norm, adv_labels, all_queries, l2s, batch_success = execute_attack(
            model,
            surrogate_models,
            images_batch,
            labels_batch,
            args,
            mean,
            std,
        )

        adv_chunks.append(adv_norm)
        pred_chunks.append(adv_labels)
        query_chunks.extend(list(all_queries))
        l2_chunks.extend(list(l2s))
        total_success += int(batch_success)

        # Avoid holding onto per-batch graphs and buffers
        del adv_norm, adv_labels, all_queries, l2s
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    adv_all = torch.cat(adv_chunks, dim=0) if adv_chunks else torch.empty(0)
    pred_all = np.concatenate(pred_chunks, axis=0) if pred_chunks else np.asarray([])

    return adv_all, pred_all, query_chunks, l2_chunks, total_success


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', type=str)
        # Accept either a single comma-separated string or multiple tokens
        parser.add_argument(
            '--surrogate_models', '--surrogate-models',
            dest='surrogate_models',
            type=str,
            nargs='+',
        )
        parser.add_argument('--dataset', type=str)
        parser.add_argument('--total-images', type=int)
        parser.add_argument('--results-path', type=str)
        parser.add_argument('--log', action='store_true')
        parser.add_argument('--targeted', action='store_true')
        parser.add_argument(
            '--batch_size', '--batch-size',
            dest='batch_size',
            type=int,
            default=None,
        )
        parser.add_argument(
            '--q_budget', '--q-budget',
            dest='q_budget',
            type=int,
            default=1000,
            required=True,
        )
        parser.add_argument('--eps', type=float, default=0.0627, required=True)
        parser.add_argument('--constraint', type=str, default='linf', required=True)
        parser.add_argument(
            '--target_label', '--target-label',
            dest='target_label',
            type=int,
            default=None,
            required=False,
        )
        parsed_args, unknown = parser.parse_known_args()
        ini_args = vars(parsed_args)
        ini_args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

        args = repo_utils.prepare_attack_arguments(ini_args, ini_args.get('attack_config', ''), 'dsa')

        target_label_for_all_samples = None
        if args.get('targeted', False) and target_label_for_all_samples is not None:
            args['target_label'] = int(target_label_for_all_samples)

        if not os.path.isdir(args['results_path']):
            os.makedirs(args['results_path'])

        logging.basicConfig(filename=os.path.join(args['results_path'], 'DSA.log'), format='%(asctime)s | %(levelname)s | %(message)s', datefmt= '%m-%d-%Y %H:%M:%S', level=logging.INFO)

        ini_stdout = sys.stdout
        sys.stdout = repo_utils.StdoutToLogging()

        print("Loading model")
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

        #load surrogate models, if not provided, he will raise an exception to abort
        if not args.get('surrogate_models'):
            raise ValueError('Surrogate models not specified')
        print("Loading surrogate models")

        surrogate_models_arg = args.get('surrogate_models', '')
        if isinstance(surrogate_models_arg, list):
            surrogate_models_arg = ' '.join(surrogate_models_arg)
        surrogate_models_names = [s.strip() for s in str(surrogate_models_arg).split(',') if s.strip()]
        surrogate_models = [repo_utils.load_surrogate_from_ckpt(name) for name in surrogate_models_names]
        print("Surrogate models loaded: ", surrogate_models_names)

        # load dataset
        spec = importlib.util.spec_from_file_location('dataset', args['dataset'])
        dataset = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dataset)
        testloader, trainloader = dataset.dataLoader()

        print("loading images")
        images, labels = repo_utils.get_images_labels_from_dataLoader(testloader, args['device'], args['total_images'])
        images, labels, classified_labels, _, orig_indices = repo_utils.filter_correctly_classified(
            images, labels, classified_labels
        )
        if images is None or (torch.is_tensor(images) and images.shape[0] == 0):
            print("No correctly classified samples found; skipping DSA.")
            dsa_dict = {
                'attack_name': 'DSA',
                'perturbed_image': [],
                'perturbed_label': [],
                'total_queries': [],
                'l2': [],
                'success': 0,
                'max_perturbation': 0.0,
                'orig_index': []
            }
            repo_utils.save_statistics_of_attack(dsa_dict, labels, classified_labels, args['results_path'])
            repo_utils.save_images_attack(dsa_dict, labels, os.path.join(args['results_path'], 'perturbed_images'))
            sys.stdout = ini_stdout
            raise SystemExit(0)

        print("Normalizing images")
        images_norm = repo_utils.normalize_image(images.clone(), dataset.MEAN, dataset.STD).to(args['device'])
        
        print("Executing attack")
        adversarial_images, adversarial_labels, all_queries, l2s, total_success = execute_attack_batched(
            model,
            surrogate_models,
            images_norm,
            labels,
            args,
            dataset.MEAN,
            dataset.STD,
        )


        #remove normalization for saving and visualization
        adv_raw_tensor = None
        try:
            if isinstance(adversarial_images, torch.Tensor):
                denorm_imgs = repo_utils.remove_normalization(adversarial_images.clone().cpu(), dataset.MEAN, dataset.STD)
                adv_raw_tensor = denorm_imgs
                perturbed_images = denorm_imgs.cpu().numpy()
            else:
                adv_np = np.asarray(adversarial_images)
                try:
                    adv_tensor = torch.from_numpy(adv_np)
                except Exception:
                    adv_tensor = torch.tensor(adv_np)
                denorm_imgs = repo_utils.remove_normalization(adv_tensor.clone().cpu(), dataset.MEAN, dataset.STD)
                adv_raw_tensor = denorm_imgs
                perturbed_images = denorm_imgs.cpu().numpy()
        except Exception:
            perturbed_images = np.asarray(adversarial_images)

        perturbed_labels = np.asarray(adversarial_labels)

        # Max perturbation (Linf) in raw pixel space [0,1]
        # Compute global max over all images/pixels.
        try:
            orig_raw = images.detach().cpu().clone()
            orig_raw = torch.clamp(orig_raw, 0.0, 1.0)

            if adv_raw_tensor is None:
                adv_raw = torch.tensor(np.asarray(perturbed_images))
            else:
                adv_raw = adv_raw_tensor.detach().cpu().clone()
            adv_raw = torch.clamp(adv_raw, 0.0, 1.0)

            maxp_val = float((adv_raw - orig_raw).abs().max().item())
        except Exception:
            maxp_val = 0.0
        dsa_dict = {
            'attack_name': 'DSA',
            'perturbed_image': perturbed_images,
            'perturbed_label': adversarial_labels,
            'total_queries': all_queries,
            'l2': l2s,
            'success': total_success,
            'max_perturbation': maxp_val,
            'orig_index': orig_indices
        }

        repo_utils.save_statistics_of_attack(dsa_dict, labels, classified_labels, args['results_path'])
        repo_utils.save_images_attack(dsa_dict, labels, os.path.join(args['results_path'], 'perturbed_images'))

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
