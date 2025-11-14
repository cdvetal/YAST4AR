import sys
import os

# get repo path for imports
abs_path = os.path.abspath(os.path.realpath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.dirname(abs_path)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

import torch
from torch.autograd import grad
import numpy as np

from attacks.Attack import Attack
import utils.func_utils as utils

import importlib
import dill
import argparse
import logging



def Top1Criterion(x, y, model, mean=(0, 0, 0), std=(1, 1, 1)):
    """Returns True if model prediction is in top1"""
    x_ = x.clone()
    if x_.dim() == 3:
        for i in range(x_.size(0)):
            x_[i, :, :] = (x_[i, :, :] - mean[i]) / std[i]
    else:
        for i in range(x_.size(1)):
            x_[:, i, :, :] = (x_[:, i, :, :] - mean[i]) / std[i]
    return model(x_).topk(1)[1].view(-1) == y


def Top5Criterion(x, y, model, mean=(0, 0, 0), std=(1, 1, 1)):
    """Returns True if model prediction is in top5"""
    x_ = x.clone()
    if x_.dim() == 3:
        for i in range(x_.size(0)):
            x_[i, :, :] = (x_[i, :, :] - mean[i]) / std[i]
    else:
        for i in range(x_.size(1)):
            x_[:, i, :, :] = (x_[:, i, :, :] - mean[i]) / std[i]
    return (model(x_).topk(5)[1] == y.view(-1, 1)).any(dim=-1)


def initialize(x, y, criterion, max_iters=1e3, bounds=(0, 1)):
    """Generates random perturbations of clean images until images have incorrect label.

    If the image is already mis-classified, then it is not perturbed."""
    xpert = x.clone()
    dt = 0.01

    correct = criterion(xpert, y)
    k = 0
    while correct.sum() > 0:
        l = correct.sum()
        xpert[correct] = x[correct] + (1.01) ** k * dt * torch.randn(l, *xpert.shape[1:], device=xpert.device)
        xpert[correct].clamp_(*bounds)
        correct = criterion(xpert, y)

        k += 1
        if k > max_iters:
            raise ValueError('failed to initialize: maximum iterations reached')

    return xpert


########################################################################

class LogBarrier(Attack):

    def __init__(self, model, criterion=Top1Criterion, initialize=initialize, norm=2,
                 verbose=False, mean=(0, 0, 0), std=(1, 1, 1), **kwargs):
        """Attack a model using the log barrier constraint to enforce mis-classification.

        Arguments:
            model: PyTorch model, takes batch of inputs and returns logits
            criterion: function which takes in images and labels and a model, and returns
                a boolean vector, which is True is model prediction is correct
                For example, the Top1 or Top5 classification criteria
            initialize (optional): function which takes in images, labels and a model, and
                returns mis-classified images (an initial starting guess) (default: clipped Gaussians)
            norm (optional): norm to measure adversarial distance with (default: 2)
            verbobose (optional): if True (default), display status during attack

        Keyword arguments:
            bounds: tuple, image bounds (default (0,1))
            dt: step size (default: 0.01)
            alpha: initial Lagrange multiplier of log barrier penalty (default: 0.1)
            beta: shrink parameter of Lagrange multiplier after each inner loop (default: 0.75)
            gamma: back track parameter (default: 0.5)
            max_outer: maximum number of outer loops (default: 15)
            tol: inner loop stopping criteria (default: 1e-6)
            max_inner: maximum number of inner loop iterations (default: 500)
            T: softmax temperature in L-infinity norm approximation (default: 500)

        Returns:
            images: adversarial images mis-classified by the model
        """
        super().__init__()

        self.model = model
        self.criterion = lambda x, y: criterion(x, y, model, mean, std)
        self.initialize = initialize
        self.labels = None
        self.original_images = None
        self.perturbed_images = None
        self.mean = mean
        self.std = std

        if not (norm == 2 or norm == np.inf):
            raise ValueError('norm must be either 2 or np.inf')
        self.norm = norm
        self.verbose = verbose

        config = {'bounds': (0, 1),
                  'dt': 0.01,
                  'alpha': 0.1,
                  'beta': 0.75,
                  'gamma': 0.5,
                  'max_outer': 15,
                  'tol': 1e-6,
                  'max_inner': int(5e2),
                  'T': 500.}
        config.update(kwargs)

        self.hyperparams = config

    def perturb(self, x, y):
        self.labels = y
        self.original_images = x

        config = self.hyperparams
        model = self.model
        criterion = self.criterion

        bounds, dt, alpha0, beta, gamma, max_outer, tol, max_inner, T = (
            config['bounds'], config['dt'], config['alpha'], config['beta'],
            config['gamma'], config['max_outer'], config['tol'], config['max_inner'],
            config['T'])

        Nb = len(y)
        ix = torch.arange(Nb, device=x.device)

        imshape = x.shape[1:]
        PerturbedImages = torch.full(x.shape, np.nan, device=x.device)

        mis0 = criterion(x, y)

        xpert = initialize(x, y, criterion)

        xpert[~mis0] = x[~mis0]
        xold = xpert.clone()
        xbest = xpert.clone()
        diffBest = torch.full((Nb,), np.inf, device=x.device)
        xpert.requires_grad_(True)

        total_iterations = 0

        for k in range(max_outer):
            alpha = alpha0 * beta ** k

            diff = (xpert - x).view(Nb, -1).norm(self.norm, -1)
            update = diff > 0
            for j in range(max_inner):
                total_iterations += 1
                p_ = xpert.clone()
                if p_.dim() == 3:
                    for i in range(p_.size(0)):
                        p_[i, :, :] = (p_[i, :, :] - self.mean[i]) / self.std[i]
                else:
                    for i in range(p_.size(1)):
                        p_[:, i, :, :] = (p_[:, i, :, :] - self.mean[i]) / self.std[i]
                p = model(p_).softmax(dim=-1)

                pdiff = p.max(dim=-1)[0] - p[ix, y]
                s = -torch.log(pdiff).sum()
                g = grad(alpha * s, xpert)[0]  # TODO: use only one grad when norm==Linf
                if self.norm == 2:
                    with torch.no_grad():
                        xpert[update] = xpert[update].mul(1 - dt).add(g[update], alpha = -dt).add(
                            x[update], alpha = dt).clamp_(*bounds)
                elif self.norm == np.inf:
                    Nb_ = xpert[update].shape[0]
                    xpert_, x_ = xpert[update].view(Nb_, -1), x[update].view(Nb_, -1)
                    z_ = (xpert_ - x_)
                    z = torch.abs(z_)

                    # smooth approximation of Linf norm
                    ex_ = ((z * T).softmax(dim=-1) * z).sum(dim=-1)

                    ginf = grad(ex_.sum(), xpert_)[0]

                    with torch.no_grad():
                        GradientStep = ginf.view(Nb_, *imshape) + g[update]
                        xpert[update] = xpert[update].add(GradientStep, alpha = -dt).clamp(*bounds)

                with torch.no_grad():
                    # backtrack
                    c = criterion(xpert, y)
                    while c.any():
                        xpert.data[c] = xpert.data[c].clone().mul(1 - gamma).add(xold[c], alpha = gamma)
                        c = criterion(xpert, y)

                    diff = (xpert - x).view(Nb, -1).norm(self.norm, -1)
                    boolDiff = diff <= diffBest
                    xbest[boolDiff] = xpert[boolDiff]
                    diffBest[boolDiff] = diff[boolDiff]

                    iterdiff = (xpert - xold).view(Nb, -1).norm(self.norm, -1)
                    # med = diff.median()

                    xold = xpert.clone()

                if self.verbose:
                    sys.stdout.write('[%2d outer, %4d inner] median & max distance: (%4.4f, %4.4f)\r'
                                     % (k, j, diffBest.median(), diffBest.max()))

                if not iterdiff.abs().max() > tol:
                    break

        if self.verbose:
            sys.stdout.write('\n')

        switched = ~criterion(xbest, y)
        PerturbedImages[switched] = xbest.detach()[switched]

        self.perturbed_images = PerturbedImages

        return PerturbedImages, total_iterations


def execute_attack(model, images, args, mean, std):
    total_success = 0

    all_original_images, all_labels, all_adversarial_images, all_adversarial_labels, all_queries, all_l2s = [], [], [], [], [], []


    attack_params = {'bounds': (args["lower_bound"], args["upper_bound"]),
                'dt': args["dt"],
                'alpha': args["alpha"],
                'beta': args["beta"],
                'gamma': args["gamma"],
                'max_outer': args["max_outer"],
                'tol': args["tol"],
                'max_inner': args["max_inner"],
                'T': args["T"]}

    
    if 'batch_size' in args and args['batch_size'] > 0:
        b_size = args['batch_size']
    else:
        print("ERROR: Batch size not define")
        exit(1)

    logbarrier_attack = LogBarrier(model=model, norm=args["norm"], mean=mean, std=std, verbose = 'log' in args, **attack_params) 
    for i in range(args['total_images'] // b_size):
        images_batch = images[i * b_size:b_size * (i + 1)]
        labels_batch = labels[i * b_size:b_size * (i + 1)]
        xpert, total_queries = logbarrier_attack.perturb(images_batch.to(args['device']),
                                                            labels_batch.to(args['device']))
        pred_logbarrier = model(utils.normalize_image(xpert, mean, std)).argmax(dim=1).cpu()
        success = (pred_logbarrier != labels_batch.cpu()).float().mean().item() * 100

        l2_norms = []
        for ii in range(len(xpert)):
            l2_norms.append(utils.calculate_l2_norm(xpert.cpu().detach().numpy()[ii],
                                                images_batch.cpu().detach().numpy()[ii]))


        all_l2s.extend(l2_norms)
        all_original_images.extend(images_batch.cpu().detach().numpy())
        all_adversarial_images.extend(xpert.cpu().detach().numpy())
        all_labels.extend(labels_batch.cpu())
        all_adversarial_labels.extend(pred_logbarrier.cpu())
        all_queries.extend([total_queries] * len(images_batch.cpu()))


        success = sum((labels_batch.cpu() != pred_logbarrier.cpu()).float() == 1.)
        total_success += success.item()
        
        if 'log' in args: print("Images done: {}/{}; Success: {:.2f}%".format(b_size * (i + 1), args['total_images'], success.item() / b_size * 100))
    
    return all_adversarial_images, all_adversarial_labels, all_queries, all_l2s, total_success


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--norm', type=int, help='')
    parser.add_argument('--lower-bound', type=float, help='')
    parser.add_argument('--upper-bound', type=float, help='')
    parser.add_argument('--dt', type=float, help='')
    parser.add_argument('--alpha', type=float, help='')
    parser.add_argument('--beta', type=float, help='')
    parser.add_argument('--gamma', type=float, help='')
    parser.add_argument('--max-outer', type=int, help='')
    parser.add_argument('--tol', type=float, help='')
    parser.add_argument('--max-inner', type=int, help='')
    parser.add_argument('--T', type=int, help='')
    parser.add_argument('--batch-size', type=int, help='')

    parser.add_argument('--model', type=str, default = 'Path for binary file containing the model')
    parser.add_argument('--dataset', type=str, default = 'Path for dataset loader file')
    parser.add_argument('--total-images', type=int, help='')
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--attack-config', type=str, default = '', help='Config file to be passed in instead of arguments')
    parser.add_argument('--results-path', type=str, default = '', help="Path to store results")

    ini_args = vars(parser.parse_args())

    ini_args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    # initial argument verifications
    args = utils.prepare_attack_arguments(ini_args, ini_args['attack_config'], 'logbarrier')


    try:
        # setup logger
        if not os.path.isdir(args['results_path']):
            os.makedirs(args['results_path'])

        logging.basicConfig(filename=os.path.join(args['results_path'], 'LogBarrier.log'), format='%(asctime)s | %(levelname)s | %(message)s', datefmt= '%m-%d-%Y %H:%M:%S', level=logging.INFO)

        ini_stdout = sys.stdout
        sys.stdout = utils.StdoutToLogging()
        
        print("Loading DeepFool")
        

        # load model
        with open(args['model'], "rb") as file:
            serialized_data = file.read()
            data = dill.loads(serialized_data)
            #model_name = data['model_name']
            model = data['model']
            classified_labels = data['classified_labels']

        print("Model loaded")

        # load dataset
        spec = importlib.util.spec_from_file_location("dataset", args['dataset'])
        dataset = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dataset)
        testloader, trainloader = dataset.dataLoader()

        images, labels = utils.get_images_labels_from_dataLoader(testloader, args['device'], args['total_images'])
        
        print("Dataset loaded")

        print("Running LogBarrier on {}".format(args['device']))
        
        adversarial_images, adversarial_labels, all_queries, l2s, total_success = execute_attack(model, images, args, dataset.MEAN, dataset.STD)
        
        
        logbarrier_dict = {
            'attack_name': 'LogBarrier',
            'perturbed_image': adversarial_images,
            'perturbed_label': adversarial_labels,
            'total_queries': all_queries,
            'l2': l2s,
            'success': total_success
        }

        print("Saving Results")

        utils.save_statistics_of_attack(logbarrier_dict, labels, classified_labels, args['results_path'])

        utils.save_images_attack(logbarrier_dict, labels, os.path.join(args['results_path'], 'perturbed_images'))

    finally:
        sys.stdout = ini_stdout
