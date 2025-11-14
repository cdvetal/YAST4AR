import sys
import os

# get repo path for imports
abs_path = os.path.abspath(os.path.realpath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.dirname(abs_path)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

import logging
import utils.func_utils as utils
import importlib
import dill
import pdb
import argparse

import torch as ch
import torchvision.transforms.functional as F
from torch.nn.modules import Upsample

from attacks.Attack import Attack
from torch.utils import data




class BanditsPrior(Attack):
    def __init__(self, images, targets, max_queries, fd_eta, image_lr, online_lr, mode,
                 exploration, tile_size, epsilon, batch_size, log_progress, nes, tiling,
                 gradient_iters, model_to_fool, dataset_size, mean, std, device):

        super().__init__()

        self.images = images
        self.targets = targets
        self.max_queries = max_queries
        self.fd_eta = fd_eta
        self.image_lr = image_lr
        self.online_lr = online_lr
        self.mode = mode
        self.exploration = exploration
        self.tile_size = tile_size
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.log_progress = log_progress
        self.nes = nes
        self.tiling = tiling
        self.gradient_iters = gradient_iters
        self.model_to_fool = model_to_fool
        self.dataset_size = dataset_size
        self.mean = mean
        self.std = std
        self.device = device

    def perturb(self):
        return self.make_adversarial_examples(self.images, self.targets, self.max_queries, self.fd_eta, self.image_lr,
                                        self.online_lr, self.mode, self.exploration, self.tile_size, self.epsilon,
                                        self.batch_size, self.log_progress, self.nes, self.tiling, self.gradient_iters,
                                        self.model_to_fool, self.dataset_size, self.mean, self.std, self.device)


    def __norm(self, t):
        assert len(t.shape) == 4
        norm_vec = ch.sqrt(t.pow(2).sum(dim=[1, 2, 3])).view(-1, 1, 1, 1)
        norm_vec += (norm_vec == 0).float() * 1e-8
        return norm_vec

    ###
    # Different optimization steps
    # All take the form of func(x, g, lr)
    # eg: exponentiated gradients
    # l2/linf: projected gradient descent
    ###

    def __eg_step(self, x, g, lr):
        real_x = (x + 1) / 2  # from [-1, 1] to [0, 1]
        pos = real_x * ch.exp(lr * g)
        neg = (1 - real_x) * ch.exp(-lr * g)
        new_x = pos / (pos + neg)
        return new_x * 2 - 1


    def __linf_step(self, x, g, lr):
        return x + lr * ch.sign(g)


    def __l2_prior_step(self, x, g, lr):
        new_x = x + lr * g / self.__norm(g)
        norm_new_x = self.__norm(new_x)
        norm_mask = (norm_new_x < 1.0).float()
        return new_x * norm_mask + (1 - norm_mask) * new_x / norm_new_x


    def __gd_prior_step(self, x, g, lr):
        return x + lr * g


    def __l2_image_step(self, x, g, lr):
        return x + lr * g / self.__norm(g)


    ##
    # Projection steps for l2 and linf constraints:
    # All take the form of func(new_x, old_x, epsilon)
    ##

    def __l2_proj(self, image, eps):
        orig = image.clone()

        def proj(new_x):
            delta = new_x - orig
            out_of_bounds_mask = (self.__norm(delta) > eps).float()
            x = (orig + eps * delta / self.__norm(delta)) * out_of_bounds_mask
            x += new_x * (1 - out_of_bounds_mask)
            return x

        return proj


    def __linf_proj(self, image, eps):
        orig = image.clone()

        def proj(new_x):
            return orig + ch.clamp(new_x - orig, -eps, eps)

        return proj


    ##
    # Main functions
    ##

    def make_adversarial_examples(self, image, true_label, max_queries, fd_eta, image_lr, online_lr, mode, exploration, tile_size,
                                epsilon, batch_size, log_progress, nes, tiling, gradient_iters, model_to_fool,
                                IMAGENET_SL, mean, std, device):
        '''
        The main process for generating adversarial examples with priors.
        '''

        # Initial setup
        prior_size = IMAGENET_SL if not tiling else tile_size
        upsampler = Upsample(size=(IMAGENET_SL, IMAGENET_SL))
        total_queries = ch.zeros(batch_size)
        prior = ch.zeros(batch_size, 3, prior_size, prior_size)
        dim = prior.nelement() / batch_size
        prior_step = self.__gd_prior_step if mode == 'l2' else self.__eg_step
        image_step = self.__l2_image_step if mode == 'l2' else self.__linf_step
        proj_maker = self.__l2_proj if mode == 'l2' else self.__linf_proj
        proj_step = proj_maker(image, epsilon)


        # Loss function
        criterion = ch.nn.CrossEntropyLoss(reduction='none')

        def normalized_eval(x):
            with ch.no_grad():
                x_copy = x.clone()
                x_copy = ch.stack([F.normalize(x_copy[i], mean, std) for i in range(batch_size)])
                logits = model_to_fool(x_copy.to(device))
                x_copy.cpu()
                return logits.cpu()

        L = lambda x: criterion(normalized_eval(x), true_label)
        # losses = L(image)

        # Original classifications
        orig_images = image.clone()
        orig_classes = model_to_fool(image.to(device)).argmax(1).cpu()
        correct_classified_mask = (orig_classes == true_label).float()
        total_ims = correct_classified_mask.sum()
        not_dones_mask = correct_classified_mask.clone()

        t = 0
        while not ch.any(total_queries > max_queries):
            t += gradient_iters * 2
            if t >= max_queries:
                break
            if not nes:
                ## Updating the prior:
                # Create noise for exporation, estimate the gradient, and take a PGD step
                exp_noise = exploration * ch.randn_like(prior) / (dim ** 0.5)
                # Query deltas for finite difference estimator
                q1 = upsampler(prior + exp_noise)
                q2 = upsampler(prior - exp_noise)
                # Loss points for finite difference estimator
                l1 = L(image + fd_eta * q1 / self.__norm(q1))  # L(prior + c*noise)
                l2 = L(image + fd_eta * q2 / self.__norm(q2))  # L(prior - c*noise)
                # Finite differences estimate of directional derivative
                est_deriv = (l1 - l2) / (fd_eta * exploration)
                # 2-query gradient estimate
                est_grad = est_deriv.view(-1, 1, 1, 1) * exp_noise
                # Update the prior with the estimated gradient
                prior = prior_step(prior, est_grad, online_lr)
            else:
                prior = ch.zeros_like(image)
                for _ in range(gradient_iters):
                    exp_noise = ch.randn_like(image) / (dim ** 0.5)
                    est_deriv = (L(image + fd_eta * exp_noise) - L(image - fd_eta * exp_noise)) / fd_eta
                    prior += est_deriv.view(-1, 1, 1, 1) * exp_noise

                # Preserve images that are already done,
                # Unless we are specifically measuring gradient estimation
                prior = prior * not_dones_mask.view(-1, 1, 1, 1)

            ## Update the image:
            # take a pgd step using the prior
            new_im = image_step(image, upsampler(prior * correct_classified_mask.view(-1, 1, 1, 1)), image_lr)
            image = proj_step(new_im)
            image = ch.clamp(image, 0, 1)
            if mode == 'l2':
                if not ch.all(self.__norm(image - orig_images) <= epsilon + 1e-3):
                    pdb.set_trace()
            else:
                if not (image - orig_images).max() <= epsilon + 1e-3:
                    pdb.set_trace()

            ## Continue query count
            total_queries += 2 * gradient_iters * not_dones_mask
            not_dones_mask = (not_dones_mask * ((normalized_eval(image).argmax(1) == true_label).float()))

            ## Logging stuff
            new_losses = L(image)
            success_mask = ((1 - not_dones_mask) * correct_classified_mask)
            num_success = success_mask.sum()
            current_success_rate = (num_success / correct_classified_mask.sum()).cpu().item()
            success_queries = ((success_mask * total_queries).sum() / num_success).cpu().item()
            not_done_loss = ((new_losses * not_dones_mask).sum() / not_dones_mask.sum()).cpu().item()
            max_curr_queries = total_queries.max().cpu().item()
            if log_progress:
                print("Queries: %d | Success rate: %f | Average queries: %f" % (
                    max_curr_queries, current_success_rate, success_queries))

            if current_success_rate == 1.0:
                break

        return success_queries, correct_classified_mask.sum().cpu().item(), current_success_rate, orig_images.cpu().numpy(), \
            image.cpu().detach().numpy(), total_queries.cpu().numpy(), correct_classified_mask.cpu().numpy(), success_mask.cpu().numpy()        


def execute_attack(model, images, args, mean, std):
    nes = True if 'nes' in args else False
    tiling = True if 'tiling' in args else False
    dataset_size = images[0].shape[1]
    
    total_success, total_images, sum_queries = 0, 0, 0

    all_l2s, all_original_images, all_adversarial_images, all_labels, all_adversarial_labels, all_queries_total = [], [], [], [], [], []

    if 'batch_size' in args and args['batch_size'] > 0:
        b_size = args['batch_size']
    else:
        print("ERROR: Batch size not define")
        exit(1)
    

    for i in range(args['total_images'] // b_size):
        images_batch = images[i * b_size : b_size * (i + 1)]
        labels_batch = labels[i * b_size : b_size * (i + 1)]

        banditsprior_attack = BanditsPrior(images_batch.cpu(), labels_batch.cpu(), args['max_queries'], args['fd_eta'], args['image_lr'], args['online_lr'],
                        args['mode'], args['exploration'], args['tile_size'], args['epsilon'], b_size, 'log' in args, nes, tiling,
                        args['gradient_iters'], model.module.to(args['device']), dataset_size, mean, std, args['device'])

        ncc, average_queries, success_rate, images_orig, images_adv, all_queries, correctly_classified, success = banditsprior_attack.perturb()

        model = model.to(args['device'])
        images_adv = ch.from_numpy(images_adv)
        images_adv = images_adv.to(args['device'])
        pred_bandits = model(utils.normalize_image(images_adv, mean, std)).argmax(dim=1).cpu()
        success = (pred_bandits.cpu() != labels_batch.cpu()).float().mean().item() * 100

        l2_norms = []
        for ii in range(len(images_adv)):
            l2_norms.append(utils.calculate_l2_norm(images_adv.cpu().detach().numpy()[ii],
                                                images_batch.cpu().detach().numpy()[ii]))
        
        all_l2s.extend(l2_norms)
        all_original_images.extend(images_orig)
        all_adversarial_images.extend(images_adv.cpu().detach().numpy())
        all_labels.extend(labels_batch.cpu())
        all_adversarial_labels.extend(pred_bandits.cpu().detach().numpy())
        all_queries_total.extend(all_queries)
        
        success = sum((labels_batch.cpu() != pred_bandits.cpu()).float() == 1.)
        total_success += success.item()
        total_images += len(labels_batch.cpu())
        sum_queries += all_queries.max()

        del banditsprior_attack

        if 'log' in args: print("Images done: {}/{}; Success: {:.2f}%".format(b_size * (i + 1), args['total_images'], success.item() / b_size * 100))

    if 'log' in args: print('Bandits Prior done: Success: {:.2f}%'.format(total_success / len(all_l2s) * 100))

    return all_adversarial_images, all_adversarial_labels, all_queries_total , all_l2s, total_success

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-queries', type=int)
    parser.add_argument('--fd-eta', type=float, help='\eta, used to estimate the derivative via finite differences')
    parser.add_argument('--image-lr', type=float, help='Learning rate for the image (itperative attack)')
    parser.add_argument('--online-lr', type=float, help='Learning rate for the prior')
    parser.add_argument('--mode', type=str, help='Which lp constraint to run bandits [linf|l2]')
    parser.add_argument('--exploration', type=float, help='\delta, parameterizes the exploration to be done around the prior')
    parser.add_argument('--tile-size', type=int, help='the side length of each tile (for the tiling prior)')
    parser.add_argument('--epsilon', type=float, help='the lp perturbation bound')
    parser.add_argument('--nes', action='store_true')
    parser.add_argument('--tiling', action='store_true')
    parser.add_argument('--gradient-iters', type=int)
    parser.add_argument('--batch-size', type=int, help='batch size for bandits')  
    #parser.add_argument('--json-config', type=str, help='a config file to be passed in instead of arguments')
      
    parser.add_argument('--model', type=str, default = 'Path for binary file containing the model')
    parser.add_argument('--dataset', type=str, default = 'Path for dataset loader file')
    parser.add_argument('--total-images', type=int, help='')
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--attack-config', type=str, default = '', help='Config file to be passed in instead of arguments')
    parser.add_argument('--results-path', type=str, default = '', help="Path to store results")

    ini_args = vars(parser.parse_args())

    ini_args['device'] = 'cuda' if ch.cuda.is_available() else 'cpu'

    
    # initial argument verifications
    args = utils.prepare_attack_arguments(ini_args, ini_args['attack_config'], 'deepfool')


    try:
        # setup logger
        if not os.path.isdir(args['results_path']):
            os.makedirs(args['results_path'])

        logging.basicConfig(filename=os.path.join(args['results_path'], 'BanditsPrior.log'), format='%(asctime)s | %(levelname)s | %(message)s', datefmt= '%m-%d-%Y %H:%M:%S', level=logging.INFO)

        ini_stdout = sys.stdout
        sys.stdout = utils.StdoutToLogging()

        print("Loading BanditsPrior")


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

        print("Running BanditsPrior on {}".format(args['device']))

        adversarial_images, adversarial_labels, all_queries, l2s, total_success = execute_attack(model, images, args, dataset.MEAN, dataset.STD)

        
        bandits_prior_dict = {
            'attack_name': 'BanditsPrior',
            'perturbed_image': adversarial_images,
            'perturbed_label': adversarial_labels,
            'total_queries': all_queries,
            'l2': l2s,
            'success': total_success
        }

        print("Saving Results")

        utils.save_statistics_of_attack(bandits_prior_dict, labels, classified_labels, args['results_path'])

        utils.save_images_attack(bandits_prior_dict, labels, os.path.join(args['results_path'], 'perturbed_images'))


    finally:
        sys.stdout = ini_stdout

