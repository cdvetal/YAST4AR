import sys
import os

# get repo path for imports
abs_path = os.path.abspath(os.path.realpath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.dirname(abs_path)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

import torch
import numpy as np
import time
import math
import argparse
import logging
import importlib
import dill
from typing import List

import torchvision.transforms as transforms
from torch.autograd import Variable

from attacks.Attack import Attack
from attacks.GeoDA.SubNoise import SubNoise

from attacks.GeoDA import GeoDAUtils

import utils.static_vars as static
import utils.func_utils as utils


class GeoDA(Attack):
    def __init__(self, dist: str, device: torch.device, tol=0.0001, sigma=0.0002,
                mu=0.6, Q_max=5000, sub_dim=75, search_space='sub', verbose_control=True):
        """GeoDA class

        Parameters
        ----------

        dist :
            Norm distance ('linf', 'l2', 'l1')
        delta : int
            Original max values of pixels of the original image (default = 255)
        device : torch.device
            Device to use (cpu or a gpu device)
        verbose_control : str
            Verbose control (default = 'Yes')
        Q_max : int
            Maximum number of queries (default = 5000)
        sub_dim : int
            Attack hyperparameter (default = 75)
        tol : double
            Attack hyperparameter (default = 0.0001)
        sigma : double
            Attack hyperparameter (default = 0.0002)
        mu : double
            Attack hyperparameter (default = 0.6)
        search_space : str
            Search space (default = 'sub')

        """
        super().__init__()

        self.dist = dist
        self.device = device
        self.verbose_control = verbose_control
        self.Q_max = Q_max
        self.sub_dim = sub_dim
        self.tol = tol
        self.sigma = sigma
        self.mu = mu
        self.search_space = search_space


    def perturb(self, delta, im_orig, net, img_width: int, img_height: int, mean: List[float], std: List[float],
                ground_truth_label: int, grad_estimator_batch_size: int, temp_folder_path = static.PATH_TEMP):


        """Attack a model using the GeoDA Attack.

        Parameters
        ----------

        delta : int
            Original max values of pixels of the original image (default = 255)
        im_orig :
            Original image to perturb (Normalized!)
        net :
            Target pytorch model
        device : torch.device
            Device to use (cpu or a gpu device)
        grad_estimator_batch_size : int
            Batch size
        verbose_control : str
            Verbose control (default = 'Yes')
        img_width : int
            Width of the original image
        img_height : int
            Height of the original image
        mean : list
            Mean of the images of the dataset
        std : list
            Standard Deviation of the images of the dataset
        ground_truth_label :
            The ground truth label of the original image
        temp_folder_path : 
            folder to stop temporary file

        Returns
        -------
        pertimage:
            The adversarial image generated
        x_opt_inverse:

        normalized_image:

        pert_norm:
            The norm perturbation
        """

        ####################################################################################################

        def is_adversarial(given_image, orig_label):
            predict_label = torch.argmax(net.forward(Variable(given_image, requires_grad=True)).data).item()

            return predict_label != orig_label

        def find_random_adversarial(image, epsilon=1000):
            num_calls = 1

            step = 0.02
            perturbed = x_0

            while is_adversarial(perturbed, orig_label) == 0:
                pert = torch.randn([1, 3, img_width, img_height])
                pert = pert.to(self.device)

                perturbed = image + num_calls * step * pert
                perturbed = GeoDAUtils.clip_image_values(perturbed, lb, ub)
                perturbed = perturbed.to(self.device)
                num_calls += 1

            return perturbed, num_calls

        def bin_search(x_0, x_random, tol):
            num_calls = 0
            adv = x_random
            cln = x_0

            while True:

                mid = (cln + adv) / 2.0
                num_calls += 1

                if is_adversarial(mid, orig_label):
                    adv = mid
                else:
                    cln = mid

                if torch.norm(adv - cln).cpu().numpy() < tol:
                    break

            return adv, num_calls

        def black_grad_batch(x_boundary, q_max, sigma, random_noises, batch_size, original_label):
            grad_tmp = []  # estimated gradients in each estimate_batch
            z = []  # sign of grad_tmp
            outs = []
            num_batchs = math.ceil(q_max / batch_size)
            last_batch = q_max - (num_batchs - 1) * batch_size
            EstNoise = SubNoise(batch_size, sub_basis_torch).to(self.device)
            all_noises = []
            for j in range(num_batchs):
                if j == num_batchs - 1:
                    EstNoise_last = SubNoise(last_batch, sub_basis_torch).to(self.device)
                    current_batch = EstNoise_last(img_width, img_height)
                    current_batch_np = current_batch.cpu().numpy()
                    noisy_boundary = [x_boundary[0, :, :,
                                    :].cpu().numpy()] * last_batch + sigma * current_batch.cpu().numpy()

                else:
                    current_batch = EstNoise(img_width, img_height)
                    current_batch_np = current_batch.cpu().numpy()
                    noisy_boundary = [x_boundary[0, :, :,
                                    :].cpu().numpy()] * batch_size + sigma * current_batch.cpu().numpy()

                all_noises.append(current_batch_np)

                noisy_boundary_tensor = torch.tensor(noisy_boundary).to(self.device)

                predict_labels = torch.argmax(net.forward(noisy_boundary_tensor), 1).cpu().numpy().astype(int)

                outs.append(predict_labels)
            all_noise = np.concatenate(all_noises, axis=0)
            outs = np.concatenate(outs, axis=0)

            for i, predict_label in enumerate(outs):
                if predict_label == original_label:
                    z.append(1)
                    grad_tmp.append(all_noise[i])
                else:
                    z.append(-1)
                    grad_tmp.append(-all_noise[i])

            grad = -(1 / q_max) * sum(grad_tmp)

            grad_f = torch.tensor(grad).to(self.device)[None, :, :, :]

            return grad_f, sum(z)

        def go_to_boundary(x_0, grad, x_b):
            epsilon = 5

            num_calls = 1
            perturbed = x_0

            if self.dist == 'l1' or self.dist == 'l2':
                grads = grad

            if self.dist == 'linf':
                grads = torch.sign(grad) / torch.norm(grad)

            while is_adversarial(perturbed, orig_label) == 0:

                perturbed = x_0 + (num_calls * epsilon * grads[0])
                perturbed = GeoDAUtils.clip_image_values(perturbed, lb, ub)

                num_calls += 1

                if num_calls > 100:
                    if self.verbose_control: print('failed ... ')
                    break

            return perturbed, num_calls, epsilon * num_calls

        def GeoDA(x_b, iteration, q_opt):
            q_num = 0
            grad = 0

            for i in range(iteration):

                t1 = time.time()
                random_vec_o = torch.randn(q_opt[i], 3, img_width, img_height)

                grad_oi, ratios = black_grad_batch(x_b, q_opt[i], self.sigma, random_vec_o, grad_estimator_batch_size,
                                                orig_label)
                q_num = q_num + q_opt[i]
                grad = grad_oi + grad
                x_adv, qs, eps = go_to_boundary(x_0, grad, x_b)
                q_num = q_num + qs
                x_adv, bin_query = bin_search(x_0, x_adv, self.tol)

                q_num = q_num + bin_query

                x_b = x_adv

                t2 = time.time()
                x_adv_inv = GeoDAUtils.inv_tf(x_adv.cpu().numpy()[0, :, :, :].squeeze(), mean, std)

                if self.dist == 'l1' or self.dist == 'l2':
                    dp = 'l2'
                    norm_p = np.linalg.norm(x_adv_inv - denormalized_image)

                elif self.dist == 'linf':
                    dp = self.dist

                    norm_p = np.max(abs(x_adv_inv - denormalized_image))

                if self.verbose_control:
                    message = ' (took {:.5f} seconds)'.format(t2 - t1)
                    if self.verbose_control: print('iteration -> ' + str(i) + str(message) + '     -- ' + dp + ' norm is -> ' + str(norm_p))

            x_adv = GeoDAUtils.clip_image_values(x_adv, lb, ub)

            return x_adv, q_num, grad

        def opt_query_iteration(Nq, T, eta):
            coefs = [eta ** (-2 * i / 3) for i in range(0, T)]
            coefs[0] = 1 * coefs[0]

            sum_coefs = sum(coefs)
            opt_q = [round(Nq * coefs[i] / sum_coefs) for i in range(0, T)]

            if opt_q[0] > 80:
                T = T + 1
                opt_q, T = opt_query_iteration(Nq, T, eta)
            elif opt_q[0] < 50:
                T = T - 1

                opt_q, T = opt_query_iteration(Nq, T, eta)

            return opt_q, T

        ####################################################################################################


        # Remove the normalization of the image because it is needed in several steps (This way, the user only needs to
        # insert a normalized image as input)
        denormalized_image = torch.clone(im_orig)
        denormalized_image = denormalized_image.cpu().numpy()

        for i in range(3):
            denormalized_image[i, :, :] = denormalized_image[i, :, :] * std[i] + mean[i]

        denormalized_image = np.transpose(denormalized_image, (1, 2, 0))

        # Image with original values as a numpy array to get the valid bounds
        orig = (denormalized_image * 255).astype(int)

        lb, ub = GeoDAUtils.valid_bounds(orig, delta)

        lb = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])(lb)
        ub = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])(ub)

        lb = lb[None, :, :, :].to(self.device)
        ub = ub[None, :, :, :].to(self.device)


        if self.search_space == 'sub':
            if self.verbose_control: print('Check if DCT basis available ...', )
            
            file = 'n_{}_2d_dct_basis_{}.npy'.format(img_width, self.sub_dim)
            path = os.path.join(temp_folder_path, file)
            
            if os.path.isfile(path):
                if self.verbose_control: print('Yes, we already have it ...')
                sub_basis = np.load(path).astype(np.float32)
            else:
                if self.verbose_control: print('Generating dct basis ......')
                sub_basis = GeoDAUtils.generate_2d_dct_basis(self.sub_dim, img_width, folder_path=temp_folder_path).astype(np.float32)


            estimate_batch = grad_estimator_batch_size
            sub_basis_torch = torch.from_numpy(sub_basis).to(self.device)
            EstNoise = SubNoise(estimate_batch, sub_basis_torch).to(self.device)
            random_vectors = EstNoise(img_width, img_height)

        x_0 = im_orig[None, :, :, :].to(self.device)

        orig_label = torch.argmax(net.forward(Variable(x_0, requires_grad=True)).data).item()

        if ground_truth_label != int(orig_label):
            if self.verbose_control: print('Already missclassified ... Lets try another one!')
            return denormalized_image, -1, denormalized_image, -1, x_0, 0

        else:

            x_random, query_random_1 = find_random_adversarial(x_0, epsilon=100)
            is_adversarial(x_random, orig_label)

            # Binary search
            x_boundary, query_binsearch_2 = bin_search(x_0, x_random, self.tol)
            x_b = x_boundary

            is_adversarial(x_boundary, orig_label)
            query_rnd = query_binsearch_2 + query_random_1

            # Run over iterations
            iteration = round(self.Q_max / 500)
            q_opt_it = int(self.Q_max - iteration * 25)
            q_opt_iter, iterate = opt_query_iteration(q_opt_it, iteration, self.mu)
            q_opt_it = int(self.Q_max - iterate * 25)
            q_opt_iter, iterate = opt_query_iteration(q_opt_it, iteration, self.mu)
            if self.verbose_control:
                print('#################################################################')
                print('Start: The GeoDA will be run for:' + ' Iterations = ' + str(iterate) + ', Query = ' + str(
                        self.Q_max) + ', Norm = ' + str(self.dist) + ', Space = ' + str(self.search_space))
                print('#################################################################')

            t3 = time.time()
            x_adv, query_o, gradient = GeoDA(x_b, iterate, q_opt_iter)
            t4 = time.time()
            message = ' took {:.5f} seconds'.format(t4 - t3)
            total_queries = query_o + query_rnd
            qmessage = ' with query = ' + str(total_queries)

            x_opt_inverse = GeoDAUtils.inv_tf(x_adv.cpu().numpy()[0, :, :, :].squeeze(), mean, std)
            if self.verbose_control:
                print('#################################################################')
                print('End: The GeoDA algorithm' + message + qmessage)
                print('#################################################################')

            if self.dist == 'l2' or self.dist == 'linf':
                adv_label = torch.argmax(net.forward(Variable(x_adv, requires_grad=True)).data).item()

                pert_norm = abs(x_opt_inverse - denormalized_image) / np.linalg.norm(abs(x_opt_inverse - denormalized_image))

                pert_norm_abs = (x_opt_inverse - denormalized_image) / np.linalg.norm((x_opt_inverse - denormalized_image))

                pertimage = denormalized_image + 30 * pert_norm_abs

                return denormalized_image, pertimage, x_opt_inverse, pert_norm, x_adv, total_queries


def execute_attack(model, images, args, mean, std, image_size):
    total_success = 0

    all_l2s, all_original_images, all_adversarial_images, all_labels, all_adversarial_labels, all_queries = [], [], [], [], [], []

    geodaattack = GeoDA(dist=args["dist"], device=args['device'], sigma=args['sigma'], tol=args['tol'], 
                            sub_dim=args['sub_dim'], Q_max=args['Q_max'], 
                            search_space=args['search_space'], verbose_control='log' in args)
    
    for i in range(args['total_images']):
        input_image = utils.normalize_image(images[i], mean, std)

        normalized_image, pertimage, x_opt_inverse, pert_norm, x_adv, total_queries = geodaattack.perturb(
                            delta=args['delta'], im_orig=input_image.to(args['device']), net=model.to(args['device']),
                            img_width=image_size, img_height=image_size, mean=mean, std=std,
                            ground_truth_label=labels[i].to(args['device']), grad_estimator_batch_size=args['batch_size'])       
                

        pred_geoda = model(x_adv).argmax(dim=1).cpu()[0]
        success = (pred_geoda != labels[i].cpu()).float().mean().item() * 100

        x_opt_inverse_copy = x_opt_inverse.copy()
        x_opt_inverse_copy = torch.from_numpy(np.transpose(x_opt_inverse_copy, (2, 0, 1)))
        l2 = utils.calculate_l2_norm(x_opt_inverse_copy.cpu().detach().numpy(),
                                images[i].cpu().detach().numpy())

        all_l2s.append(l2)
        all_original_images.append(np.transpose(images[i].cpu().detach().numpy(), (1, 2, 0)))
        x_opt_inverse_ = np.transpose(x_opt_inverse, (2, 0, 1))
        all_adversarial_images.append(x_opt_inverse_)
        all_labels.append(labels[i].cpu())
        all_adversarial_labels.append(pred_geoda.cpu())
        all_queries.append(total_queries)

        if labels[i].cpu() != pred_geoda.cpu():
            total_success += 1

        if 'log' in args: print(str(i + 1) + "/" + str(args['total_images']))
        if 'log' in args: print("Images done: {}/{}; Success: {:.2f}%".format(i + 1, args['total_images'], success))

    if 'log' in args: print('GeoDA done: Success: {:.2f}%'.format(total_success / len(all_l2s) * 100))

    return all_adversarial_images, all_adversarial_labels, all_queries, all_l2s, total_success


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', type=str, help='')
    parser.add_argument('--tol', type=float, help='')
    parser.add_argument('--sigma', type=float, help='')
    parser.add_argument('--mu', type=float, help='')
    parser.add_argument('--delta', type=int, help='')
    parser.add_argument('--search-space', type=str, help='')
    parser.add_argument('--sub-dim', type=int, help='')
    parser.add_argument('--Q-max', type=int, help='')
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
    args = utils.prepare_attack_arguments(ini_args, ini_args['attack_config'], 'geoda')


    try:
        # setup logger
        if not os.path.isdir(args['results_path']):
            os.makedirs(args['results_path'])

        logging.basicConfig(filename=os.path.join(args['results_path'], 'GeoDA.log'), format='%(asctime)s | %(levelname)s | %(message)s', datefmt= '%m-%d-%Y %H:%M:%S', level=logging.INFO)

        ini_stdout = sys.stdout
        sys.stdout = utils.StdoutToLogging()
        
        print("Loading GeoDA")
        

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

        print("Running DeepFool on {}".format(args['device']))

        adversarial_images, adversarial_labels, all_queries, l2s, total_success = execute_attack(model, images, args, dataset.MEAN, dataset.STD, dataset.IMAGE_SIZE)
        
        
        geoda_dict = {
            'attack_name': 'GeoDA',
            'perturbed_image': adversarial_images,
            'perturbed_label': adversarial_labels,
            'total_queries': all_queries,
            'l2': l2s,
            'success': total_success
        }

        print("Saving Results")

        utils.save_statistics_of_attack(geoda_dict, labels, classified_labels, args['results_path'])

        utils.save_images_attack(geoda_dict, labels, os.path.join(args['results_path'], 'perturbed_images'))

    finally:
        sys.stdout = ini_stdout