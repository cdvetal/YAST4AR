import sys
import os

# get repo path for imports
abs_path = os.path.abspath(os.path.realpath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.dirname(abs_path)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

import torch
import numpy as np
import argparse
import importlib
import logging
import dill

from attacks.CornerSearch.utilsCornerSearch import get_logits, get_predictions

from attacks.Attack import Attack
import utils.func_utils as utils


def onepixel_perturbation(attack, orig_x, pos, sigma):
    ''' returns a batch with the possible perturbations of the pixel in position pos '''

    if attack.type_attack == 'L0':
        if orig_x.shape[-1] == 3:
            batch_x = np.tile(orig_x, (8, 1, 1, 1))
            t = np.zeros([3])
            for counter in range(8):
                t2 = counter + 0
                for c in range(3):
                    t[c] = t2 % 2
                    t2 = (t2 - t[c]) / 2
                batch_x[counter, pos[0], pos[1]] = t.astype(np.float32)
        elif orig_x.shape[-1] == 1:
            batch_x = np.tile(orig_x, (2, 1, 1, 1))
            batch_x[0, pos[0], pos[1], 0] = 0.0
            batch_x[1, pos[0], pos[1], 0] = 1.0

    elif attack.type_attack == 'L0+Linf':
        if orig_x.shape[-1] == 3:
            batch_x = np.tile(orig_x, (8, 1, 1, 1))
            t = np.zeros([3])
            for counter in range(8):
                t2 = counter + 0
                for c in range(3):
                    t3 = t2 % 2
                    t[c] = (t3 * 2.0 - 1.0) * attack.epsilon
                    t2 = (t2 - t3) / 2
                batch_x[counter, pos[0], pos[1]] = np.clip(t.astype(np.float32) + orig_x[pos[0], pos[1]], 0.0, 1.0)
        elif orig_x.shape[-1] == 1:
            batch_x = np.tile(orig_x, (2, 1, 1, 1))
            batch_x[0, pos[0], pos[1], 0] = np.clip(batch_x[0, pos[0], pos[1], 0] - attack.epsilon, 0.0, 1.0)
            batch_x[1, pos[0], pos[1], 0] = np.clip(batch_x[1, pos[0], pos[1], 0] + attack.epsilon, 0.0, 1.0)

    elif attack.type_attack == 'L0+sigma':
        batch_x = np.tile(orig_x, (2, 1, 1, 1))
        if orig_x.shape[-1] == 3:
            batch_x[0, pos[0], pos[1]] = np.clip(
                batch_x[0, pos[0], pos[1]] * (1.0 - attack.kappa * sigma[pos[0], pos[1]]), 0.0, 1.0)
            batch_x[1, pos[0], pos[1]] = np.clip(
                batch_x[0, pos[0], pos[1]] * (1.0 + attack.kappa * sigma[pos[0], pos[1]]), 0.0, 1.0)

        elif orig_x.shape[-1] == 1:
            batch_x[0, pos[0], pos[1]] = np.clip(batch_x[0, pos[0], pos[1]] - attack.kappa * sigma[pos[0], pos[1]], 0.0,
                                                 1.0)
            batch_x[1, pos[0], pos[1]] = np.clip(batch_x[0, pos[0], pos[1]] + attack.kappa * sigma[pos[0], pos[1]], 0.0,
                                                 1.0)

    else:
        raise ValueError('unknown attack')

    return batch_x


def onepixel_perturbation_image(attack, orig_x, sigma):
    ''' returns a batch with all the possible perturbations of the image orig_x '''

    n_channels = orig_x.shape[-1]
    assert n_channels in [1, 3]
    n_corners = 2 ** n_channels if attack.type_attack in ['L0', 'L0+Linf'] else 2

    batch_x = np.zeros(
        [n_corners * orig_x.shape[0] * orig_x.shape[1], orig_x.shape[0], orig_x.shape[1], orig_x.shape[2]])
    for counter in range(orig_x.shape[0]):
        for counter2 in range(orig_x.shape[1]):
            batch_x[(counter * orig_x.shape[0] + counter2) * n_corners:(counter * orig_x.shape[
                1] + counter2) * n_corners + n_corners] = np.clip(
                onepixel_perturbation(attack, orig_x, [counter, counter2], sigma), 0.0, 1.0)

    return batch_x


def flat2square(attack, ind):
    ''' returns the position and the perturbation given the index of an image
      of the batch of all the possible perturbations '''

    if attack.type_attack in ['L0', 'L0+Linf']:
        if attack.shape_img[-1] == 3:
            new_pixel = ind % 8
            ind = (ind - new_pixel) // 8
            c = ind % attack.shape_img[1]
            r = (ind - c) // attack.shape_img[1]
            t = np.zeros([ind.shape[0], 3])
            for counter in range(3):
                t[:, counter] = new_pixel % 2
                new_pixel = (new_pixel - t[:, counter]) / 2
        elif attack.shape_img[-1] == 1:
            t = ind % 2
            ind = (ind - t) // 2
            c = ind % attack.shape_img[1]
            r = (ind - c) // attack.shape_img[1]

    elif attack.type_attack == 'L0+sigma':
        t = ind % 2
        c = ((ind - t) // 2) % attack.shape_img[1]
        r = ((ind - t) // 2 - c) // attack.shape_img[1]

    return r, c, t


def npixels_perturbation(attack, orig_x, ind, k, sigma):
    ''' creates n_iter images which differ from orig_x in at most k pixels '''

    # sampling the n_iter k-pixels perturbations
    ind2 = np.random.randint(0, attack.n_max ** 2, (attack.n_iter, k))
    ind2 = attack.n_max - np.floor(ind2 ** 0.5).astype(int) - 1

    # creating the n_iter k-pixels perturbed images
    batch_x = np.tile(orig_x, (attack.n_iter, 1, 1, 1))
    if attack.type_attack == 'L0':
        for counter in range(attack.n_iter):
            p11, p12, d1 = flat2square(attack, ind[ind2[counter]])
            batch_x[counter, p11, p12] = d1 + 0 if attack.shape_img[-1] == 3 else np.expand_dims(d1 + 0, 1)

    elif attack.type_attack == 'L0+Linf':
        for counter in range(attack.n_iter):
            p11, p12, d1 = flat2square(attack, ind[ind2[counter]])
            d1 = d1 + 0 if attack.shape_img[-1] == 3 else np.expand_dims(d1 + 0, 1)
            batch_x[counter, p11, p12] = np.clip(batch_x[counter, p11, p12] + (2.0 * d1 - 1.0) * attack.epsilon, 0.0,
                                                 1.0)

    elif attack.type_attack == 'L0+sigma':
        for counter in range(attack.n_iter):
            p11, p12, d1 = flat2square(attack, ind[ind2[counter]])
            d1 = np.expand_dims(d1, 1)
            if attack.shape_img[-1] == 3:
                batch_x[counter, p11, p12] = np.clip(
                    batch_x[counter, p11, p12] - attack.kappa * sigma[p11, p12] * (1 - d1) + attack.kappa * sigma[
                        p11, p12] * d1, 0.0, 1.0)
            elif attack.shape_img[-1] == 1:
                batch_x[counter, p11, p12] = np.clip(
                    batch_x[counter, p11, p12] - attack.kappa * sigma[p11, p12] * (1 - d1) + attack.kappa * sigma[
                        p11, p12] * d1, 0.0, 1.0)

    return batch_x


def sigma_map(x):
    ''' creates the sigma-map for the batch x '''

    sh = [4]
    sh.extend(x.shape)
    t = np.zeros(sh)
    t[0, :, :-1] = x[:, 1:]
    t[0, :, -1] = x[:, -1]
    t[1, :, 1:] = x[:, :-1]
    t[1, :, 0] = x[:, 0]
    t[2, :, :, :-1] = x[:, :, 1:]
    t[2, :, :, -1] = x[:, :, -1]
    t[3, :, :, 1:] = x[:, :, :-1]
    t[3, :, :, 0] = x[:, :, 0]

    mean1 = (t[0] + x + t[1]) / 3
    sd1 = np.sqrt(((t[0] - mean1) ** 2 + (x - mean1) ** 2 + (t[1] - mean1) ** 2) / 3)

    mean2 = (t[2] + x + t[3]) / 3
    sd2 = np.sqrt(((t[2] - mean2) ** 2 + (x - mean2) ** 2 + (t[3] - mean2) ** 2) / 3)

    sd = np.minimum(sd1, sd2)
    sd = np.sqrt(sd)

    return sd


class CSattack(Attack):
    def __init__(self, args):
        super().__init__()

        self.type_attack = args['type_attack']  # 'L0', 'L0+Linf', 'L0+sigma'
        self.n_iter = args['n_iter']  # number of iterations (N_iter in the paper)
        self.n_max = args['n_max']  # the modifications for k-pixels perturbations are sampled among the best n_max (N in the paper)
        self.epsilon = args['epsilon']  # for L0+Linf, the bound on the Linf-norm of the perturbation
        self.kappa = args['kappa']  # for L0+sigma (see kappa in the paper), larger kappa means easier and more visible attacks
        self.k = args['sparsity']  # maximum number of pixels that can be modified (k_max in the paper)
        self.size_incr = args['size_incr']  # size of progressive increment of sparsity levels to check
        self.log = args['log'] if 'log' in args else False
        self.device = args['device']

    def perturb(self, x_nat, y_nat, n_classes, model, mean, std):
        self.model = model
        self.mean = mean
        self.std = std
        total_queries = 0
        adv = np.copy(x_nat)
        fl_success = np.ones([x_nat.shape[0]])
        self.shape_img = x_nat.shape[1:]
        self.sigma = sigma_map(x_nat)
        self.n_classes = n_classes
        self.n_corners = 2 ** self.shape_img[2] if self.type_attack in ['L0', 'L0+Linf'] else 2
        corr_pred = get_predictions(self.model, x_nat, y_nat, mean=self.mean, std=self.std, device=self.device)
        total_queries += 1
        bs = self.shape_img[0] * self.shape_img[1]

        for c in range(x_nat.shape[0]):
            if corr_pred[c]:
                sigma = np.copy(self.sigma[c])
                batch_x = onepixel_perturbation_image(self, x_nat[c], sigma)
                batch_y = np.squeeze(y_nat[c])
                logit_2 = np.zeros([batch_x.shape[0], self.n_classes])
                found = False

                # checks one-pixels modifications
                for counter in range(self.n_corners):
                    logit_2[counter * bs:(counter + 1) * bs] = get_logits(self.model,
                                                                          batch_x[counter * bs:(counter + 1) * bs], mean=self.mean, std=self.std, device=self.device)
                    total_queries += 1

                    pred = logit_2[counter * bs:(counter + 1) * bs].argmax(axis=-1) == np.tile(batch_y, (bs))
                    if not pred.all() and not found:
                        ind_adv = np.where(pred.astype(int) == 0)
                        adv[c] = batch_x[counter * bs + ind_adv[0][0]]
                        found = True
                        if self.log:
                            print('Point {} - adversarial example found changing 1 pixel'.format(c))

                # creates the orderings
                t1 = np.copy(logit_2[:, batch_y])
                logit_2[:, batch_y] = -1000.0 * np.ones(np.shape(logit_2[:, batch_y]))
                t2 = np.amax(logit_2, axis=1)
                t3 = t1 - t2
                logit_3 = np.tile(np.expand_dims(t1, axis=1), (1, self.n_classes)) - logit_2
                logit_3[:, batch_y] = t3
                ind = np.argsort(logit_3, axis=0)

                # checks multiple-pixels modifications
                for n3 in range(1 + self.size_incr, self.k + 1, self.size_incr):
                    if not found:
                        for c2 in range(self.n_classes):
                            if not found:
                                ind_cl = np.copy(ind[:, c2])

                                batch_x = npixels_perturbation(self, x_nat[c], ind_cl, n3, sigma)
                                # self.model.y_input: np.tile(batch_y,(batch_x.shape[0]))})
                                pred = get_predictions(self.model, batch_x, np.tile(batch_y, (batch_x.shape[0])), mean=self.mean, std=self.std, device=self.device)
                                total_queries += 1

                                if np.sum(pred.astype(np.int32)) < self.n_iter and not found:
                                    found = True
                                    ind_adv = np.where(pred.astype(int) == 0)
                                    adv[c] = batch_x[ind_adv[0][0]]
                                    if self.log:
                                        print('Point {} - adversarial example found changing {} pixels'.format(c, np.sum(
                                            np.amax(np.abs(adv[c] - x_nat[c]) > 1e-10, axis=-1), axis=(0, 1))))

                if not found:
                    fl_success[c] = 0
                    if self.log:
                        print('Point {} - adversarial example not found'.format(c))

            else:
                if self.log:
                    print('Point {} - misclassified'.format(c))

        pixels_changed = np.sum(np.amax(np.abs(adv - x_nat) > 1e-10, axis=-1), axis=(1, 2))

        corr_pred = get_predictions(self.model, adv, y_nat, mean=self.mean, std=self.std, device=self.device)
        
        return adv, pixels_changed, fl_success, total_queries


def execute_attack(model, images, args, mean, std, num_classes):
    total_success, sum_queries = 0, 0
    
    all_queries_total, all_original_images, all_labels, all_adversarial_images, all_adversarial_labels, all_l2s = [], [], [], [], [], []

    if 'batch_size' in args and args['batch_size'] > 0:
        b_size = args['batch_size']
    else:
        print("ERROR: Batch size not define")
        exit(1)

    attack_args = {'type_attack': args['type_attack'],
            'n_iter': args['n_iter'],
            'n_max': args['n_max'],
            'kappa': args['kappa'],
            'epsilon': args['epsilon'],
            'sparsity': args['sparsity'],
            'size_incr': args['size_incr'],
            'device': args['device'],
            'log': 'log' in args}

    csAttack = CSattack(attack_args)

    for i in range(args['total_images'] // b_size):
        images_batch = images[i * b_size : b_size * (i + 1)]
        labels_batch = labels[i * b_size : b_size * (i + 1)]
        images_batch_np = np.transpose(images_batch.cpu().detach().numpy(), (0, 2, 3, 1))

        adv, pixels_changed, fl_success, total_queries = csAttack.perturb(images_batch_np, labels_batch.cpu().numpy(), n_classes=num_classes, model=model, mean=mean, std=std)

        images_perturbed = adv.copy()
        images_perturbed = torch.from_numpy(np.transpose(images_perturbed, (0, 3, 1, 2)))
        images_perturbed_normalized = utils.normalize_image(images_perturbed, mean, std)

        l2_norms = []
        for ii in range(len(images_perturbed)):
            l2_norms.append(utils.calculate_l2_norm(images_perturbed.cpu().detach().numpy()[ii],
                                                images_batch.cpu().detach().numpy()[ii]))

        pred = model(images_perturbed_normalized).argmax(dim=1).cpu()

        all_l2s.extend(l2_norms)
        all_original_images.extend(images_batch.cpu().detach().numpy())
        
        perturbed_image = images_perturbed.cpu().detach().numpy()
        all_adversarial_images.extend(perturbed_image)

        all_labels.extend(labels_batch.cpu())
        all_adversarial_labels.extend(pred.cpu())

        success = sum((labels_batch.cpu() != pred.cpu()).float() == 1.)
        total_success += success.item()
        sum_queries += total_queries
        all_queries_total.extend([total_queries] * b_size)

        if 'log' in args: print("Images done: {}/{}; Success: {:.2f}%".format(b_size * (i + 1), args['total_images'], success.item() / b_size * 100))


    if 'log' in args: print('Corner Search done: Success: {:.2f}%'.format(total_success / len(all_l2s) * 100))

    return all_adversarial_images, all_adversarial_labels, all_queries_total , all_l2s, total_success

if __name__ == '__main__':
    # get all possible parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--type-attack', type=str, help='[L0|L0+Linf|L0+sigma]')
    parser.add_argument('--n-iter', type=int, help='Number of iterations (N_iter in the paper)')
    parser.add_argument('--n-max', type=int, help='The modifications for k-pixels perturbations are sampled among the best n_max (N in the paper)')
    parser.add_argument('--epsilon', type=float, help='For L0+Linf, the bound on the Linf-norm of the perturbation')
    parser.add_argument('--kappa', type=float, help='For L0+sigma (see kappa in the paper), larger kappa means easier and more visible attacks')
    parser.add_argument('--sparsity', type=int, help='Maximum number of pixels that can be modified (k_max in the paper)')
    parser.add_argument('--size-incr', type=int, help='Size of progressive increment of sparsity levels to check')
    parser.add_argument('--batch-size', type=int, help='Number of adversarial examples')

    parser.add_argument('--model', type=str, default = 'Path for binary file containing the model')
    parser.add_argument('--dataset', type=str, default = 'Path for dataset loader file')
    parser.add_argument('--total-images', type=int, help='')
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--attack-config', type=str, default = '', help='Config file to be passed in instead of arguments')
    parser.add_argument('--results-path', type=str, default = '', help="Path to store results")


    ini_args = vars(parser.parse_args())

    ini_args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    # initial argument verifications
    args = utils.prepare_attack_arguments(ini_args, ini_args['attack_config'], 'cornersearch')


    try:
        # setup logger
        if not os.path.isdir(args['results_path']):
            os.makedirs(args['results_path'])

        logging.basicConfig(filename=os.path.join(args['results_path'], 'CornerSearch.log'), format='%(asctime)s | %(levelname)s | %(message)s', datefmt= '%m-%d-%Y %H:%M:%S', level=logging.INFO)

        ini_stdout = sys.stdout
        sys.stdout = utils.StdoutToLogging()
        
        print("Loading CornerSearch")
        

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

        print("Running CornerSearch on {}".format(args['device']))
        
        adversarial_images, adversarial_labels, all_queries, l2s, total_success = execute_attack(model, images, args, dataset.MEAN, dataset.STD, dataset.NUM_CLASSES)
        
        
        cornersearch_dict = {
            'attack_name': 'CornerSearch',
            'perturbed_image': adversarial_images,
            'perturbed_label': adversarial_labels,
            'total_queries': all_queries,
            'l2': l2s,
            'success': total_success
        }

        print("Saving Results")

        utils.save_statistics_of_attack(cornersearch_dict, labels, classified_labels, args['results_path'])

        utils.save_images_attack(cornersearch_dict, labels, os.path.join(args['results_path'], 'perturbed_images'))

    finally:
        sys.stdout = ini_stdout
        