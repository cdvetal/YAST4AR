import sys
import os

# get repo path for imports
abs_path = os.path.abspath(os.path.realpath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.dirname(abs_path)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

import numpy as np
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from torch.autograd import Variable
import copy

from attacks.Attack import Attack
from attacks.DeepFool.DeepFool import DeepFool
import utils.func_utils as utils
import utils.static_vars as static

import argparse
import dill
import importlib
import logging


def predict(image_inp, model, device):
    # image_inp = np.reshape(image_inp, (channels, image_size, image_size))
    # print(image_inp)
    model.to(device)
    new_im = torch.from_numpy(image_inp).to(device)
    if new_im.dim() == 3:
        new_im = Variable(new_im[None, :, :, :])
    output = model(new_im)
    model.cpu()
    return output.cpu()


class UAP(Attack):

    def __init__(self, device, delta=0.2, max_iter_uni=np.inf,
                 xi=10, p=np.inf, overshoot=0.02, max_iter_df=10):
        """ Attack a model using the UAP Attack.

        Parameters
        ----------

        dataset:
            Images of size MxHxWxC (M: number of images)
        f:
            feedforward function (input: images, output: values of activation BEFORE softmax)
        delta:
            controls the desired fooling rate (default = 80% fooling rate)
        max_iter_uni:
            optional other termination criterion (maximum number of iteration, default = np.inf)
        xi:
            controls the l_p magnitude of the perturbation (default = 10)
        p:
            norm to be used (FOR NOW, ONLY p = 2, and p = np.inf ARE ACCEPTED!) (default = np.inf)
        num_classes:
            num_classes (limits the number of classes to test against, by default = 10)
        overshoot:
            used as a termination criterion to prevent vanishing updates (default = 0.02)
        max_iter_df:
            maximum number of iterations for deepfool (default = 10)

        Returns
        -------
            v:
                The universal perturbation created
            fooling_rate:
                Fooling rate obtained
            n_fooled_labels:
                Number of images that can be misclassified by being perturbed with the uap perturbation

        """
        super().__init__()

        self.f = None
        self.device = device
        self.delta = delta
        self.max_iter_uni = max_iter_uni
        self.xi = xi
        self.p = p
        self.overshoot = overshoot
        self.max_iter_df = max_iter_df

    def perturb(self, model, dataset, image_size, channels, num_classes=10, log=None):
        self.f = lambda image, model, device: predict(image, model, self.device)
        #v, fooling_rate, n_fooled_labels, total_iterations \
        #    = universal_perturbation(model, self.device, dataset, self.f, image_size, channels,
        #                             self.delta, self.max_iter_uni,
        #                             self.xi, self.p, num_classes, self.overshoot, self.max_iter_df)

        #return v, fooling_rate, n_fooled_labels, total_iterations

        #def universal_perturbation(model, device, dataset, f, image_size, channels, delta=0.2, max_iter_uni=np.inf, xi=10, p=np.inf, num_classes=10,
        #                       overshoot=0.02, max_iter_df=10):
        """
        :param dataset: Images of size MxHxWxC (M: number of images)

        :param f: feedforward function (input: images, output: values of activation BEFORE softmax).

        :param delta: controls the desired fooling rate (default = 80% fooling rate)

        :param max_iter_uni: optional other termination criterion (maximum number of iteration, default = np.inf)

        :param xi: controls the l_p magnitude of the perturbation (default = 10)

        :param p: norm to be used (FOR NOW, ONLY p = 2, and p = np.inf ARE ACCEPTED!) (default = np.inf)

        :param num_classes: num_classes (limits the number of classes to test against, by default = 10)

        :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).

        :param max_iter_df: maximum number of iterations for deepfool (default = 10)

        :param log: log attack behaviour (default = None) [None, False, True] -> [No logs, Only log with number of fooled images, All logs]

        :return: the universal perturbation.
        """

        v = 0
        fooling_rate = 0.0
        n_fooled_labels = 0.0
        num_images = np.shape(dataset)[0]  # The images should be stacked ALONG FIRST DIMENSION
        #num_images = 100

        dataset = np.reshape(dataset, (-1, channels, image_size, image_size))

        itr = 0
        total_iterations = 0
        while fooling_rate < 1 - self.delta and itr < self.max_iter_uni:
            # Shuffle the dataset
            np.random.shuffle(dataset)

            if log is True: print('Starting pass number ', itr)

            # Go through the data set and compute the perturbation increments sequentially
            for k in range(0, num_images):

                cur_img = dataset[k:(k + 1), :, :, :]
                cur_img_clone = copy.deepcopy(cur_img[0])
                new_image = copy.deepcopy(cur_img) + v

                new_image = new_image[0]

                if int(np.argmax(self.f(cur_img_clone, model, self.device).detach().cpu().numpy())) == int(np.argmax(self.f(new_image, model, self.device).detach().cpu().numpy())):
                    if log is True: print('>> k = {}/{}, pass #{}'.format(k, num_images, itr))
                    # One query belonging to the UAP attack
                    total_iterations += 1

                    # Compute adversarial perturbation
                    dr, iter, _, _, _ = DeepFool().deepfool(torch.from_numpy(new_image).to(self.device), net=model, device = self.device, overshoot=self.overshoot, num_classes=num_classes)
                    # The number of queries performed by the deepfool attack
                    total_iterations += iter

                    # Make sure it converged...
                    if iter < self.max_iter_df - 1:
                        v = v + dr

                        # Project on l_p ball
                        v = self.proj_lp(v, self.xi,  self.p)

            itr = itr + 1

            # Perturb the dataset with computed perturbation
            dataset_perturbed = dataset + v

            est_labels_orig = np.zeros(num_images)
            est_labels_pert = np.zeros(num_images)

            batch_size = 10
            num_batches = np.int64(np.ceil(np.float64(num_images) / np.float64(batch_size)))

            # Compute the estimated labels in batches
            for ii in range(0, num_batches):
                m = (ii * batch_size)
                M = min((ii + 1) * batch_size, num_images)
                est_labels_orig[m:M] = np.argmax(self.f(dataset[m:M, :, :, :], model, self.device).detach().cpu().numpy(), axis=1).flatten()
                est_labels_pert[m:M] = np.argmax(self.f(dataset_perturbed[m:M, :, :, :], model, self.device).detach().cpu().numpy(), axis=1).flatten()

            for i in range(len(est_labels_pert)):
                if est_labels_pert[i] != est_labels_orig[i]:
                    n_fooled_labels += 1

            # Compute the fooling rate
            # n_fooled_labels = float(np.sum(est_labels_pert != est_labels_orig))
            fooling_rate = float(n_fooled_labels / float(num_images))
            if log is True or log is False:
                print('FOOLING RATE = {}'.format(fooling_rate))
                print("Number of fooled images {}".format(n_fooled_labels))

            break

        return v, fooling_rate, n_fooled_labels, total_iterations

    def proj_lp(self, v, xi, p):
        # Project on the lp ball centered at 0 and of radius xi

        # SUPPORTS only p = 2 and p = Inf for now
        if p == 2:
            v = v * min(1, xi / np.linalg.norm(v.flatten(1)))
            # v = v / np.linalg.norm(v.flatten(1)) * xi
        elif p == np.inf:
            v = np.sign(v) * np.minimum(abs(v), xi)
        else:
            raise ValueError('Values of p different from 2 and Inf are currently not supported...')

        return v
    
def execute_attack(model, images, args, model_name, dataset_name, mean, std, image_size, num_classes, trainloader, channels = 3):
    total_success = 0

    all_original_images, all_labels, all_adversarial_images, all_adversarial_labels, all_queries, all_l2s = [], [], [], [], [], []

    path_file_uap = os.path.join(static.PATH_TEMP, model_name + '_' + dataset_name + '_UAP_perturbation.npz')
    if os.path.isfile(path_file_uap) == 0:
        img_list = []
        for i, (image, label) in enumerate(trainloader):
            for j in image:
                img_list.append(j)
        try:
            dataset_numpy = np.stack(img_list, axis=0)
        except:
            raise ValueError('when img_size and crop_size are None, images'
                                ' in image_paths must have the same shapes.')
        norm = np.inf
        if args['norm'] == 2:
            norm = 2
        max_iter_uni = np.inf
        if args['max_iter_uni'] != -1:
            max_iter_uni = args['max_iter_uni']

        log = True if 'log' in args else None

        uapattack = UAP(delta=args["delta"], device=args['device'],
                        overshoot=args["overshoot"], max_iter_df=args["max_iter_df"],
                        max_iter_uni=max_iter_uni, p=norm, xi=args['xi'])
        v, fooling_rate, n_fooled_labels, total_iterations = uapattack.perturb(model=model, dataset=dataset_numpy, image_size=image_size,
                                                                                channels=channels, num_classes=num_classes, log=log)
                                                                                

        np.savez(path_file_uap, v=v, fooling_rate=fooling_rate,
                    n_fooled_labels=n_fooled_labels, total_iterations=total_iterations)

        del uapattack

    else:
        print('>> Found a pre-computed universal perturbation! Retrieving it from:\n\t{}'.format(str(path_file_uap)))
        np_file = np.load(path_file_uap)
        v = np_file['v']
        v = v[0]
        fooling_rate = np_file['fooling_rate']
        n_fooled_labels = np_file['n_fooled_labels']
        total_iterations = np_file['total_iterations']

    ###########################################################

    if 'batch_size' in args and args['batch_size'] > 0:
        b_size = args['batch_size']
    else:
        print("ERROR: Batch size not define")
        exit(1)

    for i in range(args['total_images'] // b_size):
        images_batch = images[i * b_size:b_size * (i + 1)]
        labels_batch = labels[i * b_size:b_size * (i + 1)]
        images_normalized = utils.normalize_image(images_batch, mean, std)

        # Add the perturbation created by the UAP attack
        # v_clipped = np.clip(invert_normalization(images_perturbed + v, dataset_name), 0, 255) - np.clip(undo_image_avg(image_original[0, :, :, :]), 0, 255)
        images_perturbed = (images_normalized.cpu().detach().numpy()) + v
        images_perturbed = torch.from_numpy(images_perturbed)
        # images = invert_normalization(images, dataset_name)
        images_perturbed = images_perturbed.to(args['device'])
        model.to(args['device'])
        pred = model(images_perturbed).argmax(dim=1).cpu()
        success = (pred != labels_batch.cpu()).float().mean().item() * 100

        pert_image = utils.remove_normalization(images_perturbed, mean, std)

        l2_norms = []
        for ii in range(len(pert_image)):
            l2_norms.append(utils.calculate_l2_norm(pert_image.cpu().detach().numpy()[ii],
                                                images_batch.cpu().detach().numpy()[ii]))

        all_l2s.extend(l2_norms)
        all_original_images.extend(images_batch.cpu().detach().numpy())
        all_adversarial_images.extend(pert_image.cpu().detach().numpy())
        all_labels.extend(labels_batch.cpu())
        all_adversarial_labels.extend(pred)
        all_queries.extend([total_iterations] * len(images_batch.cpu()))

        success = sum((labels_batch.cpu() != pred).float() == 1.)
        total_success += success.item()

        if 'log' in args: print("Images done: {}/{}; Success: {:.2f}%".format(b_size * (i + 1), args['total_images'], success.item() / b_size * 100))

    if 'log' in args: print('UAP done: Success: {:.2f}%'.format(total_success / len(all_l2s) * 100))

    if os.path.exists(path_file_uap):
        os.remove(path_file_uap)

    return all_adversarial_images, all_adversarial_labels, all_queries , all_l2s, total_success

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--overshoot', type=float, help='')
    parser.add_argument('--xi', type=int, help='')
    parser.add_argument('--delta', type=float, help='')
    parser.add_argument('--max-iter-df', type=int, help='')
    parser.add_argument('--max-iter-uni', type=int, help='')
    parser.add_argument('--norm', type=str, help='')
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
    args = utils.prepare_attack_arguments(ini_args, ini_args['attack_config'], 'uap')


    try:
        # setup logger
        if not os.path.isdir(args['results_path']):
            os.makedirs(args['results_path'])

        logging.basicConfig(filename=os.path.join(args['results_path'], 'UAP.log'), format='%(asctime)s | %(levelname)s | %(message)s', datefmt= '%m-%d-%Y %H:%M:%S', level=logging.INFO)

        ini_stdout = sys.stdout
        sys.stdout = utils.StdoutToLogging()
        
        print("Loading UAP")
        

        # load model
        with open(args['model'], "rb") as file:
            serialized_data = file.read()
            data = dill.loads(serialized_data)
            model_name = data['model_name']
            model = data['model']
            classified_labels = data['classified_labels']

        print("Model loaded")
        print("Model name: {}".format(model_name))
        # load dataset
        spec = importlib.util.spec_from_file_location("dataset", args['dataset'])
        dataset = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dataset)
        testloader, trainloader = dataset.dataLoader()

        images, labels = utils.get_images_labels_from_dataLoader(testloader, args['device'], args['total_images'])
        
        print("Dataset loaded")

        print("Running UAP on {}".format(args['device']))
        
        adversarial_images, adversarial_labels, all_queries, l2s, total_success = execute_attack(model, images, args, model_name, \
                                                                        dataset.DATASET_NAME, dataset.MEAN, dataset.STD, dataset.IMAGE_SIZE, dataset.NUM_CLASSES, trainloader, len(dataset.MEAN))

        uap_dict = {
            'attack_name': 'UAP',
            'perturbed_image': adversarial_images,
            'perturbed_label': adversarial_labels,
            'total_queries': all_queries,
            'l2': l2s,
            'success': total_success
        }

        print("Saving Results")

        utils.save_statistics_of_attack(uap_dict, labels, classified_labels, args['results_path'])

        utils.save_images_attack(uap_dict, labels, os.path.join(args['results_path'], 'perturbed_images'))

    finally:
        sys.stdout = ini_stdout
