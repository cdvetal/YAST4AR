import sys
import os

# get repo path for imports
abs_path = os.path.abspath(os.path.realpath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.dirname(abs_path)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

import collections
import dill
import importlib

import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
import argparse
import logging
 
import utils.func_utils as utils
from attacks.Attack import Attack


class DeepFool(Attack):

    def __init__(self, overshoot=0.02, max_iter=50):
        """DeepFool class

            Arguments:
                overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02)
                max_iter: maximum number of iterations for deepfool (default = 50)

            """

        super().__init__()

        self.max_iter = max_iter
        self.overshoot = overshoot


    def perturb(self, model, image, num_classes, device):
        """Attack a model using the DeepFool attack.

            Arguments:
                image: Image of size HxWx3
                model: network (input: images, output: values of activation **BEFORE** softmax)
                num_classes: num_classes (limits the number of classes to test against, by default = 10)

            Returns:
                minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
            """
        r, loop_i, label_orig, label_pert, pert_image = \
            self.deepfool(image=image, net=model, device=device, num_classes=num_classes, overshoot=self.overshoot,
                     max_iter=self.max_iter)

        return r, loop_i, label_orig, label_pert, pert_image


    def deepfool(self, image, net, device, num_classes=10, overshoot=0.02, max_iter=50):
        """
            :param image: Image of size HxWx3 (Normalized)
            :param net: network (input: images, output: values of activation **BEFORE** softmax).
            :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
            :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
            :param max_iter: maximum number of iterations for deepfool (default = 50)
            :return: minimal perturbation that fools the classifier
            :return: number of iterations that it required
            :return: new estimated_label
            :return: perturbed image
        """

        if device == "cuda":
            # print("Using GPU")
            image = image.cuda()
            net = net.cuda()
        else:
            image = image.cpu()
            net = net.cpu()
            # print("Using CPU")

        f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
        I = (np.array(f_image)).flatten().argsort()[::-1]

        I = I[0:num_classes]
        label = I[0]

        input_shape = image.cpu().numpy().shape
        pert_image = copy.deepcopy(image)
        w = np.zeros(input_shape)
        r_tot = np.zeros(input_shape)

        loop_i = 0

        x = Variable(pert_image[None, :], requires_grad=True)
        fs = net.forward(x)
        fs_list = [fs[0, I[k]] for k in range(num_classes)]
        k_i = label

        while k_i == label and loop_i < max_iter:

            pert = np.inf
            fs[0, I[0]].backward(retain_graph=True)
            grad_orig = x.grad.data.cpu().numpy().copy()

            for k in range(1, num_classes):
                self.zero_gradients(x)

                fs[0, I[k]].backward(retain_graph=True)
                cur_grad = x.grad.data.cpu().numpy().copy()

                # set new w_k and new f_k
                w_k = cur_grad - grad_orig
                f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

                pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())

                # determine which w_k to use
                if pert_k < pert:
                    pert = pert_k
                    w = w_k

            # compute r_i and r_tot
            # Added 1e-4 for numerical stability
            r_i = (pert + 1e-4) * w / np.linalg.norm(w)
            r_tot = np.float32(r_tot + r_i)

            if device == "cuda":
                pert_image = image + (1 + overshoot) * torch.from_numpy(r_tot).cuda()
            else:
                pert_image = image + (1 + overshoot) * torch.from_numpy(r_tot)

            # pert_image = image + (1 + overshoot) * torch.from_numpy(r_tot)

            x = Variable(pert_image, requires_grad=True)
            fs = net.forward(x)
            k_i = np.argmax(fs.data.cpu().numpy().flatten())

            loop_i += 1

        r_tot = (1 + overshoot) * r_tot

        return r_tot, loop_i, label, k_i, pert_image[0]


    def zero_gradients(self, x):
        if isinstance(x, torch.Tensor):
            if x.grad is not None:
                x.grad.detach_()
                x.grad.zero_()
        elif isinstance(x, collections.abc.Iterable):
            for elem in x:
                self.zero_gradients(elem)


def execute_attack(model, images, args, mean, std, num_classes):
    total_success = 0

    all_original_images, all_labels, all_adversarial_images, all_adversarial_labels, all_queries, all_l2s = [], [], [], [], [], []

    for i in range(args['total_images']):
        normalized_image = utils.normalize_image(images[i], mean, std)
        deepfoolAttack = DeepFool(overshoot=args["overshoot"], max_iter=args["max_iter"])
        r, loop_i, label_orig, label_pert, pert_image_normalized \
            = deepfoolAttack.perturb(model=model.to(args['device']), image=normalized_image.to(args['device']),
                                        num_classes=num_classes, device = args['device'])

        
        success = 100 if label_orig != label_pert else 0


        pert_image = utils.remove_normalization(pert_image_normalized.cpu(), mean, std)

        l2 = utils.calculate_l2_norm(pert_image.cpu().detach().numpy(), images[i].cpu().detach().numpy())


        all_l2s.append(l2)
        all_original_images.append(images[i].cpu().detach().numpy())
        all_adversarial_images.append(pert_image.cpu().detach().numpy())
        all_labels.append(label_orig)
        all_adversarial_labels.append(label_pert)
        all_queries.append(loop_i)

        if label_orig != label_pert:
            total_success += 1

        if 'log' in args: print("Images done: {}/{}; Success: {:.2f}%".format(i + 1, args['total_images'], success))

    if 'log' in args: print('DeepFool done: Success: {:.2f}%'.format(total_success / len(all_l2s) * 100))

    return all_adversarial_images, all_adversarial_labels, all_queries, all_l2s, total_success


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--overshoot', type=float, help='')
    parser.add_argument('--max-iter', type=int, help='')

    parser.add_argument('--model', type=str, default = 'Path for binary file containing the model')
    parser.add_argument('--dataset', type=str, default = 'Path for dataset loader file')
    parser.add_argument('--total-images', type=int, help='')
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--attack-config', type=str, default = '', help='Config file to be passed in instead of arguments')
    parser.add_argument('--results-path', type=str, default = '', help="Path to store results")

    ini_args = vars(parser.parse_args())

    ini_args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'


    # initial argument verifications
    args = utils.prepare_attack_arguments(ini_args, ini_args['attack_config'], 'deepfool')


    try:
        # setup logger
        if not os.path.isdir(args['results_path']):
            os.makedirs(args['results_path'])

        logging.basicConfig(filename=os.path.join(args['results_path'], 'DeepFool.log'), format='%(asctime)s | %(levelname)s | %(message)s', datefmt= '%m-%d-%Y %H:%M:%S', level=logging.INFO)

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

        print("Running DeepFool on {}".format(args['device']))
        
        adversarial_images, adversarial_labels, all_queries, l2s, total_success = execute_attack(model, images, args, dataset.MEAN, dataset.STD, dataset.NUM_CLASSES)
        
        
        deepfool_dict = {
            'attack_name': 'DeepFool',
            'perturbed_image': adversarial_images,
            'perturbed_label': adversarial_labels,
            'total_queries': all_queries,
            'l2': l2s,
            'success': total_success
        }

        print("Saving Results")

        utils.save_statistics_of_attack(deepfool_dict, labels, classified_labels, args['results_path'])

        utils.save_images_attack(deepfool_dict, labels, os.path.join(args['results_path'], 'perturbed_images'))

    finally:
        sys.stdout = ini_stdout
