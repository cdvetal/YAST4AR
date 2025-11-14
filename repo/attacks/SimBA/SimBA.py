import sys
import os

# get repo path for imports
abs_path = os.path.abspath(os.path.realpath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.dirname(abs_path)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

import torch
import torch.nn.functional as F

from attacks.Attack import Attack
import utils.func_utils as utils
import attacks.SimBA.utilsSimBA as utilsSimBA

import logging
import argparse
import dill
import importlib

class SimBA(Attack):

    def __init__(self, device, mean, std):
        """Attack a model using the GeoDA Attack.
            Arguments:
                device: Device to use ('cuda' or 'cpu')
                model: Model to attack
                mean: Mean of the dataset to use
                std: Standard deviation of the dataset to use
                image_size: Size of the images of the dataset
        """

        super().__init__()

        self.image_size = None
        self.device = device
        self.model = None
        self.mean = mean
        self.std = std

    def expand_vector(self, x, size):
        batch_size = x.size(0)
        x = x.view(-1, 3, size, size)
        z = torch.zeros(batch_size, 3, self.image_size, self.image_size, device=self.device)
        z[:, :, :size, :size] = x
        return z

    def normalize(self, x):
        return utils.normalize_image(x, self.mean, self.std)

    def get_probs(self, x, y):
        output = self.model(self.normalize(x)).to(self.device)
        probs = torch.index_select(F.softmax(output, dim=-1).data, 1, y)
        return torch.diag(probs).to(self.device)

    def get_preds(self, x):
        output = self.model(self.normalize(x)).to(self.device)
        _, preds = output.data.max(1)
        return preds

    # 20-line implementation of SimBA for single image input
    def simba_single(self, x, y, num_iters=10000, epsilon=0.2, targeted=False):
        n_dims = x.view(1, -1).size(1)
        perm = torch.randperm(n_dims)
        x = x.unsqueeze(0)
        last_prob = self.get_probs(x, y)
        for i in range(num_iters):
            diff = torch.zeros(n_dims)
            diff[perm[i]] = epsilon
            left_prob = self.get_probs((x - diff.view(x.size())).clamp(0, 1), y)
            if targeted != (left_prob < last_prob):
                x = (x - diff.view(x.size())).clamp(0, 1)
                last_prob = left_prob
            else:
                right_prob = self.get_probs((x + diff.view(x.size())).clamp(0, 1), y)
                if targeted != (right_prob < last_prob):
                    x = (x + diff.view(x.size())).clamp(0, 1)
                    last_prob = right_prob
            if i % 10 == 0:
                print(last_prob)
        return x.squeeze()

    # runs simba on a batch of images <images_batch> with true labels (for untargeted attack) or target labels
    # (for targeted attack) <labels_batch>
    def perturb(self, model, image_size, images_batch, labels_batch, freq_dims, stride, epsilon, linf_bound=0.0,
                order: str = 'rand', targeted: bool = False, pixel_attack: bool = False, log_every: int = 0,
                num_iters: int = 0):
        """Attack a model using the Simba Batch Attack.

            Arguments:
                images_batch: Batch of images to attack (Not normalized)
                labels_batch: Ground truth labels of the images to attack
                freq_dims:
                stride:
                epsilon:
                linf_bound:
                order: (default = 'rand')
                targeted: Define if the attack is targeted or untargeted (default = False)
                pixel_attack: Define if the attack is pixel or DCT (default = True)
                log_every: Logging iterations (default = 0)
                num_iters: Number of iterations to run the attack (default = 0)
            Returns:
                expanded:
                probs:
                succs:
                queries:
                l2_norms:
                linf_norms:
        """

        self.model = model
        self.model.eval()
        self.image_size = image_size
        
        batch_size = images_batch.size(0)
        image_size = images_batch.size(2)
        assert self.image_size == image_size
        if order == 'rand':
            n_dims = 3 * freq_dims * freq_dims
        else:
            n_dims = 3 * image_size * image_size
        if num_iters > 0:
            max_iters = int(min(n_dims, num_iters))
        else:
            max_iters = int(n_dims)
        # sample a random ordering for coordinates independently per batch element
        if order == 'rand':
            indices = torch.randperm(3 * freq_dims * freq_dims)[:max_iters]
        elif order == 'diag':
            indices = utilsSimBA.diagonal_order(image_size, 3)[:max_iters]
        elif order == 'strided':
            indices = utilsSimBA.block_order(image_size, 3, initial_size=freq_dims, stride=stride)[:max_iters]
        else:
            indices = utilsSimBA.block_order(image_size, 3)[:max_iters]
        if order == 'rand':
            expand_dims = freq_dims
        else:
            expand_dims = image_size
        n_dims = 3 * expand_dims * expand_dims
        x = torch.zeros(batch_size, n_dims, device=self.device)
        # logging tensors
        probs = torch.zeros(batch_size, max_iters, device=self.device)
        succs = torch.zeros(batch_size, max_iters, device=self.device)
        queries = torch.zeros(batch_size, max_iters, device=self.device)
        l2_norms = torch.zeros(batch_size, max_iters, device=self.device)
        linf_norms = torch.zeros(batch_size, max_iters, device=self.device)
        prev_probs = self.get_probs(images_batch, labels_batch)
        preds = self.get_preds(images_batch)
        if pixel_attack:
            trans = lambda z: z
        else:
            trans = lambda z: utilsSimBA.block_idct(z, block_size=image_size, linf_bound=linf_bound)
        remaining_indices = torch.arange(0, batch_size).long()
        for k in range(max_iters):
            dim = indices[k]
            expanded = (images_batch[remaining_indices] + trans(
                self.expand_vector(x[remaining_indices], expand_dims))).clamp(0, 1)
            perturbation = trans(self.expand_vector(x, expand_dims))
            l2_norms[:, k] = perturbation.view(batch_size, -1).norm(2, 1)
            linf_norms[:, k] = perturbation.view(batch_size, -1).abs().max(1)[0]
            preds_next = self.get_preds(expanded)
            preds[remaining_indices] = preds_next
            if targeted:
                remaining = preds.ne(labels_batch).to(self.device)
            else:
                remaining = preds.eq(labels_batch).to(self.device)
            # check if all images are misclassified and stop early
            if remaining.sum() == 0:
                adv = (images_batch + trans(self.expand_vector(x, expand_dims))).clamp(0, 1)
                probs_k = self.get_probs(adv, labels_batch)
                probs[:, k:] = probs_k.unsqueeze(1).repeat(1, max_iters - k)
                succs[:, k:] = torch.ones(batch_size, max_iters - k)
                queries[:, k:] = torch.zeros(batch_size, max_iters - k)
                break
            remaining_indices = torch.arange(0, batch_size, device=remaining.device)[remaining].long()
            if k > 0:
                succs[:, k - 1] = ~remaining
            diff = torch.zeros(remaining.sum(), n_dims, device=self.device)
            diff[:, dim] = epsilon
            
            left_vec = x[remaining_indices] - diff # .to('cpu')
            right_vec = x[remaining_indices] + diff
            # trying negative direction
            adv = (images_batch[remaining_indices] + trans(self.expand_vector(left_vec, expand_dims))).clamp(0, 1)
            left_probs = self.get_probs(adv, labels_batch[remaining_indices])
            queries_k = torch.zeros(batch_size, device=self.device)
            # increase query count for all images
            queries_k[remaining_indices] += 1
            if targeted:
                improved = left_probs.gt(prev_probs[remaining_indices])
            else:
                improved = left_probs.lt(prev_probs[remaining_indices])
            # only increase query count further by 1 for images that did not improve in adversarial loss
            if improved.sum() < remaining_indices.size(0):
                queries_k[remaining_indices[~improved]] += 1
            # try positive directions
            adv = (images_batch[remaining_indices] + trans(self.expand_vector(right_vec, expand_dims))).clamp(0, 1)
            right_probs = self.get_probs(adv, labels_batch[remaining_indices])
            if targeted:
                right_improved = right_probs.gt(torch.max(prev_probs[remaining_indices], left_probs))
            else:
                right_improved = right_probs.lt(torch.min(prev_probs[remaining_indices], left_probs))
            probs_k = prev_probs.clone()
            # update x depending on which direction improved
            if improved.sum() > 0:
                left_indices = remaining_indices[improved]
                left_mask_remaining = improved.unsqueeze(1).repeat(1, n_dims)
                x[left_indices] = left_vec[left_mask_remaining].view(-1, n_dims)
                probs_k[left_indices] = left_probs[improved]
            if right_improved.sum() > 0:
                right_indices = remaining_indices[right_improved]
                right_mask_remaining = right_improved.unsqueeze(1).repeat(1, n_dims)
                x[right_indices] = right_vec[right_mask_remaining].view(-1, n_dims)
                probs_k[right_indices] = right_probs[right_improved]
            probs[:, k] = probs_k
            queries[:, k] = queries_k
            prev_probs = probs[:, k]
            if log_every > 0 and ((k + 1) % log_every == 0 or k == max_iters - 1):
                print('Iteration %d: queries = %.4f, prob = %.4f, remaining = %.4f' % (
                    k + 1, queries.sum(1).mean(), probs[:, k].mean(), remaining.float().mean()))
        expanded = (images_batch + trans(self.expand_vector(x, expand_dims))).clamp(0, 1)
        preds = self.get_preds(expanded)
        if targeted:
            remaining = preds.ne(labels_batch)
        else:
            remaining = preds.eq(labels_batch)
        succs[:, max_iters - 1] = ~remaining
        return expanded, probs, succs, queries, l2_norms, linf_norms, preds


def execute_attack(model, images, args, mean, std, image_size):
    total_success = 0

    all_original_images, all_labels, all_adversarial_images, all_adversarial_labels, all_queries, all_l2s = [], [], [], [], [], []

    attacker = SimBA(args['device'], mean, std)

    if 'batch_size' in args and args['batch_size'] > 0:
        b_size = args['batch_size']
    else:
        print("ERROR: Batch size not define")
        exit(1)

    for i in range(args['total_images'] // b_size):
        images_batch = images[i * b_size:b_size * (i + 1)]
        labels_batch = labels[i * b_size:b_size * (i + 1)]

        adv, probs, succs, queries, l2_norms, linf_norms, preds = \
            attacker.perturb(model=model, image_size=image_size,
                                images_batch=images_batch.to(args['device']), labels_batch=labels_batch.to(args['device']),
                                freq_dims=args["freq_dims"],
                                stride=args["stride"], epsilon=args["epsilon"],
                                linf_bound=args["linf_bound"], order=args["order"], targeted= args['targeted'],
                                pixel_attack= args['pixel_attack'], log_every=args["log_every"],
                                num_iters=args["num_iters"])

        success = (preds.cpu() != labels_batch.cpu()).float().mean().item() * 100

        l2 = []
        for ii in range(len(adv)):
            l2.append(utils.calculate_l2_norm(adv.cpu().detach().numpy()[ii],
                                        images_batch.cpu().detach().numpy()[ii]))

    
        all_l2s.extend(l2)
        all_original_images.extend(images_batch.cpu().detach().numpy())
        all_adversarial_images.extend(adv.cpu().detach().numpy())
        all_labels.extend(labels_batch.cpu())
        all_adversarial_labels.extend(preds.cpu())
        all_queries.extend(queries.sum(1))


        success = sum((labels_batch.cpu() != preds.cpu()).float() == 1.)
        total_success += success.item()

        if 'log' in args: print("Images done: {}/{}; Success: {:.2f}%".format(b_size * (i + 1), args['total_images'], success.item() / b_size * 100))

    all_queries = [int(t.item()) for t in all_queries]

    if 'log' in args: print('SimBA done: Success: {:.2f}%'.format(total_success / len(all_l2s) * 100))

    return all_adversarial_images, all_adversarial_labels, all_queries, all_l2s, total_success



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freq-dims', type=int, help='')
    parser.add_argument('--stride', type=int, help='')
    parser.add_argument('--epsilon', type=float, help='')
    parser.add_argument('--linf-bound', type=float, help='')
    parser.add_argument('--order', type=str, help='')
    parser.add_argument('--targeted', action='store_true')
    parser.add_argument('--log-every', type=int, help='')
    parser.add_argument('--num-iters', type=int, help='')
    parser.add_argument('--pixel-attack', action='store_true')
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
    args = utils.prepare_attack_arguments(ini_args, ini_args['attack_config'], 'simba')

    try:
        # setup logger
        if not os.path.isdir(args['results_path']):
            os.makedirs(args['results_path'])

        logging.basicConfig(filename=os.path.join(args['results_path'], 'SimBA.log'), format='%(asctime)s | %(levelname)s | %(message)s', datefmt= '%m-%d-%Y %H:%M:%S', level=logging.INFO)

        ini_stdout = sys.stdout
        sys.stdout = utils.StdoutToLogging()
        
        print("Loading SimBA")
        

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

        print("Running SimBA on {}".format(args['device']))
        
        adversarial_images, adversarial_labels, all_queries, l2s, total_success = execute_attack(model, images, args, dataset.MEAN, dataset.STD, dataset.IMAGE_SIZE)
        
        
        simba_dict = {
            'attack_name': 'SimBA',
            'perturbed_image': adversarial_images,
            'perturbed_label': adversarial_labels,
            'total_queries': all_queries,
            'l2': l2s,
            'success': total_success
        }

        print("Saving Results")

        utils.save_statistics_of_attack(simba_dict, labels, classified_labels, args['results_path'])

        utils.save_images_attack(simba_dict, labels, os.path.join(args['results_path'], 'perturbed_images'))

    finally:
        sys.stdout = ini_stdout

