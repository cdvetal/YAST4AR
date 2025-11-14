import sys
import os

# get repo path for imports
abs_path = os.path.abspath(os.path.realpath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.dirname(abs_path)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

import numpy as np
import time
import torch

import argparse
import logging
import importlib
import dill

import utils.func_utils as utils

class SquareAttack:
    def __init__(self, model, device, mean, std, eps=0.05, n_iters=10000, p_init=0.05,
                 metrics_path='default_square_output.npy', targeted=False, verbose=False):

        self.model = model
        self.eps = eps
        self.n_iters = n_iters
        self.p_init = p_init
        self.metrics_path = metrics_path
        self.targeted = targeted
        self.loss_type = 'margin_loss' if not targeted else 'cross_entropy'
        self.device = device
        self.mean = mean
        self.std = std
        self.verbose = verbose

    def perturb(self, x, y, constraint):
        if constraint == 'l2':
            n_queries, x_best, metrics = self.square_attack_l2(x=x, y=y)
            return n_queries, x_best, metrics
        else:
            n_queries, x_best = self.square_attack_linf(x=x, y=y)
            return n_queries, x_best, []


    #########################################################################################

    def predict(self, model, image, mean, std, device):
        with torch.no_grad():
            image_copy = image.copy()
            image_copy = torch.from_numpy(image_copy).float().to(device)

            if image_copy.dim() == 3:
                for i in range(image_copy.size(0)):
                    image_copy[i, :, :] = (image_copy[i, :, :] - mean[i]) / std[i]
            else:
                for i in range(image_copy.size(1)):
                    image_copy[:, i, :, :] = (image_copy[:, i, :, :] - mean[i]) / std[i]

            logits = model(image_copy)

        return logits.cpu().detach().numpy()

    
    def softmax(self, x):
        """Softmax function created by the Square Attack authors."""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    
    def _loss(self, y, logits, targeted, loss_type):
        """Loss function created by the Square Attack authors."""
        """ Implements the margin loss (difference between the correct and 2nd best class). """
        if loss_type == 'margin_loss':
            preds_correct_class = (logits * y).sum(1, keepdims=True)
            diff = preds_correct_class - logits  # difference between the correct class and all other classes
            diff[y] = np.inf  # to exclude zeros coming from f_correct - f_correct
            margin = diff.min(1, keepdims=True)
            loss = margin * -1 if targeted else margin
        elif loss_type == 'cross_entropy':
            probs = self.softmax(logits)
            loss = -np.log(probs[y])
            loss = loss * -1 if not targeted else loss
        else:
            raise ValueError('Wrong loss.')
        return loss.flatten()


    def p_selection(self, p_init, it, n_iters):
        """ Piece-wise constant schedule for p (the fraction of pixels changed on every iteration). """
        it = int(it / n_iters * 10000)

        if 10 < it <= 50:
            p = p_init / 2
        elif 50 < it <= 200:
            p = p_init / 4
        elif 200 < it <= 500:
            p = p_init / 8
        elif 500 < it <= 1000:
            p = p_init / 16
        elif 1000 < it <= 2000:
            p = p_init / 32
        elif 2000 < it <= 4000:
            p = p_init / 64
        elif 4000 < it <= 6000:
            p = p_init / 128
        elif 6000 < it <= 8000:
            p = p_init / 256
        elif 8000 < it <= 10000:
            p = p_init / 512
        else:
            p = p_init

        return p


    def pseudo_gaussian_pert_rectangles(self, x, y):
        delta = np.zeros([x, y])
        x_c, y_c = x // 2 + 1, y // 2 + 1

        counter2 = [x_c - 1, y_c - 1]
        for counter in range(0, max(x_c, y_c)):
            delta[max(counter2[0], 0):min(counter2[0] + (2 * counter + 1), x),
            max(0, counter2[1]):min(counter2[1] + (2 * counter + 1), y)] += 1.0 / (counter + 1) ** 2

            counter2[0] -= 1
            counter2[1] -= 1

        delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))

        return delta

    
    def meta_pseudo_gaussian_pert(self, s):
        delta = np.zeros([s, s])
        n_subsquares = 2
        if n_subsquares == 2:
            delta[:s // 2] = self.pseudo_gaussian_pert_rectangles(s // 2, s)
            delta[s // 2:] = self.pseudo_gaussian_pert_rectangles(s - s // 2, s) * (-1)
            delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))
            if np.random.rand(1) > 0.5: delta = np.transpose(delta)

        elif n_subsquares == 4:
            delta[:s // 2, :s // 2] = self.pseudo_gaussian_pert_rectangles(s // 2, s // 2) * np.random.choice([-1, 1])
            delta[s // 2:, :s // 2] = self.pseudo_gaussian_pert_rectangles(s - s // 2, s // 2) * np.random.choice([-1, 1])
            delta[:s // 2, s // 2:] = self.pseudo_gaussian_pert_rectangles(s // 2, s - s // 2) * np.random.choice([-1, 1])
            delta[s // 2:, s // 2:] = self.pseudo_gaussian_pert_rectangles(s - s // 2, s - s // 2) * np.random.choice([-1, 1])
            delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))

        return delta

    #########################################################################################
    

    def square_attack_l2(self, x, y):
        """ The L2 square attack """
        np.random.seed(0)

        min_val, max_val = 0, 1
        c, h, w = x.shape[1:]
        n_features = c * h * w
        n_ex_total = x.shape[0]

        ### initialization
        delta_init = np.zeros(x.shape)
        s = h // 5
        if self.verbose: print('Initial square side={} for bumps'.format(s))
        sp_init = (h - s * 5) // 2
        center_h = sp_init + 0
        for counter in range(h // s):
            center_w = sp_init + 0
            for counter2 in range(w // s):
                delta_init[:, :, center_h:center_h + s, center_w:center_w + s] += self.meta_pseudo_gaussian_pert(s).reshape(
                    [1, 1, s, s]) * np.random.choice([-1, 1], size=[x.shape[0], c, 1, 1])
                center_w += s
            center_h += s

        x_best = np.clip(x + delta_init / np.sqrt(np.sum(delta_init ** 2, axis=(1, 2, 3), keepdims=True)) * self.eps, 0, 1)

        logits = self.predict(model=self.model, image=x_best, mean=self.mean, std=self.std, device=self.device)
        loss_min = self._loss(y, logits, self.targeted, loss_type=self.loss_type)
        margin_min = self._loss(y, logits, self.targeted, loss_type='margin_loss')
        n_queries = np.ones(x.shape[0])  # ones because we have already used 1 query

        time_start = time.time()
        s_init = int(np.sqrt(self.p_init * n_features / c))
        metrics = np.zeros([self.n_iters, 7])
        for i_iter in range(self.n_iters):
            idx_to_fool = (margin_min > 0.0)

            acc = (margin_min > 0.0).sum() / n_ex_total
            if acc == 0:
                break

            x_curr, x_best_curr = x[idx_to_fool], x_best[idx_to_fool]
            y_curr, margin_min_curr = y[idx_to_fool], margin_min[idx_to_fool]
            loss_min_curr = loss_min[idx_to_fool]
            delta_curr = x_best_curr - x_curr

            p = self.p_selection(self.p_init, i_iter, self.n_iters)
            s = max(int(round(np.sqrt(p * n_features / c))), 3)

            if s % 2 == 0:
                s += 1

            s2 = s + 0
            ### window_1
            center_h = np.random.randint(0, h - s)
            center_w = np.random.randint(0, w - s)
            new_deltas_mask = np.zeros(x_curr.shape)
            new_deltas_mask[:, :, center_h:center_h + s, center_w:center_w + s] = 1.0

            ### window_2
            center_h_2 = np.random.randint(0, h - s2)
            center_w_2 = np.random.randint(0, w - s2)
            new_deltas_mask_2 = np.zeros(x_curr.shape)
            new_deltas_mask_2[:, :, center_h_2:center_h_2 + s2, center_w_2:center_w_2 + s2] = 1.0
            norms_window_2 = np.sqrt(
                np.sum(delta_curr[:, :, center_h_2:center_h_2 + s2, center_w_2:center_w_2 + s2] ** 2, axis=(-2, -1),
                    keepdims=True))

            ### compute total norm available
            curr_norms_window = np.sqrt(
                np.sum(((x_best_curr - x_curr) * new_deltas_mask) ** 2, axis=(2, 3), keepdims=True))
            curr_norms_image = np.sqrt(np.sum((x_best_curr - x_curr) ** 2, axis=(1, 2, 3), keepdims=True))
            mask_2 = np.maximum(new_deltas_mask, new_deltas_mask_2)
            norms_windows = np.sqrt(np.sum((delta_curr * mask_2) ** 2, axis=(2, 3), keepdims=True))

            ### create the updates
            new_deltas = np.ones([x_curr.shape[0], c, s, s])
            new_deltas = new_deltas * self.meta_pseudo_gaussian_pert(s).reshape([1, 1, s, s])
            new_deltas *= np.random.choice([-1, 1], size=[x_curr.shape[0], c, 1, 1])
            old_deltas = delta_curr[:, :, center_h:center_h + s, center_w:center_w + s] / (1e-10 + curr_norms_window)
            new_deltas += old_deltas
            new_deltas = new_deltas / np.sqrt(np.sum(new_deltas ** 2, axis=(2, 3), keepdims=True)) * (
                np.maximum(self.eps ** 2 - curr_norms_image ** 2, 0) / c + norms_windows ** 2) ** 0.5
            delta_curr[:, :, center_h_2:center_h_2 + s2, center_w_2:center_w_2 + s2] = 0.0  # set window_2 to 0
            delta_curr[:, :, center_h:center_h + s, center_w:center_w + s] = new_deltas + 0  # update window_1

            hps_str = 's={}->{}'.format(s_init, s)
            x_new = x_curr + delta_curr / np.sqrt(np.sum(delta_curr ** 2, axis=(1, 2, 3), keepdims=True)) * self.eps
            x_new = np.clip(x_new, min_val, max_val)
            curr_norms_image = np.sqrt(np.sum((x_new - x_curr) ** 2, axis=(1, 2, 3), keepdims=True))

            logits = self.predict(model=self.model, image=x_new, mean=self.mean, std=self.std, device=self.device)
            loss = self._loss(y_curr, logits, self.targeted, loss_type=self.loss_type)
            margin = self._loss(y_curr, logits, self.targeted, loss_type='margin_loss')

            idx_improved = loss < loss_min_curr
            loss_min[idx_to_fool] = idx_improved * loss + ~idx_improved * loss_min_curr
            margin_min[idx_to_fool] = idx_improved * margin + ~idx_improved * margin_min_curr

            idx_improved = np.reshape(idx_improved, [-1, *[1] * len(x.shape[:-1])])
            x_best[idx_to_fool] = idx_improved * x_new + ~idx_improved * x_best_curr
            n_queries[idx_to_fool] += 1

            acc = (margin_min > 0.0).sum() / n_ex_total
            acc_corr = (margin_min > 0.0).mean()
            mean_nq, mean_nq_ae, median_nq, median_nq_ae = np.mean(n_queries), np.mean(
                n_queries[margin_min <= 0]), np.median(n_queries), np.median(n_queries[margin_min <= 0])

            time_total = time.time() - time_start
            if self.verbose:
                print(
                    '{}: acc={:.2%} acc_corr={:.2%} avg#q_ae={:.1f} med#q_ae={:.1f} {}, n_ex={}, {:.0f}s, loss={:.3f}, max_pert={:.1f}, impr={:.0f}'.
                        format(i_iter + 1, acc, acc_corr, mean_nq_ae, median_nq_ae, hps_str, x.shape[0], time_total,
                            np.mean(margin_min), np.amax(curr_norms_image), np.sum(idx_improved)))
            metrics[i_iter] = [acc, acc_corr, mean_nq, mean_nq_ae, median_nq, margin_min.mean(), time_total]
            #if (i_iter <= 500 and i_iter % 500) or (i_iter > 100 and i_iter % 500) or i_iter + 1 == self.n_iters or acc == 0:
            #    np.save(metrics_path, metrics)
            if acc == 0:
                curr_norms_image = np.sqrt(np.sum((x_best - x) ** 2, axis=(1, 2, 3), keepdims=True))
                if self.verbose: print('Maximal norm of the perturbations: {:.5f}'.format(np.amax(curr_norms_image)))
                break

        curr_norms_image = np.sqrt(np.sum((x_best - x) ** 2, axis=(1, 2, 3), keepdims=True))
        if self.verbose: print('Maximal norm of the perturbations: {:.5f}'.format(np.amax(curr_norms_image)))

        return n_queries, x_best, metrics


    def square_attack_linf(self, x, y):
        """ The Linf square attack """
        np.random.seed(0)  # important to leave it here as well
        min_val, max_val = 0, 1 if x.max() <= 1 else 255
        c, h, w = x.shape[1:]
        n_features = c * h * w
        n_ex_total = x.shape[0]

        # [c, 1, w], i.e. vertical stripes work best for untargeted attacks
        init_delta = np.random.choice([-self.eps, self.eps], size=[x.shape[0], c, 1, w])
        x_best = np.clip(x + init_delta, min_val, max_val)

        logits = self.predict(self.model, x_best, self.mean, self.std, self.device)
        loss_min = self._loss(y, logits, self.targeted, loss_type=self.loss_type)
        margin_min = self._loss(y, logits, self.targeted, loss_type='margin_loss')
        n_queries = np.ones(x.shape[0])  # ones because we have already used 1 query

        time_start = time.time()
        metrics = np.zeros([self.n_iters, 7])
        for i_iter in range(self.n_iters - 1):
            idx_to_fool = (margin_min > 0) #| (margin_min <= 0).all()
        
            acc = (margin_min > 0.0).sum() / n_ex_total
            if acc == 0:
                break

            x_curr, x_best_curr, y_curr = x[idx_to_fool], x_best[idx_to_fool], y[idx_to_fool]
            loss_min_curr, margin_min_curr = loss_min[idx_to_fool], margin_min[idx_to_fool]
            deltas = x_best_curr - x_curr

            p = self.p_selection(self.p_init, i_iter, self.n_iters)
            for i_img in range(x_best_curr.shape[0]):
                s = int(round(np.sqrt(p * n_features / c)))
                s = min(max(s, 1), h - 1)  # at least c x 1 x 1 window is taken and at most c x h-1 x h-1
                center_h = np.random.randint(0, h - s)
                center_w = np.random.randint(0, w - s)

                x_curr_window = x_curr[i_img, :, center_h:center_h + s, center_w:center_w + s]
                x_best_curr_window = x_best_curr[i_img, :, center_h:center_h + s, center_w:center_w + s]
                # prevent trying out a delta if it doesn't change x_curr (e.g. an overlapping patch)
                while np.sum(np.abs(np.clip(x_curr_window + deltas[i_img, :, center_h:center_h + s, center_w:center_w + s],
                                            min_val, max_val) - x_best_curr_window) < 10 ** -7) == c * s * s:
                    deltas[i_img, :, center_h:center_h + s, center_w:center_w + s] = np.random.choice([-self.eps, self.eps],
                                                                                                    size=[c, 1, 1])

            x_new = np.clip(x_curr + deltas, min_val, max_val)

            logits = self.predict(self.model, x_new, self.mean, self.std, self.device)
            loss = self._loss(y_curr, logits, self.targeted, loss_type=self.loss_type)
            margin = self._loss(y_curr, logits, self.targeted, loss_type='margin_loss')

            idx_improved = loss < loss_min_curr
            loss_min[idx_to_fool] = idx_improved * loss + ~idx_improved * loss_min_curr
            margin_min[idx_to_fool] = idx_improved * margin + ~idx_improved * margin_min_curr
            idx_improved = np.reshape(idx_improved, [-1, *[1] * len(x.shape[:-1])])
            x_best[idx_to_fool] = idx_improved * x_new + ~idx_improved * x_best_curr
            n_queries[idx_to_fool] += 1

            acc = (margin_min > 0.0).sum() / n_ex_total
            acc_corr = (margin_min > 0.0).mean()
            mean_nq, mean_nq_ae, median_nq_ae = np.mean(n_queries), np.mean(n_queries[margin_min <= 0]), np.median(
                n_queries[margin_min <= 0])
            avg_margin_min = np.mean(margin_min)
            time_total = time.time() - time_start
            if self.verbose:
                print(
                    '{}: acc={:.2%} acc_corr={:.2%} avg#q_ae={:.2f} med#q={:.1f}, avg_margin={:.2f} (n_ex={}, eps={:.3f}, {:.2f}s)'.
                        format(i_iter + 1, acc, acc_corr, mean_nq_ae, median_nq_ae, avg_margin_min, x.shape[0], self.eps, time_total))

            metrics[i_iter] = [acc, acc_corr, mean_nq, mean_nq_ae, median_nq_ae, margin_min.mean(), time_total]
            # if (i_iter <= 500 and i_iter % 20 == 0) or (
            #         i_iter > 100 and i_iter % 50 == 0) or i_iter + 1 == n_iters or acc == 0:
            #     np.save(metrics_path, metrics)
            if acc == 0:
                break

        return n_queries, x_best


def execute_attack(model, images, args, mean, std, num_classes):
    total_success = 0

    all_original_images, all_labels, all_adversarial_images, all_adversarial_labels, all_queries, all_l2s = [], [], [], [], [], []


    squareattack = SquareAttack(model=model, device=args['device'], mean=mean, std=std, p_init=args['p_init'],
                                n_iters=args['n_iters'], targeted=args["targeted"], verbose = 'log' in args)

    if 'batch_size' in args and args['batch_size'] > 0:
        b_size = args['batch_size']
    else:
        print("ERROR: Batch size not define")
        exit(1)
        
    for i in range(args['total_images'] // b_size):
        images_batch = images[i * b_size:b_size * (i + 1)]
        labels_batch = labels[i * b_size:b_size * (i + 1)]

        y_target = utils.random_classes_except_current(labels_batch.cpu(),
                                                        n_cls=num_classes) if args["targeted"] else labels_batch.cpu()
        y_target = utils.dense_to_onehot(y_target, n_cls=num_classes)

        n_queries, x_best, metrics = squareattack.perturb(x=images_batch.cpu().detach().numpy(),
                                                            y=y_target,
                                                            constraint=args['constraint'])

        images_perturbed = torch.from_numpy(x_best).float()
        images_perturbed = images_perturbed.to(args['device'])

        images_perturbed_normalized = images_perturbed.clone()
        images_perturbed_normalized = utils.normalize_image(images_perturbed_normalized, mean, std)
        pred = model(images_perturbed_normalized).argmax(dim=1).cpu()

        l2_norms = []
        for ii in range(len(images_perturbed)):
            l2_norms.append(utils.calculate_l2_norm(images_perturbed.cpu().detach().numpy()[ii],
                                                images_batch.cpu().detach().numpy()[ii]))


        all_l2s.extend(l2_norms)
        all_original_images.extend(images_batch.cpu().detach().numpy())
        all_adversarial_images.extend(images_perturbed.cpu().detach().numpy())
        all_labels.extend(labels_batch.cpu())
        all_adversarial_labels.extend(pred)
        all_queries.extend(n_queries)

        success = sum((labels_batch.cpu() != pred).float() == 1.)
        total_success += success.item()

        if 'log' in args: print("Images done: {}/{}; Success: {:.2f}%".format(b_size * (i + 1), args['total_images'], success.item() / b_size * 100))

    if 'log' in args: print('Square attack done: Success: {:.2f}%'.format(total_success / len(all_l2s) * 100))

    return all_adversarial_images, all_adversarial_labels, all_queries, all_l2s, total_success


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eps', type=int, help='')
    parser.add_argument('--n-iters', type=int, help='')
    parser.add_argument('--p-init', type=float, help='')
    parser.add_argument('--targeted', action='store_true', help='')
    parser.add_argument('--constraint', type=str, help='')
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
    args = utils.prepare_attack_arguments(ini_args, ini_args['attack_config'], 'squareattack')


    try:
        # setup logger
        if not os.path.isdir(args['results_path']):
            os.makedirs(args['results_path'])

        logging.basicConfig(filename=os.path.join(args['results_path'], 'SquareAttack.log'), format='%(asctime)s | %(levelname)s | %(message)s', datefmt= '%m-%d-%Y %H:%M:%S', level=logging.INFO)

        ini_stdout = sys.stdout
        sys.stdout = utils.StdoutToLogging()
        
        print("Loading SquareAttack")
        

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

        print("Running SquareAttack on {}".format(args['device']))
        
        adversarial_images, adversarial_labels, all_queries, l2s, total_success = execute_attack(model, images, args, dataset.MEAN, dataset.STD, dataset.NUM_CLASSES)
        
        
        squareattack_dict = {
            'attack_name': 'SquareAttack',
            'perturbed_image': adversarial_images,
            'perturbed_label': adversarial_labels,
            'total_queries': all_queries,
            'l2': l2s,
            'success': total_success
        }

        print("Saving Results")

        utils.save_statistics_of_attack(squareattack_dict, labels, classified_labels, args['results_path'])

        utils.save_images_attack(squareattack_dict, labels, os.path.join(args['results_path'], 'perturbed_images'))

    finally:
        sys.stdout = ini_stdout
