import torch
import numpy as np
import torchvision
import algorithm.attack as attack
from utilities.loss_function import list as lossf_list

import matplotlib.pyplot as plt
import time


def make_grid(sample):
    img = torchvision.utils.make_grid(sample)
    return img.detach().numpy()


class Configuration:
    def __init__(self, on=False, name="none", eps=0, eta=0, alp=0, c=0, lr=0, iter=0, sigma=0, stop=False):
        self.on = on
        self.name = name
        self.eps = eps
        self.eta = eta
        self.c = c
        self.alp = alp
        self.lr = lr
        self.iter = iter  # steps
        self.sigma = sigma
        self.stop = stop
        return


def prediction(model, x):
    output = model(x)
    _, hx = output.data.max(1)
    return hx


def test(device, classes, target_model, sub_model, test_loader):
    sub_model.to(device)

    dgm_cfg  = Configuration(False, "dgm", eps=0.005, alp=0.001, iter=300, c=0.3)
    sqba_cfg = Configuration(True, "sqba", iter=250)

    count = 0
    success_cnt = np.zeros(2).astype(int)
    queries = np.zeros(2).astype(int)
    throughput = np.zeros(2)
    predict = np.zeros(2).astype(int)
    distance = np.zeros(2)
    algorithm = []

    for data, true_class in test_loader:

        x = data.clone()
        y = true_class.clone()

        pix_min = torch.min(x.flatten()).to(device)
        pix_max = torch.max(x.flatten()).to(device)

        x = x.to(device)
        y = y.to(device)

        h = prediction(target_model, x)
        if h != y:
            continue

        idx = 0

        count += 1
        if count > 1001:
            break

        def evaluation(idx, alg, name):
            algorithm.append(name)
            t0 = time.perf_counter()
            ladv, lquery, iter0, iter1 = alg.untarget(x, y)
            t1 = time.perf_counter()

            loutput = target_model(ladv)
            lpred = loutput.max(1, keepdim=True)[1]
            predict[idx] = (lpred.item())
            queries[idx] = lquery
            throughput[idx] = (t1 - t0)
            distance[idx] = (torch.norm(torch.abs(ladv - x)) / torch.norm(x))

            if lpred.item() != y:
                success_cnt[idx] += 1

        if dgm_cfg.on:
            cfg = dgm_cfg
            alg = attack.DGM_L2(device, target_model, eps=cfg.eps, min=pix_min, max=pix_max)
            evaluation(idx, alg, cfg.name)
            idx += 1

        if sqba_cfg.on:
            cfg = sqba_cfg
            # Resolve loss function name on the provided sub_model (handle DataParallel)
            lossF = None
            try:
                loss_name = None
                if hasattr(sub_model, 'loss') and sub_model.loss is not None:
                    loss_name = sub_model.loss
                elif hasattr(sub_model, 'module') and hasattr(sub_model.module, 'loss') and sub_model.module.loss is not None:
                    loss_name = sub_model.module.loss

                if loss_name:
                    lossF = lossf_list(loss_name).to(device)
            except Exception:
                lossF = None

            if lossF is None:
                lossF = torch.nn.CrossEntropyLoss().to(device)

            alg = attack.SQBA(device, model=target_model, sub_model=sub_model, lossF=lossF,  q_budgets=[cfg.iter], stop=False)
            evaluation(idx, alg, cfg.name)
            idx += 1

        if count == 1 or show_model == 1:
            show_model = 0
            for i in range(idx):
                print("{}[{} - {} {}] ".format(i, algorithm[i], target_model.name, sub_model.name), end='')
            print("")

        print("{}[{}]- ".format(count, classes[true_class.item()]), end='')
        for i in range(idx):
            print("{}[{}, {:.3f}, {:.3f}, {}] ".format(success_cnt[i],
                                                       classes[predict[i]], distance[i], throughput[i], queries[i]), end='')
        print("")


def run_on_images(target_model, images, labels, device='cpu', q_budget=250, max_images=None):
    """Run SQBA attack on a provided batch of images and return results in the standard format.

    Parameters:
    - target_model: torch model (already on device)
    - images: torch.Tensor shape (N,C,H,W) on cpu or device
    - labels: torch.Tensor shape (N,) with integer labels
    - device: 'cpu' or 'cuda'
    - q_budget: query budget per sample
    - max_images: optional limit on number of images to process

    Returns: adversarial_images (list numpy arrays), adversarial_labels (list ints), queries (list ints), l2s (list floats), total_success (int)
    """
    import time
    adv_images = []
    adv_labels = []
    queries = []
    l2s = []
    total_success = 0

    # ensure model in eval and on device
    try:
        target_model.to(device)
    except Exception:
        pass
    target_model.eval()

    n = images.shape[0]
    if max_images is not None:
        n = min(n, max_images)

    # attempt to build a loss function if possible (handle DataParallel wrapper)
    lossF = None
    try:
        loss_name = None
        if hasattr(target_model, 'loss') and target_model.loss is not None:
            loss_name = target_model.loss
        elif hasattr(target_model, 'module') and hasattr(target_model.module, 'loss') and target_model.module.loss is not None:
            loss_name = target_model.module.loss

        if loss_name:
            lossF = lossf_list(loss_name).to(device)
    except Exception:
        lossF = None

    # fallback to a standard PyTorch loss if mapping not found
    if lossF is None:
        lossF = torch.nn.CrossEntropyLoss().to(device)

    for i in range(n):
        x = images[i:i+1].to(device)
        y = labels[i:i+1].to(device)

        try:
            alg = attack.SQBA(device, model=target_model, sub_model=target_model, lossF=lossF, q_budgets=[q_budget], stop=False)
            ladv, lquery, iter0, iter1 = alg.untarget(x, y)
        except Exception as e :
            # fallback: no adversarial found
            import traceback
            traceback.print_exc()
            print(f"SQBA attack failed on sample {i}: {e}")
            ladv = x.clone()
            lquery = 0
            raise Exception(f"SQBA attack failed on sample {i}: {e}")

        # ensure tensor
        if isinstance(ladv, np.ndarray):
            ladv_t = torch.from_numpy(ladv).float().to(device)
        else:
            ladv_t = ladv.to(device) if hasattr(ladv, 'to') else torch.tensor(ladv, device=device)

        with torch.no_grad():
            pred = target_model(ladv_t).argmax(dim=1).cpu().detach()

        # compute l2
        try:
            l2_val = float((torch.norm(torch.abs(ladv - x)) / torch.norm(x)))#float(torch.norm((ladv_t - x).view(1, -1), p=2).item())
        except Exception:
            try:
                l2_val = float(np.linalg.norm((ladv_t.cpu().detach().numpy() - x.cpu().detach().numpy()).ravel()))
            except Exception:
                l2_val = 0.0

        adv_images.append(ladv_t.cpu().detach().numpy()[0])
        adv_labels.append(int(pred.item()))
        queries.append(int(lquery))
        l2s.append(float(l2_val))

        if int(pred.item()) != int(labels[i].cpu().item()):
            total_success += 1

    return adv_images, adv_labels, queries, l2s, total_success

