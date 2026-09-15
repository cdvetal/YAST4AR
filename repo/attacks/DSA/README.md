# Transferability of White-box Perturbations: Query-Efficient Adversarial Attacks against Commercial DNN Services

This is a PyTorch implementation of DSA, a powerful black-box adversarial attack, as described in the following:
> Meng Shen, Changyue Li, Qi Li, Hao Lu, Liehuang Zhu, and Ke Xu. Transferability of Whitebox Perturbations: Query-Efficient Adversarial Attacks against Commercial DNN Services. Proceedings of the 33rd USENIX Conference on Security Symposium (USENIX Security), 2024.

This repository aims to provide a simple demo to show how to use DSA to attack neural networks.

*Note that this demo may not consume the full query budget. With this demo, you can obtain a candidate adversarial example, which can be used as the starting point with the remaining query budget for existing query-based attacks, e.g., [SurFree](https://openaccess.thecvf.com/content/CVPR2021/papers/Maho_SurFree_A_Fast_Surrogate-Free_Black-Box_Attack_CVPR_2021_paper.pdf), and [HSJA](https://ieeexplore.ieee.org/document/9152788).*



#  🔥Design

The key components of this repository are as follows:

The `./demo.py` launches the adversarial attack against VGG16 using three substitute models, i.e.,  DenseNet, Inc-v3, IncRes-v2. These models are pre-trained and automatically downloaded on the first load.

The `./attack/dsa.py` is the main implementation of the DSA method.

The `./config/attack_config.py` contains the key parameters of the attack.



# 🚀QuickStart 

## Requirements

DSA is tested with Python >= 3.10 and  PyTorch >= 1.12. You can see the versions we currently use for testing in the [Compatibility section](#Compatibility) below, but newer versions are in general expected to work.

Other dependencies can be installed using the following command:

```bash 
pip install timm==0.9.16 pretrainedmodels==0.7.4 yacs==0.1.8 six==1.16 numpy==1.23.5
```

## Config

**Data**: As this demo uses two images with different labels from ImageNet, it is required to set your ImageNet path in `cfg.dataset.root_dir` . If ImageNet is unavailable, you can manually load two images by changing the function `load_images` in  `demo.py`.

**Model**: You can change `cfg.model.root_path` to indicate the download path of pre-trained models, which is the default torch_home when set to None. You can set different model types by changing `target_model` and `substitute_models`, and see available model types in [pretrainedmodels](https://github.com/Cadene/pretrained-models.pytorch) and [timm](https://github.com/huggingface/pytorch-image-models).

## Run

You can use the following command to launch the attack.

```bash
python demo.py
```




# 📄 Citation
If you use this repository for your work, please cite our paper using the following BibTeX entries:
>@inproceedings {298048, 
author = {Meng Shen and Changyue Li and Qi Li and Hao Lu and Liehuang Zhu and Ke Xu}, 
title = {Transferability of White-box Perturbations: {Query-Efficient} Adversarial Attacks against Commercial {DNN} Services}, 
booktitle = {33rd USENIX Security Symposium (USENIX Security 24)}, 
year = {2024},
isbn = {978-1-939133-44-1},
address = {Philadelphia, PA},
pages = {2991--3008},
url = {https://www.usenix.org/conference/usenixsecurity24/presentation/shen-meng}, 
publisher = {USENIX Association},
month = aug
}



# 🎉Compatibility

We tested with the following versions:

* Python: 3.10, 3.11
* PyTorch: 1.12, 1.13, 2.0, 2.1
* Numpy:  1.23, 1.24



# 💡 Questions?

If you have a question or need help, please feel free to open an issue on GitHub, or contact me at lichangyue98@163.com.

