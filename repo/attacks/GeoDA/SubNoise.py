import torch.nn as nn
import torch

class SubNoise(nn.Module):
    """given subspace x and the number of noises, generate sub noises"""

    # x is the subspace basis
    def __init__(self, num_noises, x):
        self.num_noises = num_noises
        self.x = x
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        super(SubNoise, self).__init__()

    def forward(self, img_width, img_height):
        r = torch.zeros([img_width * img_height, 3 * self.num_noises], dtype=torch.float32)
        noise = torch.randn([self.x.shape[1], 3 * self.num_noises], dtype=torch.float32).to(self.device)
        sub_noise = torch.transpose(torch.mm(self.x, noise), 0, 1)
        r = sub_noise.view([self.num_noises, 3, img_width, img_height])

        r_list = r
        return r_list
    