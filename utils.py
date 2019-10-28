# -*- coding:utf-8 -*-
# @Time     : 2019-09-24 17:20
# @Author   : Richardo Mu
# @FILE     : class.PY
# @Software : PyCharm
# import torchvision
import torch
from torch import nn
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1,shape)

