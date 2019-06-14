from __future__ import division
from __future__ import print_function

import torch.nn as nn

def conv2d(channels_in, channels_out, kernel_size=3, stride=1, bias = True):
    return nn.Sequential(
        nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=bias),
        nn.LeakyReLU(0.1,inplace=True)
    )

def deconv2d(channels_in, channels_out, kernel_size=4, stride=2, padding=1, bias=True):
    return nn.Sequential(
        nn.ConvTranspose2d(channels_in, channels_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.LeakyReLU(0.1,inplace=True)
    )