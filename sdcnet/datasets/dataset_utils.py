from __future__ import division
from __future__ import print_function

import torch

class StaticRandomCrop(object):
    """
    Helper function for random spatial crop
    """
    def __init__(self, size, image_shape):
        h, w = image_shape
        self.th, self.tw = size
        self.h1 = torch.randint(0, h - self.th + 1, (1,)).item()
        self.w1 = torch.randint(0, w - self.tw + 1, (1,)).item()

    def __call__(self, img):
        return img[self.h1:(self.h1 + self.th), self.w1:(self.w1 + self.tw), :]
