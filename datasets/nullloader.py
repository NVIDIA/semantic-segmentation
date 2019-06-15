"""
Null Loader
"""
import numpy as np
import torch
from torch.utils import data

num_classes = 19
ignore_label = 255

class NullLoader(data.Dataset):
    """
    Null Dataset for Performance
    """
    def __init__(self,crop_size):
        self.imgs = range(200)
        self.crop_size = crop_size

    def __getitem__(self, index):
    	#Return img, mask, name
        return torch.FloatTensor(np.zeros((3,self.crop_size,self.crop_size))), torch.LongTensor(np.zeros((self.crop_size,self.crop_size))), 'img' + str(index)

    def __len__(self):
        return len(self.imgs)