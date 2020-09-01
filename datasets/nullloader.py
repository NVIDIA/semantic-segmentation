"""
Copyright 2020 Nvidia Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

Null Loader
"""
from config import cfg
from runx.logx import logx
from datasets.base_loader import BaseLoader
from datasets.utils import make_dataset_folder
from datasets import uniform
import numpy as np
import torch
from torch.utils import data

class Loader(BaseLoader):
    """
    Null Dataset for Performance
    """
    num_classes = 19
    ignore_label = 255
    trainid_to_name = {}
    color_mapping = []
 
    def __init__(self, mode, quality=None, joint_transform_list=None,
                 img_transform=None, label_transform=None, eval_folder=None):
        super(Loader, self).__init__(quality=quality,
                                     mode=mode,
                                     joint_transform_list=joint_transform_list,
                                     img_transform=img_transform,
                                     label_transform=label_transform)

    def __getitem__(self, index):
        # return img, mask, img_name, scale_float
        crop_size = cfg.DATASET.CROP_SIZE
        if ',' in crop_size:
            crop_size = [int(x) for x in crop_size.split(',')]
        else:
            crop_size = int(crop_size)
            crop_size = [crop_size, crop_size]
        
        img = torch.FloatTensor(np.zeros([3] + crop_size))
        mask = torch.LongTensor(np.zeros(crop_size))
        img_name = f'img{index}'
        scale_float = 0.0
        return img, mask, img_name, scale_float

    def __len__(self):
        return 3000
