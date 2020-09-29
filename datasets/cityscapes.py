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
"""
import os
import os.path as path

from config import cfg
from runx.logx import logx
from datasets.base_loader import BaseLoader
import datasets.cityscapes_labels as cityscapes_labels
import datasets.uniform as uniform
from datasets.utils import make_dataset_folder


def cities_cv_split(root, split, cv_split):
    """
    Find cities that correspond to a given split of the data. We split the data
    such that a given city belongs to either train or val, but never both. cv0
    is defined to be the default split.

     all_cities = [x x x x x x x x x x x x]
     val:
       split0     [x x x                  ]
       split1     [        x x x          ]
       split2     [                x x x  ]
     trn:
       split0     [      x x x x x x x x x]
       split1     [x x x x       x x x x x]
       split2     [x x x x x x x x        ]

    split - train/val/test
    cv_split - 0,1,2,3

    cv_split == 3 means use train + val
    """
    trn_path = path.join(root, 'leftImg8bit_trainvaltest/leftImg8bit', 'train')
    val_path = path.join(root, 'leftImg8bit_trainvaltest/leftImg8bit', 'val')

    trn_cities = ['train/' + c for c in os.listdir(trn_path)]
    trn_cities = sorted(trn_cities)  # sort to insure reproducibility
    val_cities = ['val/' + c for c in os.listdir(val_path)]

    all_cities = val_cities + trn_cities

    if cv_split == 3:
        logx.msg('cv split {} {} {}'.format(split, cv_split, all_cities))
        return all_cities

    num_val_cities = len(val_cities)
    num_cities = len(all_cities)

    offset = cv_split * num_cities // cfg.DATASET.CV_SPLITS
    cities = []
    for j in range(num_cities):
        if j >= offset and j < (offset + num_val_cities):
            if split == 'val':
                cities.append(all_cities[j])
        else:
            if split == 'train':
                cities.append(all_cities[j])

    logx.msg('cv split {} {} {}'.format(split, cv_split, cities))
    return cities


def coarse_cities(root):
    """
    Find coarse cities
    """
    split = 'train_extra'
    coarse_path = path.join(root, 'leftImg8bit_trainextra/leftImg8bit',
                            split)
    coarse_cities = [f'{split}/' + c for c in os.listdir(coarse_path)]

    logx.msg(f'found {len(coarse_cities)} coarse cities')
    return coarse_cities


class Loader(BaseLoader):
    num_classes = 19
    ignore_label = 255
    trainid_to_name = {}
    color_mapping = []

    def __init__(self, mode, quality='fine', joint_transform_list=None,
                 img_transform=None, label_transform=None, eval_folder=None):

        super(Loader, self).__init__(quality=quality, mode=mode,
                                     joint_transform_list=joint_transform_list,
                                     img_transform=img_transform,
                                     label_transform=label_transform)

        ######################################################################
        # Cityscapes-specific stuff:
        ######################################################################
        self.root = cfg.DATASET.CITYSCAPES_DIR
        self.id_to_trainid = cityscapes_labels.label2trainid
        self.trainid_to_name = cityscapes_labels.trainId2name
        self.fill_colormap()
        img_ext = 'png'
        mask_ext = 'png'
        img_root = path.join(self.root, 'leftImg8bit_trainvaltest/leftImg8bit')
        mask_root = path.join(self.root, 'gtFine_trainvaltest/gtFine')
        if mode == 'folder':
            self.all_imgs = make_dataset_folder(eval_folder)
        else:
            self.fine_cities = cities_cv_split(self.root, mode, cfg.DATASET.CV)
            self.all_imgs = self.find_cityscapes_images(
                self.fine_cities, img_root, mask_root, img_ext, mask_ext)

        logx.msg(f'cn num_classes {self.num_classes}')
        self.fine_centroids = uniform.build_centroids(self.all_imgs,
                                                      self.num_classes,
                                                      self.train,
                                                      cv=cfg.DATASET.CV,
                                                      id2trainid=self.id_to_trainid)
        self.centroids = self.fine_centroids

        if cfg.DATASET.COARSE_BOOST_CLASSES and mode == 'train':
            self.coarse_cities = coarse_cities(self.root)
            img_root = path.join(self.root,
                                 'leftImg8bit_trainextra/leftImg8bit')
            mask_root = path.join(self.root, 'gtCoarse', 'gtCoarse')
            self.coarse_imgs = self.find_cityscapes_images(
                self.coarse_cities, img_root, mask_root, img_ext, mask_ext,
                fine_coarse='gtCoarse')

            if cfg.DATASET.CLASS_UNIFORM_PCT:   
                
                custom_coarse = (cfg.DATASET.CUSTOM_COARSE_PROB is not None)
                self.coarse_centroids = uniform.build_centroids(
                    self.coarse_imgs, self.num_classes, self.train,
                    coarse=(not custom_coarse), custom_coarse=custom_coarse,
                    id2trainid=self.id_to_trainid)

                for cid in cfg.DATASET.COARSE_BOOST_CLASSES:
                    self.centroids[cid].extend(self.coarse_centroids[cid])
            else:
                self.all_imgs.extend(self.coarse_imgs)

        self.build_epoch()

    def disable_coarse(self):
        """
        Turn off using coarse images in training
        """
        self.centroids = self.fine_centroids

    def only_coarse(self):
        """
        Turn on using coarse images in training
        """
        print('==============+Running Only Coarse+===============')
        self.centroids = self.coarse_centroids

    def find_cityscapes_images(self, cities, img_root, mask_root, img_ext,
                               mask_ext, fine_coarse='gtFine'):
        """
        Find image and segmentation mask files and return a list of
        tuples of them.

        Inputs:
        img_root: path to parent directory of train/val/test dirs
        mask_root: path to parent directory of train/val/test dirs
        img_ext: image file extension
        mask_ext: mask file extension
        cities: a list of cities, each element in the form of 'train/a_city'
          or 'val/a_city', for example.
        """
        items = []
        for city in cities:
            img_dir = '{root}/{city}'.format(root=img_root, city=city)
            for file_name in os.listdir(img_dir):
                basename, ext = os.path.splitext(file_name)
                assert ext == '.' + img_ext, '{} {}'.format(ext, img_ext)
                full_img_fn = os.path.join(img_dir, file_name)
                basename, ext = file_name.split('_leftImg8bit')
                if cfg.DATASET.CUSTOM_COARSE_PROB and fine_coarse != 'gtFine':
                    mask_fn = f'{basename}_leftImg8bit.png'
                    cc_path = cfg.DATASET.CITYSCAPES_CUSTOMCOARSE
                    full_mask_fn = os.path.join(cc_path, city, mask_fn)
                    os.path.isfile(full_mask_fn)
                else:
                    mask_fn = f'{basename}_{fine_coarse}_labelIds{ext}'
                    full_mask_fn = os.path.join(mask_root, city, mask_fn)
                items.append((full_img_fn, full_mask_fn))

        logx.msg('mode {} found {} images'.format(self.mode, len(items)))

        return items

    def fill_colormap(self):
        palette = [128, 64, 128,
                   244, 35, 232,
                   70, 70, 70,
                   102, 102, 156,
                   190, 153, 153,
                   153, 153, 153,
                   250, 170, 30,
                   220, 220, 0,
                   107, 142, 35,
                   152, 251, 152,
                   70, 130, 180,
                   220, 20, 60,
                   255, 0, 0,
                   0, 0, 142,
                   0, 0, 70,
                   0, 60, 100,
                   0, 80, 100,
                   0, 0, 230,
                   119, 11, 32]
        zero_pad = 256 * 3 - len(palette)
        for i in range(zero_pad):
            palette.append(0)
        self.color_mapping = palette
