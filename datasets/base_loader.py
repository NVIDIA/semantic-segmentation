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

Generic dataloader base class
"""
import os
import glob
import numpy as np
import torch

from PIL import Image
from torch.utils import data
from config import cfg
from datasets import uniform
from runx.logx import logx
from utils.misc import tensor_to_pil


class BaseLoader(data.Dataset):
    def __init__(self, quality, mode, joint_transform_list, img_transform,
                 label_transform):

        super(BaseLoader, self).__init__()
        self.quality = quality
        self.mode = mode
        self.joint_transform_list = joint_transform_list
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.train = mode == 'train'
        self.id_to_trainid = {}
        self.centroids = None
        self.all_imgs = None
        self.drop_mask = np.zeros((1024, 2048))
        self.drop_mask[15:840, 14:2030] = 1.0

    def build_epoch(self):
        """
        For class uniform sampling ... every epoch, we want to recompute
        which tiles from which images we want to sample from, so that the
        sampling is uniformly random.
        """
        self.imgs = uniform.build_epoch(self.all_imgs,
                                        self.centroids,
                                        self.num_classes,
                                        self.train)

    @staticmethod
    def find_images(img_root, mask_root, img_ext, mask_ext):
        """
        Find image and segmentation mask files and return a list of
        tuples of them.
        """
        img_path = '{}/*.{}'.format(img_root, img_ext)
        imgs = glob.glob(img_path)
        items = []
        for full_img_fn in imgs:
            img_dir, img_fn = os.path.split(full_img_fn)
            img_name, _ = os.path.splitext(img_fn)
            full_mask_fn = '{}.{}'.format(img_name, mask_ext)
            full_mask_fn = os.path.join(mask_root, full_mask_fn)
            assert os.path.exists(full_mask_fn)
            items.append((full_img_fn, full_mask_fn))
        return items

    def disable_coarse(self):
        pass

    def colorize_mask(self, image_array):
        """
        Colorize the segmentation mask
        """
        new_mask = Image.fromarray(image_array.astype(np.uint8)).convert('P')
        new_mask.putpalette(self.color_mapping)
        return new_mask

    def dump_images(self, img_name, mask, centroid, class_id, img):
        img = tensor_to_pil(img)
        outdir = 'new_dump_imgs_{}'.format(self.mode)
        os.makedirs(outdir, exist_ok=True)
        if centroid is not None:
            dump_img_name = '{}_{}'.format(self.trainid_to_name[class_id],
                                           img_name)
        else:
            dump_img_name = img_name
        out_img_fn = os.path.join(outdir, dump_img_name + '.png')
        out_msk_fn = os.path.join(outdir, dump_img_name + '_mask.png')
        out_raw_fn = os.path.join(outdir, dump_img_name + '_mask_raw.png')
        mask_img = self.colorize_mask(np.array(mask))
        raw_img = Image.fromarray(np.array(mask))
        img.save(out_img_fn)
        mask_img.save(out_msk_fn)
        raw_img.save(out_raw_fn)

    def do_transforms(self, img, mask, centroid, img_name, class_id):
        """
        Do transformations to image and mask

        :returns: image, mask
        """
        scale_float = 1.0

        if self.joint_transform_list is not None:
            for idx, xform in enumerate(self.joint_transform_list):
                if idx == 0 and centroid is not None:
                    # HACK! Assume the first transform accepts a centroid
                    outputs = xform(img, mask, centroid)
                else:
                    outputs = xform(img, mask)

                if len(outputs) == 3:
                    img, mask, scale_float = outputs
                else:
                    img, mask = outputs

        if self.img_transform is not None:
            img = self.img_transform(img)

        if cfg.DATASET.DUMP_IMAGES:
            self.dump_images(img_name, mask, centroid, class_id, img)

        if self.label_transform is not None:
            mask = self.label_transform(mask)

        return img, mask, scale_float

    def read_images(self, img_path, mask_path, mask_out=False):
        img = Image.open(img_path).convert('RGB')
        if mask_path is None or mask_path == '':
            w, h = img.size
            mask = np.zeros((h, w))
        else:
            mask = Image.open(mask_path)

        drop_out_mask = None
        # This code is specific to cityscapes
        if(cfg.DATASET.CITYSCAPES_CUSTOMCOARSE in mask_path):

            gtCoarse_mask_path = mask_path.replace(cfg.DATASET.CITYSCAPES_CUSTOMCOARSE, os.path.join(cfg.DATASET.CITYSCAPES_DIR, 'gtCoarse/gtCoarse') )
            gtCoarse_mask_path = gtCoarse_mask_path.replace('leftImg8bit','gtCoarse_labelIds')          
            gtCoarse=np.array(Image.open(gtCoarse_mask_path))



        img_name = os.path.splitext(os.path.basename(img_path))[0]

        mask = np.array(mask)
        if (mask_out):
            mask = self.drop_mask * mask

        mask = mask.copy()
        for k, v in self.id_to_trainid.items():
            binary_mask = (mask == k) #+ (gtCoarse == k)
            if ('refinement' in mask_path) and cfg.DROPOUT_COARSE_BOOST_CLASSES != None and v in cfg.DROPOUT_COARSE_BOOST_CLASSES and binary_mask.sum() > 0 and 'vidseq' not in mask_path:
                binary_mask += (gtCoarse == k)
                binary_mask[binary_mask >= 1] = 1
                mask[binary_mask] = gtCoarse[binary_mask]
            mask[binary_mask] = v


        mask = Image.fromarray(mask.astype(np.uint8))
        return img, mask, img_name

    def __getitem__(self, index):
        """
        Generate data:

        :return:
        - image: image, tensor
        - mask: mask, tensor
        - image_name: basename of file, string
        """
        # Pick an image, fill in defaults if not using class uniform
        if len(self.imgs[index]) == 2:
            img_path, mask_path = self.imgs[index]
            centroid = None
            class_id = None
        else:
            img_path, mask_path, centroid, class_id = self.imgs[index]

        mask_out = cfg.DATASET.MASK_OUT_CITYSCAPES and \
            cfg.DATASET.CUSTOM_COARSE_PROB is not None and \
            'refinement' in mask_path

        img, mask, img_name = self.read_images(img_path, mask_path,
                                               mask_out=mask_out)

        ######################################################################
        # Thresholding is done when using coarse-labelled Cityscapes images
        ######################################################################
        if 'refinement' in mask_path:
            
            mask = np.array(mask)
            prob_mask_path = mask_path.replace('.png', '_prob.png')
            # put it in 0 to 1
            prob_map = np.array(Image.open(prob_mask_path)) / 255.0
            prob_map_threshold = (prob_map < cfg.DATASET.CUSTOM_COARSE_PROB)
            mask[prob_map_threshold] = cfg.DATASET.IGNORE_LABEL
            mask = Image.fromarray(mask.astype(np.uint8))

        img, mask, scale_float = self.do_transforms(img, mask, centroid,
                                                    img_name, class_id)

        return img, mask, img_name, scale_float

    def __len__(self):
        return len(self.imgs)

    def calculate_weights(self):
        raise BaseException("not supported yet")
