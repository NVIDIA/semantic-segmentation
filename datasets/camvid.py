"""
Camvid Dataset Loader
"""

import os
import sys
import numpy as np
from PIL import Image
from torch.utils import data
import logging
import datasets.uniform as uniform
import json
from config import cfg


# trainid_to_name = cityscapes_labels.trainId2name
# id_to_trainid = cityscapes_labels.label2trainid
num_classes = 11
ignore_label = 11
root = cfg.DATASET.CAMVID_DIR

palette = [128, 128, 128, 
            128, 0, 0, 
            192, 192, 128, 
            128, 64, 128,
            0, 0, 192, 
            128, 128, 0,
            192, 128, 128, 
            64, 64, 128,
            64, 0, 128, 
            64, 64, 0, 
            0, 128, 192]


CAMVID_CLASSES = ['Sky',
                  'Building',
                  'Column-Pole',
                  'Road',
                  'Sidewalk',
                  'Tree',
                  'Sign-Symbol',
                  'Fence',
                  'Car',
                  'Pedestrain',
                  'Bicyclist',
                  'Void']

CAMVID_CLASS_COLORS = [
    (128, 128, 128),
    (128, 0, 0),
    (192, 192, 128),
    (128, 64, 128),
    (0, 0, 192),
    (128, 128, 0),
    (192, 128, 128),
    (64, 64, 128),
    (64, 0, 128),
    (64, 64, 0),
    (0, 128, 192),
    (0, 0, 0),
]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def add_items(img_path, mask_path, aug_img_path, aug_mask_path, mode, maxSkip):

    c_items = os.listdir(img_path)
    c_items.sort()
    items = []
    aug_items = []

    for it in c_items:
        item = (os.path.join(img_path, it), os.path.join(mask_path, it))
        items.append(item)
        if mode != 'test' and maxSkip > 0:
            seq_info = it.split("_")
            cur_seq_id = seq_info[-1][:-4]

            if seq_info[0] == "0001TP":
                prev_seq_id = "%06d" % (int(cur_seq_id) - maxSkip)
                next_seq_id = "%06d" % (int(cur_seq_id) + maxSkip)
            elif seq_info[0] == "0006R0":
                prev_seq_id = "f%05d" % (int(cur_seq_id[1:]) - maxSkip)
                next_seq_id = "f%05d" % (int(cur_seq_id[1:]) + maxSkip)
            else:
                prev_seq_id = "%05d" % (int(cur_seq_id) - maxSkip)
                next_seq_id = "%05d" % (int(cur_seq_id) + maxSkip)

            prev_it = seq_info[0] + "_" + prev_seq_id + '.png'
            next_it = seq_info[0] + "_" + next_seq_id + '.png'

            prev_item = (os.path.join(aug_img_path, prev_it), os.path.join(aug_mask_path, prev_it))
            next_item = (os.path.join(aug_img_path, next_it), os.path.join(aug_mask_path, next_it))
            if os.path.isfile(prev_item[0]) and os.path.isfile(prev_item[1]):
                aug_items.append(prev_item)
            if os.path.isfile(next_item[0]) and os.path.isfile(next_item[1]):
                aug_items.append(next_item)
    return items, aug_items

def make_dataset(quality, mode, maxSkip=0, cv_split=0, hardnm=0):
    
    items = []
    aug_items = []
    assert quality == 'semantic' 
    assert mode in ['train', 'val', 'trainval', 'test']

    # img_dir_name = "SegNet/CamVid"
    original_img_dir = "LargeScale/CamVid"
    augmented_img_dir = "camvid_aug3/CamVid"

    img_path = os.path.join(root, original_img_dir, 'train')
    mask_path = os.path.join(root, original_img_dir, 'trainannot')
    aug_img_path = os.path.join(root, augmented_img_dir, 'train')
    aug_mask_path = os.path.join(root, augmented_img_dir, 'trainannot')

    train_items, train_aug_items = add_items(img_path, mask_path, aug_img_path, aug_mask_path, mode, maxSkip)
    logging.info('Camvid has a total of {} train images'.format(len(train_items)))

    img_path = os.path.join(root, original_img_dir, 'val')
    mask_path = os.path.join(root, original_img_dir, 'valannot')
    aug_img_path = os.path.join(root, augmented_img_dir, 'val')
    aug_mask_path = os.path.join(root, augmented_img_dir, 'valannot')

    val_items, val_aug_items = add_items(img_path, mask_path, aug_img_path, aug_mask_path, mode, maxSkip)
    logging.info('Camvid has a total of {} validation images'.format(len(val_items)))

    if mode == 'test':
        img_path = os.path.join(root, original_img_dir, 'test')
        mask_path = os.path.join(root, original_img_dir, 'testannot')
        test_items, test_aug_items = add_items(img_path, mask_path, aug_img_path, aug_mask_path, mode, maxSkip)
        logging.info('Camvid has a total of {} test images'.format(len(test_items)))

    if mode == 'train':
        items = train_items
    elif mode == 'val':
        items = val_items
    elif mode == 'trainval':
        items = train_items + val_items
        aug_items = train_aug_items + val_aug_items
    elif mode == 'test':
        items = test_items
        aug_items = []
    else:
        logging.info('Unknown mode {}'.format(mode))
        sys.exit()

    logging.info('Camvid-{}: {} images'.format(mode, len(items)))

    return items, aug_items

class CAMVID(data.Dataset):

    def __init__(self, quality, mode, maxSkip=0, joint_transform_list=None,
                 transform=None, target_transform=None, dump_images=False,
                 class_uniform_pct=0, class_uniform_tile=0, test=False, 
                 cv_split=None, scf=None, hardnm=0):

        self.quality = quality
        self.mode = mode
        self.maxSkip = maxSkip
        self.joint_transform_list = joint_transform_list
        self.transform = transform
        self.target_transform = target_transform
        self.dump_images = dump_images
        self.class_uniform_pct = class_uniform_pct
        self.class_uniform_tile = class_uniform_tile
        self.scf = scf
        self.hardnm = hardnm
        self.cv_split = cv_split
        self.centroids = []

        self.imgs, self.aug_imgs = make_dataset(quality, mode, self.maxSkip, cv_split=self.cv_split, hardnm=self.hardnm)
        assert len(self.imgs), 'Found 0 images, please check the data set'

        # Centroids for GT data
        if self.class_uniform_pct > 0:
            json_fn = 'camvid_tile{}_cv{}_{}.json'.format(self.class_uniform_tile, self.cv_split, self.mode)

            if os.path.isfile(json_fn):
                with open(json_fn, 'r') as json_data:
                    centroids = json.load(json_data)
                self.centroids = {int(idx): centroids[idx] for idx in centroids}
            else:
                self.centroids = uniform.class_centroids_all(
                        self.imgs,
                        num_classes,
                        id2trainid=None,
                        tile_size=class_uniform_tile)
                with open(json_fn, 'w') as outfile:
                    json.dump(self.centroids, outfile, indent=4)

            self.fine_centroids = self.centroids.copy()

            if self.maxSkip > 0:
                json_fn = 'camvid_tile{}_cv{}_{}_skip{}.json'.format(self.class_uniform_tile, self.cv_split, self.mode, self.maxSkip)
                if os.path.isfile(json_fn):
                    with open(json_fn, 'r') as json_data:
                        centroids = json.load(json_data)
                    self.aug_centroids = {int(idx): centroids[idx] for idx in centroids}
                else:
                    self.aug_centroids = uniform.class_centroids_all(
                            self.aug_imgs,
                            num_classes,
                            id2trainid=None,
                            tile_size=class_uniform_tile)
                    with open(json_fn, 'w') as outfile:
                        json.dump(self.aug_centroids, outfile, indent=4)

                for class_id in range(num_classes):
                    self.centroids[class_id].extend(self.aug_centroids[class_id])

        self.build_epoch()

    def build_epoch(self, cut=False):

        if self.class_uniform_pct > 0:
            if cut:
                self.imgs_uniform = uniform.build_epoch(self.imgs,
                                                self.fine_centroids,
                                                num_classes,
                                                cfg.CLASS_UNIFORM_PCT)
            else:
                self.imgs_uniform = uniform.build_epoch(self.imgs,
                                                self.centroids,
                                                num_classes,
                                                cfg.CLASS_UNIFORM_PCT)
        else:
            self.imgs_uniform = self.imgs
            

    def __getitem__(self, index):
        elem = self.imgs_uniform[index]
        centroid = None
        if len(elem) == 4:
            img_path, mask_path, centroid, class_id = elem
        else:
            img_path, mask_path = elem

        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # Image Transformations
        if self.joint_transform_list is not None:
            for idx, xform in enumerate(self.joint_transform_list):
                if idx == 0 and centroid is not None:
                    # HACK
                    # We assume that the first transform is capable of taking
                    # in a centroid
                    img, mask = xform(img, mask, centroid)
                else:
                    img, mask = xform(img, mask)

        # Debug
        if self.dump_images and centroid is not None:
            outdir = './dump_imgs_{}'.format(self.mode)
            os.makedirs(outdir, exist_ok=True)
            dump_img_name = trainid_to_name[class_id] + '_' + img_name
            out_img_fn = os.path.join(outdir, dump_img_name + '.png')
            out_msk_fn = os.path.join(outdir, dump_img_name + '_mask.png')
            mask_img = colorize_mask(np.array(mask))
            img.save(out_img_fn)
            mask_img.save(out_msk_fn)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return img, mask, img_name

    def __len__(self):
        return len(self.imgs_uniform)


