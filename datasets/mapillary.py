"""
Mapillary Dataset Loader
"""
from PIL import Image
from torch.utils import data
import os
import numpy as np
import json
import datasets.uniform as uniform
from config import cfg

num_classes = 65
ignore_label = 65
root = cfg.DATASET.MAPILLARY_DIR
config_fn = os.path.join(root, 'config.json')
id_to_ignore_or_group = {}
color_mapping = []
id_to_trainid = {}


def colorize_mask(image_array):
    """
    Colorize a segmentation mask
    """
    new_mask = Image.fromarray(image_array.astype(np.uint8)).convert('P')
    new_mask.putpalette(color_mapping)
    return new_mask


def make_dataset(quality, mode):
    """
    Create File List
    """
    assert (quality == 'semantic' and mode in ['train', 'val'])
    img_dir_name = None
    if quality == 'semantic':
        if mode == 'train':
            img_dir_name = 'training'
        if mode == 'val':
            img_dir_name = 'validation'
        mask_path = os.path.join(root, img_dir_name, 'labels')
    else:
        raise BaseException("Instance Segmentation Not support")

    img_path = os.path.join(root, img_dir_name, 'images')
    print(img_path)
    if quality != 'video':
        imgs = sorted([os.path.splitext(f)[0] for f in os.listdir(img_path)])
        msks = sorted([os.path.splitext(f)[0] for f in os.listdir(mask_path)])
        assert imgs == msks

    items = []
    c_items = os.listdir(img_path)
    if '.DS_Store' in c_items:
        c_items.remove('.DS_Store')

    for it in c_items:
        if quality == 'video':
            item = (os.path.join(img_path, it), os.path.join(img_path, it))
        else:
            item = (os.path.join(img_path, it),
                    os.path.join(mask_path, it.replace(".jpg", ".png")))
        items.append(item)
    return items


def gen_colormap():
    """
    Get Color Map from file
    """
    global color_mapping

    # load mapillary config
    with open(config_fn) as config_file:
        config = json.load(config_file)
    config_labels = config['labels']

    # calculate label color mapping
    colormap = []
    id2name = {}
    for i in range(0, len(config_labels)):
        colormap = colormap + config_labels[i]['color']
        id2name[i] = config_labels[i]['readable']
    color_mapping = colormap
    return id2name


class Mapillary(data.Dataset):
    def __init__(self, quality, mode, joint_transform_list=None,
                 transform=None, target_transform=None, dump_images=False,
                 class_uniform_pct=0, class_uniform_tile=768, test=False):
        """
        class_uniform_pct = Percent of class uniform samples. 1.0 means fully uniform.
                            0.0 means fully random.
        class_uniform_tile_size = Class uniform tile size
        """
        self.quality = quality
        self.mode = mode
        self.joint_transform_list = joint_transform_list
        self.transform = transform
        self.target_transform = target_transform
        self.dump_images = dump_images
        self.class_uniform_pct = class_uniform_pct
        self.class_uniform_tile = class_uniform_tile
        self.id2name = gen_colormap()
        self.imgs_uniform = None
        for i in range(num_classes):
            id_to_trainid[i] = i

        # find all images
        self.imgs = make_dataset(quality, mode)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        if test:
            np.random.shuffle(self.imgs)
            self.imgs = self.imgs[:200]

        if self.class_uniform_pct:
            json_fn = 'mapillary_tile{}.json'.format(self.class_uniform_tile)
            if os.path.isfile(json_fn):
                with open(json_fn, 'r') as json_data:
                    centroids = json.load(json_data)
                self.centroids = {int(idx): centroids[idx] for idx in centroids}
            else:
                # centroids is a dict (indexed by class) of lists of centroids
                self.centroids = uniform.class_centroids_all(
                    self.imgs,
                    num_classes,
                    id2trainid=None,
                    tile_size=self.class_uniform_tile)
                with open(json_fn, 'w') as outfile:
                    json.dump(self.centroids, outfile, indent=4)
        else:
            self.centroids = []
        self.build_epoch()

    def build_epoch(self):
        if self.class_uniform_pct != 0:
            self.imgs_uniform = uniform.build_epoch(self.imgs,
                                                    self.centroids,
                                                    num_classes,
                                                    self.class_uniform_pct)
        else:
            self.imgs_uniform = self.imgs

    def __getitem__(self, index):
        if len(self.imgs_uniform[index]) == 2:
            img_path, mask_path = self.imgs_uniform[index]
            centroid = None
            class_id = None
        else:
            img_path, mask_path, centroid, class_id = self.imgs_uniform[index]
        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        mask = np.array(mask)
        mask_copy = mask.copy()
        for k, v in id_to_ignore_or_group.items():
            mask_copy[mask == k] = v
        mask = Image.fromarray(mask_copy.astype(np.uint8))

        # Image Transformations
        if self.joint_transform_list is not None:
            for idx, xform in enumerate(self.joint_transform_list):
                if idx == 0 and centroid is not None:
                    # HACK! Assume the first transform accepts a centroid
                    img, mask = xform(img, mask, centroid)
                else:
                    img, mask = xform(img, mask)

        if self.dump_images:
            outdir = 'dump_imgs_{}'.format(self.mode)
            os.makedirs(outdir, exist_ok=True)
            if centroid is not None:
                dump_img_name = self.id2name[class_id] + '_' + img_name
            else:
                dump_img_name = img_name
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

    def calculate_weights(self):
        raise BaseException("not supported yet")
