"""
Uniform sampling of classes.
For all images, for all classes, generate centroids around which to sample.

All images are divided into tiles.
For each tile, a class can be present or not. If it is
present, calculate the centroid of the class and record it.

We would like to thank Peter Kontschieder for the inspiration of this idea.
"""

import logging
from collections import defaultdict
from PIL import Image
import numpy as np
from scipy import ndimage
from tqdm import tqdm

pbar = None

class Point():
    """
    Point Class For X and Y Location
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y


def calc_tile_locations(tile_size, image_size):
    """
    Divide an image into tiles to help us cover classes that are spread out.
    tile_size: size of tile to distribute
    image_size: original image size
    return: locations of the tiles
    """
    image_size_y, image_size_x = image_size
    locations = []
    for y in range(image_size_y // tile_size):
        for x in range(image_size_x // tile_size):
            x_offs = x * tile_size
            y_offs = y * tile_size
            locations.append((x_offs, y_offs))
    return locations


def class_centroids_image(item, tile_size, num_classes, id2trainid):
    """
    For one image, calculate centroids for all classes present in image.
    item: image, image_name
    tile_size:
    num_classes:
    id2trainid: mapping from original id to training ids
    return: Centroids are calculated for each tile.
    """
    image_fn, label_fn = item
    centroids = defaultdict(list)
    mask = np.array(Image.open(label_fn))
    image_size = mask.shape
    tile_locations = calc_tile_locations(tile_size, image_size)

    mask_copy = mask.copy()
    if id2trainid:
        for k, v in id2trainid.items():
            mask[mask_copy == k] = v

    for x_offs, y_offs in tile_locations:
        patch = mask[y_offs:y_offs + tile_size, x_offs:x_offs + tile_size]
        for class_id in range(num_classes):
            if class_id in patch:
                patch_class = (patch == class_id).astype(int)
                centroid_y, centroid_x = ndimage.measurements.center_of_mass(patch_class)
                centroid_y = int(centroid_y) + y_offs
                centroid_x = int(centroid_x) + x_offs
                centroid = (centroid_x, centroid_y)
                centroids[class_id].append((image_fn, label_fn, centroid, class_id))
    pbar.update(1)
    return centroids



def pooled_class_centroids_all(items, num_classes, id2trainid, tile_size=1024):
    """
    Calculate class centroids for all classes for all images for all tiles.
    items: list of (image_fn, label_fn)
    tile size: size of tile
    returns: dict that contains a list of centroids for each class
    """
    from multiprocessing.dummy import Pool
    from functools import partial
    pool = Pool(32)
    global pbar
    pbar = tqdm(total=len(items), desc='pooled centroid extraction')
    class_centroids_item = partial(class_centroids_image,
                                   num_classes=num_classes,
                                   id2trainid=id2trainid,
                                   tile_size=tile_size)

    centroids = defaultdict(list)
    new_centroids = pool.map(class_centroids_item, items)
    pool.close()
    pool.join()

    # combine each image's items into a single global dict
    for image_items in new_centroids:
        for class_id in image_items:
            centroids[class_id].extend(image_items[class_id])
    return centroids


def unpooled_class_centroids_all(items, num_classes, tile_size=1024):
    """
    Calculate class centroids for all classes for all images for all tiles.
    items: list of (image_fn, label_fn)
    tile size: size of tile
    returns: dict that contains a list of centroids for each class
    """
    centroids = defaultdict(list)
    global pbar
    pbar = tqdm(total=len(items), desc='centroid extraction')
    for image, label in items:
        new_centroids = class_centroids_image((image, label),
                                              tile_size,
                                              num_classes)
        for class_id in new_centroids:
            centroids[class_id].extend(new_centroids[class_id])

    return centroids


def class_centroids_all(items, num_classes, id2trainid, tile_size=1024):
    """
    intermediate function to call pooled_class_centroid
    """

    pooled_centroids = pooled_class_centroids_all(items, num_classes,
                                                  id2trainid, tile_size)
    return pooled_centroids


def random_sampling(alist, num):
    """
    Randomly sample num items from the list
    alist: list of centroids to sample from
    num: can be larger than the list and if so, then wrap around
    return: class uniform samples from the list
    """
    sampling = []
    len_list = len(alist)
    assert len_list, 'len_list is zero!'
    indices = np.arange(len_list)
    np.random.shuffle(indices)

    for i in range(num):
        item = alist[indices[i % len_list]]
        sampling.append(item)
    return sampling


def build_epoch(imgs, centroids, num_classes, class_uniform_pct):
    """
    Generate an epochs-worth of crops using uniform sampling. Needs to be called every
    imgs: list of imgs
    centroids:
    num_classes:
    class_uniform_pct: class uniform sampling percent ( % of uniform images in one epoch )
    """
    logging.info("Class Uniform Percentage: %s", str(class_uniform_pct))
    num_epoch = int(len(imgs))

    logging.info('Class Uniform items per Epoch:%s', str(num_epoch))
    num_per_class = int((num_epoch * class_uniform_pct) / num_classes)
    num_rand = num_epoch - num_per_class * num_classes
    # create random crops
    imgs_uniform = random_sampling(imgs, num_rand)

    # now add uniform sampling
    for class_id in range(num_classes):
        string_format = "cls %d len %d"% (class_id, len(centroids[class_id]))
        logging.info(string_format)
    for class_id in range(num_classes):
        centroid_len = len(centroids[class_id])
        if centroid_len == 0:
            pass
        else:
            class_centroids = random_sampling(centroids[class_id], num_per_class)
            imgs_uniform.extend(class_centroids)

    return imgs_uniform
