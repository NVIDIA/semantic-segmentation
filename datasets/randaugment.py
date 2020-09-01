# this code from: https://github.com/ildoonet/pytorch-randaugment
# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
import random
import numpy as np
import torch

from PIL import Image, ImageOps, ImageEnhance, ImageDraw
from config import cfg


fillmask = cfg.DATASET.IGNORE_LABEL
fillcolor = (0, 0, 0)


def affine_transform(pair, affine_params):
    img, mask = pair
    img = img.transform(img.size, Image.AFFINE, affine_params,
                        resample=Image.BILINEAR, fillcolor=fillcolor)
    mask = mask.transform(mask.size, Image.AFFINE, affine_params,
                          resample=Image.NEAREST, fillcolor=fillmask)
    return img, mask


def ShearX(pair, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return affine_transform(pair, (1, v, 0, 0, 1, 0))


def ShearY(pair, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return affine_transform(pair, (1, 0, 0, v, 1, 0))


def TranslateX(pair, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    img, _ = pair
    v = v * img.size[0]
    return affine_transform(pair, (1, 0, v, 0, 1, 0))


def TranslateY(pair, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    img, _ = pair
    v = v * img.size[1]
    return affine_transform(pair, (1, 0, 0, 0, 1, v))


def TranslateXAbs(pair, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    if random.random() > 0.5:
        v = -v
    return affine_transform(pair, (1, 0, v, 0, 1, 0))


def TranslateYAbs(pair, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    if random.random() > 0.5:
        v = -v
    return affine_transform(pair, (1, 0, 0, 0, 1, v))


def Rotate(pair, v):  # [-30, 30]
    assert -30 <= v <= 30
    if random.random() > 0.5:
        v = -v
    img, mask = pair
    img = img.rotate(v, fillcolor=fillcolor)
    mask = mask.rotate(v, resample=Image.NEAREST, fillcolor=fillmask)
    return img, mask


def AutoContrast(pair, _):
    img, mask = pair
    return ImageOps.autocontrast(img), mask


def Invert(pair, _):
    img, mask = pair
    return ImageOps.invert(img), mask


def Equalize(pair, _):
    img, mask = pair
    return ImageOps.equalize(img), mask


def Flip(pair, _):  # not from the paper
    img, mask = pair
    return ImageOps.mirror(img), ImageOps.mirror(mask)


def Solarize(pair, v):  # [0, 256]
    img, mask = pair
    assert 0 <= v <= 256
    return ImageOps.solarize(img, v), mask


def Posterize(pair, v):  # [4, 8]
    img, mask = pair
    assert 4 <= v <= 8
    v = int(v)
    return ImageOps.posterize(img, v), mask


def Posterize2(pair, v):  # [0, 4]
    img, mask = pair
    assert 0 <= v <= 4
    v = int(v)
    return ImageOps.posterize(img, v), mask


def Contrast(pair, v):  # [0.1,1.9]
    img, mask = pair
    assert 0.1 <= v <= 1.9
    return ImageEnhance.Contrast(img).enhance(v), mask


def Color(pair, v):  # [0.1,1.9]
    img, mask = pair
    assert 0.1 <= v <= 1.9
    return ImageEnhance.Color(img).enhance(v), mask


def Brightness(pair, v):  # [0.1,1.9]
    img, mask = pair
    assert 0.1 <= v <= 1.9
    return ImageEnhance.Brightness(img).enhance(v), mask


def Sharpness(pair, v):  # [0.1,1.9]
    img, mask = pair
    assert 0.1 <= v <= 1.9
    return ImageEnhance.Sharpness(img).enhance(v), mask


def Cutout(pair, v):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return pair
    img, mask = pair
    v = v * img.size[0]
    return CutoutAbs(img, v), mask


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    ImageDraw.Draw(img).rectangle(xy, color)
    return img


def Identity(pair, v):
    return pair


def augment_list():  # 16 oeprations and their ranges
    # https://github.com/google-research/uda/blob/master/image/randaugment/policies.py#L57
    l = [
        (Identity, 0., 1.0),
        (ShearX, 0., 0.3),  # 0
        (ShearY, 0., 0.3),  # 1
        (TranslateX, 0., 0.33),  # 2
        (TranslateY, 0., 0.33),  # 3
        (Rotate, 0, 30),  # 4
        (AutoContrast, 0, 1),  # 5
        (Invert, 0, 1),  # 6
        (Equalize, 0, 1),  # 7
        (Solarize, 0, 110),  # 8
        (Posterize, 4, 8),  # 9
        # (Contrast, 0.1, 1.9),  # 10
        (Color, 0.1, 1.9),  # 11
        (Brightness, 0.1, 1.9),  # 12
        (Sharpness, 0.1, 1.9),  # 13
        # (Cutout, 0, 0.2),  # 14
        # (SamplePairing(imgs), 0, 0.4),  # 15
        # (Flip, 1, 1),
    ]
    return l


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class RandAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m      # [0, 30]
        self.augment_list = augment_list()

    def __call__(self, img, mask):
        pair = img, mask
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            pair = op(pair, val)

        return pair
