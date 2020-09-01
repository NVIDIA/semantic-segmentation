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

Dataset setup and loaders
"""

import importlib
import torchvision.transforms as standard_transforms

import transforms.joint_transforms as joint_transforms
import transforms.transforms as extended_transforms
from torch.utils.data import DataLoader

from config import cfg, update_dataset_cfg, update_dataset_inst
from runx.logx import logx
from datasets.randaugment import RandAugment


def setup_loaders(args):
    """
    Setup Data Loaders[Currently supports Cityscapes, Mapillary and ADE20kin]
    input: argument passed by the user
    return:  training data loader, validation data loader loader,  train_set
    """

    # TODO add error checking to make sure class exists
    logx.msg(f'dataset = {args.dataset}')

    mod = importlib.import_module('datasets.{}'.format(args.dataset))
    dataset_cls = getattr(mod, 'Loader')

    logx.msg(f'ignore_label = {dataset_cls.ignore_label}')

    update_dataset_cfg(num_classes=dataset_cls.num_classes,
                       ignore_label=dataset_cls.ignore_label)

    ######################################################################
    # Define transformations, augmentations
    ######################################################################

    # Joint transformations that must happen on both image and mask
    if ',' in args.crop_size:
        args.crop_size = [int(x) for x in args.crop_size.split(',')]
    else:
        args.crop_size = int(args.crop_size)
    train_joint_transform_list = [
        # TODO FIXME: move these hparams into cfg
        joint_transforms.RandomSizeAndCrop(args.crop_size,
                                           False,
                                           scale_min=args.scale_min,
                                           scale_max=args.scale_max,
                                           full_size=args.full_crop_training,
                                           pre_size=args.pre_size)]
    train_joint_transform_list.append(
        joint_transforms.RandomHorizontallyFlip())

    if args.rand_augment is not None:
        N, M = [int(i) for i in args.rand_augment.split(',')]
        assert isinstance(N, int) and isinstance(M, int), \
            f'Either N {N} or M {M} not integer'
        train_joint_transform_list.append(RandAugment(N, M))

    ######################################################################
    # Image only augmentations
    ######################################################################
    train_input_transform = []

    if args.color_aug:
        train_input_transform += [extended_transforms.ColorJitter(
            brightness=args.color_aug,
            contrast=args.color_aug,
            saturation=args.color_aug,
            hue=args.color_aug)]
    if args.bblur:
        train_input_transform += [extended_transforms.RandomBilateralBlur()]
    elif args.gblur:
        train_input_transform += [extended_transforms.RandomGaussianBlur()]

    mean_std = (cfg.DATASET.MEAN, cfg.DATASET.STD)
    train_input_transform += [standard_transforms.ToTensor(),
                              standard_transforms.Normalize(*mean_std)]
    train_input_transform = standard_transforms.Compose(train_input_transform)

    val_input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

    target_transform = extended_transforms.MaskToTensor()

    if args.jointwtborder:
        target_train_transform = \
            extended_transforms.RelaxedBoundaryLossToTensor()
    else:
        target_train_transform = extended_transforms.MaskToTensor()

    if args.eval == 'folder':
        val_joint_transform_list = None
    elif 'mapillary' in args.dataset:
        if args.pre_size is None:
            eval_size = 2177
        else:
            eval_size = args.pre_size
        if cfg.DATASET.MAPILLARY_CROP_VAL:
            val_joint_transform_list = [
                joint_transforms.ResizeHeight(eval_size),
                joint_transforms.CenterCropPad(eval_size)]
        else:
            val_joint_transform_list = [
                joint_transforms.Scale(eval_size)]
    else:
        val_joint_transform_list = None

    if args.eval is None or args.eval == 'val':
        val_name = 'val'
    elif args.eval == 'trn':
        val_name = 'train'
    elif args.eval == 'folder':
        val_name = 'folder'
    else:
        raise 'unknown eval mode {}'.format(args.eval)

    ######################################################################
    # Create loaders
    ######################################################################
    val_set = dataset_cls(
        mode=val_name,
        joint_transform_list=val_joint_transform_list,
        img_transform=val_input_transform,
        label_transform=target_transform,
        eval_folder=args.eval_folder)

    update_dataset_inst(dataset_inst=val_set)

    if args.apex:
        from datasets.sampler import DistributedSampler
        val_sampler = DistributedSampler(val_set, pad=False, permutation=False,
                                         consecutive_sample=False)
    else:
        val_sampler = None

    val_loader = DataLoader(val_set, batch_size=args.bs_val,
                            num_workers=args.num_workers // 2,
                            shuffle=False, drop_last=False,
                            sampler=val_sampler)

    if args.eval is not None:
        # Don't create train dataloader if eval
        train_set = None
        train_loader = None
    else:
        train_set = dataset_cls(
            mode='train',
            joint_transform_list=train_joint_transform_list,
            img_transform=train_input_transform,
            label_transform=target_train_transform)

        if args.apex:
            from datasets.sampler import DistributedSampler
            train_sampler = DistributedSampler(train_set, pad=True,
                                               permutation=True,
                                               consecutive_sample=False)
            train_batch_size = args.bs_trn
        else:
            train_sampler = None
            train_batch_size = args.bs_trn * args.ngpu

        train_loader = DataLoader(train_set, batch_size=train_batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=(train_sampler is None),
                                  drop_last=True, sampler=train_sampler)

    return train_loader, val_loader, train_set
