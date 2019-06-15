#!/usr/bin/env bash

    # Example on Cityscapes
     python -m torch.distributed.launch --nproc_per_node=8 scripts/train.py --dataset cityscapes \
        --arch network.deepv3.DeepSRNX50V3PlusD_m1 \
        --snapshot ~/nfs-mount/results/best_mapillary_ResNextSE50_50239.pth \
        --gblur \
        --rlx_off_epoch 100 \
        --class_uniform_pct 0.5 \
        --scale_max 2.0 \
        --repoly 1.5  \
        --class_uniform_tile 1024 \
        --syncbn \
        --lr_schedule scl-poly \
        --tb_path /tmp/ \
        --sgd \
        --crop_size 896 \
        --scale_min 0.5 \
        --color_aug 0.25 \
        --poly_exp 1.0 \
        --max_epoch 175 \
        --coarse_boost_classes 14,15,16,3,12,17,4 \
        --rescale 1.0 \
        --ckpt /tmp/ \
        --jointwtborder \
        --cv 2 \
        --strict_bdr_cls 5,6,7,11,12,17,18 \
        --lr 0.001 \
        --wt_bound 1.0 \
        --bs_mult 2 \
        --apex \
        --max_cu_epoch 150 \
        --fp16
