#!/usr/bin/env bash

    # Example on Mapillary
     python -m torch.distributed.launch --nproc_per_node=8 train.py \
        --dataset mapillary \
        --arch network.deepv3.DeepWV3Plus \
        --class_uniform_pct 0.5 \
        --class_uniform_tile 1024 \
        --syncbn \
        --sgd \
        --lr 2e-2 \
        --lr_schedule poly \
        --poly_exp 1.0 \
        --crop_size 896 \
        --scale_min 0.5 \
        --scale_max 2.0 \
        --color_aug 0.25 \
        --gblur \
        --max_epoch 175 \
        --img_wt_loss \
        --wt_bound 6.0 \
        --bs_mult 2 \
        --apex \
        --exp mapillary_pretrain \
        --ckpt ./logs/ \
        --tb_path ./logs/ 
