#!/usr/bin/env bash
echo "Running inference on" ${1}
echo "Saving Results :" ${2}
PYTHONPATH=$PWD:$PYTHONPATH python3 eval.py \
    --dataset kitti \
    --arch network.deepv3.DeepWV3Plus \
    --mode semantic \
    --split test \
    --inference_mode sliding \
    --cv_split 0 \
    --scales 1.5,2.0,2.5 \
    --crop_size 368 \
    --dump_images \
    --snapshot ${1} \
    --ckpt_path ${2}
