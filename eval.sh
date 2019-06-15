#!/usr/bin/env bash
echo "Running inference on" ${1}
echo "Saving Results :" ${2}
PYTHONPATH=$PWD:$PYTHONPATH python3 scripts/eval.py \
    --arch network.deepv3.DeepWV3Plus \
    --scales 1.0 \
    --split val \
    --cv_split 2 \
    --dump_images \
    --ckpt_path ${2} \
    --snapshot ${1}
