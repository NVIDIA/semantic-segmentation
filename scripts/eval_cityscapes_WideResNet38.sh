#!/usr/bin/env bash
echo "Running inference on" ${1}
echo "Saving Results :" ${2}
sleep 10
PYTHONPATH=$PWD:$PYTHONPATH python eval.py \
	--dataset cityscapes \
    --arch network.deepv3.DeepWV3Plus \
    --inference_mode sliding \
    --scales 0.5,1.0,2.0 \
    --split train \
    --cv_split $3 \
    --dump_images \
    --ckpt_path ${2} \
    --snapshot ${1}
