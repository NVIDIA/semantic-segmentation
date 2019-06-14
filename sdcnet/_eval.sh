#!/usr/bin/env bash
# Run SDC2DRecon on Cityscapes dataset

# Root folder of cityscapes images
VAL_FILE=~/data/tmp/tinycs
SDC2DREC_CHECKPOINT=../pretrained_models/sdc_cityscapes_vrec.pth.tar
FLOWNET2_CHECKPOINT=../pretrained_models/FlowNet2_checkpoint.pth.tar

python3 main.py \
    --eval \
    --sequence_length 2 \
    --save ./ \
    --name __evalrun \
    --val_n_batches 1 \
    --write_images \
    --dataset FrameLoader \
    --model SDCNet2DRecon \
    --val_file ${VAL_FILE} \
    --resume ${SDC2DREC_CHECKPOINT} \
    --flownet2_checkpoint ${FLOWNET2_CHECKPOINT}

