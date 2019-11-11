# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash
set -e -x

CODE=src
DATA=/media/owen/ba9d40b5-89de-4832-bad4-156b118e4a66/chickendata_out/preprocessed

EXP=chickenNet
export MASTER_PORT=29500

python3 ${CODE}/train_decoder.py \
    --data ${DATA}/chickendata \
    --checkpoint checkpoints/pretrained_musicnet/bestmodel_5.pth \
    --batch-size 2 \
    --lr-decay 0.995 \
    --epoch-len 1000 \
    --epochs 50 \
    --num-workers 1 \
    --lr 1e-3 \
    --seq-len 12000 \
    --expName ${EXP} \
    --latent-d 64 \
    --layers 14 \
    --blocks 4 \
    --data-aug \
    --grad-clip 1
