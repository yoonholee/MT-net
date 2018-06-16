#!/usr/bin/env bash

for lr in .01 .04 .1
do
python main.py \
    --datasource=sinusoid --metatrain_iterations=60000 \
    --meta_batch_size=4 --update_lr=$lr --norm=None --resume=True \
    --update_batch_size=10 --use_T=True --use_M=True --share_M=True \
    --logdir=logs/sine
done

# For example, to use T-net:
# --use_T=True --use_M=False --share_M=False
#
# Original MAML is recovered by using:
# --use_T=False --use_M=False --share_M=False
