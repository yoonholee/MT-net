#!/usr/bin/env bash

# miniImagenet with MT-nets and hyperparameters from MAML
python main.py \
    --datasource=miniimagenet --metatrain_iterations=60000 \
    --meta_batch_size=4 --update_batch_size=1 \
    --num_updates=5 --logdir=logs/miniimagenet5way \
    --update_lr=.01 --resume=True --num_filters=32 --max_pool=True \
    --use_T=True --use_M=True --share_M=True

# works well even with single gradient step
python main.py \
    --datasource=miniimagenet --metatrain_iterations=60000 \
    --meta_batch_size=4 --update_batch_size=1 \
    --num_updates=1 --logdir=logs/miniimagenet5way \
    --update_lr=.4 --resume=True --num_filters=32 --max_pool=True \
    --use_T=True --use_M=True --share_M=True
