#!/usr/bin/env bash

# Omniglot 5-way with MT-net
python main.py \
    --datasource=omniglot --metatrain_iterations=40000 \
    --meta_batch_size=32 --update_batch_size=1\
    --num_classes=5 --num_updates=1 --logdir=logs/omniglot20way \
    --update_lr=.4 --use_T=True --use_M=True --share_M=True

# Omniglot 20-way with MT-net
python main.py \
    --datasource=omniglot --metatrain_iterations=40000 \
    --meta_batch_size=16 --update_batch_size=1\
    --num_classes=20 --num_updates=1 --logdir=logs/omniglot20way \
    --update_lr=.1 --use_T=True --use_M=True --share_M=True
