#!/usr/bin/env bash

python main.py \
    --datasource=polynomial --metatrain_iterations=60000 --update_batch_size=10 \
    --meta_batch_size=4 --norm=None --logdir=logs/poly --poly_order=0 \
    --use_T=True --use_M=True --share_M=True

python main.py \
    --datasource=polynomial --metatrain_iterations=60000 --update_batch_size=10 \
    --meta_batch_size=4 --norm=None --logdir=logs/poly --poly_order=1 \
    --use_T=True --use_M=True --share_M=True

python main.py \
    --datasource=polynomial --metatrain_iterations=60000 --update_batch_size=10 \
    --meta_batch_size=4 --norm=None --logdir=logs/poly --poly_order=2 \
    --use_T=True --use_M=True --share_M=True
