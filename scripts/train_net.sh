#!/bin/bash

# Training
CUDA_VISIBLE_DEVICES=0 python train_net_wandb.py \
    --config-file configs/faster_rcnn.yaml \
    --exp-name rcnn_2 \
    OUTPUT_DIR output/rcnn_2

# Resume training
# CUDA_VISIBLE_DEVICES=0 python train_net_wandb.py \
#     --config-file output/rcnn_1/config.yaml \
#     --exp-name rcnn_1 \
#     --resume \
#     OUTPUT_DIR output/rcnn_1

# Visualize preprocessed images
# python visualize_data.py \
#     --config-file configs/faster_rcnn.yaml \
#     --source dataloader \
#     --num-imgs 100 \
#     --output-dir saved