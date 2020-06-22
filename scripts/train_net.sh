#!/bin/bash

# Training
# CUDA_VISIBLE_DEVICES=0 python train_net_wandb.py \
#     --config-file configs/faster_rcnn.yaml \
#     --exp-name rcnn_2_resume \
#     OUTPUT_DIR output/rcnn_2_resume \
#     MODEL.WEIGHTS output/rcnn_2/model_final.pth

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

# Evaluation
python train_net.py \
    --eval-only \
    --config-file output/eval/rcnn/config.yaml \
    OUTPUT_DIR output/eval/rcnn \
    MODEL.WEIGHTS output/rcnn_2/model_final.pth