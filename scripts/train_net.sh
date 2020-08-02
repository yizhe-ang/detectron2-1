#!/bin/bash

# Training
# CUDA_VISIBLE_DEVICES=0 python train_net_wandb.py \
#     --config-file configs/faster_rcnn_bet365.yaml \
#     --exp-name rcnn_bet365 \
#     MODEL.WEIGHTS output/rcnn_2_resume/model_0004999.pth \
#     OUTPUT_DIR output/rcnn_bet365

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
# python train_net.py \
#     --eval-only \
#     --config-file output/eval/retina/config.yaml \
#     OUTPUT_DIR output/eval/retina \
#     MODEL.WEIGHTS output/retina/model_0067499.pth

# Visualize Predictions
# python visualize_json_results.py \
#     --input output/eval/eval_annos_yolo.json \
#     --output saved/yolo \
#     --dataset benign_eval \
#     --conf-threshold 0.05

# Run DAG
python run_DAG.py \
    --cfg-path output/rcnn_2/config.yaml \
    --weights-path output/rcnn_2_resume/model_0004999.pth \
    --results-save-path coco.json \
    --vis-save-dir saved
