# Getting Started

# Installation
```
git clone https://github.com/lemonwaffle/detectron2-1.git
```
To install Detectron2 and its dependencies, refer to the official [installation instructions](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).

# Directory
Make sure the data files are organized as follows:
```
├── GETTING_STARTED.md
├── LICENSE
├── README.md
├── configs  # Contains config files that control training parameters
├── data
│   └── benign_data
│       ├── benign_database  # Contains all the images
│       ├── coco_test.json  # Test annotations in COCO format
│       └── coco_train.json  # Train annotations in COCO format
├── detectron2_1
│   ├── __init__.py
│   └── datasets.py  # Registers train and test datasets
├── requirements.txt
└── train_net.py  # Main entry point for model training
```

# Config
Each training run is completely defined by customizable parameters in its configuration file, with a few templates already specified in the [configs](./configs) folder.

For example, all the existing config files train the models with pretrained COCO weights:
- `cascade_mask_rcnn.yaml`: Cascade Mask R-CNN model with ResNet50 backbone.
- `faster_rcnn.yaml`: Faster R-CNN model with ResNet50 backbone.
- `retinanet.yaml`: RetinaNet model with ResNet50 backbone.

Other types of models and their respective configs and pretrained weights can be found in the official Detectron2 [Model Zoo](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md).

While you can refer to the [config reference](https://detectron2.readthedocs.io/modules/config.html#config-references) for a full list of available parameters and what they mean, I've annotated some of them in the existing configs, and some notable ones to customize are:
- `SOLVER.IMS_PER_BATCH`: Batch size
- `SOLVER.BASE_LR`: Base learning rate
- `SOLVER.STEPS`: The iteration number to decrease learning rate by GAMMA
- `SOLVER.MAX_ITER`: Total number of training iterations
- `SOLVER.CHECKPOINT_PERIOD`: Saves checkpoint every number of steps
- `INPUT.MIN_SIZE_TRAIN`: Image input sizes
- `TEST.EVAL_PERIOD`: The period (in terms of steps) to evaluate the model during training
- `OUTPUT_DIR`: Specify output directory to save checkpoints, logs, results etc.

# Training
To train on a single gpu:
```
python train_net.py \
    --config-file configs/retinanet.yaml
```

To train on multiple gpus:
```
python train_net.py \
    --num-gpus 4 \
    --config-file configs/retinanet.yaml
```

To resume training from a checkpoint (finds last checkpoint from cfg.OUTPUT_DIR)
```
python train_net.py \
    --config-file config.yaml \  # Config file of halted run
    --resume
```

To see all options:
```
python train_net.py -h
```

# Visualization
## Training Examples
To visualize annotations of raw training images:
```
python visualize_data.py \
    --config-file config.yaml \  # Uses dataset specified in cfg.DATASETS.TRAIN
    --source annotation \
    --num-imgs 1000 \  # Num of images to save; don't specify if want to save all
    --output-dir /dir/to/save/images
```

To visualize annotations of training images after preprocessing/augs:
```
python visualize_data.py \
    --config-file config.yaml \  # Uses dataset specified in cfg.DATASETS.TRAIN
    --source dataloader \
    --num-imgs 1000 \  # Num of images to save; don't specify if want to save all
    --output-dir /dir/to/save/images
```

## Predictions
To visualize JSON predictions:
```
python visualize_json_results.py \
    --input coco_instances_results.json \  # JSON file produced by model
    --output /dir/to/save/images \
    --dataset benign_test \  # Name of dataset
    --conf-threshold 0.5  # Default 0.5
```

## Training Curves
Losses and metrics are automatically logged to TensorBoard.
```
tensorboard --logdir {path}
```

# Evaluation
This command only runs evaluation on the test dataset:
```
python train_net.py \
    --eval-only \
    --config-file configs/retinanet.yaml \
    MODEL.WEIGHTS /path/to/checkpoint_file  # Path to trained checkpoint
```

# Dataset Details
29k images