"""Inference on a single image.
"""

from argparse import ArgumentParser

import cv2
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from PIL import Image

import detectron2_1


def inference(img_path, config_path, weights_path, output_path, conf_threshold=0.05):
    # Configure weights and confidence threshold
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_threshold

    # Initialize model
    predictor = DefaultPredictor(cfg)

    # Load image as numpy array
    im = cv2.imread(img_path)

    # Perform inference
    outputs = predictor(im)

    # Set dataset categories
    # FIXME Specifc to this task
    MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes = ["box", "logo"]

    # Draw instance predictions
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # Image with instance predictions as numpy array
    pred = out.get_image()

    # Save image with instance predictions
    Image.fromarray(pred).save(output_path)


def get_args():
    parser = ArgumentParser()

    parser.add_argument("--img-path", help="Path to image to perform inference on")
    parser.add_argument("--config-path", help="Path to config file of model")
    parser.add_argument("--weights-path", help="Path to model weights")
    parser.add_argument(
        "--output-path", help="Path to save image with instance predictions"
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default="0.05",
        help="Confidence threshold of predictions, default 0.05",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    inference(
        args.img_path,
        args.config_path,
        args.weights_path,
        args.output_path,
        args.conf_threshold,
    )
