"""Registers datasets.
"""
from pathlib import Path

from detectron2.data.datasets import register_coco_instances

# Define dataset paths
data_dir = Path("data")

benign_data_dir = data_dir / "benign_data"
benign_img_dir = benign_data_dir / "benign_database"

benign_train_coco_path = benign_data_dir / "coco_train.json"
benign_test_coco_path = benign_data_dir / "coco_test.json"

# Register benign train and test sets
register_coco_instances("benign_train", {}, benign_train_coco_path, benign_img_dir)
register_coco_instances("benign_test", {}, benign_test_coco_path, benign_img_dir)
