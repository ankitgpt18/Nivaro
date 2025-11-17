from __future__ import annotations

import argparse

from ultralytics import YOLO

from src.config import load_config
from src.utils.logger import configure_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run validation on trained weights.")
    parser.add_argument("--train-config", default="configs/train.yaml", help="Training config path.")
    parser.add_argument("--weights", default="models/best.pt", help="Weights to evaluate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.train_config)
    data_config = cfg["paths"]["data_config"]
    logger = configure_logger("validate")

    model = YOLO(args.weights)
    metrics = model.val(data=data_config)
    logger.info("Validation metrics: %s", metrics)


if __name__ == "__main__":
    main()

