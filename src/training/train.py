from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from src.config import load_config
from src.utils.logger import configure_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO detector for Nivaro.")
    parser.add_argument(
        "--config",
        default="configs/train.yaml",
        help="Training configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    logger = configure_logger("train")

    training_cfg = cfg["training"]
    paths_cfg = cfg["paths"]
    logging_cfg = cfg.get("logging", {})
    hardware_cfg = cfg.get("hardware", {})

    model = YOLO(training_cfg["model"])
    logger.info("Loaded model weights: %s", training_cfg["model"])

    train_results = model.train(
        data=paths_cfg["data_config"],
        epochs=training_cfg["epochs"],
        imgsz=training_cfg["imgsz"],
        batch=training_cfg["batch"],
        workers=training_cfg["workers"],
        patience=training_cfg["patience"],
        optimizer=training_cfg["optimizer"],
        lr0=training_cfg["lr0"],
        lrf=training_cfg["lrf"],
        momentum=training_cfg["momentum"],
        weight_decay=training_cfg["weight_decay"],
        warmup_epochs=training_cfg["warmup_epochs"],
        warmup_momentum=training_cfg["warmup_momentum"],
        warmup_bias_lr=training_cfg["warmup_bias_lr"],
        box=training_cfg["box"],
        cls=training_cfg["cls"],
        dfl=training_cfg["dfl"],
        project=paths_cfg["project_dir"],
        name=paths_cfg["name"],
        device=hardware_cfg.get("device", 0),
        deterministic=hardware_cfg.get("deterministic", False),
    )
    logger.info("Training complete: %s", train_results)

    if logging_cfg.get("use_wandb", False):
        try:
            import wandb  # type: ignore

            wandb.config.update(cfg)
        except ImportError:
            logger.warning("Weights & Biases not installed; skipping logging.")


if __name__ == "__main__":
    main()

