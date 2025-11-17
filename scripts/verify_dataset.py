from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.config import load_config
from src.data.dataset_utils import SplitStats, summarize_dataset
from src.utils.logger import configure_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate dataset integrity.")
    parser.add_argument(
        "--config",
        default="configs/dataset.yaml",
        help="Path to dataset configuration file.",
    )
    parser.add_argument(
        "--output",
        default="reports/dataset_audit.csv",
        help="Destination CSV file for summary stats.",
    )
    return parser.parse_args()


def stats_to_rows(stats: list[SplitStats]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in stats:
        rows.append(
            {
                "split": item.split,
                "class": item.class_name,
                "images": str(item.images),
                "labels": str(item.labels),
                "missing_labels": str(item.missing_labels),
                "invalid_labels": str(item.invalid_labels),
            }
        )
    return rows


def print_summary(stats: list[SplitStats]) -> None:
    total_images = sum(item.images for item in stats)
    total_missing = sum(item.missing_labels for item in stats)
    total_invalid = sum(item.invalid_labels for item in stats)
    logger = configure_logger("verify_dataset")

    logger.info("Dataset summary:")
    for item in stats:
        logger.info(
            "[%s | %s] images=%d labels=%d missing=%d invalid=%d",
            item.split,
            item.class_name,
            item.images,
            item.labels,
            item.missing_labels,
            item.invalid_labels,
        )

    logger.info(
        "Totals -> images=%d missing=%d invalid=%d",
        total_images,
        total_missing,
        total_invalid,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    dataset_root = Path(cfg["dataset_root"])
    classes = list(cfg["classes"].keys())
    splits = list(cfg["splits"].values())
    structure = cfg.get("structure", {})
    image_subdir = structure.get("image_subdir", "images")
    label_subdir = structure.get("label_subdir", "labels")
    class_map = cfg["classes"]
    min_columns = cfg.get("label_format", {}).get("min_columns", 5)

    stats = summarize_dataset(
        dataset_root=dataset_root,
        classes=classes,
        splits=splits,
        image_subdir=image_subdir,
        label_subdir=label_subdir,
        class_map=class_map,
        min_columns=min_columns,
    )
    print_summary(stats)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "class", "images", "labels", "missing_labels", "invalid_labels"],
        )
        writer.writeheader()
        for row in stats_to_rows(stats):
            writer.writerow(row)


if __name__ == "__main__":
    main()

