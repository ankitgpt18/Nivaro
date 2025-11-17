from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from tqdm import tqdm

from src.config import load_config
from src.data.dataset_utils import expected_label_path, list_image_files
from src.utils.logger import configure_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flatten class-wise dataset structure into YOLO-compatible splits."
    )
    parser.add_argument(
        "--config",
        default="configs/dataset.yaml",
        help="Dataset configuration file.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/yolo",
        help="Destination root for consolidated dataset.",
    )
    parser.add_argument(
        "--strategy",
        choices=["copy", "symlink"],
        default="copy",
        help="Copy files (default) or create symlinks (faster, requires admin on Windows).",
    )
    return parser.parse_args()


def ensure_split_dirs(root: Path, split: str) -> tuple[Path, Path]:
    img_dir = root / split / "images"
    lbl_dir = root / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    return img_dir, lbl_dir


def transfer_file(src: Path, dst: Path, strategy: str) -> None:
    if dst.exists():
        return
    if strategy == "symlink":
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    logger = configure_logger("prepare_dataset")

    dataset_root = Path(cfg["dataset_root"])
    output_root = Path(args.output)
    classes = list(cfg["classes"].keys())
    splits = list(cfg["splits"].values())
    structure = cfg.get("structure", {})
    image_subdir = structure.get("image_subdir", "images")
    label_subdir = structure.get("label_subdir", "labels")

    total_pairs = 0
    for cls in classes:
        for split in splits:
            image_dir = dataset_root / cls / split / image_subdir
            label_dir = dataset_root / cls / split / label_subdir
            if not image_dir.exists():
                continue
            images = list_image_files(image_dir)
            total_pairs += len(images)
            target_img_dir, target_lbl_dir = ensure_split_dirs(output_root, split)

            for image_path in tqdm(
                images,
                desc=f"{cls}-{split}",
                unit="img",
                leave=False,
            ):
                label_path = expected_label_path(image_path, label_dir)
                if not label_path.exists():
                    logger.warning("Missing label for %s", image_path)
                    continue

                new_stem = f"{cls}_{image_path.stem}"
                dest_image = target_img_dir / f"{new_stem}{image_path.suffix.lower()}"
                dest_label = target_lbl_dir / f"{new_stem}.txt"
                transfer_file(image_path, dest_image, args.strategy)
                transfer_file(label_path, dest_label, args.strategy)

    logger.info(
        "Prepared dataset -> %s (strategy=%s, pairs=%d)",
        output_root,
        args.strategy,
        total_pairs,
    )


if __name__ == "__main__":
    main()

