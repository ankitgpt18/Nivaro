from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass
class LabelIssue:
    file: Path
    line_number: int
    message: str


@dataclass
class SplitStats:
    split: str
    class_name: str
    images: int = 0
    labels: int = 0
    missing_labels: int = 0
    invalid_labels: int = 0
    class_counts: Dict[int, int] = field(default_factory=dict)


def list_image_files(path: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([p for p in path.glob("*") if p.suffix.lower() in exts])


def expected_label_path(image_path: Path, label_dir: Path) -> Path:
    return label_dir / f"{image_path.stem}.txt"


def parse_yolo_line(line: str, min_columns: int = 5) -> Tuple[int, List[float]]:
    parts = line.strip().split()
    if len(parts) < min_columns:
        raise ValueError(f"Expected >= {min_columns} columns, got {len(parts)}")
    class_id = int(parts[0])
    values = [float(value) for value in parts[1:5]]
    return class_id, values


def validate_label_file(
    label_file: Path,
    class_ids: Iterable[int],
    min_columns: int = 5,
) -> Tuple[List[LabelIssue], Dict[int, int]]:
    issues: List[LabelIssue] = []
    counts: Dict[int, int] = {}
    valid_classes = set(class_ids)

    with label_file.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                class_id, _ = parse_yolo_line(stripped, min_columns=min_columns)
            except ValueError as exc:
                issues.append(LabelIssue(label_file, idx, str(exc)))
                continue

            if class_id not in valid_classes:
                issues.append(
                    LabelIssue(
                        label_file,
                        idx,
                        f"class_id {class_id} not present in dataset config",
                    )
                )
                continue

            counts[class_id] = counts.get(class_id, 0) + 1

    return issues, counts


def collect_split_stats(
    dataset_root: Path,
    class_name: str,
    split: str,
    image_subdir: str,
    label_subdir: str,
    class_map: Dict[str, int],
    min_columns: int = 5,
) -> SplitStats:
    image_dir = dataset_root / class_name / split / image_subdir
    label_dir = dataset_root / class_name / split / label_subdir
    stats = SplitStats(split=split, class_name=class_name)

    if not image_dir.exists():
        return stats

    images = list_image_files(image_dir)
    stats.images = len(images)

    label_dir.mkdir(parents=True, exist_ok=True)
    for image_path in images:
        label_path = expected_label_path(image_path, label_dir)
        if not label_path.exists():
            stats.missing_labels += 1
            continue

        stats.labels += 1
        issues, counts = validate_label_file(
            label_path,
            class_ids=class_map.values(),
            min_columns=min_columns,
        )
        stats.invalid_labels += len(issues)
        for cls_id, amount in counts.items():
            stats.class_counts[cls_id] = stats.class_counts.get(cls_id, 0) + amount

    return stats


def summarize_dataset(
    dataset_root: Path,
    classes: Sequence[str],
    splits: Sequence[str],
    image_subdir: str,
    label_subdir: str,
    class_map: Dict[str, int],
    min_columns: int = 5,
) -> List[SplitStats]:
    summaries: List[SplitStats] = []
    for cls in classes:
        for split in splits:
            summaries.append(
                collect_split_stats(
                    dataset_root=dataset_root,
                    class_name=cls,
                    split=split,
                    image_subdir=image_subdir,
                    label_subdir=label_subdir,
                    class_map=class_map,
                    min_columns=min_columns,
                )
            )
    return summaries

