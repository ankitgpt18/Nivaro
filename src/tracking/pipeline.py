from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Dict, List, Optional

from ultralytics import YOLO

from src.config import load_config
from src.utils.logger import configure_logger

MetadataFn = Callable[[int], Dict[str, float | int | str]]


class TrackingPipeline:
    """Wrapper around Ultralytics YOLO tracking for consistent logging."""

    def __init__(
        self,
        detector_weights: str,
        tracker_config: str = "configs/tracking.yaml",
        imgsz: int = 1280,
        conf: float = 0.25,
        iou: float = 0.45,
        device: str | int = "cpu",
    ) -> None:
        self.logger = configure_logger("tracking")
        self.model = YOLO(detector_weights)
        self.cfg = load_config(tracker_config)
        self.tracker_cfg = self.cfg["tracker"]
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device

    def _result_to_rows(
        self,
        result,
        frame_index: int,
        metadata: Optional[Dict[str, float | int | str]] = None,
    ) -> List[Dict[str, float | int | str]]:
        rows: List[Dict[str, float | int | str]] = []
        boxes = result.boxes
        if boxes is None:
            return rows

        xyxy = boxes.xyxy.cpu().tolist()
        confs = boxes.conf.cpu().tolist()
        cls_ids = boxes.cls.cpu().tolist()
        track_ids = boxes.id.cpu().tolist() if boxes.id is not None else [None] * len(xyxy)

        for idx, bbox in enumerate(xyxy):
            row: Dict[str, float | int | str] = {
                "frame": frame_index,
                "track_id": int(track_ids[idx]) if track_ids[idx] is not None else -1,
                "class_id": int(cls_ids[idx]),
                "confidence": float(confs[idx]),
                "xmin": float(bbox[0]),
                "ymin": float(bbox[1]),
                "xmax": float(bbox[2]),
                "ymax": float(bbox[3]),
                "source": result.path,
            }
            if metadata:
                row.update(metadata)
            rows.append(row)
        return rows

    def run(
        self,
        source: str,
        output_csv: str | Path | None = None,
        metadata_fn: Optional[MetadataFn] = None,
    ) -> List[Dict[str, float | int | str]]:
        tracker_type = self.tracker_cfg.get("type", "bytetrack")
        tracker_yaml = f"{tracker_type}.yaml" if tracker_type.endswith(".yaml") is False else tracker_type
        rows: List[Dict[str, float | int | str]] = []

        results = self.model.track(
            source=source,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            tracker=tracker_yaml,
            device=self.device,
            stream=True,
            persist=True,
        )

        for frame_index, result in enumerate(results):
            metadata = metadata_fn(frame_index) if metadata_fn else None
            rows.extend(self._result_to_rows(result, frame_index, metadata))

        if output_csv:
            csv_path = Path(output_csv)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else [])
                if rows:
                    writer.writeheader()
                    writer.writerows(rows)
            self.logger.info("Saved tracking log -> %s", csv_path)

        return rows

