from __future__ import annotations

from pathlib import Path
from typing import List

from ultralytics import YOLO

from src.utils.logger import configure_logger


class Detector:
    def __init__(
        self,
        weights: str,
        imgsz: int = 1280,
        conf: float = 0.25,
        iou: float = 0.45,
        device: str | int = "cpu",
    ) -> None:
        self.model = YOLO(weights)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device
        self.logger = configure_logger("detector")

    def predict(self, source: str | Path):
        return self.model(
            source,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

    def predict_batch(self, sources: List[str | Path]):
        outputs = []
        for src in sources:
            outputs.append(self.predict(src))
        return outputs

