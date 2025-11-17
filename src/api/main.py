from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from src.config import load_config
from src.detection.predictor import Detector
from src.tracking.pipeline import TrackingPipeline
from src.utils.logger import configure_logger

app = FastAPI(title="Nivaro Civic Issue Detection API")
logger = configure_logger("api")
cfg = load_config("configs/api.yaml")

detector = Detector(
    weights=cfg["inference"]["model_weights"],
    imgsz=cfg["inference"]["imgsz"],
    conf=cfg["inference"]["conf"],
    iou=cfg["inference"]["iou"],
    device=cfg["inference"]["device"],
)
tracker = TrackingPipeline(
    detector_weights=cfg["inference"]["model_weights"],
    tracker_config=cfg["inference"]["tracker_config"],
    imgsz=cfg["inference"]["imgsz"],
    conf=cfg["inference"]["conf"],
    iou=cfg["inference"]["iou"],
    device=cfg["inference"]["device"],
)


@app.get("/health")
async def healthcheck() -> dict:
    return {"status": "ok"}


def _format_detection(results) -> List[dict]:
    formatted: List[dict] = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        xyxy = boxes.xyxy.cpu().tolist()
        confs = boxes.conf.cpu().tolist()
        cls_ids = boxes.cls.cpu().tolist()
        for idx, bbox in enumerate(xyxy):
            formatted.append(
                {
                    "class_id": int(cls_ids[idx]),
                    "confidence": float(confs[idx]),
                    "bbox": {
                        "xmin": float(bbox[0]),
                        "ymin": float(bbox[1]),
                        "xmax": float(bbox[2]),
                        "ymax": float(bbox[3]),
                    },
                }
            )
    return formatted


@app.post("/detect")
async def detect_endpoint(file: UploadFile = File(...)) -> JSONResponse:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    results = detector.predict(tmp_path)
    formatted = _format_detection(results)
    tmp_path.unlink(missing_ok=True)
    return JSONResponse({"detections": formatted})


@app.post("/track")
async def track_endpoint(file: UploadFile = File(...)) -> JSONResponse:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    rows = tracker.run(str(tmp_path))
    tmp_path.unlink(missing_ok=True)
    return JSONResponse({"tracks": rows})

