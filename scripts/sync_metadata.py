from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.utils.logger import configure_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge detection rows with GPS metadata.")
    parser.add_argument("--detections", required=True, help="CSV from tracking pipeline.")
    parser.add_argument("--gps", required=True, help="CSV with timestamp, lat, lon columns.")
    parser.add_argument("--output", default="data/exports/detections_enriched.csv")
    return parser.parse_args()


def load_gps(path: Path) -> dict:
    gps_map = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            gps_map[row["timestamp"]] = row
    return gps_map


def main() -> None:
    args = parse_args()
    logger = configure_logger("sync_metadata")
    detections_path = Path(args.detections)
    gps_path = Path(args.gps)

    gps_map = load_gps(gps_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with detections_path.open("r", encoding="utf-8") as det_handle, output_path.open(
        "w", newline="", encoding="utf-8"
    ) as out_handle:
        reader = csv.DictReader(det_handle)
        fieldnames = list(reader.fieldnames or []) + ["lat", "lon"]
        writer = csv.DictWriter(out_handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            timestamp = row.get("timestamp")
            gps_row = gps_map.get(timestamp, {})
            row["lat"] = gps_row.get("lat")
            row["lon"] = gps_row.get("lon")
            writer.writerow(row)

    logger.info("Enriched detections saved -> %s", output_path)


if __name__ == "__main__":
    main()

