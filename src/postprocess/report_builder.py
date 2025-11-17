from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

import folium
from shapely.geometry import Point, mapping

from src.config import load_config
from src.utils.logger import configure_logger


class ReportBuilder:
    def __init__(self, config_path: str = "configs/reporting.yaml") -> None:
        self.cfg = load_config(config_path)["report"]
        self.logger = configure_logger("report")

    def _dedupe(self, rows: Iterable[Dict], distance_m: float) -> List[Dict]:
        deduped: List[Dict] = []
        for row in rows:
            lat = row.get("lat")
            lon = row.get("lon")
            if lat is None or lon is None:
                deduped.append(row)
                continue
            point = Point(float(lon), float(lat))
            duplicate = False
            for existing in deduped:
                existing_point = Point(float(existing["lon"]), float(existing["lat"]))
                if existing_point.distance(point) * 111_139 <= distance_m:
                    duplicate = True
                    break
            if not duplicate:
                deduped.append(row)
        return deduped

    def _write_csv(self, rows: List[Dict]) -> Path:
        csv_path = Path(self.cfg["output_csv"])
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)
        return csv_path

    def _write_geojson(self, rows: List[Dict]) -> Path:
        geojson_path = Path(self.cfg["output_geojson"])
        geojson_path.parent.mkdir(parents=True, exist_ok=True)
        features = []
        for row in rows:
            lat = row.get("lat")
            lon = row.get("lon")
            if lat is None or lon is None:
                continue
            point = Point(float(lon), float(lat))
            features.append(
                {
                    "type": "Feature",
                    "geometry": mapping(point),
                    "properties": row,
                }
            )
        geojson = {"type": "FeatureCollection", "features": features}
        geojson_path.write_text(json.dumps(geojson, indent=2), encoding="utf-8")
        return geojson_path

    def _write_map(self, rows: List[Dict]) -> Path:
        html_path = Path(self.cfg["map"]["output_html"])
        html_path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            html_path.write_text("<p>No rows to map.</p>", encoding="utf-8")
            return html_path
        center_lat = float(rows[0].get("lat", 0))
        center_lon = float(rows[0].get("lon", 0))
        fmap = folium.Map(location=[center_lat, center_lon], zoom_start=14)
        for row in rows:
            lat = row.get("lat")
            lon = row.get("lon")
            if lat is None or lon is None:
                continue
            popup = f\"\"\"{row.get('class_id', 'unknown')} | conf={row.get('confidence', 0):.2f}\"\"\"
            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                fill=True,
                popup=popup,
            ).add_to(fmap)
        fmap.save(html_path)
        return html_path

    def build(self, rows: List[Dict]) -> Dict[str, Path]:
        if not rows:
            raise ValueError("No detection rows supplied.")

        dedupe_distance = float(self.cfg.get("dedupe_distance_m", 5))
        processed = self._dedupe(rows, dedupe_distance)
        outputs = {
            "csv": self._write_csv(processed),
            "geojson": self._write_geojson(processed),
        }
        if self.cfg.get("map", {}).get("enabled", False):
            outputs["map_html"] = self._write_map(processed)
        self.logger.info("Generated report artifacts: %s", outputs)
        return outputs

