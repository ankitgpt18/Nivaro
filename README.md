# Nivaro Civic Issue Detection

End-to-end computer vision solution for Vadodara Municipal Corporation to detect potholes, stray cattle, and garbage from bike-mounted camera streams, track every occurrence, and log GPS/time metadata for dispatch teams.

## Structure
- `src/` core Python packages (data, training, tracking, reporting, API)
- `configs/` Hydra/YAML configs for training, inference, reporting
- `data/` raw/processed datasets plus exports
- `scripts/` automation utilities (verification, preparation, validation)
- `notebooks/` exploratory analysis and evaluation reports
- `experiments/` training logs, checkpoints, WANDB runs
- `models/` frozen weights for deployment
- `reports/` generated evaluation, maps, and client documents
- `docs/` deep project documentation

## Quickstart
1. `python -m venv .venv`
2. `.\\.venv\\Scripts\\activate` (PowerShell)
3. `pip install -r requirements.txt`
4. Organize labelled data under `data/dataset/<class>/<split>/{images,labels}`
5. `python scripts/verify_dataset.py --config configs/dataset.yaml`
6. `python scripts/prepare_yolo_dataset.py --strategy copy`
7. `python -m src.training.train --config configs/train.yaml`

## Targets
- mAP50 ≥ 0.5 per class, recall ≥ 0.6
- BYTETrack ID persistence ≥ 90% across clips
- CSV + GeoJSON logging (timestamp, GPS, severity) for each detection
- FastAPI microservice for bike/edge ingestion and civic dashboards

