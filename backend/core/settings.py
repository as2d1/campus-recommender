"""Backend settings."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

APP_NAME = "Campus Forum Recommender Backend"
API_PREFIX = "/api"

CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
