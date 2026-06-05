"""Configuration for the campus recommender."""

from __future__ import annotations

from pathlib import Path

from src.taxonomy import ALLOWED_BOARDS as ALL_BOARDS
from src.taxonomy import BOARD_DEFAULT_TAGS, BOARD_TOPIC_MAP

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DEFAULT_ZANAO_INPUT = RAW_DATA_DIR / "zanao_100_posts.jsonl"
DEFAULT_DATASET_OUTPUT = DATA_DIR
DEFAULT_SQLITE_PATH = PROJECT_ROOT.parent / "zanao.sqlite"

DEFAULT_NUM_POSTS = 2000
DEFAULT_NUM_USERS = 500
DEFAULT_NUM_BEHAVIORS = 30000
DEFAULT_NUM_COMMENTS = 6000
DEFAULT_NUM_QUESTIONNAIRES = 150
DEFAULT_RANDOM_SEED = 2026

ACTION_WEIGHTS = {
    "expose": 0,
    "skip": -1,
    "view": 1,
    "like": 2,
    "comment": 3,
    "collect": 4,
    "dislike": -2,
    "report": -4,
}
