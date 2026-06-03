"""Data loading utilities for the campus forum recommender.

The project now uses normalized CSV tables. ``load_all_data`` reads the
standard tables required by preprocessing and feature engineering.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class CampusData:
    users: pd.DataFrame
    posts: pd.DataFrame
    post_stats: pd.DataFrame
    comments: pd.DataFrame
    tags: pd.DataFrame
    post_tags: pd.DataFrame
    behaviors: pd.DataFrame
    questionnaire: pd.DataFrame
    user_profiles: pd.DataFrame


def read_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV file with a Chinese-friendly default encoding."""

    return pd.read_csv(path, encoding="utf-8-sig")


def read_optional_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV file if it exists; otherwise return an empty dataframe."""

    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame()
    return read_csv(csv_path)


def load_all_data(data_dir: str | Path = DATA_DIR) -> CampusData:
    """Load the normalized campus forum data tables.

    Required tables:
    users.csv, posts.csv, post_stats.csv, comments.csv, tags.csv,
    post_tags.csv, user_behaviors.csv, questionnaire.csv.

    For compatibility with the previous step, ``behaviors.csv`` is accepted
    when ``user_behaviors.csv`` has not been generated yet.
    """

    data_path = Path(data_dir)
    behavior_path = data_path / "user_behaviors.csv"
    if not behavior_path.exists() and (data_path / "behaviors.csv").exists():
        behavior_path = data_path / "behaviors.csv"

    required_files = {
        "users": data_path / "users.csv",
        "posts": data_path / "posts.csv",
        "post_stats": data_path / "post_stats.csv",
        "comments": data_path / "comments.csv",
        "tags": data_path / "tags.csv",
        "post_tags": data_path / "post_tags.csv",
        "behaviors": behavior_path,
        "questionnaire": data_path / "questionnaire.csv",
    }
    missing = [str(path) for path in required_files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing data files. Run `python scripts/mock_data.py` first: "
            + ", ".join(missing)
        )

    return CampusData(
        users=read_csv(required_files["users"]),
        posts=read_csv(required_files["posts"]),
        post_stats=read_csv(required_files["post_stats"]),
        comments=read_csv(required_files["comments"]),
        tags=read_csv(required_files["tags"]),
        post_tags=read_csv(required_files["post_tags"]),
        behaviors=read_csv(required_files["behaviors"]),
        questionnaire=read_csv(required_files["questionnaire"]),
        user_profiles=read_optional_csv(data_path / "user_profiles.csv"),
    )


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    """Save a dataframe as UTF-8 with BOM, so Excel/WPS can read Chinese text."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
