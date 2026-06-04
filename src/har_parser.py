"""HAR parser interface for converting raw forum traffic to normalized tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


STANDARD_TABLES = [
    "users",
    "posts",
    "post_stats",
    "comments",
    "tags",
    "post_tags",
    "user_behaviors",
    "questionnaire",
]


def empty_standard_tables() -> dict[str, pd.DataFrame]:
    """Return empty normalized tables with stable keys."""

    return {name: pd.DataFrame() for name in STANDARD_TABLES}


def parse_har_to_tables(har_path: str | Path) -> dict[str, pd.DataFrame]:
    """Parse a HAR file into normalized tables.

    This placeholder keeps the expected parser contract. It can be expanded to
    parse category lists, thread lists, hot threads, thread details, comments,
    and hot tags from real HAR files.
    """

    if not Path(har_path).exists():
        raise FileNotFoundError(f"HAR file not found: {har_path}")
    return empty_standard_tables()
