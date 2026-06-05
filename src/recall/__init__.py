"""Recall channel utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.taxonomy import split_tags


RECALL_COLUMNS = ["user_id", "post_id", "recall_score", "recall_source"]
POSITIVE_ACTIONS = {"view", "like", "comment", "collect"}


def normalize_score(series: pd.Series) -> pd.Series:
    """Min-max normalize a score series with a stable zero fallback."""

    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    min_value = values.min()
    max_value = values.max()
    if np.isclose(max_value, min_value):
        return pd.Series(np.ones(len(values)), index=series.index)
    return (values - min_value) / (max_value - min_value)


def get_interacted_posts(behaviors: pd.DataFrame, user_id: str, positive_only: bool = False) -> set[str]:
    """Return posts the user has interacted with."""

    user_rows = behaviors[behaviors["user_id"].astype(str).eq(str(user_id))]
    if positive_only:
        user_rows = user_rows[user_rows["is_positive"].astype(int).eq(1)]
    return set(user_rows["post_id"].astype(str))


def format_recall_result(user_id: str, rows: pd.DataFrame, source: str, score_column: str, top_k: int) -> pd.DataFrame:
    """Format recall output to the unified schema."""

    if rows.empty:
        return pd.DataFrame(columns=RECALL_COLUMNS)
    result = rows.copy()
    result["user_id"] = str(user_id)
    result["recall_score"] = pd.to_numeric(result[score_column], errors="coerce").fillna(0.0)
    result["recall_source"] = source
    result = result.sort_values("recall_score", ascending=False).head(top_k)
    return result[RECALL_COLUMNS].reset_index(drop=True)
