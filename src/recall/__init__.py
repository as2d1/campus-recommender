"""Recall channel utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


RECALL_COLUMNS = ["user_id", "post_id", "recall_score", "recall_source"]
POSITIVE_ACTIONS = {"view", "like", "comment", "collect"}


def split_tags(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [tag.strip() for tag in str(value).split("|") if tag.strip()]


def filter_valid_posts(posts: pd.DataFrame) -> pd.DataFrame:
    """Filter deleted, blocked, or abnormal posts."""

    valid = posts.copy()
    if "status" in valid.columns:
        valid = valid[valid["status"].fillna("normal").eq("normal")]
    if "report_status" in valid.columns:
        valid = valid[~valid["report_status"].fillna("normal").isin(["blocked", "deleted", "abnormal", "违规", "suspect"])]
    if "finish_status" in valid.columns:
        valid = valid[~valid["finish_status"].fillna("open").isin(["blocked", "deleted"])]
    return valid.drop_duplicates("post_id", keep="last").reset_index(drop=True)


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

    if behaviors.empty:
        return set()
    user_rows = behaviors[behaviors["user_id"].astype(str).eq(str(user_id))]
    if positive_only:
        if "is_positive" in user_rows.columns:
            user_rows = user_rows[user_rows["is_positive"].fillna(0).astype(int).eq(1)]
        else:
            user_rows = user_rows[user_rows["action_type"].isin(POSITIVE_ACTIONS)]
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
