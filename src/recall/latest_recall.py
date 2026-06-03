"""Latest post recall."""

from __future__ import annotations

import pandas as pd

from src.recall import filter_valid_posts, format_recall_result


def latest_recall(
    user_id: str,
    posts: pd.DataFrame,
    top_k: int = 50,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Recall recent posts by ``latest_score = 1 / (1 + post_age_hours)``."""

    if now is None:
        now = pd.Timestamp.now()
    candidates = filter_valid_posts(posts)
    candidates["publish_time"] = pd.to_datetime(candidates["publish_time"], errors="coerce")
    if "post_age_hours" not in candidates.columns:
        candidates["post_age_hours"] = (now - candidates["publish_time"]).dt.total_seconds().clip(lower=0) / 3600
    candidates["post_age_hours"] = pd.to_numeric(candidates["post_age_hours"], errors="coerce").fillna(0.0)
    candidates["latest_score"] = 1.0 / (1.0 + candidates["post_age_hours"])
    return format_recall_result(user_id, candidates, "latest", "latest_score", top_k)
