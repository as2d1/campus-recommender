"""Latest post recall."""

from __future__ import annotations

import pandas as pd

from src.recall import format_recall_result


def latest_recall(
    user_id: str,
    posts: pd.DataFrame,
    top_k: int = 50,
) -> pd.DataFrame:
    """Recall recent posts by ``latest_score = 1 / (1 + post_age_hours)``."""

    candidates = posts.drop_duplicates("post_id", keep="last").copy()
    candidates["latest_score"] = 1.0 / (1.0 + candidates["post_age_hours"])
    return format_recall_result(user_id, candidates, "latest", "latest_score", top_k)
