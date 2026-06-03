"""Hot post recall."""

from __future__ import annotations

import pandas as pd

from src.recall import filter_valid_posts, format_recall_result


def hot_recall(
    user_id: str,
    posts: pd.DataFrame,
    post_stats: pd.DataFrame,
    top_k: int = 50,
) -> pd.DataFrame:
    """Recall globally hot posts by ``final_hot_score``."""

    valid_posts = filter_valid_posts(posts)
    stats = post_stats.copy()
    if "final_hot_score" not in stats.columns:
        stats["final_hot_score"] = 0.0
    candidates = valid_posts.copy()
    if "final_hot_score" not in candidates.columns:
        candidates = candidates.merge(
            stats[["post_id", "final_hot_score"]].drop_duplicates("post_id", keep="last"),
            on="post_id",
            how="left",
        )
    candidates["final_hot_score"] = pd.to_numeric(candidates["final_hot_score"], errors="coerce").fillna(0.0)
    return format_recall_result(user_id, candidates, "hot", "final_hot_score", top_k)
