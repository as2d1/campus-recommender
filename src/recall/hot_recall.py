"""Hot post recall."""

from __future__ import annotations

import pandas as pd

from src.recall import format_recall_result


def hot_recall(
    user_id: str,
    posts: pd.DataFrame,
    top_k: int = 50,
) -> pd.DataFrame:
    """Recall globally hot posts by ``final_hot_score``."""

    candidates = posts.drop_duplicates("post_id", keep="last").copy()
    return format_recall_result(user_id, candidates, "hot", "final_hot_score", top_k)
