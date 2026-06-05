"""Content-based recall using TF-IDF post vectors."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from src.recall import format_recall_result, get_interacted_posts


def build_user_content_vector(
    user_id: str,
    behaviors: pd.DataFrame,
    post_text_matrix: sparse.csr_matrix,
    post_id_to_index: dict[str, int],
) -> sparse.csr_matrix | None:
    """Build one user content vector from positive feedback posts."""

    user_rows = behaviors[behaviors["user_id"].astype(str).eq(str(user_id))].copy()
    user_rows = user_rows[user_rows["is_positive"].astype(int).eq(1)]
    if user_rows.empty:
        return None

    vectors = []
    weights = []
    for row in user_rows.itertuples(index=False):
        post_id = str(row.post_id)
        if post_id not in post_id_to_index:
            continue
        action_weight = float(getattr(row, "action_weight", 1.0) or 1.0)
        dwell_time = float(getattr(row, "dwell_time", 0.0) or 0.0)
        weight = max(action_weight, 1.0) * (1.0 + np.log1p(max(dwell_time, 0.0)) / 10.0)
        vectors.append(post_text_matrix[post_id_to_index[post_id]])
        weights.append(weight)

    if not vectors:
        return None
    stacked = sparse.vstack(vectors)
    weight_array = np.array(weights, dtype=float)
    return sparse.csr_matrix(stacked.multiply(weight_array.reshape(-1, 1)).sum(axis=0) / weight_array.sum())


def content_recall(
    user_id: str,
    posts: pd.DataFrame,
    behaviors: pd.DataFrame,
    post_text_matrix: sparse.csr_matrix,
    post_id_to_index: dict[str, int],
    top_k: int = 50,
    exclude_interacted: bool = True,
) -> pd.DataFrame:
    """Recall posts similar to the user's positive-history content vector."""

    user_vector = build_user_content_vector(user_id, behaviors, post_text_matrix, post_id_to_index)
    if user_vector is None:
        return pd.DataFrame(columns=["user_id", "post_id", "recall_score", "recall_source"])

    candidates = posts.drop_duplicates("post_id", keep="last").copy()
    if exclude_interacted:
        seen = get_interacted_posts(behaviors, user_id)
        candidates = candidates[~candidates["post_id"].astype(str).isin(seen)]
    candidates = candidates[candidates["post_id"].astype(str).isin(post_id_to_index)]
    if candidates.empty:
        return pd.DataFrame(columns=["user_id", "post_id", "recall_score", "recall_source"])

    candidate_indices = [post_id_to_index[str(post_id)] for post_id in candidates["post_id"]]
    sims = cosine_similarity(user_vector, post_text_matrix[candidate_indices]).ravel()
    candidates = candidates.copy()
    candidates["content_score"] = sims.clip(min=0)
    return format_recall_result(user_id, candidates, "content", "content_score", top_k)
