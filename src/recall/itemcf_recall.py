"""ItemCF collaborative filtering recall."""

from __future__ import annotations

import math

import pandas as pd

from src.recall import format_recall_result, get_interacted_posts


def build_positive_interactions(behaviors: pd.DataFrame) -> pd.DataFrame:
    """Keep positive feedback and normalize behavior weights."""

    rows = behaviors.copy()
    rows = rows[rows["is_positive"].astype(int).eq(1)]
    rows["action_weight"] = pd.to_numeric(rows["action_weight"], errors="coerce").fillna(1.0)
    rows["interaction_weight"] = rows["action_weight"].clip(lower=1.0)
    return rows[["user_id", "post_id", "interaction_weight"]]


def build_item_similarity(positive_interactions: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Calculate item-item similarity from user co-occurrence."""

    user_posts = positive_interactions.groupby("user_id")[["post_id", "interaction_weight"]].apply(
        lambda df: list(zip(df["post_id"].astype(str), df["interaction_weight"].astype(float)))
    )
    item_norm: dict[str, float] = {}
    co_counts: dict[str, dict[str, float]] = {}

    for interactions in user_posts:
        if not interactions:
            continue
        user_norm = math.log1p(len(interactions))
        for post_i, weight_i in interactions:
            item_norm[post_i] = item_norm.get(post_i, 0.0) + weight_i * weight_i
            co_counts.setdefault(post_i, {})
            for post_j, weight_j in interactions:
                if post_i == post_j:
                    continue
                co_counts[post_i][post_j] = co_counts[post_i].get(post_j, 0.0) + (weight_i * weight_j) / user_norm

    similarity: dict[str, dict[str, float]] = {}
    for post_i, related in co_counts.items():
        similarity[post_i] = {}
        for post_j, score in related.items():
            denominator = math.sqrt(item_norm.get(post_i, 1.0) * item_norm.get(post_j, 1.0))
            similarity[post_i][post_j] = score / denominator if denominator > 0 else 0.0
    return similarity


def itemcf_recall(
    user_id: str,
    behaviors: pd.DataFrame,
    posts: pd.DataFrame,
    top_k: int = 50,
    item_similarity: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Recall posts similar to the user's positive interacted posts."""

    positive = build_positive_interactions(behaviors)
    if positive.empty:
        return pd.DataFrame(columns=["user_id", "post_id", "recall_score", "recall_source"])
    if item_similarity is None:
        item_similarity = build_item_similarity(positive)

    valid_post_ids = set(posts["post_id"].astype(str))
    seen = get_interacted_posts(behaviors, user_id)
    user_positive = positive[positive["user_id"].astype(str).eq(str(user_id))]
    if user_positive.empty:
        return pd.DataFrame(columns=["user_id", "post_id", "recall_score", "recall_source"])

    scores: dict[str, float] = {}
    for row in user_positive.itertuples(index=False):
        source_post = str(row.post_id)
        source_weight = float(row.interaction_weight)
        for candidate_post, sim in item_similarity.get(source_post, {}).items():
            if candidate_post in seen or candidate_post not in valid_post_ids:
                continue
            scores[candidate_post] = scores.get(candidate_post, 0.0) + sim * source_weight

    if not scores:
        return pd.DataFrame(columns=["user_id", "post_id", "recall_score", "recall_source"])
    candidates = pd.DataFrame({"post_id": list(scores.keys()), "itemcf_score": list(scores.values())})
    return format_recall_result(user_id, candidates, "itemcf", "itemcf_score", top_k)
