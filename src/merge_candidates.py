"""Candidate merging for multi-channel recall results."""

from __future__ import annotations

import pandas as pd


MERGED_COLUMNS = [
    "user_id",
    "post_id",
    "merged_recall_score",
    "recall_sources",
    "source_count",
    "hot_score",
    "final_hot_score",
    "board",
    "tags",
    "topic_type",
    "publish_time",
    "post_age_hours",
]

DEFAULT_RECALL_WEIGHTS = {
    "hot": 0.8,
    "latest": 0.7,
    "content": 1.0,
    "itemcf": 1.0,
    "profile": 0.9,
    "time_scene": 0.7,
}


def minmax_normalize(group: pd.DataFrame) -> pd.DataFrame:
    """Normalize recall_score within one recall channel."""

    group = group.copy()
    values = pd.to_numeric(group["recall_score"], errors="coerce").fillna(0.0)
    min_value = values.min()
    max_value = values.max()
    if max_value == min_value:
        group["normalized_recall_score"] = 1.0
    else:
        group["normalized_recall_score"] = (values - min_value) / (max_value - min_value)
    return group


def prepare_recall_results(recall_results: list[pd.DataFrame]) -> pd.DataFrame:
    """Concat recall results and validate the unified schema."""

    frames = [df.copy() for df in recall_results if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame(columns=["user_id", "post_id", "recall_score", "recall_source"])
    merged = pd.concat(frames, ignore_index=True)
    required = {"user_id", "post_id", "recall_score", "recall_source"}
    missing = required - set(merged.columns)
    if missing:
        raise ValueError(f"Recall result missing columns: {sorted(missing)}")
    merged["user_id"] = merged["user_id"].astype(str)
    merged["post_id"] = merged["post_id"].astype(str)
    merged["recall_source"] = merged["recall_source"].astype(str)
    merged["recall_score"] = pd.to_numeric(merged["recall_score"], errors="coerce").fillna(0.0)
    return merged


def build_post_feature_table(posts: pd.DataFrame) -> pd.DataFrame:
    """Build post metadata table for candidate merging."""

    post_features = posts.drop_duplicates("post_id", keep="last").copy()
    for column in ["hot_score", "final_hot_score", "post_age_hours"]:
        post_features[column] = pd.to_numeric(post_features[column], errors="coerce").fillna(0.0)
    return post_features[
        [
            "post_id",
            "hot_score",
            "final_hot_score",
            "board",
            "tags",
            "topic_type",
            "publish_time",
            "post_age_hours",
        ]
    ].copy()


def merge_recall_candidates(
    recall_results: list[pd.DataFrame],
    posts: pd.DataFrame,
    recall_weights: dict[str, float] | None = None,
    source_count_bonus_weight: float = 0.05,
    top_k_candidates: int = 200,
) -> pd.DataFrame:
    """Merge multi-channel recall outputs into a deduplicated candidate set."""

    recall_weights = recall_weights or DEFAULT_RECALL_WEIGHTS
    raw = prepare_recall_results(recall_results)
    if raw.empty:
        return pd.DataFrame(columns=MERGED_COLUMNS)

    raw = pd.concat(
        [minmax_normalize(group) for _, group in raw.groupby("recall_source", sort=False)],
        ignore_index=True,
    )
    raw["source_weight"] = raw["recall_source"].map(recall_weights).fillna(0.5)
    raw["weighted_score"] = raw["normalized_recall_score"] * raw["source_weight"]

    grouped = (
        raw.groupby(["user_id", "post_id"])
        .agg(
            weighted_score=("weighted_score", "sum"),
            recall_sources=("recall_source", lambda values: ",".join(sorted(set(values)))),
            source_count=("recall_source", lambda values: len(set(values))),
        )
        .reset_index()
    )
    grouped["merged_recall_score"] = grouped["weighted_score"] + source_count_bonus_weight * grouped["source_count"]
    grouped = grouped.drop(columns=["weighted_score"])

    post_features = build_post_feature_table(posts)
    candidates = grouped.merge(post_features, on="post_id", how="inner")
    candidates = candidates.sort_values("merged_recall_score", ascending=False).head(top_k_candidates)

    for column in MERGED_COLUMNS:
        if column not in candidates.columns:
            candidates[column] = 0 if column in {"merged_recall_score", "source_count", "hot_score", "final_hot_score", "post_age_hours"} else ""
    return candidates[MERGED_COLUMNS].reset_index(drop=True)


def merge_candidates(*args, **kwargs) -> pd.DataFrame:
    """Backward-compatible alias."""

    return merge_recall_candidates(*args, **kwargs)
