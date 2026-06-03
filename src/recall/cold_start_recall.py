"""Cold-start recall."""

from __future__ import annotations

import pandas as pd

from src.recall import filter_valid_posts, format_recall_result, split_tags
from src.recall.hot_recall import hot_recall
from src.recall.latest_recall import latest_recall
from src.recall.scene_recall import time_scene_recall


def questionnaire_recall(
    user_id: str,
    questionnaire: pd.DataFrame,
    posts: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    """Recall by selected tags and boards from a new-user questionnaire."""

    rows = questionnaire[questionnaire["user_id"].astype(str).eq(str(user_id))]
    if rows.empty:
        return pd.DataFrame(columns=["user_id", "post_id", "recall_score", "recall_source"])
    q = rows.iloc[0]
    selected_tags = set(split_tags(q.get("selected_tags", "")))
    selected_boards = set(split_tags(q.get("selected_boards", "")))

    candidates = filter_valid_posts(posts).copy()
    scores = []
    for _, post in candidates.iterrows():
        post_tags = set(split_tags(post.get("tags", "")))
        tag_score = len(selected_tags & post_tags) / max(len(selected_tags | post_tags), 1)
        board_score = 0.6 if str(post.get("board", "")) in selected_boards else 0.0
        scores.append(tag_score + board_score)
    candidates["cold_start_score"] = scores
    candidates = candidates[candidates["cold_start_score"] > 0]
    return format_recall_result(user_id, candidates, "cold_start", "cold_start_score", top_k)


def fallback_cold_start_recall(
    user_id: str,
    posts: pd.DataFrame,
    post_stats: pd.DataFrame,
    top_k: int,
    time_period: str,
    scene: str,
) -> pd.DataFrame:
    """Fallback for new users without questionnaire."""

    channels = [
        hot_recall(user_id, posts, post_stats, top_k=top_k),
        latest_recall(user_id, posts, top_k=top_k),
        time_scene_recall(user_id, posts, time_period=time_period, scene=scene, top_k=top_k),
    ]
    merged = pd.concat(channels, ignore_index=True)
    if merged.empty:
        return pd.DataFrame(columns=["user_id", "post_id", "recall_score", "recall_source"])
    merged["recall_score"] = pd.to_numeric(merged["recall_score"], errors="coerce").fillna(0.0)
    merged = (
        merged.groupby(["user_id", "post_id"], as_index=False)["recall_score"]
        .max()
        .sort_values("recall_score", ascending=False)
        .head(top_k)
    )
    merged["recall_source"] = "cold_start"
    return merged[["user_id", "post_id", "recall_score", "recall_source"]].reset_index(drop=True)


def cold_start_recall(
    user_id: str,
    users: pd.DataFrame,
    questionnaire: pd.DataFrame,
    posts: pd.DataFrame,
    post_stats: pd.DataFrame,
    top_k: int = 50,
    time_period: str = "晚上",
    scene: str = "普通浏览",
) -> pd.DataFrame:
    """Cold-start recall for new users."""

    user_rows = users[users["user_id"].astype(str).eq(str(user_id))]
    if user_rows.empty:
        return pd.DataFrame(columns=["user_id", "post_id", "recall_score", "recall_source"])
    user = user_rows.iloc[0]
    is_new_user = int(user.get("is_new_user", 0)) == 1
    questionnaire_filled = int(user.get("questionnaire_filled", 0)) == 1
    if not is_new_user:
        return pd.DataFrame(columns=["user_id", "post_id", "recall_score", "recall_source"])

    if questionnaire_filled:
        result = questionnaire_recall(user_id, questionnaire, posts, top_k=top_k)
        if not result.empty:
            return result

    return fallback_cold_start_recall(
        user_id=user_id,
        posts=posts,
        post_stats=post_stats,
        top_k=top_k,
        time_period=time_period,
        scene=scene,
    )
