"""Time scene recall."""

from __future__ import annotations

import pandas as pd

from src.recall import filter_valid_posts, format_recall_result, split_tags


def time_scene_score(post: pd.Series, time_period: str) -> float:
    tags = set(split_tags(post.get("tags", "")))
    board = str(post.get("board", ""))
    text = f"{board}|{post.get('title', '')}|{post.get('content', '')}|{post.get('tags', '')}"
    score = 0.0
    if time_period == "中午" and (board in {"校园趣事", "打听求助"} or {"食堂", "拼饭"} & tags):
        score += 1.0
    if time_period == "晚上" and board in {"打听求助", "恋爱交友", "校园趣事"}:
        score += 0.9
    return score


def time_scene_recall(
    user_id: str,
    posts: pd.DataFrame,
    time_period: str = "晚上",
    top_k: int = 50,
) -> pd.DataFrame:
    """Recall posts suitable for current time period and campus scene."""

    candidates = filter_valid_posts(posts).copy()
    candidates["time_scene_score"] = candidates.apply(lambda post: time_scene_score(post, time_period), axis=1)
    candidates = candidates[candidates["time_scene_score"] > 0]
    return format_recall_result(user_id, candidates, "time_scene", "time_scene_score", top_k)
