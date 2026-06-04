"""Time and location scene recall."""

from __future__ import annotations

import pandas as pd

from src.recall import filter_valid_posts, format_recall_result, split_tags


def time_scene_score(post: pd.Series, time_period: str, scene: str) -> float:
    tags = set(split_tags(post.get("tags", "")))
    board = str(post.get("board", ""))
    text = f"{board}|{post.get('title', '')}|{post.get('content', '')}|{post.get('tags', '')}"
    score = 0.0
    if time_period == "中午" and (board in {"校园趣事", "打听求助"} or {"食堂", "拼饭"} & tags):
        score += 1.0
    if time_period == "晚上" and board in {"打听求助", "恋爱交友", "校园趣事"}:
        score += 0.9
    if scene == "考试周" and (board == "打听求助" or any(key in text for key in ["复习资料", "课程重点", "考试经验"])):
        score += 1.1
    if scene == "开学季" and (board in {"打听求助", "校园趣事"} or any(key in text for key in ["选课经验", "新生攻略", "招新"])):
        score += 1.0
    if scene == "毕业季" and (board in {"校园招聘", "兼职招聘", "二手闲置"} or any(key in text for key in ["就业", "租房", "毕业"])):
        score += 1.0
    return score


def location_scene_score(post: pd.Series, location: str) -> float:
    tags = set(split_tags(post.get("tags", "")))
    board = str(post.get("board", ""))
    scope = str(post.get("location_scope", ""))
    score = 0.25 if scope == "全校" or scope == location else 0.0
    if location == "食堂区" and board in {"校园趣事", "打听求助"}:
        score += 1.0
    if location == "教学区" and (board == "打听求助" or {"考试", "课程资料"} & tags):
        score += 1.0
    if location == "宿舍区" and (board in {"二手闲置", "校园趣事", "打听求助"} or {"租房", "宿舍"} & tags):
        score += 1.0
    if location == "图书馆" and (board == "打听求助" or {"课程资料", "复习资料"} & tags):
        score += 1.0
    if location == "校外" and (board in {"兼职招聘", "校园招聘"} or {"租房", "兼职"} & tags):
        score += 1.0
    return score


def time_scene_recall(
    user_id: str,
    posts: pd.DataFrame,
    time_period: str = "晚上",
    scene: str = "普通浏览",
    top_k: int = 50,
) -> pd.DataFrame:
    """Recall posts suitable for current time period and campus scene."""

    candidates = filter_valid_posts(posts).copy()
    candidates["time_scene_score"] = candidates.apply(lambda post: time_scene_score(post, time_period, scene), axis=1)
    candidates = candidates[candidates["time_scene_score"] > 0]
    return format_recall_result(user_id, candidates, "time_scene", "time_scene_score", top_k)


def location_scene_recall(
    user_id: str,
    posts: pd.DataFrame,
    location: str = "教学区",
    top_k: int = 50,
) -> pd.DataFrame:
    """Recall posts matching the user's simulated campus location."""

    candidates = filter_valid_posts(posts).copy()
    candidates["location_scene_score"] = candidates.apply(lambda post: location_scene_score(post, location), axis=1)
    candidates = candidates[candidates["location_scene_score"] > 0]
    return format_recall_result(user_id, candidates, "location_scene", "location_scene_score", top_k)
