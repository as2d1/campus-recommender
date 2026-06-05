"""Preprocess normalized campus forum recommendation data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data_loader import CampusData
from src.taxonomy import topic_for_board


ACTION_WEIGHTS = {
    "expose": 0,
    "skip": -1,
    "view": 1,
    "like": 2,
    "comment": 3,
    "collect": 4,
    "dislike": -2,
    "report": -4,
}
STRONG_POSITIVE_ACTIONS = {"like", "comment", "collect"}
WEAK_POSITIVE_DWELL_SECONDS = 30
SHORT_DWELL_SECONDS = 8

@dataclass(frozen=True)
class PreprocessResult:
    users: pd.DataFrame
    posts: pd.DataFrame
    post_stats: pd.DataFrame
    comments: pd.DataFrame
    tags: pd.DataFrame
    post_tags: pd.DataFrame
    behaviors: pd.DataFrame
    preferences: pd.DataFrame
    user_profiles: pd.DataFrame
    samples: pd.DataFrame
    train_samples: pd.DataFrame
    test_samples: pd.DataFrame


def get_time_period(timestamp: pd.Timestamp) -> str:
    if pd.isna(timestamp):
        return "未知"
    hour = int(timestamp.hour)
    if 5 <= hour < 11:
        return "早上"
    if 11 <= hour < 14:
        return "中午"
    if 14 <= hour < 18:
        return "下午"
    if 18 <= hour < 23:
        return "晚上"
    return "深夜"


def normalize_board_and_topic(posts: pd.DataFrame) -> pd.DataFrame:
    posts = posts.copy()
    posts["board"] = posts["board"].astype(str)
    mapped_topic = posts["board"].map(topic_for_board)
    posts["topic_type"] = posts["topic_type"].where(posts["topic_type"].astype(str).str.len() > 0, mapped_topic)
    posts["topic_type"] = posts["topic_type"].fillna(mapped_topic)
    return posts


def parse_time_columns(data: CampusData) -> CampusData:
    users = data.users.copy()
    posts = data.posts.copy()
    post_stats = data.post_stats.copy()
    comments = data.comments.copy()
    behaviors = data.behaviors.copy()
    user_profiles = data.user_profiles.copy()

    posts["publish_time"] = pd.to_datetime(posts["publish_time"], errors="coerce")
    post_stats["update_time"] = pd.to_datetime(post_stats["update_time"], errors="coerce")
    comments["publish_time"] = pd.to_datetime(comments["publish_time"], errors="coerce")
    behaviors["timestamp"] = pd.to_datetime(behaviors["timestamp"], errors="coerce")
    if not user_profiles.empty:
        user_profiles["last_update_time"] = pd.to_datetime(user_profiles["last_update_time"], errors="coerce")

    return CampusData(
        users=users,
        posts=posts,
        post_stats=post_stats,
        comments=comments,
        tags=data.tags.copy(),
        post_tags=data.post_tags.copy(),
        behaviors=behaviors,
        preferences=data.preferences.copy(),
        user_profiles=user_profiles,
    )


def clean_users(users: pd.DataFrame) -> pd.DataFrame:
    users = users.copy()
    users = users[users["status"].eq("normal")]
    return users.drop_duplicates("user_id", keep="last").reset_index(drop=True)


def prepare_post_stats(post_stats: pd.DataFrame, posts: pd.DataFrame) -> pd.DataFrame:
    stats = post_stats.copy()
    count_columns = [
        "view_count",
        "like_count",
        "comment_count",
        "collect_count",
        "dislike_count",
        "hot_val",
        "hot_rank",
    ]
    for column in count_columns:
        stats[column] = pd.to_numeric(stats[column], errors="coerce").fillna(0)

    stats = stats.drop_duplicates("post_id", keep="last")
    stats = stats.merge(posts[["post_id", "post_age_hours"]], on="post_id", how="right")
    stats[count_columns] = stats[count_columns].fillna(0)
    stats["hot_score"] = (
        stats["view_count"] * 0.2
        + stats["like_count"] * 0.3
        + stats["comment_count"] * 0.3
        + stats["collect_count"] * 0.2
    )
    stats["final_hot_score"] = stats["hot_score"] / (1 + stats["post_age_hours"].fillna(0) / 24)
    stats["quality_score"] = (
        stats["like_count"] * 0.35
        + stats["comment_count"] * 0.30
        + stats["collect_count"] * 0.35
    ) / stats["view_count"].replace(0, np.nan)
    stats["quality_score"] = stats["quality_score"].fillna(0).clip(0, 1)
    return stats.drop(columns=["post_age_hours"], errors="ignore").reset_index(drop=True)


def clean_posts(posts: pd.DataFrame, post_stats: pd.DataFrame, now: pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if now is None:
        now = pd.Timestamp.now()

    posts = posts.copy()
    posts[["title", "content", "tags", "topic_type"]] = posts[["title", "content", "tags", "topic_type"]].fillna("")
    posts = normalize_board_and_topic(posts)
    posts["publish_time"] = posts["publish_time"].fillna(now)
    posts["post_age_hours"] = ((now - posts["publish_time"]).dt.total_seconds().clip(lower=0) / 3600)
    posts["content_length"] = posts["content"].fillna("").astype(str).str.len()
    posts["img_count"] = pd.to_numeric(posts["img_count"], errors="coerce").fillna(0).astype(int)
    posts["has_image"] = (pd.to_numeric(posts["has_image"], errors="coerce").fillna(0).astype(int) | (posts["img_count"] > 0)).astype(int)

    posts = posts.drop_duplicates("post_id", keep="last").reset_index(drop=True)

    post_stats = prepare_post_stats(post_stats, posts)
    stat_columns = [
        "view_count",
        "like_count",
        "dislike_count",
        "comment_count",
        "collect_count",
        "hot_val",
        "update_time",
        "hot_rank",
        "hot_score",
        "final_hot_score",
        "quality_score",
    ]
    posts = posts.drop(columns=stat_columns, errors="ignore")
    posts = posts.merge(post_stats, on="post_id", how="left")
    return posts, post_stats


def clean_behaviors(behaviors: pd.DataFrame, users: pd.DataFrame, posts: pd.DataFrame) -> pd.DataFrame:
    behaviors = behaviors.copy()
    behaviors = behaviors.dropna(subset=["user_id", "post_id", "timestamp"])
    behaviors = behaviors[behaviors["user_id"].isin(set(users["user_id"]))]
    behaviors = behaviors[behaviors["post_id"].isin(set(posts["post_id"]))]
    behaviors["dwell_time"] = pd.to_numeric(behaviors["dwell_time"], errors="coerce").fillna(0).astype(int)
    behaviors["action_weight"] = behaviors["action_type"].map(ACTION_WEIGHTS).fillna(0).astype(int)
    behaviors["time_period"] = behaviors["time_period"].where(
        behaviors["time_period"].astype(str).str.len() > 0,
        behaviors["timestamp"].apply(get_time_period),
    )
    behaviors["is_positive"] = behaviors.apply(label_behavior, axis=1)
    behaviors = behaviors.drop_duplicates(
        subset=["user_id", "post_id", "action_type", "timestamp"],
        keep="last",
    )
    return behaviors.sort_values("timestamp").reset_index(drop=True)


def label_behavior(row: pd.Series) -> int:
    action = str(row["action_type"])
    dwell_time = int(row["dwell_time"])
    if action in STRONG_POSITIVE_ACTIONS:
        return 1
    if action == "view" and dwell_time >= WEAK_POSITIVE_DWELL_SECONDS:
        return 1
    if action in {"skip", "dislike", "report"} or dwell_time <= SHORT_DWELL_SECONDS:
        return 0
    return 0


def build_behavior_samples(behaviors: pd.DataFrame) -> pd.DataFrame:
    samples = behaviors[behaviors["action_type"].isin({"view", "like", "comment", "collect", "skip", "dislike", "report"})].copy()
    samples["label"] = samples["is_positive"].astype(int)
    samples["sample_source"] = "behavior"
    return samples[
        [
            "user_id",
            "post_id",
            "timestamp",
            "action_type",
            "dwell_time",
            "time_period",
            "label",
            "sample_source",
        ]
    ]


def negative_sampling(
    users: pd.DataFrame,
    posts: pd.DataFrame,
    behaviors: pd.DataFrame,
    positive_count: int,
    negative_ratio: int = 4,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    all_posts = np.array(posts["post_id"].unique())
    interacted = behaviors.groupby("user_id")["post_id"].apply(set).to_dict()
    latest_time = behaviors["timestamp"].max()
    if pd.isna(latest_time):
        latest_time = pd.Timestamp.now()
    total_negative = positive_count * negative_ratio
    per_user = max(1, int(np.ceil(total_negative / max(len(users), 1))))

    rows: list[dict[str, object]] = []
    for user in users.itertuples(index=False):
        seen = interacted.get(user.user_id, set())
        candidates = np.array([post_id for post_id in all_posts if post_id not in seen])
        if len(candidates) == 0:
            continue
        sample_size = min(per_user, len(candidates))
        for post_id in rng.choice(candidates, size=sample_size, replace=False):
            rows.append(
                {
                    "user_id": user.user_id,
                    "post_id": post_id,
                    "timestamp": latest_time + pd.Timedelta(seconds=1),
                    "action_type": "negative_sample",
                    "dwell_time": 0,
                    "time_period": get_time_period(latest_time),
                    "label": 0,
                    "sample_source": "negative_sampling",
                }
            )
            if len(rows) >= total_negative:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def build_labeled_samples(
    users: pd.DataFrame,
    posts: pd.DataFrame,
    behaviors: pd.DataFrame,
    negative_ratio: int = 4,
    random_state: int = 42,
) -> pd.DataFrame:
    behavior_samples = build_behavior_samples(behaviors)
    positive_count = int(behavior_samples["label"].sum())
    sampled_negatives = negative_sampling(
        users=users,
        posts=posts,
        behaviors=behaviors,
        positive_count=max(positive_count, 1),
        negative_ratio=negative_ratio,
        random_state=random_state,
    )
    samples = pd.concat([behavior_samples, sampled_negatives], ignore_index=True)
    samples = samples.drop_duplicates(subset=["user_id", "post_id", "label"], keep="last")
    return samples.sort_values("timestamp").reset_index(drop=True)


def split_train_test_by_time(samples: pd.DataFrame, test_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    samples = samples.sort_values("timestamp").reset_index(drop=True)
    split_index = max(1, int(len(samples) * (1 - test_ratio)))
    return samples.iloc[:split_index].reset_index(drop=True), samples.iloc[split_index:].reset_index(drop=True)


def clean_side_tables(data: CampusData, valid_posts: pd.DataFrame, valid_users: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    valid_post_ids = set(valid_posts["post_id"])
    valid_user_ids = set(valid_users["user_id"])
    comments = data.comments.copy()
    comments = comments[comments["post_id"].isin(valid_post_ids)]

    post_tags = data.post_tags.copy()
    post_tags = post_tags[post_tags["post_id"].isin(valid_post_ids)].drop_duplicates()

    preferences = data.preferences.copy()
    if not preferences.empty:
        preferences = preferences[preferences["user_id"].isin(valid_user_ids)].drop_duplicates("user_id", keep="last")

    profiles = data.user_profiles.copy()
    if not profiles.empty:
        profiles = profiles[profiles["user_id"].isin(valid_user_ids)].drop_duplicates("user_id", keep="last")

    return (
        comments.reset_index(drop=True),
        data.tags.copy().drop_duplicates("tag_id"),
        post_tags.reset_index(drop=True),
        preferences.reset_index(drop=True),
        profiles.reset_index(drop=True),
    )


def preprocess_all(
    data: CampusData,
    negative_ratio: int = 4,
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> PreprocessResult:
    """Run the full preprocessing workflow for normalized tables."""

    data = parse_time_columns(data)
    users = clean_users(data.users)
    posts, post_stats = clean_posts(data.posts, data.post_stats)
    behaviors = clean_behaviors(data.behaviors, users, posts)
    comments, tags, post_tags, preferences, user_profiles = clean_side_tables(data, posts, users)
    samples = build_labeled_samples(
        users=users,
        posts=posts,
        behaviors=behaviors,
        negative_ratio=negative_ratio,
        random_state=random_state,
    )
    train_samples, test_samples = split_train_test_by_time(samples, test_ratio=test_ratio)
    return PreprocessResult(
        users=users,
        posts=posts,
        post_stats=post_stats,
        comments=comments,
        tags=tags,
        post_tags=post_tags,
        behaviors=behaviors,
        preferences=preferences,
        user_profiles=user_profiles,
        samples=samples,
        train_samples=train_samples,
        test_samples=test_samples,
    )
