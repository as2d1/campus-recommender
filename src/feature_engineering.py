"""Feature engineering for normalized campus forum recommendation data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import jieba
except ImportError:  # pragma: no cover
    jieba = None

from src.data_loader import save_dataframe
from src.user_profile import UserProfile, get_time_period, split_tags


TRAINING_SAMPLE_COLUMNS = [
    "user_id",
    "post_id",
    "label",
    "grade",
    "college",
    "major",
    "campus",
    "board",
    "tags",
    "topic_type",
    "time_period",
    "location",
    "location_scope",
    "hot_score",
    "final_hot_score",
    "quality_score",
    "content_sim_score",
    "itemcf_score",
    "profile_match_score",
    "time_scene_score",
    "location_scene_score",
    "multi_interest_score",
    "source_count",
    "recall_sources",
    "post_age_hours",
    "dwell_time",
]


@dataclass(frozen=True)
class TextVectorBundle:
    vectorizer: TfidfVectorizer
    post_text_matrix: sparse.csr_matrix
    post_id_to_index: dict[str, int]


def tokenize_chinese_text(text: str) -> list[str]:
    text = "" if pd.isna(text) else str(text)
    if jieba is None:
        return [char for char in text if char.strip()]
    return [word.strip() for word in jieba.lcut(text) if word.strip()]


def build_post_text(posts: pd.DataFrame) -> pd.Series:
    return (
        posts["title"].fillna("")
        + " "
        + posts["content"].fillna("")
        + " "
        + posts["tags"].fillna("").str.replace("|", " ", regex=False)
    )


def fit_post_text_vectors(posts: pd.DataFrame, max_features: int = 800) -> TextVectorBundle:
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize_chinese_text,
        token_pattern=None,
        max_features=max_features,
        min_df=1,
    )
    matrix = vectorizer.fit_transform(build_post_text(posts))
    post_id_to_index = {str(post_id): idx for idx, post_id in enumerate(posts["post_id"])}
    return TextVectorBundle(vectorizer, matrix, post_id_to_index)


def add_post_features(posts: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    if now is None:
        now = pd.Timestamp.now()
    posts = posts.copy()
    posts["publish_time"] = pd.to_datetime(posts["publish_time"], errors="coerce").fillna(now)
    posts["post_age_hours"] = ((now - posts["publish_time"]).dt.total_seconds().clip(lower=0) / 3600)
    posts["content_length"] = posts["content"].fillna("").astype(str).str.len()
    for column in ["view_count", "like_count", "comment_count", "collect_count", "hot_score", "final_hot_score", "quality_score"]:
        if column not in posts.columns:
            posts[column] = 0
        posts[column] = pd.to_numeric(posts[column], errors="coerce").fillna(0)
    posts["hot_score"] = (
        posts["view_count"] * 0.2
        + posts["like_count"] * 0.3
        + posts["comment_count"] * 0.3
        + posts["collect_count"] * 0.2
    )
    posts["final_hot_score"] = posts["hot_score"] / (1 + posts["post_age_hours"] / 24)
    posts["quality_score"] = posts["quality_score"].fillna(0).clip(0, 1)
    posts["keyword_list"] = posts.apply(
        lambda row: row["keyword_list"] if str(row.get("keyword_list", "")).strip() else "|".join(split_tags(row["tags"])[:5]),
        axis=1,
    )
    return posts


def build_user_features(users: pd.DataFrame, user_profiles: pd.DataFrame) -> pd.DataFrame:
    user_columns = [
        "user_id",
        "grade",
        "college",
        "major",
        "campus",
        "interest_tags",
        "default_location",
    ]
    features = users[user_columns].copy()
    features = features.rename(columns={"default_location": "preferred_location"})
    if not user_profiles.empty:
        profile_columns = [
            "user_id",
            "long_term_tags",
            "short_term_tags",
            "preferred_boards",
            "active_time_period",
            "preferred_location",
        ]
        existing_columns = [column for column in profile_columns if column in user_profiles.columns]
        features = features.merge(user_profiles[existing_columns], on="user_id", how="left", suffixes=("", "_profile"))
        if "preferred_location_profile" in features.columns:
            features["preferred_location"] = features["preferred_location_profile"].fillna(features["preferred_location"])
            features = features.drop(columns=["preferred_location_profile"])
    for column in ["long_term_tags", "short_term_tags", "preferred_boards", "active_time_period"]:
        if column not in features.columns:
            features[column] = ""
    return features.fillna("")


def build_post_feature_table(posts: pd.DataFrame) -> pd.DataFrame:
    post_columns = [
        "post_id",
        "author_id",
        "board",
        "tags",
        "topic_type",
        "title",
        "content",
        "keyword_list",
        "publish_time",
        "post_age_hours",
        "hot_score",
        "final_hot_score",
        "quality_score",
        "content_length",
        "has_image",
        "img_count",
        "has_contact_info",
        "location_scope",
    ]
    for column in post_columns:
        if column not in posts.columns:
            posts[column] = 0 if column in {"post_age_hours", "hot_score", "final_hot_score", "quality_score", "content_length", "has_image", "img_count", "has_contact_info"} else ""
    return posts[post_columns].copy()


def tag_overlap_score(user_tags: object, profile_tags: object, post_tags: object) -> float:
    user_tag_set = set(split_tags(user_tags)) | set(split_tags(profile_tags))
    post_tag_set = set(split_tags(post_tags))
    if not user_tag_set or not post_tag_set:
        return 0.0
    return len(user_tag_set & post_tag_set) / len(user_tag_set | post_tag_set)


def is_major_related(row: pd.Series) -> int:
    text = f"{row.get('board', '')}|{row.get('tags', '')}|{row.get('title', '')}|{row.get('content', '')}"
    college = str(row.get("college", ""))
    major = str(row.get("major", ""))
    if "计算机" in college or major in {"软件工程", "人工智能", "数据科学", "网络工程"}:
        return int(any(keyword in text for keyword in ["408", "竞赛", "科研", "课程资料", "实习"]))
    if "管理" in college:
        return int(any(keyword in text for keyword in ["就业", "简历", "二手交易", "兼职", "社团"]))
    if "公共卫生" in college or "医学" in major:
        return int(any(keyword in text for keyword in ["考研", "科研", "考试", "课程资料", "实习"]))
    return int(any(keyword in text for keyword in ["活动", "交友", "学习经验", "社团"]))


def location_match(user_location: object, location_scope: object) -> int:
    if str(location_scope) == "全校":
        return 1
    return int(str(user_location) == str(location_scope))


def get_semester_phase(timestamp: pd.Timestamp) -> str:
    month = timestamp.month
    day = timestamp.day
    if month in {8, 9}:
        return "开学季"
    if month in {1, 6} or (month == 12 and day >= 15):
        return "考试周"
    if month in {5, 6}:
        return "毕业季"
    return "平时"


def time_scene_score(row: pd.Series) -> float:
    tags = set(split_tags(row.get("tags", "")))
    board = str(row.get("board", ""))
    period = str(row.get("time_period", ""))
    scene = str(row.get("scene", ""))
    semester = get_semester_phase(pd.Timestamp(row.get("timestamp", pd.Timestamp.now())))
    score = 0.0
    if period == "中午" and (board == "食堂生活" or {"食堂", "拼饭"} & tags):
        score += 0.35
    if period == "晚上" and (board in {"学习交流", "考研保研", "课程评价"} or {"课程资料", "考研"} & tags):
        score += 0.25
    if semester == "考试周" and (board in {"学习交流", "考研保研", "课程评价"} or {"考试", "复习资料"} & tags):
        score += 0.35
    if scene == "开学季" and ({"社团", "招新", "新生攻略"} & tags):
        score += 0.25
    return min(score, 1.0)


def multi_interest_score(profile: UserProfile | None, post_id: str, text_bundle: TextVectorBundle) -> float:
    if profile is None or not profile.multi_interest_vectors or post_id not in text_bundle.post_id_to_index:
        return 0.0
    post_vector = text_bundle.post_text_matrix[text_bundle.post_id_to_index[post_id]]
    scores = []
    for direction, user_vector in profile.multi_interest_vectors.items():
        weight = profile.multi_interest_weights.get(direction, 1.0)
        scores.append(float(cosine_similarity(user_vector, post_vector)[0, 0]) * np.log1p(weight))
    return max(float(max(scores)), 0.0) if scores else 0.0


def build_behavior_statistics(behaviors: pd.DataFrame, posts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    enriched = behaviors.merge(posts[["post_id", "board", "tags"]], on="post_id", how="left")
    enriched["is_positive"] = enriched["is_positive"].fillna(0).astype(int)
    board_stats = (
        enriched.groupby(["user_id", "board"])["is_positive"]
        .agg(user_board_positive="sum", user_board_total="count")
        .reset_index()
    )
    board_stats["user_board_ctr"] = board_stats["user_board_positive"] / board_stats["user_board_total"]

    tag_rows = []
    for row in enriched.itertuples(index=False):
        for tag in split_tags(row.tags):
            tag_rows.append({"user_id": row.user_id, "tag": tag, "is_positive": row.is_positive})
    if tag_rows:
        tag_stats = (
            pd.DataFrame(tag_rows)
            .groupby(["user_id", "tag"])["is_positive"]
            .agg(user_tag_positive="sum", user_tag_total="count")
            .reset_index()
        )
        tag_stats["user_tag_ctr"] = tag_stats["user_tag_positive"] / tag_stats["user_tag_total"]
    else:
        tag_stats = pd.DataFrame(columns=["user_id", "tag", "user_tag_ctr"])
    return board_stats, tag_stats


def user_tag_ctr_for_row(row: pd.Series, tag_stats: pd.DataFrame) -> float:
    post_tags = split_tags(row.get("tags", ""))
    if not post_tags or tag_stats.empty:
        return 0.0
    matched = tag_stats[tag_stats["user_id"].eq(row["user_id"]) & tag_stats["tag"].isin(post_tags)]
    return float(matched["user_tag_ctr"].mean()) if not matched.empty else 0.0


def build_training_features(
    samples: pd.DataFrame,
    users: pd.DataFrame,
    posts: pd.DataFrame,
    behaviors: pd.DataFrame,
    user_profiles_df: pd.DataFrame,
    profiles: dict[str, UserProfile],
    text_bundle: TextVectorBundle,
) -> pd.DataFrame:
    user_features = build_user_features(users, user_profiles_df)
    post_features = build_post_feature_table(add_post_features(posts))
    features = samples.merge(user_features, on="user_id", how="left")
    features = features.merge(post_features, on="post_id", how="left")

    if "time_period" not in features.columns:
        features["time_period"] = features["timestamp"].apply(get_time_period)
    features["location"] = features["location"].fillna(features["preferred_location"]).fillna("教学区")
    features["device_type"] = features.get("device_type", "unknown")
    features["scene"] = features.get("scene", "普通浏览")
    features["is_weekend"] = features["timestamp"].dt.dayofweek.isin([5, 6]).astype(int)
    features["location_match"] = features.apply(lambda row: location_match(row["location"], row["location_scope"]), axis=1)
    features["location_scene_score"] = features["location_match"].astype(float)
    features["time_scene_score"] = features.apply(time_scene_score, axis=1)

    board_stats, tag_stats = build_behavior_statistics(behaviors, posts)
    features = features.merge(board_stats[["user_id", "board", "user_board_ctr"]], on=["user_id", "board"], how="left")
    features["user_tag_ctr"] = features.apply(lambda row: user_tag_ctr_for_row(row, tag_stats), axis=1)
    features["is_frequent_board"] = (features["user_board_ctr"].fillna(0) >= 0.5).astype(int)
    features["is_preferred_tag"] = (features["user_tag_ctr"].fillna(0) >= 0.5).astype(int)
    features["major_related"] = features.apply(is_major_related, axis=1)
    features["profile_match_score"] = features.apply(
        lambda row: tag_overlap_score(row["interest_tags"], row.get("long_term_tags", ""), row["tags"]),
        axis=1,
    )
    features["multi_interest_score"] = features.apply(
        lambda row: multi_interest_score(profiles.get(str(row["user_id"])), str(row["post_id"]), text_bundle),
        axis=1,
    )
    features["content_sim_score"] = features["multi_interest_score"]

    if "source_count" not in features.columns:
        features["source_count"] = 0
    if "recall_sources" not in features.columns:
        features["recall_sources"] = "unknown"
    features["itemcf_score"] = features.get("itemcf_score", 0.0)

    for column in TRAINING_SAMPLE_COLUMNS:
        if column not in features.columns:
            features[column] = 0 if column not in {"recall_sources"} else "unknown"
    numeric_columns = [
        "hot_score",
        "final_hot_score",
        "quality_score",
        "content_sim_score",
        "itemcf_score",
        "profile_match_score",
        "time_scene_score",
        "location_scene_score",
        "multi_interest_score",
        "source_count",
        "post_age_hours",
        "dwell_time",
    ]
    features[numeric_columns] = features[numeric_columns].fillna(0)
    return features


def build_ranking_features(*args, **kwargs) -> pd.DataFrame:
    """Backward-compatible alias for the previous step."""

    return build_training_features(*args, **kwargs)


def save_training_samples(training_samples: pd.DataFrame, output_path: str) -> None:
    save_dataframe(training_samples[TRAINING_SAMPLE_COLUMNS], output_path)
