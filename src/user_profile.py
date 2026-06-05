"""User profile construction for the campus forum recommender.

This module builds in-memory user profiles from normalized users, posts, and
user behavior logs. It focuses on interpretable profile features that can be
reused by recall, ranking, and later online profile updates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json

import numpy as np
import pandas as pd
from scipy import sparse

from src.taxonomy import INTEREST_DIRECTIONS, infer_interest_direction, join_tags, split_tags


ACTION_WEIGHTS = {
    "expose": 0.0,
    "view": 1.0,
    "like": 2.0,
    "comment": 3.0,
    "collect": 4.0,
    "skip": -1.0,
    "dislike": -2.0,
    "report": -4.0,
}

POSITIVE_ACTIONS = {"like", "comment", "collect"}
PROFILE_COLUMNS = [
    "user_id",
    "long_term_tags",
    "short_term_tags",
    "preferred_boards",
    "active_time_period",
    "learning_interest_weight",
    "life_interest_weight",
    "social_interest_weight",
    "career_interest_weight",
    "avg_dwell_time",
    "click_rate",
    "like_rate",
    "collect_rate",
    "comment_rate",
    "last_update_time",
]

@dataclass
class UserProfile:
    user_id: str
    long_term_tags: list[str] = field(default_factory=list)
    short_term_tags: list[str] = field(default_factory=list)
    preferred_boards: list[str] = field(default_factory=list)
    active_time_period: str = "未知"
    board_ctr: dict[str, float] = field(default_factory=dict)
    tag_ctr: dict[str, float] = field(default_factory=dict)
    multi_interest_vectors: dict[str, sparse.csr_matrix] = field(default_factory=dict)
    multi_interest_weights: dict[str, float] = field(default_factory=dict)


def parse_json_list(value: object) -> list[str]:
    """Parse a stored JSON list from USER_PREFERENCES."""

    if pd.isna(value):
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item).strip() for item in parsed if str(item).strip()]


def get_time_period(timestamp: pd.Timestamp) -> str:
    """Map a timestamp to a coarse activity period."""

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


def is_positive_behavior(row: pd.Series) -> bool:
    """Whether a behavior should count as positive feedback."""

    action = str(row.get("action_type", ""))
    dwell_time = int(row.get("dwell_time", 0) or 0)
    if "is_positive" in row and not pd.isna(row["is_positive"]):
        return int(row["is_positive"]) == 1
    return action in POSITIVE_ACTIONS or (action == "view" and dwell_time >= 30)


def time_decay_weight(timestamp: pd.Timestamp, now: pd.Timestamp) -> float:
    """Recent behaviors receive larger weight."""

    if pd.isna(timestamp):
        return 0.0
    days_diff = max((now - pd.Timestamp(timestamp)).days, 0)
    return 1.0 / (1.0 + days_diff)


def prepare_behavior_features(posts: pd.DataFrame, behaviors: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Join behavior logs with post metadata and add reusable weighting fields."""

    if now is None:
        now = pd.Timestamp.now()
    post_columns = ["post_id", "board", "tags", "topic_type"]
    enriched = behaviors.copy()
    enriched["timestamp"] = pd.to_datetime(enriched["timestamp"], errors="coerce")
    enriched["dwell_time"] = pd.to_numeric(enriched.get("dwell_time", 0), errors="coerce").fillna(0)
    if "time_period" not in enriched.columns:
        enriched["time_period"] = enriched["timestamp"].apply(get_time_period)
    enriched = enriched.merge(posts[post_columns], on="post_id", how="left")
    enriched["action_weight"] = enriched["action_type"].map(ACTION_WEIGHTS).fillna(
        pd.to_numeric(enriched.get("action_weight", 0), errors="coerce").fillna(0)
    )
    enriched["is_positive"] = enriched.apply(is_positive_behavior, axis=1).astype(int)
    enriched["time_weight"] = enriched["timestamp"].apply(lambda value: time_decay_weight(value, now))
    enriched["final_weight"] = enriched["action_weight"] * enriched["time_weight"]
    enriched["positive_weight"] = enriched["final_weight"].clip(lower=0)
    enriched["interest_direction"] = enriched.apply(
        lambda row: infer_interest_direction(row.get("board", ""), row.get("tags", ""), row.get("topic_type", "")),
        axis=1,
    )
    return enriched


def weighted_tag_ranking(df: pd.DataFrame, top_n: int = 8) -> list[str]:
    """Rank tags by behavior weight."""

    scores: dict[str, float] = {}
    for row in df.itertuples(index=False):
        weight = float(getattr(row, "positive_weight", 0.0))
        if weight <= 0:
            continue
        for tag in split_tags(getattr(row, "tags", "")):
            scores[tag] = scores.get(tag, 0.0) + weight
    return [tag for tag, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_n]]


def weighted_value_ranking(df: pd.DataFrame, column: str, top_n: int = 4) -> list[str]:
    """Rank boards or time periods by positive weighted frequency."""

    if df.empty or column not in df.columns:
        return []
    weighted = df.copy()
    weighted["_weight"] = weighted["positive_weight"].where(weighted["positive_weight"] > 0, 1.0)
    scores = weighted.groupby(column)["_weight"].sum().sort_values(ascending=False)
    return scores.head(top_n).index.dropna().astype(str).tolist()


def calculate_interest_weights(user_behaviors: pd.DataFrame) -> dict[str, float]:
    """Calculate normalized learning/life/social/career interest weights."""

    raw_scores = {direction: 0.0 for direction in INTEREST_DIRECTIONS}
    for row in user_behaviors.itertuples(index=False):
        direction = getattr(row, "interest_direction", "life")
        signed_weight = float(getattr(row, "final_weight", 0.0))
        raw_scores[direction] = raw_scores.get(direction, 0.0) + signed_weight

    clipped_scores = {key: max(value, 0.0) for key, value in raw_scores.items()}
    total = sum(clipped_scores.values())
    if total <= 0:
        return {key: 0.0 for key in INTEREST_DIRECTIONS}
    return {key: round(value / total, 6) for key, value in clipped_scores.items()}


def preference_interest_weights(preferred_boards: list[str], preferred_tags: list[str]) -> dict[str, float]:
    """Derive initial interest weights from cold-start board/tag choices."""

    raw_scores = {direction: 0.0 for direction in INTEREST_DIRECTIONS}
    for board in preferred_boards:
        raw_scores[infer_interest_direction(board, "", "")] += 1.0
    for tag in preferred_tags:
        raw_scores[infer_interest_direction("", tag, "")] += 0.5
    total = sum(raw_scores.values())
    if total <= 0:
        return {direction: 0.0 for direction in INTEREST_DIRECTIONS}
    return {direction: round(value / total, 6) for direction, value in raw_scores.items()}


def calculate_action_rates(user_behaviors: pd.DataFrame) -> dict[str, float]:
    """Calculate behavior rates for one user."""

    total = max(len(user_behaviors), 1)
    return {
        "click_rate": round((user_behaviors["action_type"].eq("view").sum()) / total, 6),
        "like_rate": round((user_behaviors["action_type"].eq("like").sum()) / total, 6),
        "collect_rate": round((user_behaviors["action_type"].eq("collect").sum()) / total, 6),
        "comment_rate": round((user_behaviors["action_type"].eq("comment").sum()) / total, 6),
    }


def build_user_profile_table(
    users: pd.DataFrame,
    posts: pd.DataFrame,
    behaviors: pd.DataFrame,
    preferences: pd.DataFrame | None = None,
    now: pd.Timestamp | None = None,
    recent_days: int = 7,
    recent_n: int = 30,
) -> pd.DataFrame:
    """Build the user profile dataframe used by recall and ranking."""

    if now is None:
        now = pd.Timestamp.now()
    enriched = prepare_behavior_features(posts, behaviors, now=now)
    preference_map = {}
    if preferences is not None and not preferences.empty:
        preference_map = preferences.drop_duplicates("user_id", keep="last").set_index("user_id").to_dict("index")
    rows: list[dict[str, object]] = []

    for user in users.itertuples(index=False):
        user_id = str(user.user_id)
        user_behaviors = enriched[enriched["user_id"].eq(user_id)].copy()
        positive = user_behaviors[user_behaviors["is_positive"].eq(1)]

        recent_cutoff = now - pd.Timedelta(days=recent_days)
        recent = positive[positive["timestamp"] >= recent_cutoff]
        if recent.empty:
            recent = positive.sort_values("timestamp", ascending=False).head(recent_n)

        preference = preference_map.get(user_id, {})
        preference_boards = parse_json_list(preference.get("selected_boards_json", "[]"))
        preference_tags = parse_json_list(preference.get("selected_tags_json", "[]"))
        preference_seed_tags = preference_tags + preference_boards + [
            infer_interest_direction(board, "", "") for board in preference_boards
        ]

        interest_weights = calculate_interest_weights(user_behaviors)
        if sum(interest_weights.values()) <= 0:
            interest_weights = preference_interest_weights(preference_boards, preference_tags)
        rates = calculate_action_rates(user_behaviors)
        view_behaviors = user_behaviors[user_behaviors["action_type"].eq("view")]
        avg_dwell_time = float(view_behaviors["dwell_time"].mean()) if not view_behaviors.empty else 0.0

        long_term_tags = weighted_tag_ranking(positive, top_n=8)
        if not long_term_tags:
            long_term_tags = preference_seed_tags
        short_term_tags = weighted_tag_ranking(recent, top_n=6)
        if not short_term_tags:
            short_term_tags = long_term_tags[:6]

        preferred_boards = weighted_value_ranking(
            user_behaviors[user_behaviors["action_type"].isin(["view", "like", "comment", "collect"])],
            "board",
            top_n=4,
        )
        if not preferred_boards:
            preferred_boards = preference_boards
        active_time_period = weighted_value_ranking(user_behaviors, "time_period", top_n=1)

        rows.append(
            {
                "user_id": user_id,
                "long_term_tags": join_tags(long_term_tags),
                "short_term_tags": join_tags(short_term_tags),
                "preferred_boards": join_tags(preferred_boards),
                "active_time_period": active_time_period[0] if active_time_period else "未知",
                "learning_interest_weight": interest_weights["learning"],
                "life_interest_weight": interest_weights["life"],
                "social_interest_weight": interest_weights["social"],
                "career_interest_weight": interest_weights["career"],
                "avg_dwell_time": round(avg_dwell_time, 6),
                **rates,
                "last_update_time": now.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    return pd.DataFrame(rows, columns=PROFILE_COLUMNS)


def build_user_profiles(
    users: pd.DataFrame,
    posts: pd.DataFrame,
    behaviors: pd.DataFrame,
    preferences: pd.DataFrame | None = None,
    user_profiles_df: pd.DataFrame | None = None,
    post_id_to_index: dict[str, int] | None = None,
    text_matrix: sparse.csr_matrix | None = None,
    now: pd.Timestamp | None = None,
) -> dict[str, UserProfile]:
    """Build in-memory user profile objects.

    This keeps compatibility with the existing feature engineering module. If
    TF-IDF vectors are supplied, multi-interest vectors are also attached by
    delegating to ``src.multi_interest``.
    """

    if now is None:
        now = pd.Timestamp.now()
    profile_df = (
        user_profiles_df.copy()
        if user_profiles_df is not None and not user_profiles_df.empty
        else build_user_profile_table(users, posts, behaviors, preferences=preferences, now=now)
    )
    profile_rows = profile_df.set_index("user_id").to_dict("index") if not profile_df.empty else {}

    multi_vectors = {}
    multi_weights = {}
    if post_id_to_index is not None and text_matrix is not None:
        from src.multi_interest import build_multi_interest_representations

        bundle = build_multi_interest_representations(
            users=users,
            posts=posts,
            behaviors=behaviors,
            post_text_matrix=text_matrix,
            post_id_to_index=post_id_to_index,
            now=now,
        )
        multi_vectors = bundle.user_vectors
        multi_weights = bundle.user_direction_weights

    profiles: dict[str, UserProfile] = {}
    for user in users.itertuples(index=False):
        user_id = str(user.user_id)
        row = profile_rows.get(user_id, {})
        long_term_tags = split_tags(row.get("long_term_tags", ""))
        short_term_tags = split_tags(row.get("short_term_tags", ""))
        profiles[user_id] = UserProfile(
            user_id=user_id,
            long_term_tags=long_term_tags,
            short_term_tags=short_term_tags,
            preferred_boards=split_tags(row.get("preferred_boards", "")),
            active_time_period=str(row.get("active_time_period", "未知")),
            multi_interest_vectors=multi_vectors.get(user_id, {}),
            multi_interest_weights=multi_weights.get(user_id, {}),
        )
    return profiles
