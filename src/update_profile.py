"""Incremental user behavior and profile update utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_loader import DATA_DIR, load_all_data, save_dataframe
from src.feature_engineering import fit_post_text_vectors
from src.multi_interest import build_multi_interest_representations
from src.preprocess import get_time_period
from src.user_profile import ACTION_WEIGHTS, build_user_profile_table, save_user_profiles


BEHAVIOR_PATH = DATA_DIR / "user_behaviors.csv"
PROFILE_PATH = DATA_DIR / "user_profiles.csv"


def build_behavior_row(
    user_id: str,
    post_id: str,
    action_type: str,
    timestamp: str | None = None,
    dwell_time: int = 0,
    time_period: str | None = None,
    location: str = "教学区",
    device_type: str = "mobile",
    scene: str = "普通浏览",
    source: str = "incremental",
) -> dict[str, object]:
    """Build one normalized behavior row."""

    ts = pd.Timestamp(timestamp) if timestamp else pd.Timestamp.now()
    action_weight = ACTION_WEIGHTS.get(action_type, 0.0)
    is_positive = int(action_type in {"like", "comment", "collect"} or (action_type == "view" and dwell_time >= 30))
    return {
        "behavior_id": f"b_inc_{int(ts.timestamp())}",
        "user_id": user_id,
        "post_id": post_id,
        "action_type": action_type,
        "action_weight": action_weight,
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "dwell_time": dwell_time,
        "time_period": time_period or get_time_period(ts),
        "location": location,
        "device_type": device_type,
        "scene": scene,
        "is_positive": is_positive,
        "source": source,
    }


def append_behavior(behavior: dict[str, object], behavior_path: str | Path = BEHAVIOR_PATH) -> pd.DataFrame:
    """Append a new behavior to user_behaviors.csv."""

    behavior_path = Path(behavior_path)
    if behavior_path.exists():
        behaviors = pd.read_csv(behavior_path, encoding="utf-8-sig")
    else:
        behaviors = pd.DataFrame()
    behaviors = pd.concat([behaviors, pd.DataFrame([behavior])], ignore_index=True)
    save_dataframe(behaviors, behavior_path)
    return behaviors


def rebuild_profiles_and_multi_interest(user_id: str | None = None):
    """Rebuild user_profiles.csv and return updated multi-interest bundle."""

    data = load_all_data()
    profile_df = build_user_profile_table(data.users, data.posts, data.behaviors)
    save_user_profiles(profile_df, PROFILE_PATH)
    text_bundle = fit_post_text_vectors(data.posts)
    multi_bundle = build_multi_interest_representations(
        users=data.users,
        posts=data.posts,
        behaviors=data.behaviors,
        post_text_matrix=text_bundle.post_text_matrix,
        post_id_to_index=text_bundle.post_id_to_index,
    )
    if user_id:
        print(profile_df[profile_df["user_id"].astype(str).eq(str(user_id))].to_string(index=False))
    return profile_df, multi_bundle


def update_user_profile_with_new_behavior(
    user_id: str,
    post_id: str,
    action_type: str,
    timestamp: str | None = None,
    dwell_time: int = 0,
    time_period: str | None = None,
    location: str = "教学区",
    device_type: str = "mobile",
    scene: str = "普通浏览",
):
    """Append new behavior and update user profile artifacts."""

    behavior = build_behavior_row(
        user_id=user_id,
        post_id=post_id,
        action_type=action_type,
        timestamp=timestamp,
        dwell_time=dwell_time,
        time_period=time_period,
        location=location,
        device_type=device_type,
        scene=scene,
    )
    append_behavior(behavior)
    return rebuild_profiles_and_multi_interest(user_id=user_id)


if __name__ == "__main__":
    data = load_all_data()
    uid = str(data.users.iloc[0]["user_id"])
    pid = str(data.posts.iloc[0]["post_id"])
    update_user_profile_with_new_behavior(uid, pid, "view", dwell_time=60)
