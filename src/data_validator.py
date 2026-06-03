"""Validation utilities for generated standard CSV tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "users.csv": [
        "user_id", "nickname", "user_level", "user_level_title", "grade", "college", "major",
        "campus", "interest_tags", "register_time", "is_new_user", "questionnaire_filled",
        "default_location", "status",
    ],
    "posts.csv": [
        "post_id", "author_id", "cate_id", "board", "title", "content", "img_count",
        "has_image", "publish_time", "post_age_hours", "tags", "topic_type", "location_scope",
        "price", "need_pay", "has_contact_info", "report_status", "finish_status", "status",
        "content_length", "keyword_list",
    ],
    "post_stats.csv": [
        "post_id", "view_count", "like_count", "comment_count", "collect_count", "dislike_count",
        "hot_val", "hot_rank", "hot_score", "final_hot_score", "quality_score", "update_time",
    ],
    "user_behaviors.csv": [
        "behavior_id", "user_id", "post_id", "action_type", "action_weight", "timestamp",
        "dwell_time", "time_period", "location", "device_type", "scene", "is_positive",
    ],
}


def validate_dataset(data_dir: str | Path) -> tuple[bool, list[str]]:
    data_dir = Path(data_dir)
    messages: list[str] = []
    ok = True
    tables = {}
    for filename, required in REQUIRED_COLUMNS.items():
        path = data_dir / filename
        if not path.exists():
            ok = False
            messages.append(f"[FAIL] missing {filename}")
            continue
        df = pd.read_csv(path, encoding="utf-8-sig")
        tables[filename] = df
        missing = [column for column in required if column not in df.columns]
        if missing:
            ok = False
            messages.append(f"[FAIL] {filename} missing columns: {missing}")
        else:
            messages.append(f"[OK] {filename} columns complete")

    if "users.csv" in tables:
        users = tables["users.csv"]
        if users["user_id"].isna().any():
            ok = False
            messages.append("[FAIL] users.csv has empty user_id")
        if not set(users["status"].dropna().unique()).issubset({"normal", "blocked"}):
            ok = False
            messages.append("[FAIL] users.csv status has illegal values")

    if "posts.csv" in tables:
        posts = tables["posts.csv"]
        if posts["post_id"].isna().any():
            ok = False
            messages.append("[FAIL] posts.csv has empty post_id")
        if not set(posts["status"].dropna().unique()).issubset({"normal", "deleted", "blocked"}):
            ok = False
            messages.append("[FAIL] posts.csv status has illegal values")
        if pd.to_datetime(posts["publish_time"], errors="coerce").isna().any():
            ok = False
            messages.append("[FAIL] posts.csv publish_time has unparseable values")

    if "post_stats.csv" in tables:
        stats = tables["post_stats.csv"]
        numeric_cols = ["view_count", "like_count", "comment_count", "collect_count", "dislike_count", "hot_score", "final_hot_score"]
        for column in numeric_cols:
            if pd.to_numeric(stats[column], errors="coerce").isna().any():
                ok = False
                messages.append(f"[FAIL] post_stats.csv {column} has non-numeric values")

    if "user_behaviors.csv" in tables:
        behaviors = tables["user_behaviors.csv"]
        if behaviors[["user_id", "post_id"]].isna().any().any():
            ok = False
            messages.append("[FAIL] user_behaviors.csv has empty user_id or post_id")
        if pd.to_datetime(behaviors["timestamp"], errors="coerce").isna().any():
            ok = False
            messages.append("[FAIL] user_behaviors.csv timestamp has unparseable values")
        positive_rate = behaviors["is_positive"].astype(int).mean() if len(behaviors) else 0
        if not 0.05 <= positive_rate <= 0.95:
            ok = False
            messages.append(f"[WARN] positive rate may be unreasonable: {positive_rate:.4f}")
        user_counts = behaviors.groupby("user_id").size()
        if not user_counts.empty and user_counts.min() < 1:
            ok = False
            messages.append("[FAIL] some users have no behavior")
        messages.append(f"[OK] behavior positive_rate={positive_rate:.4f}")

    if "posts.csv" in tables and "post_stats.csv" in tables:
        stats = tables["post_stats.csv"]
        if (stats["view_count"].fillna(0).astype(float) < 0).any():
            ok = False
            messages.append("[FAIL] some posts have negative view_count")

    return ok, messages


def write_validation_report(data_dir: str | Path, output_path: str | Path) -> tuple[bool, list[str]]:
    ok, messages = validate_dataset(data_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = ["# Dataset Validation Report", "", f"Overall status: {'PASS' if ok else 'CHECK REQUIRED'}", ""]
    content.extend(f"- {message}" for message in messages)
    output_path.write_text("\n".join(content), encoding="utf-8")
    return ok, messages
