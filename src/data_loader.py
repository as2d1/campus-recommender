"""Data loading utilities for the campus forum recommender.

The recommender reads the real Zanao SQLite database and adapts its tables to
the normalized dataframe shape expected by preprocessing and ranking code.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3

import pandas as pd

from src.taxonomy import ALLOWED_BOARDS, tags_for_post, topic_for_board


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_SQLITE_PATH = PROJECT_ROOT.parent / "zanao.sqlite"
IMAGE_CDN_BASE = "https://b1.cdn.zanao.com/"
IMAGE_CDN_SUFFIX = "@!common"


@dataclass(frozen=True)
class CampusData:
    users: pd.DataFrame
    posts: pd.DataFrame
    post_stats: pd.DataFrame
    comments: pd.DataFrame
    tags: pd.DataFrame
    post_tags: pd.DataFrame
    behaviors: pd.DataFrame
    preferences: pd.DataFrame
    user_profiles: pd.DataFrame


def ensure_app_tables(conn: sqlite3.Connection) -> None:
    """Create app-owned tables that augment the imported Zanao data."""

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS USER_PREFERENCES (
            user_id TEXT PRIMARY KEY,
            selected_boards_json TEXT NOT NULL DEFAULT '[]',
            selected_tags_json TEXT NOT NULL DEFAULT '[]',
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        )
        """
    )
    conn.commit()


def _read_table(conn: sqlite3.Connection, table_name: str) -> pd.DataFrame:
    return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)


def _parse_zanao_time(epoch: pd.Series, text_time: pd.Series | None = None) -> pd.Series:
    parsed = pd.to_datetime(epoch, unit="s", errors="coerce")
    if text_time is not None:
        parsed_text = pd.to_datetime(text_time, format="%Y/%m/%d %H:%M", errors="coerce")
        parsed = parsed.where(parsed.notna(), parsed_text)
    return parsed


def _build_tags(posts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tag_names = sorted({tag for value in posts["tags"].dropna() for tag in str(value).split("|") if tag})
    tags = pd.DataFrame(
        [{"tag_id": f"tag_{index + 1:03d}", "tag_name": tag_name, "view_count": 0} for index, tag_name in enumerate(tag_names)]
    )
    post_tags = []
    for row in posts[["post_id", "tags"]].itertuples(index=False):
        for tag_name in dict.fromkeys([tag for tag in str(row.tags).split("|") if tag]):
            if tag_name:
                post_tags.append({"post_id": row.post_id, "tag_name": tag_name})
    return tags, pd.DataFrame(post_tags)


def normalize_image_url(path: object) -> str:
    text = str(path or "").strip()
    if not text:
        return ""
    if text.startswith(("http://", "https://")):
        if "cdn.zanao.com/" in text and "@" not in text.rsplit("/", 1)[-1]:
            return f"{text}{IMAGE_CDN_SUFFIX}"
        return text
    return f"{IMAGE_CDN_BASE}{text.lstrip('/')}{IMAGE_CDN_SUFFIX}"


def _parse_image_paths(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = [part.strip() for part in text.replace("\n", ",").split(",")]
    if not isinstance(parsed, list):
        return []
    return [str(item).strip() for item in parsed if str(item).strip()]


def _adapt_users(users: pd.DataFrame) -> pd.DataFrame:
    adapted = users.copy()
    adapted["status"] = adapted["is_forbid"].fillna(0).astype(int).map(lambda value: "blocked" if value else "normal")
    return adapted


def _merge_event_visitors(users: pd.DataFrame, behaviors: pd.DataFrame) -> pd.DataFrame:
    if behaviors.empty:
        return users
    existing = set(users["user_id"].astype(str))
    visitor_ids = set(behaviors["user_id"].dropna().astype(str))
    missing = sorted(visitor_ids - existing)
    if not missing:
        return users
    visitor_rows = pd.DataFrame(
        {
            "user_id": missing,
            "nickname": ["匿名访客"] * len(missing),
            "headimgurl_hash": [""] * len(missing),
            "is_forbid": [0] * len(missing),
        }
    )
    return pd.concat([users, _adapt_users(visitor_rows)], ignore_index=True)


def _merge_preference_users(users: pd.DataFrame, preferences: pd.DataFrame) -> pd.DataFrame:
    if preferences.empty:
        return users
    existing = set(users["user_id"].astype(str))
    preference_ids = set(preferences["user_id"].dropna().astype(str))
    missing = sorted(preference_ids - existing)
    if not missing:
        return users
    rows = pd.DataFrame(
        {
            "user_id": missing,
            "nickname": ["冷启动用户"] * len(missing),
            "headimgurl_hash": [""] * len(missing),
            "is_forbid": [0] * len(missing),
        }
    )
    return pd.concat([users, _adapt_users(rows)], ignore_index=True)


def _adapt_posts(posts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered = posts[posts["cate_name"].isin(ALLOWED_BOARDS)].copy()
    filtered = filtered.rename(columns={"thread_id": "post_id", "user_id": "author_id", "cate_name": "board"})
    filtered["topic_type"] = filtered["board"].map(topic_for_board)
    filtered["publish_time"] = _parse_zanao_time(filtered["p_time"], filtered.get("pt_time"))
    filtered["tags"] = filtered["board"].map(tags_for_post)
    image_source = filtered["image_paths_json"] if "image_paths_json" in filtered.columns else pd.Series(["[]"] * len(filtered), index=filtered.index)
    filtered["image_paths"] = image_source.apply(_parse_image_paths)
    filtered["image_urls"] = filtered["image_paths"].apply(lambda paths: [normalize_image_url(path) for path in paths if normalize_image_url(path)])
    filtered["img_count"] = pd.to_numeric(filtered["img_count"], errors="coerce").fillna(0).astype(int)
    filtered["img_count"] = filtered.apply(lambda row: max(int(row["img_count"]), len(row["image_paths"])), axis=1)
    filtered["has_image"] = (filtered["img_count"] > 0).astype(int)
    filtered["collect_count"] = pd.to_numeric(filtered["mark_count"], errors="coerce").fillna(0).astype(int)

    stats = filtered[
        [
            "post_id",
            "view_count",
            "like_count",
            "dislike_count",
            "comment_count",
            "collect_count",
            "hot_val",
            "publish_time",
        ]
    ].copy()
    stats["update_time"] = stats["publish_time"]
    stats["hot_rank"] = 0
    stats = stats.drop(columns=["publish_time"])
    return filtered, stats


def _adapt_comments(comments: pd.DataFrame) -> pd.DataFrame:
    adapted = comments.rename(columns={"thread_id": "post_id", "post_time_text": "publish_time"}).copy()
    adapted["publish_time"] = _parse_zanao_time(adapted["post_time"], adapted.get("publish_time"))
    return adapted


def _adapt_behaviors(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(
            columns=[
                "behavior_id",
                "user_id",
                "post_id",
                "action_type",
                "timestamp",
                "dwell_time",
                "time_period",
                "source",
            ]
        )
    adapted = events.rename(
        columns={
            "id": "behavior_id",
            "visitor_id": "user_id",
            "thread_id": "post_id",
            "event_time": "timestamp",
        }
    ).copy()
    adapted["action_type"] = adapted["event_type"].replace({"favorite": "collect"})
    adapted["timestamp"] = _parse_zanao_time(adapted["timestamp"])
    adapted["dwell_time"] = (pd.to_numeric(adapted["duration_ms"], errors="coerce").fillna(0) / 1000).round().astype(int)
    adapted["time_period"] = ""
    adapted["source"] = "sqlite"
    return adapted


def load_all_data(sqlite_path: str | Path = DEFAULT_SQLITE_PATH) -> CampusData:
    """Load and normalize the real Zanao SQLite database."""

    db_path = Path(sqlite_path)
    if db_path.is_dir():
        db_path = DEFAULT_SQLITE_PATH
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite database not found: {db_path}")

    with sqlite3.connect(db_path) as conn:
        ensure_app_tables(conn)
        users = _adapt_users(_read_table(conn, "USERS"))
        posts, post_stats = _adapt_posts(_read_table(conn, "POSTS"))
        comments = _adapt_comments(_read_table(conn, "COMMENTS"))
        behaviors = _adapt_behaviors(_read_table(conn, "USER_EVENTS"))
        preferences = _read_table(conn, "USER_PREFERENCES")

    users = _merge_event_visitors(users, behaviors)
    users = _merge_preference_users(users, preferences)
    tags, post_tags = _build_tags(posts)
    return CampusData(
        users=users,
        posts=posts,
        post_stats=post_stats,
        comments=comments,
        tags=tags,
        post_tags=post_tags,
        behaviors=behaviors,
        preferences=preferences,
        user_profiles=pd.DataFrame(),
    )
