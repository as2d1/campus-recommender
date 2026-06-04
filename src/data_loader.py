"""Data loading utilities for the campus forum recommender.

The recommender reads the real Zanao SQLite database and adapts its tables to
the normalized dataframe shape expected by preprocessing and ranking code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_SQLITE_PATH = PROJECT_ROOT.parent / "zanao.sqlite"
ALLOWED_BOARDS = ["打听求助", "恋爱交友", "校园趣事", "兼职招聘", "校园招聘", "二手闲置"]
BOARD_TOPIC_MAP = {
    "打听求助": "生活",
    "恋爱交友": "社交",
    "校园趣事": "社交",
    "兼职招聘": "发展",
    "校园招聘": "发展",
    "二手闲置": "生活",
}


@dataclass(frozen=True)
class CampusData:
    users: pd.DataFrame
    posts: pd.DataFrame
    post_stats: pd.DataFrame
    comments: pd.DataFrame
    tags: pd.DataFrame
    post_tags: pd.DataFrame
    behaviors: pd.DataFrame
    user_profiles: pd.DataFrame


def _read_table(conn: sqlite3.Connection, table_name: str) -> pd.DataFrame:
    return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)


def _parse_zanao_time(epoch: pd.Series, text_time: pd.Series | None = None) -> pd.Series:
    parsed = pd.to_datetime(epoch, unit="s", errors="coerce")
    if text_time is not None:
        parsed_text = pd.to_datetime(text_time, format="%Y/%m/%d %H:%M", errors="coerce")
        parsed = parsed.where(parsed.notna(), parsed_text)
    return parsed


def _build_tags(posts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tag_names = sorted(set(posts["board"].dropna().astype(str)) | set(posts["topic_type"].dropna().astype(str)))
    tags = pd.DataFrame(
        [{"tag_id": f"tag_{index + 1:03d}", "tag_name": tag_name, "view_count": 0} for index, tag_name in enumerate(tag_names)]
    )
    post_tags = []
    for row in posts[["post_id", "board", "topic_type"]].itertuples(index=False):
        for tag_name in dict.fromkeys([row.board, row.topic_type]):
            if tag_name:
                post_tags.append({"post_id": row.post_id, "tag_name": tag_name})
    return tags, pd.DataFrame(post_tags)


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


def _adapt_posts(posts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered = posts[posts["cate_name"].isin(ALLOWED_BOARDS)].copy()
    filtered = filtered.rename(columns={"thread_id": "post_id", "user_id": "author_id", "cate_name": "board"})
    filtered["topic_type"] = filtered["board"].map(BOARD_TOPIC_MAP).fillna("生活")
    filtered["publish_time"] = _parse_zanao_time(filtered["p_time"], filtered.get("pt_time"))
    filtered["tags"] = filtered["board"] + "|" + filtered["topic_type"]
    filtered["has_image"] = (pd.to_numeric(filtered["img_count"], errors="coerce").fillna(0) > 0).astype(int)
    filtered["price"] = 0
    filtered["need_pay"] = 0
    filtered["has_contact_info"] = 0
    filtered["report_status"] = "normal"
    filtered["finish_status"] = "open"
    filtered["status"] = "normal"
    filtered["keyword_list"] = ""
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
    adapted["status"] = "normal"
    adapted["is_hide"] = 0
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
        users = _adapt_users(_read_table(conn, "USERS"))
        posts, post_stats = _adapt_posts(_read_table(conn, "POSTS"))
        comments = _adapt_comments(_read_table(conn, "COMMENTS"))
        behaviors = _adapt_behaviors(_read_table(conn, "USER_EVENTS"))

    users = _merge_event_visitors(users, behaviors)
    tags, post_tags = _build_tags(posts)
    return CampusData(
        users=users,
        posts=posts,
        post_stats=post_stats,
        comments=comments,
        tags=tags,
        post_tags=post_tags,
        behaviors=behaviors,
        user_profiles=pd.DataFrame(),
    )
