"""Parse Zanao JSONL seed posts into normalized intermediate tables."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import BOARD_DEFAULT_TAGS, BOARD_TOPIC_MAP


def first_non_empty(*values: Any, default: Any = "") -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        return value
    return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        if value in [None, ""]:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in [None, ""]:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_img_paths(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if isinstance(value, str) and value:
        return [value]
    return []


def hash_id(prefix: str, value: Any) -> str:
    digest = hashlib.md5(str(value).encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def parse_time(value: Any) -> pd.Timestamp:
    if value in [None, ""]:
        return pd.Timestamp.now()
    text = str(value)
    if text.isdigit():
        # Some Zanao relative timestamps are unix-like seconds but may be future-shifted.
        parsed = pd.to_datetime(int(text), unit="s", errors="coerce")
        return parsed if not pd.isna(parsed) else pd.Timestamp.now()
    parsed = pd.to_datetime(text, errors="coerce")
    return parsed if not pd.isna(parsed) else pd.Timestamp.now()


def map_category(cate_name: Any) -> tuple[str, str]:
    return BOARD_TOPIC_MAP.get(str(cate_name), ("校园趣事", "社交"))


def infer_status(detail: dict[str, Any], list_item: dict[str, Any]) -> str:
    report_status = str(first_non_empty(detail.get("report_status"), list_item.get("report_status"), default="0"))
    delete_status = str(first_non_empty(detail.get("delete_status"), default="10"))
    status = str(first_non_empty(detail.get("status"), default="10"))
    is_forbid = str(first_non_empty(detail.get("is_forbid"), list_item.get("is_forbid"), default="0"))
    if delete_status not in {"0", "10", "normal"}:
        return "deleted"
    if report_status not in {"0", "10", "normal"} or is_forbid == "1" or status in {"deleted", "blocked"}:
        return "blocked"
    return "normal"


def extract_detail(record: dict[str, Any]) -> dict[str, Any]:
    return record.get("detail", {}).get("data", {}).get("detail", {}) or {}


def extract_seed_tags(board: str, title: str, content: str) -> list[str]:
    tags = list(BOARD_DEFAULT_TAGS.get(board, []))[:2]
    text = f"{title} {content}"
    keyword_rules = {
        "南校": "南校",
        "东校": "东校",
        "宿舍": "宿舍",
        "食堂": "食堂",
        "图书馆": "图书馆",
        "课程": "课程资料",
        "搭子": "搭子",
        "二手": "二手闲置",
        "求问": "求问",
        "考研": "考研",
        "实习": "实习",
        "租": "租房",
    }
    for keyword, tag in keyword_rules.items():
        if keyword in text and tag not in tags:
            tags.append(tag)
    return tags[:5] or ["校园趣事"]


def parse_comments(comment_list: list[dict[str, Any]], post_id: str, author_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, comment in enumerate(comment_list or []):
        comment_id = str(first_non_empty(comment.get("comment_id"), comment.get("id"), default=f"{post_id}_c{idx}"))
        nickname = first_non_empty(comment.get("nickname"), default=f"comment_user_{idx}")
        user_id = str(first_non_empty(comment.get("user_id"), default=hash_id("comment_user", nickname)))
        publish_time = parse_time(first_non_empty(comment.get("publish_time"), comment.get("p_time"), default=""))
        rows.append(
            {
                "comment_id": comment_id,
                "post_id": post_id,
                "user_id": user_id,
                "root_comment_id": "",
                "reply_comment_id": "",
                "content": str(first_non_empty(comment.get("content"), default="")),
                "like_count": to_int(comment.get("like_count"), 0),
                "dislike_count": to_int(comment.get("dislike_count"), 0),
                "publish_time": publish_time.strftime("%Y-%m-%d %H:%M:%S"),
                "is_author": int(user_id == author_id),
                "is_hide": int(bool(comment.get("is_hide", False))),
                "status": "normal",
            }
        )
        for reply_idx, reply in enumerate(comment.get("reply_list") or []):
            reply_id = str(first_non_empty(reply.get("comment_id"), reply.get("id"), default=f"{comment_id}_r{reply_idx}"))
            reply_nickname = first_non_empty(reply.get("nickname"), default=f"reply_user_{reply_idx}")
            reply_user_id = str(first_non_empty(reply.get("user_id"), default=hash_id("comment_user", reply_nickname)))
            rows.append(
                {
                    "comment_id": reply_id,
                    "post_id": post_id,
                    "user_id": reply_user_id,
                    "root_comment_id": comment_id,
                    "reply_comment_id": comment_id,
                    "content": str(first_non_empty(reply.get("content"), default="")),
                    "like_count": to_int(reply.get("like_count"), 0),
                    "dislike_count": to_int(reply.get("dislike_count"), 0),
                    "publish_time": parse_time(first_non_empty(reply.get("publish_time"), reply.get("p_time"), default="")).strftime("%Y-%m-%d %H:%M:%S"),
                    "is_author": int(reply_user_id == author_id),
                    "is_hide": int(bool(reply.get("is_hide", False))),
                    "status": "normal",
                }
            )
    return rows


def parse_zanao_jsonl(input_path: str | Path) -> dict[str, pd.DataFrame]:
    """Parse Zanao JSONL into raw normalized dataframes."""

    posts: list[dict[str, Any]] = []
    users: list[dict[str, Any]] = []
    comments: list[dict[str, Any]] = []
    tag_rows: dict[str, dict[str, Any]] = {}
    post_tags: list[dict[str, Any]] = []
    stats: list[dict[str, Any]] = []
    now = pd.Timestamp.now()

    with Path(input_path).open("r", encoding="utf-8", errors="replace") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            list_item = record.get("list_item", {}) or {}
            detail = extract_detail(record)
            post_id = str(first_non_empty(detail.get("thread_id"), list_item.get("thread_id"), record.get("thread_id"), default=f"seed_{line_no}"))
            author_id = str(first_non_empty(detail.get("user_id"), default=hash_id("seed_user", first_non_empty(detail.get("nickname"), list_item.get("nickname"), post_id))))
            title = str(first_non_empty(detail.get("title"), list_item.get("title"), default="校园帖子"))
            content = str(first_non_empty(detail.get("content"), list_item.get("content"), default=""))
            cate_id = str(first_non_empty(detail.get("cate_id"), list_item.get("cate_id"), default=""))
            cate_name = first_non_empty(detail.get("cate_name"), list_item.get("cate_name"), default="校园趣事")
            board, topic_type = map_category(cate_name)
            img_paths = normalize_img_paths(first_non_empty(detail.get("img_paths"), list_item.get("img_paths"), default=[]))
            publish_time = parse_time(first_non_empty(detail.get("pt_time"), detail.get("post_time"), list_item.get("post_time"), list_item.get("p_time"), default=""))
            post_age_hours = max((now - publish_time).total_seconds() / 3600, 0)
            view_count = to_int(first_non_empty(detail.get("view_count"), list_item.get("view_count"), default=0))
            like_count = to_int(first_non_empty(detail.get("like_num"), list_item.get("l_count"), default=0))
            comment_count = to_int(list_item.get("c_count"), 0)
            collect_count = to_int(detail.get("mark_num"), 0)
            dislike_count = to_int(detail.get("dislike_num"), 0)
            tags = extract_seed_tags(board, title, content)
            has_contact_info = int(any(first_non_empty(detail.get(key), default="") for key in ["contact_person", "contact_phone", "contact_qq", "contact_wx"]))
            status = infer_status(detail, list_item)
            price = to_float(first_non_empty(detail.get("post_price"), detail.get("price"), default=0.0))
            need_pay = int(bool(first_non_empty(detail.get("need_pay"), list_item.get("need_pay"), default=False)))
            report_status = str(first_non_empty(detail.get("report_status"), list_item.get("report_status"), default="0"))
            finish_status = str(first_non_empty(detail.get("finish_status"), list_item.get("finish_status"), default="10"))

            posts.append(
                {
                    "post_id": post_id,
                    "author_id": author_id,
                    "cate_id": cate_id,
                    "board": board,
                    "title": title,
                    "content": content,
                    "img_count": len(img_paths),
                    "has_image": int(len(img_paths) > 0),
                    "publish_time": publish_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "post_age_hours": round(post_age_hours, 4),
                    "tags": "|".join(tags),
                    "topic_type": topic_type,
                    "price": price,
                    "need_pay": need_pay,
                    "has_contact_info": has_contact_info,
                    "report_status": report_status,
                    "finish_status": finish_status,
                    "status": status,
                    "content_length": len(content),
                    "keyword_list": "|".join(tags[:5]),
                }
            )
            nickname = str(first_non_empty(detail.get("nickname"), list_item.get("nickname"), default=f"用户{line_no}"))
            users.append(
                {
                    "user_id": author_id,
                    "nickname": nickname,
                    "status": "normal",
                }
            )
            hot_score = view_count * 0.2 + like_count * 0.3 + comment_count * 0.3 + collect_count * 0.2
            final_hot_score = hot_score / (1 + post_age_hours / 24)
            quality_score = (like_count * 0.35 + collect_count * 0.35 + comment_count * 0.30 - dislike_count * 0.20) / max(view_count, 1)
            stats.append(
                {
                    "post_id": post_id,
                    "view_count": view_count,
                    "like_count": like_count,
                    "comment_count": comment_count,
                    "collect_count": collect_count,
                    "dislike_count": dislike_count,
                    "hot_val": to_float(first_non_empty(list_item.get("hot_val"), default=hot_score)),
                    "hot_rank": 0,
                    "hot_score": round(hot_score, 4),
                    "final_hot_score": round(final_hot_score, 4),
                    "quality_score": round(max(quality_score, 0), 4),
                    "update_time": now.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            comments.extend(parse_comments(list_item.get("comment_list") or [], post_id, author_id))
            for tag in tags:
                if tag not in tag_rows:
                    tag_rows[tag] = {
                        "tag_id": f"tag_{len(tag_rows) + 1:04d}",
                        "tag_name": tag,
                        "tag_type": topic_type,
                        "view_count": view_count,
                        "source": "content_extract",
                    }
                else:
                    tag_rows[tag]["view_count"] += view_count
                post_tags.append(
                    {
                        "post_id": post_id,
                        "tag_id": tag_rows[tag]["tag_id"],
                        "tag_name": tag,
                        "tag_weight": 1.0,
                        "source": "extract",
                    }
                )

    posts_df = pd.DataFrame(posts).drop_duplicates("post_id", keep="last")
    stats_df = pd.DataFrame(stats).drop_duplicates("post_id", keep="last")
    if not stats_df.empty:
        stats_df = stats_df.sort_values("final_hot_score", ascending=False).reset_index(drop=True)
        stats_df["hot_rank"] = range(1, len(stats_df) + 1)
    return {
        "raw_posts_df": posts_df,
        "raw_users_df": pd.DataFrame(users).drop_duplicates("user_id", keep="last"),
        "raw_comments_df": pd.DataFrame(comments).drop_duplicates("comment_id", keep="last") if comments else pd.DataFrame(),
        "raw_tags_df": pd.DataFrame(tag_rows.values()),
        "raw_post_tags_df": pd.DataFrame(post_tags).drop_duplicates(["post_id", "tag_name"], keep="last"),
        "raw_post_stats_df": stats_df,
    }
