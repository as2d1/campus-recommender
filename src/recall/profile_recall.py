"""User profile based recall."""

from __future__ import annotations

import pandas as pd

from src.recall import filter_valid_posts, format_recall_result, split_tags


def grade_bonus(grade: str, post: pd.Series) -> float:
    """Simple grade-topic matching rules."""

    text = f"{post.get('board', '')}|{post.get('tags', '')}|{post.get('title', '')}|{post.get('content', '')}"
    if grade == "大一":
        return 0.25 if any(key in text for key in ["新生攻略", "社团", "课程评价", "选课"]) else 0.0
    if grade == "大二":
        return 0.25 if any(key in text for key in ["竞赛", "课程资料", "社团"]) else 0.0
    if grade == "大三":
        return 0.30 if any(key in text for key in ["考研", "实习", "竞赛", "保研"]) else 0.0
    if grade == "大四":
        return 0.30 if any(key in text for key in ["毕业", "就业", "租房", "二手闲置"]) else 0.0
    if str(grade).startswith("研"):
        return 0.30 if any(key in text for key in ["科研", "实习", "招聘", "租房"]) else 0.0
    return 0.0


def major_bonus(college: str, major: str, post: pd.Series) -> float:
    text = f"{post.get('board', '')}|{post.get('tags', '')}|{post.get('title', '')}|{post.get('content', '')}"
    if "计算机" in college or major in {"软件工程", "人工智能", "数据科学", "网络工程"}:
        return 0.20 if any(key in text for key in ["408", "竞赛", "科研", "课程资料", "实习"]) else 0.0
    if "管理" in college:
        return 0.15 if any(key in text for key in ["就业", "简历", "兼职", "招聘", "二手闲置"]) else 0.0
    if "公共卫生" in college or "医学" in major:
        return 0.15 if any(key in text for key in ["科研", "考研", "考试", "课程资料"]) else 0.0
    return 0.0


def profile_recall(
    user_id: str,
    users: pd.DataFrame,
    posts: pd.DataFrame,
    user_profiles: pd.DataFrame,
    post_tags: pd.DataFrame | None = None,
    top_k: int = 50,
) -> pd.DataFrame:
    """Recall posts matching user tags, boards, grade, college, and major."""

    user_rows = users[users["user_id"].astype(str).eq(str(user_id))]
    if user_rows.empty:
        return pd.DataFrame(columns=["user_id", "post_id", "recall_score", "recall_source"])
    user = user_rows.iloc[0]

    profile_rows = user_profiles[user_profiles["user_id"].astype(str).eq(str(user_id))] if not user_profiles.empty else pd.DataFrame()
    profile = profile_rows.iloc[0] if not profile_rows.empty else pd.Series(dtype=object)

    interest_tags = set(split_tags(user.get("interest_tags", "")))
    interest_tags |= set(split_tags(profile.get("long_term_tags", "")))
    interest_tags |= set(split_tags(profile.get("short_term_tags", "")))
    preferred_boards = set(split_tags(profile.get("preferred_boards", "")))

    candidates = filter_valid_posts(posts).copy()
    board_sizes = candidates.groupby("board").size().to_dict()
    scores = []
    for _, post in candidates.iterrows():
        post_tag_set = set(split_tags(post.get("tags", "")))
        if post_tags is not None and not post_tags.empty:
            extra_tags = post_tags[post_tags["post_id"].astype(str).eq(str(post["post_id"]))]["tag_name"].astype(str)
            post_tag_set |= set(extra_tags)
        tag_score = len(interest_tags & post_tag_set) / max(len(interest_tags | post_tag_set), 1)
        board = str(post.get("board", ""))
        board_score = 0.35 if board in preferred_boards else 0.0
        if board in preferred_boards and board_sizes.get(board, 0) <= 10:
            board_score += 1.2
        score = (
            tag_score * 1.2
            + board_score
            + grade_bonus(str(user.get("grade", "")), post)
            + major_bonus(str(user.get("college", "")), str(user.get("major", "")), post)
        )
        scores.append(score)

    candidates["profile_score"] = scores
    candidates = candidates[candidates["profile_score"] > 0]
    return format_recall_result(user_id, candidates, "profile", "profile_score", top_k)
