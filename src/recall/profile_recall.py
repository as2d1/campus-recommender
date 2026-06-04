"""User profile based recall."""

from __future__ import annotations

import pandas as pd

from src.recall import filter_valid_posts, format_recall_result, split_tags


def profile_recall(
    user_id: str,
    users: pd.DataFrame,
    posts: pd.DataFrame,
    user_profiles: pd.DataFrame,
    post_tags: pd.DataFrame | None = None,
    top_k: int = 50,
) -> pd.DataFrame:
    """Recall posts matching behavior-derived user tags and boards."""

    user_rows = users[users["user_id"].astype(str).eq(str(user_id))]
    if user_rows.empty:
        return pd.DataFrame(columns=["user_id", "post_id", "recall_score", "recall_source"])
    user = user_rows.iloc[0]

    profile_rows = user_profiles[user_profiles["user_id"].astype(str).eq(str(user_id))] if not user_profiles.empty else pd.DataFrame()
    profile = profile_rows.iloc[0] if not profile_rows.empty else pd.Series(dtype=object)

    profile_tags = set(split_tags(profile.get("long_term_tags", "")))
    profile_tags |= set(split_tags(profile.get("short_term_tags", "")))
    preferred_boards = set(split_tags(profile.get("preferred_boards", "")))

    candidates = filter_valid_posts(posts).copy()
    board_sizes = candidates.groupby("board").size().to_dict()
    scores = []
    for _, post in candidates.iterrows():
        post_tag_set = set(split_tags(post.get("tags", "")))
        if post_tags is not None and not post_tags.empty:
            extra_tags = post_tags[post_tags["post_id"].astype(str).eq(str(post["post_id"]))]["tag_name"].astype(str)
            post_tag_set |= set(extra_tags)
        tag_score = len(profile_tags & post_tag_set) / max(len(profile_tags | post_tag_set), 1)
        board = str(post.get("board", ""))
        board_score = 0.35 if board in preferred_boards else 0.0
        if board in preferred_boards and board_sizes.get(board, 0) <= 10:
            board_score += 1.2
        score = (
            tag_score * 1.2
            + board_score
        )
        scores.append(score)

    candidates["profile_score"] = scores
    candidates = candidates[candidates["profile_score"] > 0]
    return format_recall_result(user_id, candidates, "profile", "profile_score", top_k)
