"""Reranking rules for final recommendation lists."""

from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd

from src.user_profile import split_tags


def get_seen_posts(behaviors: pd.DataFrame, user_id: str) -> set[str]:
    """Return posts already interacted with by the user."""

    if behaviors is None or behaviors.empty:
        return set()
    rows = behaviors[behaviors["user_id"].astype(str).eq(str(user_id))]
    return set(rows["post_id"].astype(str))


def filter_candidates(
    candidates: pd.DataFrame,
    user_id: str,
    behaviors: pd.DataFrame | None = None,
    uninterested_tags: set[str] | None = None,
    exclude_seen: bool = True,
) -> pd.DataFrame:
    """Apply hard filters before reranking."""

    filtered = candidates.copy()
    if "status" in filtered.columns:
        filtered = filtered[filtered["status"].fillna("normal").eq("normal")]
    if "report_status" in filtered.columns:
        filtered = filtered[~filtered["report_status"].fillna("normal").isin(["blocked", "deleted", "abnormal", "违规", "suspect"])]
    if exclude_seen and behaviors is not None:
        seen = get_seen_posts(behaviors, user_id)
        filtered = filtered[~filtered["post_id"].astype(str).isin(seen)]
    if uninterested_tags:
        filtered = filtered[
            ~filtered["tags"].apply(lambda value: bool(set(split_tags(value)) & uninterested_tags))
        ]
    return filtered.reset_index(drop=True)


def find_user_profile(user_profiles: pd.DataFrame, user_id: str) -> pd.Series:
    """Return the user's profile row or an empty Series."""

    if user_profiles is None or user_profiles.empty or "user_id" not in user_profiles.columns:
        return pd.Series(dtype=object)
    rows = user_profiles[user_profiles["user_id"].astype(str).eq(str(user_id))]
    return rows.iloc[0] if not rows.empty else pd.Series(dtype=object)


def novelty_bonus(row: pd.Series, profile: pd.Series, existing_tag_counter: Counter[str]) -> float:
    """Reward fresh but still relevant tags."""

    post_tags = set(split_tags(row.get("tags", "")))
    long_tags = set(split_tags(profile.get("long_term_tags", "")))
    short_tags = set(split_tags(profile.get("short_term_tags", "")))
    if not post_tags:
        return 0.0

    bonus = 0.0
    new_tags = post_tags - long_tags
    if new_tags & short_tags:
        bonus += 0.08
    if new_tags and str(row.get("topic_type", "")) in {"学习", "生活", "社交", "发展"}:
        bonus += 0.03
    repeated_tag_penalty = sum(existing_tag_counter[tag] for tag in post_tags) * 0.015
    return max(bonus - repeated_tag_penalty, -0.08)


def freshness_bonus(row: pd.Series) -> float:
    """Reward timely campus posts."""

    age_hours = float(row.get("post_age_hours", 0.0) or 0.0)
    freshness = 0.12 / (1.0 + age_hours / 24.0)
    text = f"{row.get('board', '')}|{row.get('tags', '')}|{row.get('title', '')}|{row.get('content', '')}"
    if any(keyword in text for keyword in ["活动", "考试", "招聘", "实习", "校招", "通知"]):
        freshness += 0.05 / (1.0 + age_hours / 48.0)
    return freshness


def base_score_column(candidates: pd.DataFrame) -> str:
    """Use rank_score if present, otherwise merged_recall_score."""

    if "rank_score" in candidates.columns:
        return "rank_score"
    if "merged_recall_score" in candidates.columns:
        return "merged_recall_score"
    raise ValueError("Candidates must contain rank_score or merged_recall_score.")


def apply_score_adjustments(
    candidates: pd.DataFrame,
    user_id: str,
    user_profiles: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute rerank_score before diversity selection."""

    scored = candidates.copy()
    score_col = base_score_column(scored)
    scored["base_score"] = pd.to_numeric(scored[score_col], errors="coerce").fillna(0.0)
    profile = find_user_profile(user_profiles, user_id)

    tag_counter: Counter[str] = Counter()
    novelty_values = []
    for row in scored.sort_values("base_score", ascending=False).itertuples(index=False):
        series = pd.Series(row._asdict())
        bonus = novelty_bonus(series, profile, tag_counter)
        novelty_values.append((series["post_id"], bonus))
        tag_counter.update(split_tags(series.get("tags", "")))
    novelty_map = dict(novelty_values)

    scored["novelty_bonus"] = scored["post_id"].map(novelty_map).fillna(0.0)
    scored["freshness_bonus"] = scored.apply(freshness_bonus, axis=1)
    scored["diversity_adjustment"] = 0.0
    scored["rerank_score"] = (
        scored["base_score"]
        + scored["novelty_bonus"]
        + scored["freshness_bonus"]
        + scored["diversity_adjustment"]
    )
    return scored.sort_values("rerank_score", ascending=False).reset_index(drop=True)


def violates_diversity(
    row: pd.Series,
    selected: list[pd.Series],
    board_counts_top10: defaultdict[str, int],
    max_board_top10: int,
    max_consecutive_board: int,
    top_n: int,
) -> bool:
    """Check board diversity constraints."""

    board = str(row.get("board", ""))
    current_position = len(selected) + 1
    if current_position <= min(10, top_n) and board_counts_top10[board] >= max_board_top10:
        return True
    if max_consecutive_board > 0 and len(selected) >= max_consecutive_board:
        recent_boards = [str(item.get("board", "")) for item in selected[-max_consecutive_board:]]
        if all(recent_board == board for recent_board in recent_boards):
            return True
    return False


def diversity_select(
    scored: pd.DataFrame,
    top_n: int,
    max_board_top10: int = 4,
    max_consecutive_board: int = 2,
) -> pd.DataFrame:
    """Greedy diversity-aware selection."""

    remaining = [pd.Series(row._asdict()) for row in scored.itertuples(index=False)]
    selected: list[pd.Series] = []
    board_counts_top10: defaultdict[str, int] = defaultdict(int)

    while remaining and len(selected) < top_n:
        chosen_index = None
        for idx, row in enumerate(remaining):
            if not violates_diversity(row, selected, board_counts_top10, max_board_top10, max_consecutive_board, top_n):
                chosen_index = idx
                break
        if chosen_index is None:
            chosen_index = 0
        chosen = remaining.pop(chosen_index)
        if len(selected) < 10:
            board_counts_top10[str(chosen.get("board", ""))] += 1
        selected.append(chosen)

    if not selected:
        return pd.DataFrame(columns=scored.columns)
    result = pd.DataFrame(selected)
    result["final_rank"] = range(1, len(result) + 1)
    return result.reset_index(drop=True)


def rerank_candidates(
    candidates: pd.DataFrame,
    user_id: str,
    user_profiles: pd.DataFrame | None = None,
    behaviors: pd.DataFrame | None = None,
    top_n: int = 20,
    exclude_seen: bool = True,
    uninterested_tags: set[str] | None = None,
    max_board_top10: int = 4,
    max_consecutive_board: int = 2,
) -> pd.DataFrame:
    """Rerank candidates with filtering, novelty, freshness, and diversity."""

    filtered = filter_candidates(
        candidates=candidates,
        user_id=user_id,
        behaviors=behaviors,
        uninterested_tags=uninterested_tags,
        exclude_seen=exclude_seen,
    )
    if filtered.empty:
        return filtered
    scored = apply_score_adjustments(filtered, user_id, user_profiles)
    return diversity_select(
        scored=scored,
        top_n=top_n,
        max_board_top10=max_board_top10,
        max_consecutive_board=max_consecutive_board,
    )
