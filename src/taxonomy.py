"""Shared board, topic, and tag taxonomy for the recommender."""

from __future__ import annotations

import pandas as pd


BOARD_TOPIC_MAP = {
    "打听求助": "生活",
    "恋爱交友": "社交",
    "校园趣事": "社交",
    "兼职招聘": "发展",
    "校园招聘": "发展",
    "二手闲置": "生活",
}

ALLOWED_BOARDS = list(BOARD_TOPIC_MAP.keys())

BOARD_DEFAULT_TAGS = {
    "打听求助": ["求助", "课程", "宿舍", "食堂", "考试", "选课"],
    "恋爱交友": ["交友", "搭子", "活动", "拍照"],
    "校园趣事": ["校园趣事", "校园生活", "活动", "吐槽"],
    "兼职招聘": ["兼职", "招聘", "家教", "实习"],
    "校园招聘": ["校园招聘", "就业", "招聘", "简历", "面试", "校招"],
    "二手闲置": ["二手", "闲置", "教材", "电动车", "租房"],
}

INTEREST_DIRECTIONS = {
    "learning": {
        "topic_types": {"学习"},
        "boards": set(),
        "tags": {"课程", "课程资料", "考试", "考研", "保研", "学习经验", "408", "复习资料", "课程评价", "选课"},
        "profile_column": "learning_interest_weight",
    },
    "life": {
        "topic_types": {"生活"},
        "boards": {"打听求助", "二手闲置"},
        "tags": {"求助", "租房", "宿舍", "食堂", "拼饭", "二手", "二手闲置", "校园生活", "教材", "闲置", "求问"},
        "profile_column": "life_interest_weight",
    },
    "social": {
        "topic_types": {"社交"},
        "boards": {"恋爱交友", "校园趣事"},
        "tags": {"交友", "搭子", "校园趣事", "活动", "社团", "招新", "志愿服务", "校园墙", "吐槽", "拍照"},
        "profile_column": "social_interest_weight",
    },
    "career": {
        "topic_types": {"发展"},
        "boards": {"兼职招聘", "校园招聘"},
        "tags": {"实习", "就业", "兼职", "招聘", "家教", "简历", "面试", "校招"},
        "profile_column": "career_interest_weight",
    },
}


def split_tags(value: object) -> list[str]:
    """Split pipe-separated tags and drop blanks."""

    if pd.isna(value):
        return []
    return [tag.strip() for tag in str(value).split("|") if tag.strip()]


def join_tags(tags: list[str]) -> str:
    """Join tags with stable de-duplication."""

    return "|".join(dict.fromkeys([str(tag).strip() for tag in tags if str(tag).strip()]))


def topic_for_board(board: object) -> str:
    """Return the coarse topic type for a forum board."""

    return BOARD_TOPIC_MAP.get(str(board), "生活")


def tags_for_board(board: object) -> list[str]:
    """Return taxonomy tags for one board."""

    board_text = str(board)
    return [board_text, topic_for_board(board_text), *BOARD_DEFAULT_TAGS.get(board_text, [])]


def tags_for_post(board: object, extra_tags: object = "") -> str:
    """Build normalized display/algorithm tags for a post."""

    return join_tags([*tags_for_board(board), *split_tags(extra_tags)])


def cold_start_tags() -> list[str]:
    """Return a compact tag list for onboarding."""

    tags: list[str] = []
    for board in ALLOWED_BOARDS:
        tags.extend(BOARD_DEFAULT_TAGS.get(board, []))
    return list(dict.fromkeys(tags))


def infer_interest_direction(board: object = "", tags: object = "", topic_type: object = "") -> str:
    """Infer one of learning/life/social/career from topic, board, or tags."""

    topic = str(topic_type)
    board_text = str(board)
    tag_set = set(split_tags(tags))
    for direction, rules in INTEREST_DIRECTIONS.items():
        if topic in rules["topic_types"] or board_text in rules["boards"] or tag_set & rules["tags"]:
            return direction
    return "life"


def taxonomy_payload() -> dict[str, object]:
    """Return frontend-friendly taxonomy metadata."""

    return {
        "boards": ALLOWED_BOARDS,
        "board_topics": BOARD_TOPIC_MAP,
        "board_tags": BOARD_DEFAULT_TAGS,
        "cold_start_tags": cold_start_tags(),
        "interest_directions": {
            direction: {
                "topic_types": sorted(rules["topic_types"]),
                "boards": sorted(rules["boards"]),
                "tags": sorted(rules["tags"]),
            }
            for direction, rules in INTEREST_DIRECTIONS.items()
        },
    }
