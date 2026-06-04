"""Configuration for the campus recommender."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DEFAULT_ZANAO_INPUT = RAW_DATA_DIR / "zanao_100_posts.jsonl"
DEFAULT_DATASET_OUTPUT = DATA_DIR
DEFAULT_SQLITE_PATH = PROJECT_ROOT.parent / "zanao.sqlite"

DEFAULT_NUM_POSTS = 2000
DEFAULT_NUM_USERS = 500
DEFAULT_NUM_BEHAVIORS = 30000
DEFAULT_NUM_COMMENTS = 6000
DEFAULT_NUM_QUESTIONNAIRES = 150
DEFAULT_RANDOM_SEED = 2026

BOARD_TOPIC_MAP = {
    "打听求助": ("打听求助", "生活"),
    "二手闲置": ("二手闲置", "生活"),
    "恋爱交友": ("恋爱交友", "社交"),
    "校园趣事": ("校园趣事", "社交"),
    "兼职招聘": ("兼职招聘", "发展"),
    "校园招聘": ("校园招聘", "发展"),
}

ALL_BOARDS = [
    "打听求助",
    "恋爱交友",
    "校园趣事",
    "兼职招聘",
    "校园招聘",
    "二手闲置",
]

BOARD_DEFAULT_TAGS = {
    "打听求助": ["求问", "校园卡", "宿舍", "打印店", "课程资料"],
    "恋爱交友": ["交友", "搭子", "羽毛球", "拍照", "活动"],
    "校园趣事": ["校园趣事", "校园生活", "宿舍", "校车", "吐槽"],
    "兼职招聘": ["兼职", "招聘", "家教", "薪资", "实习"],
    "校园招聘": ["校园招聘", "就业", "招聘", "简历", "面试"],
    "二手闲置": ["二手闲置", "闲置", "教材", "电动车", "租房"],
}

ACTION_WEIGHTS = {
    "expose": 0,
    "skip": -1,
    "view": 1,
    "like": 2,
    "comment": 3,
    "collect": 4,
    "dislike": -2,
    "report": -4,
}
