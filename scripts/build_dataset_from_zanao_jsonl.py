"""Build standard CSV dataset from Zanao JSONL seed posts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DEFAULT_DATASET_OUTPUT,
    DEFAULT_NUM_BEHAVIORS,
    DEFAULT_NUM_COMMENTS,
    DEFAULT_NUM_POSTS,
    DEFAULT_NUM_QUESTIONNAIRES,
    DEFAULT_NUM_USERS,
    DEFAULT_RANDOM_SEED,
    DEFAULT_ZANAO_INPUT,
)
from src.large_dataset_builder import build_large_dataset
from src.seed_data_expander import ExpansionConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an experimental recommendation dataset from Zanao JSONL seeds.")
    parser.add_argument("--input", default=str(DEFAULT_ZANAO_INPUT), help="Path to zanao_100_posts.jsonl.")
    parser.add_argument("--output", default=str(DEFAULT_DATASET_OUTPUT), help="Output data directory.")
    parser.add_argument("--num_posts", type=int, default=DEFAULT_NUM_POSTS)
    parser.add_argument("--num_users", type=int, default=DEFAULT_NUM_USERS)
    parser.add_argument("--num_behaviors", type=int, default=DEFAULT_NUM_BEHAVIORS)
    parser.add_argument("--num_comments", type=int, default=DEFAULT_NUM_COMMENTS)
    parser.add_argument("--num_questionnaires", type=int, default=DEFAULT_NUM_QUESTIONNAIRES)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--skip_training_samples", action="store_true", help="Do not generate training_samples.csv.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExpansionConfig(
        num_posts=args.num_posts,
        num_users=args.num_users,
        num_behaviors=args.num_behaviors,
        num_comments=args.num_comments,
        num_questionnaires=args.num_questionnaires,
        seed=args.seed,
    )
    build_large_dataset(
        input_path=args.input,
        output_dir=args.output,
        config=config,
        build_training_samples=not args.skip_training_samples,
    )


if __name__ == "__main__":
    main()
