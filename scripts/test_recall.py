"""Smoke test for all recall channels."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_all_data
from src.feature_engineering import fit_post_text_vectors
from src.preprocess import preprocess_all
from src.recall.cold_start_recall import cold_start_recall
from src.recall.content_recall import content_recall
from src.recall.hot_recall import hot_recall
from src.recall.itemcf_recall import build_item_similarity, build_positive_interactions, itemcf_recall
from src.recall.latest_recall import latest_recall
from src.recall.profile_recall import profile_recall
from src.recall.scene_recall import location_scene_recall, time_scene_recall
from src.user_profile import build_user_profile_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run recall channel smoke test.")
    parser.add_argument("--user_id", default="", help="User id to test. Defaults to the first valid user.")
    parser.add_argument("--top_k", type=int, default=20, help="Recall size for each channel.")
    return parser.parse_args()


def print_channel(name: str, df) -> None:
    print(f"\n[{name}] top 5")
    if df.empty:
        print("(empty)")
    else:
        print(df.head(5).to_string(index=False))


def main() -> None:
    args = parse_args()
    data = load_all_data()
    result = preprocess_all(data)
    text_bundle = fit_post_text_vectors(result.posts)
    user_profiles = build_user_profile_table(result.users, result.posts, result.behaviors)

    user_id = args.user_id or str(result.users.iloc[0]["user_id"])
    user_row = result.users[result.users["user_id"].astype(str).eq(user_id)].iloc[0]
    time_period = "晚上"
    scene = "普通浏览"
    location = str(user_row.get("default_location", "教学区"))

    positive = build_positive_interactions(result.behaviors)
    item_similarity = build_item_similarity(positive)

    channels = {
        "hot": hot_recall(user_id, result.posts, result.post_stats, top_k=args.top_k),
        "latest": latest_recall(user_id, result.posts, top_k=args.top_k),
        "content": content_recall(
            user_id,
            result.posts,
            result.behaviors,
            text_bundle.post_text_matrix,
            text_bundle.post_id_to_index,
            top_k=args.top_k,
        ),
        "itemcf": itemcf_recall(
            user_id,
            result.behaviors,
            result.posts,
            top_k=args.top_k,
            item_similarity=item_similarity,
        ),
        "profile": profile_recall(
            user_id,
            result.users,
            result.posts,
            user_profiles,
            result.post_tags,
            top_k=args.top_k,
        ),
        "time_scene": time_scene_recall(user_id, result.posts, time_period=time_period, scene=scene, top_k=args.top_k),
        "location_scene": location_scene_recall(user_id, result.posts, location=location, top_k=args.top_k),
        "cold_start": cold_start_recall(
            user_id,
            result.users,
            result.questionnaire,
            result.posts,
            result.post_stats,
            top_k=args.top_k,
            time_period=time_period,
            scene=scene,
        ),
    }

    print(f"Testing recall channels for user_id={user_id}")
    for name, df in channels.items():
        print_channel(name, df)


if __name__ == "__main__":
    main()
