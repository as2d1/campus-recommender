"""Smoke test for candidate merge and reranking."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_all_data
from src.feature_engineering import fit_post_text_vectors
from src.merge_candidates import merge_recall_candidates
from src.preprocess import preprocess_all
from src.recall.content_recall import content_recall
from src.recall.hot_recall import hot_recall
from src.recall.itemcf_recall import build_item_similarity, build_positive_interactions, itemcf_recall
from src.recall.latest_recall import latest_recall
from src.recall.profile_recall import profile_recall
from src.recall.scene_recall import time_scene_recall
from src.rerank import rerank_candidates
from src.user_profile import build_user_profile_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run merge and rerank smoke test.")
    parser.add_argument("--user_id", default="", help="User id to test. Defaults to the first valid user.")
    parser.add_argument("--recall_top_k", type=int, default=30)
    parser.add_argument("--candidate_top_k", type=int, default=120)
    parser.add_argument("--top_n", type=int, default=10)
    return parser.parse_args()


def build_all_recall_results(user_id: str, result, text_bundle, user_profiles, recall_top_k: int):
    time_period = "晚上"
    positive = build_positive_interactions(result.behaviors)
    item_similarity = build_item_similarity(positive)
    return [
        hot_recall(user_id, result.posts, result.post_stats, top_k=recall_top_k),
        latest_recall(user_id, result.posts, top_k=recall_top_k),
        content_recall(
            user_id,
            result.posts,
            result.behaviors,
            text_bundle.post_text_matrix,
            text_bundle.post_id_to_index,
            top_k=recall_top_k,
        ),
        itemcf_recall(
            user_id,
            result.behaviors,
            result.posts,
            top_k=recall_top_k,
            item_similarity=item_similarity,
        ),
        profile_recall(
            user_id,
            result.users,
            result.posts,
            user_profiles,
            result.post_tags,
            top_k=recall_top_k,
        ),
        time_scene_recall(user_id, result.posts, time_period=time_period, top_k=recall_top_k),
    ]


def main() -> None:
    args = parse_args()
    data = load_all_data()
    result = preprocess_all(data)
    text_bundle = fit_post_text_vectors(result.posts)
    user_profiles = build_user_profile_table(result.users, result.posts, result.behaviors)

    user_id = args.user_id or str(result.users.iloc[0]["user_id"])
    recall_results = build_all_recall_results(user_id, result, text_bundle, user_profiles, args.recall_top_k)
    candidates = merge_recall_candidates(
        recall_results=recall_results,
        posts=result.posts,
        post_stats=result.post_stats,
        top_k_candidates=args.candidate_top_k,
    )
    final_recommendations = rerank_candidates(
        candidates=candidates,
        user_id=user_id,
        user_profiles=user_profiles,
        behaviors=result.behaviors,
        top_n=args.top_n,
    )

    print(f"user_id: {user_id}")
    print("recall sizes:", [len(df) for df in recall_results])
    print(f"merged candidates: {len(candidates)}")
    print(f"final recommendations: {len(final_recommendations)}")
    display_columns = [
        "final_rank",
        "user_id",
        "post_id",
        "rerank_score",
        "merged_recall_score",
        "recall_sources",
        "source_count",
        "board",
        "tags",
        "post_age_hours",
    ]
    existing_columns = [column for column in display_columns if column in final_recommendations.columns]
    print(final_recommendations[existing_columns].head(args.top_n).to_string(index=False))


if __name__ == "__main__":
    main()
