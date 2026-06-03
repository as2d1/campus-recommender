"""Minimal DeepFM training and ranking demo."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import DATA_DIR, load_all_data
from src.feature_engineering import fit_post_text_vectors
from src.merge_candidates import merge_recall_candidates
from src.preprocess import preprocess_all
from src.ranker import rank_candidates, train_deepfm_model
from src.recall.content_recall import content_recall
from src.recall.hot_recall import hot_recall
from src.recall.itemcf_recall import build_item_similarity, build_positive_interactions, itemcf_recall
from src.recall.latest_recall import latest_recall
from src.recall.profile_recall import profile_recall
from src.user_profile import build_user_profile_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DeepFM and rank a small candidate set.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--user_id", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training_path = Path(DATA_DIR) / "training_samples.csv"
    try:
        model, encoder, logs = train_deepfm_model(
            training_samples_path=training_path,
            epoch=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
    except ModuleNotFoundError as exc:
        print(str(exc))
        return

    data = load_all_data()
    result = preprocess_all(data)
    text_bundle = fit_post_text_vectors(result.posts)
    user_profiles = build_user_profile_table(result.users, result.posts, result.behaviors)
    user_id = args.user_id or str(result.users.iloc[0]["user_id"])

    positive = build_positive_interactions(result.behaviors)
    item_similarity = build_item_similarity(positive)
    recall_results = [
        hot_recall(user_id, result.posts, result.post_stats, top_k=20),
        latest_recall(user_id, result.posts, top_k=20),
        content_recall(
            user_id,
            result.posts,
            result.behaviors,
            text_bundle.post_text_matrix,
            text_bundle.post_id_to_index,
            top_k=20,
        ),
        itemcf_recall(
            user_id,
            result.behaviors,
            result.posts,
            top_k=20,
            item_similarity=item_similarity,
        ),
        profile_recall(user_id, result.users, result.posts, user_profiles, result.post_tags, top_k=20),
    ]
    candidates = merge_recall_candidates(recall_results, result.posts, result.post_stats, top_k_candidates=50)
    ranked = rank_candidates(
        candidate_df=candidates,
        model=model,
        encoder=encoder,
        users_df=result.users,
        posts_df=result.posts,
        post_stats_df=result.post_stats,
        user_profiles_df=user_profiles,
    )

    print("training logs:", logs)
    print("ranked candidates preview:")
    display_columns = ["user_id", "post_id", "rank_score", "merged_recall_score", "recall_sources", "board", "tags"]
    print(ranked[display_columns].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
