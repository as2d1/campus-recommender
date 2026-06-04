"""Main entry point for the campus forum recommendation prototype."""

from __future__ import annotations

import sys


def configure_console_encoding() -> None:
    """Make Windows console output tolerant of emoji and other Unicode text."""

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


configure_console_encoding()

from src.data_loader import load_all_data
from src.evaluate import evaluate_offline
from src.feature_engineering import build_training_features, fit_post_text_vectors
from src.multi_interest import build_multi_interest_representations, calculate_multi_interest_score
from src.preprocess import preprocess_all
from src.recommend import recommend_for_user
from src.train import ENCODER_PATH, MODEL_PATH, train_pipeline
from src.user_profile import build_user_profile_table, build_user_profiles


def prepare_features_and_profiles() -> tuple:
    """Run preprocessing, profile construction, multi-interest, and training samples."""

    data = load_all_data()
    result = preprocess_all(data)
    text_bundle = fit_post_text_vectors(result.posts)
    profile_table = build_user_profile_table(result.users, result.posts, result.behaviors)

    multi_bundle = build_multi_interest_representations(
        users=result.users,
        posts=result.posts,
        behaviors=result.behaviors,
        post_text_matrix=text_bundle.post_text_matrix,
        post_id_to_index=text_bundle.post_id_to_index,
    )
    profiles = build_user_profiles(
        users=result.users,
        posts=result.posts,
        behaviors=result.behaviors,
        user_profiles_df=profile_table,
        post_id_to_index=text_bundle.post_id_to_index,
        text_matrix=text_bundle.post_text_matrix,
    )
    training_features = build_training_features(
        samples=result.samples,
        users=result.users,
        posts=result.posts,
        behaviors=result.behaviors,
        user_profiles_df=profile_table,
        profiles=profiles,
        text_bundle=text_bundle,
    )
    sample_user_id = str(result.users.iloc[0]["user_id"])
    sample_post_id = str(result.posts.iloc[0]["post_id"])
    sample_score = calculate_multi_interest_score(multi_bundle, sample_user_id, sample_post_id)
    print("Data and feature preparation finished.")
    print(f"valid users: {len(result.users)}")
    print(f"valid posts: {len(result.posts)}")
    print(f"valid behaviors: {len(result.behaviors)}")
    print(f"training samples: {len(training_features)}")
    print(f"sample multi_interest_score({sample_user_id}, {sample_post_id}): {sample_score:.6f}")
    return result, profile_table


def main() -> None:
    result, _ = prepare_features_and_profiles()

    print("\nTraining DeepFM...")
    _, _, logs = train_pipeline(epoch=10, batch_size=128, learning_rate=5e-3)
    print(f"training logs: {logs}")
    print(f"model path: {MODEL_PATH}")
    print(f"encoder path: {ENCODER_PATH}")

    user_id = str(result.users.iloc[0]["user_id"])
    print(f"\nTop10 recommendations for user_id={user_id}")
    recommendations = recommend_for_user(user_id, top_n=10, log_score_stats=True)
    print(recommendations.to_string(index=False))

    print("\nOffline evaluation metrics")
    metrics = evaluate_offline(max_users=5, top_n=20)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
