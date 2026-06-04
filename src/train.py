"""Offline training workflow for the campus recommender."""

from __future__ import annotations

from pathlib import Path

from src.data_loader import PROJECT_ROOT, load_all_data
from src.feature_engineering import (
    build_training_features,
    fit_post_text_vectors,
)
from src.preprocess import preprocess_all
from src.ranker import save_encoder, save_model, train_deepfm_model
from src.user_profile import build_user_profile_table, build_user_profiles


MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "deepfm.pt"
ENCODER_PATH = MODEL_DIR / "feature_encoder.json"


def build_current_training_samples():
    """Build DeepFM training samples directly from the SQLite-backed data."""
    data = load_all_data()
    result = preprocess_all(data)
    text_bundle = fit_post_text_vectors(result.posts)
    profile_table = build_user_profile_table(result.users, result.posts, result.behaviors)
    profiles = build_user_profiles(
        users=result.users,
        posts=result.posts,
        behaviors=result.behaviors,
        user_profiles_df=profile_table,
        post_id_to_index=text_bundle.post_id_to_index,
        text_matrix=text_bundle.post_text_matrix,
    )
    return build_training_features(
        samples=result.samples,
        users=result.users,
        posts=result.posts,
        behaviors=result.behaviors,
        user_profiles_df=profile_table,
        profiles=profiles,
        text_bundle=text_bundle,
    )


def train_pipeline(
    epoch: int = 3,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    training_samples=None,
    model_path: str | Path = MODEL_PATH,
    encoder_path: str | Path = ENCODER_PATH,
):
    """Train DeepFM and save model artifacts."""

    samples = training_samples if training_samples is not None else build_current_training_samples()
    model, encoder, logs = train_deepfm_model(
        training_samples=samples,
        epoch=epoch,
        batch_size=batch_size,
        learning_rate=learning_rate,
        embed_dim=8,
        hidden_dims=[32, 16],
        dropout=0.2,
    )
    save_model(model, model_path)
    save_encoder(encoder, encoder_path)
    print(f"model saved to: {model_path}")
    print(f"encoder saved to: {encoder_path}")
    return model, encoder, logs


if __name__ == "__main__":
    train_pipeline(epoch=1, batch_size=512)
