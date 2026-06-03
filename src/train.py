"""Offline training workflow for the campus recommender."""

from __future__ import annotations

from pathlib import Path

from src.data_loader import DATA_DIR, PROJECT_ROOT, load_all_data, save_dataframe
from src.feature_engineering import (
    TRAINING_SAMPLE_COLUMNS,
    build_training_features,
    fit_post_text_vectors,
    save_training_samples,
)
from src.preprocess import preprocess_all
from src.ranker import save_encoder, save_model, train_deepfm_model
from src.user_profile import build_user_profile_table, build_user_profiles, save_user_profiles


MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "deepfm.pt"
ENCODER_PATH = MODEL_DIR / "feature_encoder.json"
TRAINING_SAMPLES_PATH = DATA_DIR / "training_samples.csv"
USER_PROFILES_PATH = DATA_DIR / "user_profiles.csv"


def ensure_training_samples(
    training_samples_path: str | Path = TRAINING_SAMPLES_PATH,
) -> Path:
    """Create training_samples.csv when it does not exist."""

    output_path = Path(training_samples_path)
    if output_path.exists():
        return output_path

    data = load_all_data()
    result = preprocess_all(data)
    text_bundle = fit_post_text_vectors(result.posts)
    profile_table = build_user_profile_table(result.users, result.posts, result.behaviors)
    save_user_profiles(profile_table, USER_PROFILES_PATH)
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
    save_training_samples(training_features, str(output_path))
    return output_path


def train_pipeline(
    epoch: int = 3,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    training_samples_path: str | Path = TRAINING_SAMPLES_PATH,
    model_path: str | Path = MODEL_PATH,
    encoder_path: str | Path = ENCODER_PATH,
):
    """Train DeepFM and save model artifacts."""

    sample_path = ensure_training_samples(training_samples_path)
    model, encoder, logs = train_deepfm_model(
        training_samples_path=sample_path,
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


def export_training_samples_snapshot(output_path: str | Path) -> None:
    """Save a clean copy of current training samples for reports."""

    sample_path = ensure_training_samples()
    import pandas as pd

    df = pd.read_csv(sample_path, encoding="utf-8-sig")
    save_dataframe(df[TRAINING_SAMPLE_COLUMNS], output_path)


if __name__ == "__main__":
    train_pipeline(epoch=1, batch_size=512)
