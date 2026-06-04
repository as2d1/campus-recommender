"""DeepFM ranker training and candidate scoring utilities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from src.models.deepfm import DeepFM
except ModuleNotFoundError:  # pragma: no cover - environment guard.
    torch = None
    nn = None
    DataLoader = None
    Dataset = object
    DeepFM = Any


SPARSE_FEATURES = [
    "user_id",
    "post_id",
    "grade",
    "college",
    "major",
    "campus",
    "board",
    "tags",
    "topic_type",
    "time_period",
    "location",
    "location_scope",
    "device_type",
    "recall_sources",
]

DENSE_FEATURES = [
    "hot_score",
    "final_hot_score",
    "quality_score",
    "content_sim_score",
    "itemcf_score",
    "profile_match_score",
    "time_scene_score",
    "location_scene_score",
    "multi_interest_score",
    "source_count",
    "post_age_hours",
]

LEAKAGE_COLUMNS = [
    "label",
    "is_positive",
    "action_type",
    "action_weight",
    "dwell_time",
    "like_count",
    "comment_count",
    "collect_count",
    "rank_score",
    "rerank_score",
]


@dataclass
class EncoderConfig:
    sparse_features: list[str]
    dense_features: list[str]
    category_maps: dict[str, dict[str, int]]
    dense_means: dict[str, float]
    dense_stds: dict[str, float]


class FeatureEncoder:
    """Dictionary encoder for sparse features and z-score scaler for dense features."""

    unknown_token = "__UNK__"

    def __init__(
        self,
        sparse_features: list[str] | None = None,
        dense_features: list[str] | None = None,
    ) -> None:
        self.sparse_features = sparse_features or SPARSE_FEATURES
        self.dense_features = dense_features or DENSE_FEATURES
        self.category_maps: dict[str, dict[str, int]] = {}
        self.dense_means: dict[str, float] = {}
        self.dense_stds: dict[str, float] = {}

    def ensure_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing feature columns with safe defaults."""

        data = df.copy()
        if "recall_source" in data.columns and "recall_sources" not in data.columns:
            data["recall_sources"] = data["recall_source"]
        for column in self.sparse_features:
            if column not in data.columns:
                data[column] = "unknown"
            data[column] = data[column].fillna("unknown").astype(str)
        for column in self.dense_features:
            if column not in data.columns:
                data[column] = 0.0
            data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0.0)
        return data

    def fit(self, df: pd.DataFrame) -> "FeatureEncoder":
        data = self.ensure_columns(df)
        for column in self.sparse_features:
            values = sorted(data[column].astype(str).unique().tolist())
            mapping = {self.unknown_token: 0}
            mapping.update({value: idx + 1 for idx, value in enumerate(values) if value != self.unknown_token})
            self.category_maps[column] = mapping
        for column in self.dense_features:
            values = pd.to_numeric(data[column], errors="coerce").fillna(0.0)
            mean = float(values.mean())
            std = float(values.std())
            self.dense_means[column] = mean
            self.dense_stds[column] = std if std > 1e-8 else 1.0
        return self

    def transform_sparse(self, df: pd.DataFrame) -> np.ndarray:
        data = self.ensure_columns(df)
        encoded = []
        for column in self.sparse_features:
            mapping = self.category_maps[column]
            encoded.append(data[column].map(mapping).fillna(0).astype("int64").to_numpy())
        return np.stack(encoded, axis=1)

    def transform_dense(self, df: pd.DataFrame) -> np.ndarray:
        data = self.ensure_columns(df)
        dense_arrays = []
        for column in self.dense_features:
            values = pd.to_numeric(data[column], errors="coerce").fillna(0.0)
            normalized = (values - self.dense_means[column]) / self.dense_stds[column]
            dense_arrays.append(normalized.astype("float32").to_numpy())
        return np.stack(dense_arrays, axis=1)

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return self.transform_sparse(df), self.transform_dense(df)

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        self.fit(df)
        return self.transform(df)

    @property
    def field_dims(self) -> list[int]:
        return [len(self.category_maps[column]) for column in self.sparse_features]

    @property
    def dense_dim(self) -> int:
        return len(self.dense_features)

    def to_config(self) -> EncoderConfig:
        return EncoderConfig(
            sparse_features=self.sparse_features,
            dense_features=self.dense_features,
            category_maps=self.category_maps,
            dense_means=self.dense_means,
            dense_stds=self.dense_stds,
        )

    @classmethod
    def from_config(cls, config: EncoderConfig) -> "FeatureEncoder":
        encoder = cls(config.sparse_features, config.dense_features)
        encoder.category_maps = config.category_maps
        encoder.dense_means = config.dense_means
        encoder.dense_stds = config.dense_stds
        return encoder


class DeepFMDataset(Dataset):
    """PyTorch dataset for DeepFM."""

    def __init__(self, sparse_x: np.ndarray, dense_x: np.ndarray, labels: np.ndarray | None = None) -> None:
        ensure_torch_available()
        self.sparse_x = torch.tensor(sparse_x, dtype=torch.long)
        self.dense_x = torch.tensor(dense_x, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32) if labels is not None else None

    def __len__(self) -> int:
        return len(self.sparse_x)

    def __getitem__(self, idx: int):
        if self.labels is None:
            return self.sparse_x[idx], self.dense_x[idx]
        return self.sparse_x[idx], self.dense_x[idx], self.labels[idx]


def ensure_torch_available() -> None:
    """Raise a clear message when PyTorch is not installed."""

    if torch is None:
        raise ModuleNotFoundError(
            "PyTorch is required for DeepFM ranking. Install it first, for example: "
            "python -m pip install torch --index-url https://download.pytorch.org/whl/cpu"
        )


def build_training_dataloader(
    samples: pd.DataFrame,
    encoder: FeatureEncoder,
    batch_size: int = 128,
    shuffle: bool = True,
) -> DataLoader:
    """Build DataLoader from a training dataframe."""

    sparse_x, dense_x = encoder.transform(samples)
    labels = pd.to_numeric(samples["label"], errors="coerce").fillna(0).astype("float32").to_numpy()
    dataset = DeepFMDataset(sparse_x, dense_x, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_deepfm_model(
    training_samples: pd.DataFrame,
    epoch: int = 3,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    embed_dim: int = 8,
    hidden_dims: list[int] | None = None,
    dropout: float = 0.2,
    device: str | None = None,
) -> tuple[DeepFM, FeatureEncoder, list[dict[str, float]]]:
    """Train a simplified DeepFM model."""

    ensure_torch_available()
    samples = training_samples.copy()
    if "label" not in samples.columns:
        raise ValueError("training samples must contain a label column.")
    encoder = FeatureEncoder()
    encoder.fit(samples)
    dataloader = build_training_dataloader(samples, encoder, batch_size=batch_size, shuffle=True)

    device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepFM(
        field_dims=encoder.field_dims,
        dense_dim=encoder.dense_dim,
        embed_dim=embed_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device_name)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    logs: list[dict[str, float]] = []

    for epoch_index in range(1, epoch + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for sparse_x, dense_x, labels in dataloader:
            sparse_x = sparse_x.to(device_name)
            dense_x = dense_x.to(device_name)
            labels = labels.to(device_name)

            optimizer.zero_grad()
            logits = model(sparse_x, dense_x)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size_actual = labels.size(0)
            total_loss += float(loss.item()) * batch_size_actual
            total_count += batch_size_actual

        avg_loss = total_loss / max(total_count, 1)
        logs.append({"epoch": float(epoch_index), "loss": avg_loss})
        print(f"epoch={epoch_index} loss={avg_loss:.6f}")

    return model, encoder, logs


def merge_candidate_features(
    candidate_df: pd.DataFrame,
    users_df: pd.DataFrame | None = None,
    posts_df: pd.DataFrame | None = None,
    post_stats_df: pd.DataFrame | None = None,
    user_profiles_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach missing user/post/stat/profile fields to candidate rows."""

    data = candidate_df.copy()
    if users_df is not None and not users_df.empty:
        user_columns = [column for column in ["user_id", "grade", "college", "major", "campus"] if column in users_df.columns]
        data = data.merge(users_df[user_columns].drop_duplicates("user_id"), on="user_id", how="left", suffixes=("", "_user"))
    if posts_df is not None and not posts_df.empty:
        post_columns = [
            column
            for column in [
                "post_id",
                "board",
                "tags",
                "topic_type",
                "location_scope",
                "post_age_hours",
                "content_length",
                "img_count",
                "has_image",
                "has_contact_info",
            ]
            if column in posts_df.columns
        ]
        data = data.merge(posts_df[post_columns].drop_duplicates("post_id"), on="post_id", how="left", suffixes=("", "_post"))
        for column in ["board", "tags", "topic_type", "location_scope", "post_age_hours", "content_length", "img_count", "has_image", "has_contact_info"]:
            post_column = f"{column}_post"
            if post_column in data.columns:
                data[column] = data[column].where(data[column].notna(), data[post_column]) if column in data.columns else data[post_column]
                data = data.drop(columns=[post_column])
    if post_stats_df is not None and not post_stats_df.empty:
        stats_columns = [
            column
            for column in [
                "post_id",
                "view_count",
                "like_count",
                "comment_count",
                "collect_count",
                "dislike_count",
                "hot_score",
                "final_hot_score",
                "quality_score",
            ]
            if column in post_stats_df.columns
        ]
        data = data.merge(post_stats_df[stats_columns].drop_duplicates("post_id"), on="post_id", how="left", suffixes=("", "_stat"))
        for column in ["view_count", "like_count", "comment_count", "collect_count", "dislike_count", "hot_score", "final_hot_score", "quality_score"]:
            stat_column = f"{column}_stat"
            if stat_column in data.columns:
                data[column] = data[column].where(data[column].notna(), data[stat_column]) if column in data.columns else data[stat_column]
                data = data.drop(columns=[stat_column])
    if user_profiles_df is not None and not user_profiles_df.empty:
        profile_columns = [
            column
            for column in [
                "user_id",
                "preferred_location",
                "learning_interest_weight",
                "life_interest_weight",
                "social_interest_weight",
                "career_interest_weight",
            ]
            if column in user_profiles_df.columns
        ]
        if len(profile_columns) > 1:
            data = data.merge(user_profiles_df[profile_columns].drop_duplicates("user_id"), on="user_id", how="left")
    return data


def rank_candidates(
    candidate_df: pd.DataFrame,
    model: DeepFM,
    encoder: FeatureEncoder,
    users_df: pd.DataFrame | None = None,
    posts_df: pd.DataFrame | None = None,
    post_stats_df: pd.DataFrame | None = None,
    user_profiles_df: pd.DataFrame | None = None,
    batch_size: int = 256,
    device: str | None = None,
    log_score_stats: bool = False,
) -> pd.DataFrame:
    """Predict rank_score for candidate dataframe."""

    ensure_torch_available()
    if candidate_df.empty:
        result = candidate_df.copy()
        result["rank_score"] = []
        return result

    features = merge_candidate_features(candidate_df, users_df, posts_df, post_stats_df, user_profiles_df)
    sparse_x, dense_x = encoder.transform(features)
    dataset = DeepFMDataset(sparse_x, dense_x)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device_name)
    model.eval()
    logits_list: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for sparse_batch, dense_batch in dataloader:
            sparse_batch = sparse_batch.to(device_name)
            dense_batch = dense_batch.to(device_name)
            batch_logits = model(sparse_batch, dense_batch)
            batch_scores = torch.sigmoid(batch_logits).cpu().numpy()
            logits_list.append(batch_logits.cpu().numpy())
            scores.append(batch_scores)

    ranked = candidate_df.copy()
    logits_array = np.concatenate(logits_list)
    score_array = np.concatenate(scores)
    ranked["rank_score"] = score_array
    ranked["rank_score_source"] = "deepfm"
    ranked.attrs["ranker_score_stats"] = {
        "logits_min": float(np.min(logits_array)),
        "logits_max": float(np.max(logits_array)),
        "logits_mean": float(np.mean(logits_array)),
        "rank_score_min": float(np.min(score_array)),
        "rank_score_max": float(np.max(score_array)),
        "rank_score_mean": float(np.mean(score_array)),
    }
    if log_score_stats:
        stats = ranked.attrs["ranker_score_stats"]
        print(
            "ranker logits: "
            f"min={stats['logits_min']:.6f}, max={stats['logits_max']:.6f}, mean={stats['logits_mean']:.6f}"
        )
        print(
            "ranker probabilities: "
            f"min={stats['rank_score_min']:.6f}, max={stats['rank_score_max']:.6f}, mean={stats['rank_score_mean']:.6f}"
        )
    return ranked.sort_values("rank_score", ascending=False).reset_index(drop=True)


def save_model(model: DeepFM, path: str | Path) -> None:
    """Save model state dict and architecture metadata."""

    ensure_torch_available()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "field_dims": model.field_dims,
        "dense_dim": model.dense_dim,
        "embed_dim": model.embed_dim,
        "hidden_dims": model.hidden_dims,
    }
    torch.save(payload, path)


def load_model(path: str | Path, hidden_dims: list[int] | None = None, device: str | None = None) -> DeepFM:
    """Load DeepFM model."""

    ensure_torch_available()
    device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(path, map_location=device_name)
    model = DeepFM(
        field_dims=payload["field_dims"],
        dense_dim=payload["dense_dim"],
        embed_dim=payload.get("embed_dim", 8),
        hidden_dims=hidden_dims or payload.get("hidden_dims"),
    )
    model.load_state_dict(payload["state_dict"])
    return model.to(device_name)


def save_encoder(encoder: FeatureEncoder, path: str | Path) -> None:
    """Save feature encoder as JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = asdict(encoder.to_config())
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def load_encoder(path: str | Path) -> FeatureEncoder:
    """Load feature encoder from JSON."""

    raw: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    config = EncoderConfig(
        sparse_features=raw["sparse_features"],
        dense_features=raw["dense_features"],
        category_maps={key: {str(k): int(v) for k, v in value.items()} for key, value in raw["category_maps"].items()},
        dense_means={key: float(value) for key, value in raw["dense_means"].items()},
        dense_stds={key: float(value) for key, value in raw["dense_stds"].items()},
    )
    return FeatureEncoder.from_config(config)
