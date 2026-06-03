"""Time-aware multi-interest user representation.

The module builds four vectors for every user:

- learning_vector
- life_vector
- social_vector
- career_vector

Each vector is a weighted average of interacted post TF-IDF vectors. The final
behavior weight is ``action_weight * time_decay`` where
``time_decay = 1 / (1 + days_diff)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from src.user_profile import ACTION_WEIGHTS, INTEREST_DIRECTIONS, infer_interest_direction, split_tags


DIRECTION_ORDER = ["learning", "life", "social", "career"]
VECTOR_COLUMNS = {
    "learning": "learning_vector",
    "life": "life_vector",
    "social": "social_vector",
    "career": "career_vector",
}


@dataclass
class MultiInterestBundle:
    """Container for reusable multi-interest vectors."""

    user_vectors: dict[str, dict[str, sparse.csr_matrix]]
    user_direction_weights: dict[str, dict[str, float]]
    post_text_matrix: sparse.csr_matrix
    post_id_to_index: dict[str, int]
    vector_dim: int

    def get_user_vector(self, user_id: str, direction: str) -> sparse.csr_matrix:
        """Return one user's vector for a direction, or a zero vector."""

        if user_id in self.user_vectors and direction in self.user_vectors[user_id]:
            return self.user_vectors[user_id][direction]
        return sparse.csr_matrix((1, self.vector_dim))

    def calculate_multi_interest_score(self, user_id: str, post_id: str) -> float:
        """Calculate max cosine similarity over four interest vectors."""

        if post_id not in self.post_id_to_index:
            return 0.0
        post_vector = self.post_text_matrix[self.post_id_to_index[post_id]]
        scores = []
        for direction in DIRECTION_ORDER:
            user_vector = self.get_user_vector(user_id, direction)
            if user_vector.nnz == 0:
                scores.append(0.0)
                continue
            score = float(cosine_similarity(user_vector, post_vector)[0, 0])
            scores.append(max(score, 0.0))
        return float(max(scores)) if scores else 0.0


def normalize_action_weight(row: pd.Series) -> float:
    """Use action_weight column when available, otherwise map action_type."""

    if "action_weight" in row and not pd.isna(row["action_weight"]):
        return float(row["action_weight"])
    return ACTION_WEIGHTS.get(str(row.get("action_type", "")), 0.0)


def time_decay(timestamp: pd.Timestamp, now: pd.Timestamp) -> float:
    """Recent behaviors get larger weights."""

    if pd.isna(timestamp):
        return 0.0
    days_diff = max((now - pd.Timestamp(timestamp)).days, 0)
    return 1.0 / (1.0 + days_diff)


def zero_direction_vectors(vector_dim: int) -> dict[str, sparse.csr_matrix]:
    """Create zero vectors for all four interest directions."""

    return {direction: sparse.csr_matrix((1, vector_dim)) for direction in DIRECTION_ORDER}


def prepare_multi_interest_behaviors(posts: pd.DataFrame, behaviors: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Join behavior logs with post metadata and calculate final weights."""

    post_meta = posts[["post_id", "board", "tags", "topic_type"]].copy()
    enriched = behaviors.copy()
    enriched["timestamp"] = pd.to_datetime(enriched["timestamp"], errors="coerce")
    enriched = enriched.merge(post_meta, on="post_id", how="left")
    enriched["interest_direction"] = enriched.apply(
        lambda row: infer_interest_direction(row.get("board", ""), row.get("tags", ""), row.get("topic_type", "")),
        axis=1,
    )
    enriched["action_weight"] = enriched.apply(normalize_action_weight, axis=1)
    enriched["time_weight"] = enriched["timestamp"].apply(lambda value: time_decay(value, now))
    enriched["final_weight"] = enriched["action_weight"] * enriched["time_weight"]
    return enriched


def build_direction_vector(
    direction_rows: pd.DataFrame,
    post_text_matrix: sparse.csr_matrix,
    post_id_to_index: dict[str, int],
    vector_dim: int,
) -> tuple[sparse.csr_matrix, float]:
    """Build weighted average vector for one user and one direction."""

    vectors = []
    weights = []
    for row in direction_rows.itertuples(index=False):
        post_id = str(row.post_id)
        if post_id not in post_id_to_index:
            continue
        final_weight = float(row.final_weight)
        if np.isclose(final_weight, 0.0):
            continue
        vectors.append(post_text_matrix[post_id_to_index[post_id]])
        weights.append(final_weight)

    if not vectors:
        return sparse.csr_matrix((1, vector_dim)), 0.0

    weight_array = np.array(weights, dtype=float)
    normalizer = float(np.abs(weight_array).sum())
    if np.isclose(normalizer, 0.0):
        return sparse.csr_matrix((1, vector_dim)), 0.0

    stacked = sparse.vstack(vectors)
    weighted_sum = stacked.multiply(weight_array.reshape(-1, 1)).sum(axis=0)
    vector = sparse.csr_matrix(weighted_sum / normalizer)
    return vector, normalizer


def initialize_from_questionnaire_or_tags(
    user: pd.Series,
    posts: pd.DataFrame,
    post_text_matrix: sparse.csr_matrix,
    post_id_to_index: dict[str, int],
    vector_dim: int,
) -> tuple[dict[str, sparse.csr_matrix], dict[str, float]]:
    """Initialize cold users by matching their interest tags with posts."""

    user_tags = set(split_tags(user.get("interest_tags", "")))
    vectors = zero_direction_vectors(vector_dim)
    weights = {direction: 0.0 for direction in DIRECTION_ORDER}
    if not user_tags:
        return vectors, weights

    candidate_posts = posts[posts["tags"].apply(lambda value: bool(user_tags & set(split_tags(value))))]
    for direction in DIRECTION_ORDER:
        direction_posts = candidate_posts[
            candidate_posts.apply(
                lambda row: infer_interest_direction(row.get("board", ""), row.get("tags", ""), row.get("topic_type", "")) == direction,
                axis=1,
            )
        ].head(20)
        post_vectors = [
            post_text_matrix[post_id_to_index[str(row.post_id)]]
            for row in direction_posts.itertuples(index=False)
            if str(row.post_id) in post_id_to_index
        ]
        if post_vectors:
            vectors[direction] = sparse.vstack(post_vectors).mean(axis=0)
            vectors[direction] = sparse.csr_matrix(vectors[direction])
            weights[direction] = float(len(post_vectors))
    return vectors, weights


def build_multi_interest_representations(
    users: pd.DataFrame,
    posts: pd.DataFrame,
    behaviors: pd.DataFrame,
    post_text_matrix: sparse.csr_matrix,
    post_id_to_index: dict[str, int],
    now: pd.Timestamp | None = None,
) -> MultiInterestBundle:
    """Build four time-aware interest vectors for each user."""

    if now is None:
        now = pd.Timestamp.now()
    vector_dim = int(post_text_matrix.shape[1])
    enriched = prepare_multi_interest_behaviors(posts, behaviors, now)

    user_vectors: dict[str, dict[str, sparse.csr_matrix]] = {}
    user_direction_weights: dict[str, dict[str, float]] = {}

    for user in users.itertuples(index=False):
        user_id = str(user.user_id)
        user_rows = enriched[enriched["user_id"].eq(user_id)]
        vectors = zero_direction_vectors(vector_dim)
        weights = {direction: 0.0 for direction in DIRECTION_ORDER}

        if user_rows.empty:
            vectors, weights = initialize_from_questionnaire_or_tags(
                pd.Series(user._asdict()),
                posts,
                post_text_matrix,
                post_id_to_index,
                vector_dim,
            )
        else:
            for direction in DIRECTION_ORDER:
                direction_rows = user_rows[user_rows["interest_direction"].eq(direction)]
                vectors[direction], weights[direction] = build_direction_vector(
                    direction_rows,
                    post_text_matrix,
                    post_id_to_index,
                    vector_dim,
                )

        user_vectors[user_id] = vectors
        user_direction_weights[user_id] = weights

    return MultiInterestBundle(
        user_vectors=user_vectors,
        user_direction_weights=user_direction_weights,
        post_text_matrix=post_text_matrix,
        post_id_to_index=post_id_to_index,
        vector_dim=vector_dim,
    )


def calculate_multi_interest_score(bundle: MultiInterestBundle, user_id: str, post_id: str) -> float:
    """Functional wrapper for candidate scoring."""

    return bundle.calculate_multi_interest_score(user_id, post_id)
