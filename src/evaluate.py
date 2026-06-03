"""Offline evaluation metrics for recommendation output."""

from __future__ import annotations

import math

import pandas as pd

from src.data_loader import load_all_data
from src.preprocess import preprocess_all
from src.recommend import recommend_for_user


def precision_at_k(recommended: list[str], truth: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(recommended[:k]) & truth) / k


def recall_at_k(recommended: list[str], truth: set[str], k: int) -> float:
    if not truth:
        return 0.0
    return len(set(recommended[:k]) & truth) / len(truth)


def hr_at_k(recommended: list[str], truth: set[str], k: int) -> float:
    return float(bool(set(recommended[:k]) & truth))


def ndcg_at_k(recommended: list[str], truth: set[str], k: int) -> float:
    dcg = 0.0
    for idx, post_id in enumerate(recommended[:k], start=1):
        if post_id in truth:
            dcg += 1.0 / math.log2(idx + 1)
    ideal_hits = min(len(truth), k)
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_offline(
    ks: list[int] | None = None,
    max_users: int = 20,
    top_n: int = 20,
) -> pd.DataFrame:
    """Evaluate recommendations against time-split positive test samples."""

    if ks is None:
        ks = [5, 10, 20]
    data = load_all_data()
    result = preprocess_all(data)
    test_positive = result.test_samples[result.test_samples["label"].astype(int).eq(1)]
    if test_positive.empty:
        behavior_positive = result.behaviors[result.behaviors["is_positive"].astype(int).eq(1)].copy()
        behavior_positive = behavior_positive.sort_values("timestamp")
        fallback_size = max(1, int(len(behavior_positive) * 0.2))
        test_positive = behavior_positive.tail(fallback_size)[["user_id", "post_id"]]
    truth_by_user = test_positive.groupby("user_id")["post_id"].apply(lambda values: set(values.astype(str))).to_dict()
    user_ids = list(truth_by_user.keys())[:max_users]
    if not user_ids:
        return pd.DataFrame(columns=["K", "Precision", "Recall", "HR", "NDCG"])

    metric_sums = {k: {"Precision": 0.0, "Recall": 0.0, "HR": 0.0, "NDCG": 0.0} for k in ks}
    evaluated_users = 0
    for user_id in user_ids:
        recs = recommend_for_user(str(user_id), top_n=top_n, exclude_seen=False)
        recommended = recs["post_id"].astype(str).tolist()
        truth = truth_by_user[user_id]
        for k in ks:
            metric_sums[k]["Precision"] += precision_at_k(recommended, truth, k)
            metric_sums[k]["Recall"] += recall_at_k(recommended, truth, k)
            metric_sums[k]["HR"] += hr_at_k(recommended, truth, k)
            metric_sums[k]["NDCG"] += ndcg_at_k(recommended, truth, k)
        evaluated_users += 1

    rows = []
    for k in ks:
        row = {"K": k}
        row.update({metric: value / evaluated_users for metric, value in metric_sums[k].items()})
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(evaluate_offline(max_users=5).to_string(index=False))
