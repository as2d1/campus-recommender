"""End-to-end recommendation orchestration."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_loader import load_all_data
from src.feature_engineering import fit_post_text_vectors
from src.merge_candidates import merge_recall_candidates
from src.preprocess import preprocess_all
from src.ranker import load_encoder, load_model, rank_candidates
from src.recall.cold_start_recall import cold_start_recall
from src.recall.content_recall import content_recall
from src.recall.hot_recall import hot_recall
from src.recall.itemcf_recall import build_item_similarity, build_positive_interactions, itemcf_recall
from src.recall.latest_recall import latest_recall
from src.recall.profile_recall import profile_recall
from src.recall.scene_recall import location_scene_recall, time_scene_recall
from src.rerank import rerank_candidates
from src.train import ENCODER_PATH, MODEL_PATH
from src.user_profile import build_user_profile_table


REASON_MAP = {
    "hot": "该帖子近期热度较高。",
    "latest": "该帖子发布时间较近，具有时效性。",
    "content": "该帖子与你最近浏览的内容相似。",
    "itemcf": "和你兴趣相似的用户也浏览过该帖子。",
    "profile": "该帖子与你的兴趣标签或常看板块相关。",
    "time_scene": "该帖子适合你当前的浏览时间段。",
    "location_scene": "该帖子与你当前所在校园场景相关。",
    "cold_start": "根据新用户问卷或热门内容为你推荐。",
}


def build_recommend_reason(recall_sources: object) -> str:
    """Generate simple recommendation explanation from recall sources."""

    sources = [source.strip() for source in str(recall_sources).split(",") if source.strip()]
    reasons = [REASON_MAP[source] for source in sources if source in REASON_MAP]
    return "；".join(reasons[:2]) if reasons else "综合你的兴趣和帖子质量进行推荐。"


def build_recall_results(user_id: str, result, text_bundle, user_profiles: pd.DataFrame, recall_top_k: int) -> list[pd.DataFrame]:
    """Run all recall channels for one user."""

    user_rows = result.users[result.users["user_id"].astype(str).eq(str(user_id))]
    if user_rows.empty:
        raise ValueError(f"user_id not found: {user_id}")
    user = user_rows.iloc[0]
    location = str(user.get("default_location", "教学区"))
    time_period = "晚上"
    scene = "普通浏览"

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
        itemcf_recall(user_id, result.behaviors, result.posts, top_k=recall_top_k, item_similarity=item_similarity),
        profile_recall(user_id, result.users, result.posts, user_profiles, result.post_tags, top_k=recall_top_k),
        time_scene_recall(user_id, result.posts, time_period=time_period, scene=scene, top_k=recall_top_k),
        location_scene_recall(user_id, result.posts, location=location, top_k=recall_top_k),
        cold_start_recall(
            user_id,
            result.users,
            result.questionnaire,
            result.posts,
            result.post_stats,
            top_k=recall_top_k,
            time_period=time_period,
            scene=scene,
        ),
    ]


def apply_ranker_if_available(
    candidates: pd.DataFrame,
    result,
    user_profiles: pd.DataFrame,
    model_path: str | Path,
    encoder_path: str | Path,
    log_score_stats: bool = False,
) -> pd.DataFrame:
    """Predict rank_score with DeepFM if artifacts exist; otherwise fallback."""

    if Path(model_path).exists() and Path(encoder_path).exists():
        model = load_model(model_path)
        encoder = load_encoder(encoder_path)
        return rank_candidates(
            candidate_df=candidates,
            model=model,
            encoder=encoder,
            users_df=result.users,
            posts_df=result.posts,
            post_stats_df=result.post_stats,
            user_profiles_df=user_profiles,
            log_score_stats=log_score_stats,
        )
    ranked = candidates.copy()
    ranked["rank_score"] = ranked["merged_recall_score"]
    ranked["rank_score_source"] = "merged_recall_fallback"
    return ranked.sort_values("rank_score", ascending=False).reset_index(drop=True)


def recommend_for_user(
    user_id: str,
    top_n: int = 10,
    recall_top_k: int = 30,
    candidate_top_k: int = 120,
    model_path: str | Path = MODEL_PATH,
    encoder_path: str | Path = ENCODER_PATH,
    exclude_seen: bool = True,
    log_score_stats: bool = False,
) -> pd.DataFrame:
    """Run recall, merge, rank, rerank, and return TopN recommendations."""

    data = load_all_data()
    result = preprocess_all(data)
    text_bundle = fit_post_text_vectors(result.posts)
    user_profiles = build_user_profile_table(result.users, result.posts, result.behaviors)

    recall_results = build_recall_results(user_id, result, text_bundle, user_profiles, recall_top_k)
    candidates = merge_recall_candidates(recall_results, result.posts, result.post_stats, top_k_candidates=candidate_top_k)
    ranked = apply_ranker_if_available(candidates, result, user_profiles, model_path, encoder_path, log_score_stats)
    reranked = rerank_candidates(
        ranked,
        user_id,
        user_profiles,
        result.behaviors,
        top_n=top_n,
        exclude_seen=exclude_seen,
    )

    post_info = result.posts[["post_id", "title", "board", "tags"]].drop_duplicates("post_id")
    final = reranked.merge(post_info, on="post_id", how="left", suffixes=("", "_post"))
    for column in ["title", "board", "tags"]:
        post_column = f"{column}_post"
        if post_column in final.columns:
            final[column] = final[column].where(final[column].notna(), final[post_column])
            final = final.drop(columns=[post_column])
    final["recommend_reason"] = final["recall_sources"].apply(build_recommend_reason)
    output_columns = [
        "post_id",
        "title",
        "board",
        "tags",
        "rank_score",
        "rerank_score",
        "recall_sources",
        "recommend_reason",
    ]
    for column in output_columns:
        if column not in final.columns:
            final[column] = ""
    return final[output_columns].head(top_n).reset_index(drop=True)


if __name__ == "__main__":
    data = load_all_data()
    result = preprocess_all(data)
    uid = str(result.users.iloc[0]["user_id"])
    print(recommend_for_user(uid, top_n=10).to_string(index=False))
