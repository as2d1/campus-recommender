"""Service facade for the campus forum recommendation algorithms.

This module is intentionally framework-free.  A FastAPI backend can create one
``CampusRecommendService`` instance and call these methods directly.
"""

from __future__ import annotations

import warnings
from pathlib import Path
import sqlite3
import time
from typing import Any

import pandas as pd

from src.data_loader import ALLOWED_BOARDS, DATA_DIR, DEFAULT_SQLITE_PATH, PROJECT_ROOT, CampusData, load_all_data
from src.feature_engineering import fit_post_text_vectors
from src.merge_candidates import merge_recall_candidates
from src.preprocess import get_time_period, preprocess_all
from src.ranker import FeatureEncoder, load_encoder, load_model, rank_candidates
from src.recall.cold_start_recall import cold_start_recall
from src.recall.content_recall import content_recall
from src.recall.hot_recall import hot_recall
from src.recall.itemcf_recall import build_item_similarity, build_positive_interactions, itemcf_recall
from src.recall.latest_recall import latest_recall
from src.recall.profile_recall import profile_recall
from src.recall.scene_recall import location_scene_recall, time_scene_recall
from src.rerank import rerank_candidates
from src.user_profile import build_user_profile_table


MODEL_PATH = PROJECT_ROOT / "models" / "deepfm.pt"
ENCODER_PATH = PROJECT_ROOT / "models" / "feature_encoder.json"

ACTION_WEIGHTS = {
    "expose": 0,
    "skip": -1,
    "view": 1,
    "like": 2,
    "comment": 3,
    "collect": 4,
    "favorite": 4,
    "dislike": -2,
    "report": -4,
}

REASON_MAP = {
    "hot": "该帖子近期热度较高。",
    "latest": "该帖子发布时间较近，具有时效性。",
    "content": "该帖子与你最近浏览的内容相似。",
    "itemcf": "和你兴趣相似的用户也浏览过该帖子。",
    "profile": "该帖子与你的兴趣标签或常看板块相关。",
    "time_scene": "该帖子适合你当前的浏览时间段。",
    "location_scene": "该帖子与你当前所在校园场景相关。",
    "cold_start": "根据新用户默认策略或问卷兴趣推荐。",
}


def _clean_value(value: Any) -> Any:
    """Convert pandas/numpy values into JSON-friendly Python values."""

    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if hasattr(value, "item"):
        return value.item()
    return value


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [{key: _clean_value(value) for key, value in row.items()} for row in df.to_dict("records")]


def _ensure_columns(df: pd.DataFrame, defaults: dict[str, Any]) -> pd.DataFrame:
    data = df.copy()
    for column, value in defaults.items():
        if column not in data.columns:
            data[column] = value
    return data


class CampusRecommendService:
    """Facade that exposes recommendation algorithms as backend-friendly APIs."""

    def __init__(
        self,
        data_dir: str | Path = DEFAULT_SQLITE_PATH,
        sqlite_path: str | Path | None = None,
        model_path: str | Path | None = None,
        encoder_path: str | Path | None = None,
        model_dir: str | Path | None = None,
    ) -> None:
        self.sqlite_path = Path(sqlite_path) if sqlite_path is not None else Path(data_dir)
        if self.sqlite_path.is_dir():
            self.sqlite_path = DEFAULT_SQLITE_PATH
        if model_dir is not None:
            model_root = Path(model_dir)
            self.model_path = Path(model_path) if model_path is not None else model_root / "deepfm.pt"
            self.encoder_path = Path(encoder_path) if encoder_path is not None else model_root / "feature_encoder.json"
        else:
            self.model_path = Path(model_path) if model_path is not None else MODEL_PATH
            self.encoder_path = Path(encoder_path) if encoder_path is not None else ENCODER_PATH
        self.model = None
        self.encoder: FeatureEncoder | None = None
        self.model_warning: str | None = None
        self.reload_data()
        self.load_ranker()

    def reload_data(self) -> None:
        """Reload SQLite data and rebuild lightweight in-memory helper objects."""

        self.data: CampusData = load_all_data(self.sqlite_path)
        self.result = preprocess_all(self.data)
        if self.result.user_profiles.empty:
            self.user_profiles = build_user_profile_table(self.result.users, self.result.posts, self.result.behaviors)
        else:
            self.user_profiles = self.result.user_profiles.copy()
        self.text_bundle = fit_post_text_vectors(self.result.posts)
        positive = build_positive_interactions(self.result.behaviors)
        self.item_similarity = build_item_similarity(positive) if not positive.empty else {}

    def load_ranker(self) -> None:
        """Load DeepFM artifacts when available; otherwise enable fallback ranking."""

        if not self.model_path.exists() or not self.encoder_path.exists():
            self.model_warning = (
                "DeepFM artifacts not found. Ranking will fallback to merged_recall_score."
            )
            warnings.warn(self.model_warning)
            return
        try:
            self.model = load_model(self.model_path)
            self.encoder = load_encoder(self.encoder_path)
            self.model_warning = None
        except Exception as exc:  # pragma: no cover - defensive service guard.
            self.model = None
            self.encoder = None
            self.model_warning = (
                f"Failed to load DeepFM artifacts: {exc}. "
                "Ranking will fallback to merged_recall_score."
            )
            warnings.warn(self.model_warning)

    def _find_user(self, user_id: str) -> pd.Series:
        rows = self.result.users[self.result.users["user_id"].astype(str).eq(str(user_id))]
        if rows.empty:
            raise ValueError(f"user_id not found: {user_id}")
        return rows.iloc[0]

    def _build_recall_results(self, user_id: str, recall_top_k: int) -> list[pd.DataFrame]:
        user = self._find_user(user_id)
        location = str(user.get("default_location", "教学区"))
        time_period = "晚上"
        scene = "普通浏览"
        is_new_user = int(user.get("is_new_user", 0) or 0) == 1

        if is_new_user:
            return [
                cold_start_recall(
                    user_id=user_id,
                    users=self.result.users,
                    questionnaire=self.result.questionnaire,
                    posts=self.result.posts,
                    post_stats=self.result.post_stats,
                    top_k=recall_top_k,
                    time_period=time_period,
                    scene=scene,
                )
            ]

        return [
            hot_recall(user_id, self.result.posts, self.result.post_stats, top_k=recall_top_k),
            latest_recall(user_id, self.result.posts, top_k=recall_top_k),
            content_recall(
                user_id,
                self.result.posts,
                self.result.behaviors,
                self.text_bundle.post_text_matrix,
                self.text_bundle.post_id_to_index,
                top_k=recall_top_k,
            ),
            itemcf_recall(
                user_id,
                self.result.behaviors,
                self.result.posts,
                top_k=recall_top_k,
                item_similarity=self.item_similarity,
            ),
            profile_recall(
                user_id,
                self.result.users,
                self.result.posts,
                self.user_profiles,
                self.result.post_tags,
                top_k=recall_top_k,
            ),
            time_scene_recall(user_id, self.result.posts, time_period=time_period, scene=scene, top_k=recall_top_k),
            location_scene_recall(user_id, self.result.posts, location=location, top_k=recall_top_k),
        ]

    def _rank_candidates(self, candidates: pd.DataFrame, log_score_stats: bool = False) -> pd.DataFrame:
        if candidates.empty:
            return candidates.copy()
        if self.model is not None and self.encoder is not None:
            return rank_candidates(
                candidate_df=candidates,
                model=self.model,
                encoder=self.encoder,
                users_df=self.result.users,
                posts_df=self.result.posts,
                post_stats_df=self.result.post_stats,
                user_profiles_df=self.user_profiles,
                log_score_stats=log_score_stats,
            )
        ranked = candidates.copy()
        ranked["rank_score"] = pd.to_numeric(ranked.get("merged_recall_score", 0.0), errors="coerce").fillna(0.0)
        ranked["rank_score_source"] = "merged_recall_fallback"
        return ranked.sort_values("rank_score", ascending=False).reset_index(drop=True)

    def _attach_post_fields(self, rows: pd.DataFrame) -> pd.DataFrame:
        posts = self.result.posts.drop_duplicates("post_id")
        stats = self.result.post_stats.drop_duplicates("post_id")
        post_columns = [
            "post_id",
            "title",
            "content",
            "board",
            "tags",
            "topic_type",
            "author_id",
            "publish_time",
        ]
        stat_columns = [
            "post_id",
            "view_count",
            "like_count",
            "comment_count",
            "collect_count",
        ]
        data = rows.copy()
        data = data.merge(posts[[c for c in post_columns if c in posts.columns]], on="post_id", how="left", suffixes=("", "_post"))
        data = data.merge(stats[[c for c in stat_columns if c in stats.columns]], on="post_id", how="left", suffixes=("", "_stat"))
        for column in ["title", "content", "board", "tags", "topic_type", "author_id", "publish_time"]:
            post_column = f"{column}_post"
            if post_column in data.columns:
                data[column] = data[column].where(data[column].notna(), data[post_column]) if column in data.columns else data[post_column]
                data = data.drop(columns=[post_column])
        for column in ["view_count", "like_count", "comment_count", "collect_count"]:
            stat_column = f"{column}_stat"
            if stat_column in data.columns:
                data[column] = data[column].where(data[column].notna(), data[stat_column]) if column in data.columns else data[stat_column]
                data = data.drop(columns=[stat_column])
            if column not in data.columns:
                data[column] = 0
            data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0).astype(int)
        return data

    def recommend_for_user(
        self,
        user_id: str,
        top_n: int = 10,
        recall_top_k: int = 30,
        candidate_top_k: int = 120,
        exclude_seen: bool = True,
        log_score_stats: bool = False,
    ) -> list[dict[str, Any]]:
        """Return TopN recommendations as JSON-friendly dictionaries."""

        self._find_user(user_id)
        recall_results = self._build_recall_results(user_id, recall_top_k)
        candidates = merge_recall_candidates(
            recall_results,
            self.result.posts,
            self.result.post_stats,
            top_k_candidates=candidate_top_k,
        )
        ranked = self._rank_candidates(candidates, log_score_stats=log_score_stats)
        reranked = rerank_candidates(
            ranked,
            user_id=user_id,
            user_profiles=self.user_profiles,
            behaviors=self.result.behaviors,
            top_n=top_n,
            exclude_seen=exclude_seen,
        )
        final = self._attach_post_fields(reranked)
        final["recommend_reason"] = final["recall_sources"].apply(self.get_recommend_reason)

        defaults = {
            "rank_score": 0.0,
            "rank_score_source": "merged_recall_fallback",
            "rerank_score": 0.0,
            "recall_sources": "",
            "recommend_reason": "",
        }
        final = _ensure_columns(final, defaults)
        output_columns = [
            "post_id",
            "title",
            "content",
            "board",
            "tags",
            "topic_type",
            "author_id",
            "publish_time",
            "view_count",
            "like_count",
            "comment_count",
            "collect_count",
            "rank_score",
            "rank_score_source",
            "rerank_score",
            "recall_sources",
            "recommend_reason",
        ]
        final = _ensure_columns(final, {column: "" for column in output_columns})
        return _records(final[output_columns].head(top_n))

    def get_post_list(
        self,
        page: int = 1,
        page_size: int = 20,
        board: str | None = None,
        keyword: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return paged posts with optional board and keyword filters."""

        page = max(int(page), 1)
        page_size = max(int(page_size), 1)
        posts = self.result.posts.copy()
        if "status" in posts.columns:
            posts = posts[posts["status"].fillna("normal").eq("normal")]
        if board:
            posts = posts[posts["board"].astype(str).eq(str(board))]
        if keyword:
            query = str(keyword).lower()
            text = posts["title"].fillna("").astype(str) + " " + posts["content"].fillna("").astype(str)
            posts = posts[text.str.lower().str.contains(query, regex=False)]

        stats = self.result.post_stats.drop_duplicates("post_id")
        stat_value_columns = ["view_count", "like_count", "comment_count", "collect_count"]
        posts = posts.drop(columns=[column for column in stat_value_columns if column in posts.columns], errors="ignore")
        posts = posts.merge(
            stats[["post_id", *stat_value_columns]],
            on="post_id",
            how="left",
        )
        for column in stat_value_columns:
            posts[column] = pd.to_numeric(posts[column], errors="coerce").fillna(0).astype(int)
        posts["content_summary"] = posts["content"].fillna("").astype(str).str.slice(0, 80)
        posts = posts.sort_values("publish_time", ascending=False)
        start = (page - 1) * page_size
        output_columns = [
            "post_id",
            "title",
            "content_summary",
            "board",
            "tags",
            "publish_time",
            "view_count",
            "like_count",
            "comment_count",
            "collect_count",
        ]
        posts = _ensure_columns(posts, {column: "" for column in output_columns})
        return _records(posts.iloc[start : start + page_size][output_columns])

    def get_post_detail(self, post_id: str) -> dict[str, Any]:
        """Return post metadata, statistics, comments, and tags."""

        post_rows = self.result.posts[self.result.posts["post_id"].astype(str).eq(str(post_id))]
        if post_rows.empty:
            raise ValueError(f"post_id not found: {post_id}")
        stat_rows = self.result.post_stats[self.result.post_stats["post_id"].astype(str).eq(str(post_id))]
        comment_rows = self.result.comments[self.result.comments["post_id"].astype(str).eq(str(post_id))].copy()
        tag_rows = self.result.post_tags[self.result.post_tags["post_id"].astype(str).eq(str(post_id))].copy()
        comment_rows = comment_rows.sort_values("publish_time", ascending=True) if "publish_time" in comment_rows.columns else comment_rows

        return {
            "post": _records(post_rows.head(1))[0],
            "stats": _records(stat_rows.head(1))[0] if not stat_rows.empty else {},
            "comments": _records(comment_rows),
            "tags": _records(tag_rows),
        }

    def get_user_profile(self, user_id: str) -> dict[str, Any]:
        """Return user base fields plus profile fields."""

        user = self._find_user(user_id)
        profile_rows = self.user_profiles[self.user_profiles["user_id"].astype(str).eq(str(user_id))]
        profile = profile_rows.iloc[0] if not profile_rows.empty else pd.Series(dtype=object)
        data = {
            "user_id": user.get("user_id"),
            "grade": user.get("grade"),
            "college": user.get("college"),
            "major": user.get("major"),
            "interest_tags": user.get("interest_tags"),
            "long_term_tags": profile.get("long_term_tags", ""),
            "short_term_tags": profile.get("short_term_tags", ""),
            "preferred_boards": profile.get("preferred_boards", ""),
            "active_time_period": profile.get("active_time_period", ""),
            "preferred_location": profile.get("preferred_location", user.get("default_location", "")),
            "learning_interest_weight": profile.get("learning_interest_weight", 0.0),
            "life_interest_weight": profile.get("life_interest_weight", 0.0),
            "social_interest_weight": profile.get("social_interest_weight", 0.0),
            "career_interest_weight": profile.get("career_interest_weight", 0.0),
        }
        return {key: _clean_value(value) for key, value in data.items()}

    def record_user_behavior(
        self,
        user_id: str,
        post_id: str,
        action_type: str,
        dwell_time: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist one user behavior to USER_EVENTS and refresh memory cache."""

        self._find_user(user_id)
        if self.result.posts[self.result.posts["post_id"].astype(str).eq(str(post_id))].empty:
            raise ValueError(f"post_id not found: {post_id}")
        if action_type not in ACTION_WEIGHTS:
            raise ValueError(f"Unsupported action_type: {action_type}")

        context = context or {}
        dwell = int(dwell_time or 0)
        timestamp = pd.Timestamp.now()
        if action_type in {"like", "comment", "collect"}:
            is_positive = 1
        elif action_type == "view" and dwell >= 10:
            is_positive = 1
        else:
            is_positive = 0

        event_type = "favorite" if action_type == "collect" else action_type
        event_time = int(time.time())
        with sqlite3.connect(self.sqlite_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO USER_EVENTS (visitor_id, thread_id, event_type, event_time, duration_ms, extra_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(user_id),
                    str(post_id),
                    event_type,
                    event_time,
                    dwell * 1000 if dwell else None,
                    "{}",
                ),
            )
            behavior_id = cursor.lastrowid
            conn.commit()

        behavior = {
            "behavior_id": behavior_id,
            "user_id": str(user_id),
            "post_id": str(post_id),
            "action_type": action_type,
            "action_weight": ACTION_WEIGHTS[action_type],
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "dwell_time": dwell,
            "time_period": context.get("time_period") or get_time_period(timestamp),
            "location": context.get("location") or self._find_user(user_id).get("default_location", "教学区"),
            "device_type": context.get("device_type", "mobile"),
            "scene": context.get("scene", "普通浏览"),
            "is_positive": is_positive,
            "source": "sqlite",
        }
        self.reload_data()
        return {key: _clean_value(value) for key, value in behavior.items()}

    def refresh_user_profile(self, user_id: str) -> dict[str, Any]:
        """Rebuild in-memory user profiles and return the refreshed user profile."""

        self._find_user(user_id)
        self.user_profiles = build_user_profile_table(self.result.users, self.result.posts, self.result.behaviors)
        self.reload_data()
        return self.get_user_profile(user_id)

    def get_hot_posts(self, top_n: int = 10) -> list[dict[str, Any]]:
        """Return globally hot posts."""

        recalled = hot_recall("_service_hot", self.result.posts, self.result.post_stats, top_k=top_n)
        data = recalled.merge(self.result.posts, on="post_id", how="left")
        data = data.merge(self.result.post_stats, on="post_id", how="left", suffixes=("", "_stat"))
        output_columns = [
            "post_id",
            "title",
            "board",
            "tags",
            "publish_time",
            "view_count",
            "like_count",
            "comment_count",
            "collect_count",
            "recall_score",
        ]
        data = _ensure_columns(data, {column: "" for column in output_columns})
        return _records(data[output_columns].head(top_n))

    def get_boards(self) -> list[dict[str, Any]]:
        """Return all boards and post counts."""

        posts = self.result.posts.copy()
        if "status" in posts.columns:
            posts = posts[posts["status"].fillna("normal").eq("normal")]
        counts = posts.groupby("board", dropna=False).size().to_dict()
        rows = pd.DataFrame([{"board": board, "post_count": int(counts.get(board, 0))} for board in ALLOWED_BOARDS])
        return _records(rows)

    def get_tags(self, top_n: int = 50) -> list[dict[str, Any]]:
        """Return popular tags."""

        if not self.result.tags.empty and "tag_name" in self.result.tags.columns:
            tags = self.result.tags.copy()
            if "view_count" not in tags.columns:
                tags["view_count"] = 0
            tags["view_count"] = pd.to_numeric(tags["view_count"], errors="coerce").fillna(0)
            tags = tags.sort_values("view_count", ascending=False).head(top_n)
            return _records(tags)

        tags = (
            self.result.post_tags.groupby("tag_name")
            .size()
            .reset_index(name="post_count")
            .sort_values("post_count", ascending=False)
            .head(top_n)
        )
        return _records(tags)

    @staticmethod
    def get_recommend_reason(recall_sources: object) -> str:
        """Build a human-readable recommendation explanation."""

        sources = [source.strip() for source in str(recall_sources).split(",") if source.strip()]
        reasons = [REASON_MAP[source] for source in sources if source in REASON_MAP]
        return "；".join(reasons[:2]) if reasons else "综合你的兴趣和帖子质量进行推荐。"
