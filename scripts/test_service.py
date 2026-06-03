"""Smoke test for CampusRecommendService.

Usage:
    python scripts/test_service.py --user_id u_0001
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


configure_console_encoding()

from src.service import CampusRecommendService  # noqa: E402


def print_json(title: str, value) -> None:
    print(f"\n[{title}]")
    print(json.dumps(value, ensure_ascii=False, indent=2))


def choose_user_id(service: CampusRecommendService, requested_user_id: str) -> str:
    users = service.result.users
    if users["user_id"].astype(str).eq(str(requested_user_id)).any():
        return str(requested_user_id)
    fallback = str(users.iloc[0]["user_id"])
    print(f"user_id={requested_user_id} not found, fallback to first valid user_id={fallback}")
    return fallback


def compact_recommendations(recommendations: list[dict], limit: int = 10) -> list[dict]:
    fields = [
        "post_id",
        "title",
        "board",
        "rank_score",
        "rank_score_source",
        "rerank_score",
        "recall_sources",
        "recommend_reason",
    ]
    return [{field: item.get(field) for field in fields} for item in recommendations[:limit]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Test CampusRecommendService.")
    parser.add_argument("--user_id", default="u_0001", help="User id to test.")
    parser.add_argument("--top_n", type=int, default=10, help="Recommendation size.")
    args = parser.parse_args()

    print("Initializing CampusRecommendService...")
    service = CampusRecommendService()
    user_id = choose_user_id(service, args.user_id)

    profile = service.get_user_profile(user_id)
    print_json("User Profile", profile)

    before = service.recommend_for_user(user_id, top_n=args.top_n, log_score_stats=True)
    print_json("Top Recommendations Before Behavior", compact_recommendations(before, args.top_n))
    if not before:
        print("No recommendations returned; stop smoke test.")
        return

    first_post_id = str(before[0]["post_id"])
    detail = service.get_post_detail(first_post_id)
    detail_preview = {
        "post": {
            "post_id": detail["post"].get("post_id"),
            "title": detail["post"].get("title"),
            "board": detail["post"].get("board"),
            "tags": detail["post"].get("tags"),
        },
        "stats": detail["stats"],
        "comment_count": len(detail["comments"]),
        "tags": detail["tags"][:5],
    }
    print_json("First Recommendation Detail", detail_preview)

    view_behavior = service.record_user_behavior(
        user_id=user_id,
        post_id=first_post_id,
        action_type="view",
        dwell_time=35,
        context={"device_type": "mobile", "scene": "service_test"},
    )
    like_behavior = service.record_user_behavior(
        user_id=user_id,
        post_id=first_post_id,
        action_type="like",
        dwell_time=35,
        context={"device_type": "mobile", "scene": "service_test"},
    )
    print_json("Recorded View Behavior", view_behavior)
    print_json("Recorded Like Behavior", like_behavior)

    refreshed_profile = service.refresh_user_profile(user_id)
    print_json("Refreshed User Profile", refreshed_profile)

    after = service.recommend_for_user(user_id, top_n=args.top_n, log_score_stats=True)
    print_json("Top Recommendations After Behavior", compact_recommendations(after, args.top_n))

    before_ids = [item["post_id"] for item in before]
    after_ids = [item["post_id"] for item in after]
    print_json(
        "Recommendation Comparison",
        {
            "before_count": len(before),
            "after_count": len(after),
            "before_post_ids": before_ids,
            "after_post_ids": after_ids,
            "service_ok": bool(before and after),
        },
    )


if __name__ == "__main__":
    main()
