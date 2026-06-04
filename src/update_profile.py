"""Incremental user behavior and profile update utilities backed by SQLite."""

from __future__ import annotations

from src.service import CampusRecommendService


def update_user_profile_with_new_behavior(
    user_id: str,
    post_id: str,
    action_type: str,
    timestamp: str | None = None,
    dwell_time: int = 0,
    time_period: str | None = None,
    location: str = "教学区",
    device_type: str = "mobile",
    scene: str = "普通浏览",
):
    """Persist a behavior event to USER_EVENTS and return the refreshed profile."""

    service = CampusRecommendService()
    service.record_user_behavior(
        user_id=user_id,
        post_id=post_id,
        action_type=action_type,
        dwell_time=dwell_time,
        context={
            "time_period": time_period,
            "location": location,
            "device_type": device_type,
            "scene": scene,
        },
    )
    return service.get_user_profile(user_id)

