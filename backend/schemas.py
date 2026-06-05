"""Pydantic schemas for the FastAPI backend."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class BehaviorContext(BaseModel):
    """Optional behavior context sent by the frontend."""

    time_period: Optional[str] = None


class BehaviorRequest(BaseModel):
    """Request body for recording a user behavior."""

    user_id: str = Field(..., min_length=1)
    post_id: str = Field(..., min_length=1)
    action_type: str = Field(..., min_length=1)
    dwell_time: Optional[float] = None
    context: Optional[BehaviorContext] = None


class PreferenceRequest(BaseModel):
    """Cold-start preference choices for a user."""

    selected_boards: list[str] = Field(default_factory=list)
    selected_tags: list[str] = Field(default_factory=list)


class PostCreateRequest(BaseModel):
    """Request body for creating one post."""

    user_id: str = Field(default="demo_user_A", min_length=1)
    board: str = Field(..., min_length=1)
    title: str = Field(..., min_length=2)
    content: str = Field(..., min_length=5)
    tags: str = ""
    anonymous: bool = False
    image_paths: list[str] = Field(default_factory=list)
    image_urls: list[str] = Field(default_factory=list)
    created_at: Optional[str] = None


class ApiResponse(BaseModel):
    """Unified API response."""

    success: bool
    message: str
    data: Any = None


def ok(data: Any = None, message: str = "ok") -> dict[str, Any]:
    return {"success": True, "message": message, "data": data}


def fail(message: str, data: Any = None) -> dict[str, Any]:
    return {"success": False, "message": message, "data": data}
