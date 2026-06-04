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


class ApiResponse(BaseModel):
    """Unified API response."""

    success: bool
    message: str
    data: Any = None


def ok(data: Any = None, message: str = "ok") -> dict[str, Any]:
    return {"success": True, "message": message, "data": data}


def fail(message: str, data: Any = None) -> dict[str, Any]:
    return {"success": False, "message": message, "data": data}
