"""User behavior API routes."""

from __future__ import annotations

from fastapi import APIRouter, Request

from backend.app import service_unavailable_response
from backend.schemas import BehaviorRequest, fail, ok


router = APIRouter(prefix="/api", tags=["behavior"])


@router.post("/behavior")
def record_user_behavior(payload: BehaviorRequest, request: Request) -> dict:
    service = request.app.state.service
    if service is None:
        return service_unavailable_response(request.app.state.service_error)
    try:
        context = payload.context.dict(exclude_none=True) if payload.context else None
        behavior = service.record_user_behavior(
            user_id=payload.user_id,
            post_id=payload.post_id,
            action_type=payload.action_type,
            dwell_time=payload.dwell_time,
            context=context,
        )
        return ok(behavior, "behavior recorded")
    except ValueError as exc:
        return fail(str(exc))
    except Exception as exc:
        return fail(f"failed to record behavior: {exc}")
