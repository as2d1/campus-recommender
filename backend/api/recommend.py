"""Recommendation API routes."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request

from backend.app import service_unavailable_response
from backend.schemas import fail, ok


router = APIRouter(prefix="/api", tags=["recommend"])


@router.get("/recommend/{user_id}")
def recommend_for_user(
    user_id: str,
    request: Request,
    top_n: int = Query(10, ge=1, le=100),
) -> dict:
    service = request.app.state.service
    if service is None:
        return service_unavailable_response(request.app.state.service_error)
    try:
        recommendations = service.recommend_for_user(user_id, top_n=top_n)
        if not recommendations:
            return ok([], "recommendation result is empty")
        return ok(recommendations)
    except ValueError as exc:
        return fail(str(exc))
    except Exception as exc:
        return fail(f"failed to get recommendations: {exc}")
