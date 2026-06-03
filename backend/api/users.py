"""User, board, and tag API routes."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request

from backend.app import service_unavailable_response
from backend.schemas import fail, ok


router = APIRouter(prefix="/api", tags=["users"])


@router.get("/users/{user_id}/profile")
def get_user_profile(user_id: str, request: Request) -> dict:
    service = request.app.state.service
    if service is None:
        return service_unavailable_response(request.app.state.service_error)
    try:
        return ok(service.get_user_profile(user_id))
    except ValueError as exc:
        return fail(str(exc))
    except Exception as exc:
        return fail(f"failed to get user profile: {exc}")


@router.post("/users/{user_id}/refresh-profile")
def refresh_user_profile(user_id: str, request: Request) -> dict:
    service = request.app.state.service
    if service is None:
        return service_unavailable_response(request.app.state.service_error)
    try:
        return ok(service.refresh_user_profile(user_id), "profile refreshed")
    except ValueError as exc:
        return fail(str(exc))
    except Exception as exc:
        return fail(f"failed to refresh user profile: {exc}")


@router.get("/boards")
def get_boards(request: Request) -> dict:
    service = request.app.state.service
    if service is None:
        return service_unavailable_response(request.app.state.service_error)
    try:
        return ok(service.get_boards())
    except Exception as exc:
        return fail(f"failed to get boards: {exc}")


@router.get("/tags")
def get_tags(
    request: Request,
    top_n: int = Query(50, ge=1, le=200),
) -> dict:
    service = request.app.state.service
    if service is None:
        return service_unavailable_response(request.app.state.service_error)
    try:
        return ok(service.get_tags(top_n=top_n))
    except Exception as exc:
        return fail(f"failed to get tags: {exc}")
