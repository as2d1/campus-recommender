"""Post API routes."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request

from backend.app import service_unavailable_response
from backend.schemas import PostCreateRequest, fail, ok


router = APIRouter(prefix="/api", tags=["posts"])


@router.get("/posts/hot")
def get_hot_posts(
    request: Request,
    top_n: int = Query(10, ge=1, le=100),
) -> dict:
    service = request.app.state.service
    if service is None:
        return service_unavailable_response(request.app.state.service_error)
    try:
        return ok(service.get_hot_posts(top_n=top_n))
    except Exception as exc:
        return fail(f"failed to get hot posts: {exc}")


@router.get("/posts")
def get_post_list(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    board: str | None = None,
    keyword: str | None = None,
) -> dict:
    service = request.app.state.service
    if service is None:
        return service_unavailable_response(request.app.state.service_error)
    try:
        board_value = board.strip() if board and board.strip() else None
        keyword_value = keyword.strip() if keyword and keyword.strip() else None
        return ok(service.get_post_list(page=page, page_size=page_size, board=board_value, keyword=keyword_value))
    except Exception as exc:
        return fail(f"failed to get post list: {exc}")


@router.get("/posts/{post_id}")
def get_post_detail(post_id: str, request: Request) -> dict:
    service = request.app.state.service
    if service is None:
        return service_unavailable_response(request.app.state.service_error)
    try:
        return ok(service.get_post_detail(post_id))
    except ValueError as exc:
        return fail(str(exc))
    except Exception as exc:
        return fail(f"failed to get post detail: {exc}")


@router.post("/posts")
def create_post(payload: PostCreateRequest, request: Request) -> dict:
    service = request.app.state.service
    if service is None:
        return service_unavailable_response(request.app.state.service_error)
    try:
        return ok(service.create_post(payload.dict()), "post created")
    except ValueError as exc:
        return fail(str(exc))
    except Exception as exc:
        return fail(f"failed to create post: {exc}")
