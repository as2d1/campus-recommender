"""FastAPI entrypoint for the campus recommender backend."""

from __future__ import annotations

import sys
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.core.settings import APP_NAME, CORS_ORIGINS, MODEL_DIR, SQLITE_PATH
from backend.schemas import fail, ok
from src.service import CampusRecommendService


def configure_console_encoding() -> None:
    """Make Windows console output tolerant of Chinese text and emoji."""

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


configure_console_encoding()

app = FastAPI(title=APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def initialize_service() -> tuple[CampusRecommendService | None, str | None]:
    """Initialize the algorithm service without crashing the backend process."""

    try:
        service = CampusRecommendService(sqlite_path=SQLITE_PATH, model_dir=MODEL_DIR)
        return service, service.model_warning
    except Exception as exc:
        return None, f"CampusRecommendService initialization failed: {exc}"


app.state.service, app.state.service_error = initialize_service()


def get_service_or_error() -> tuple[CampusRecommendService | None, str | None]:
    service = getattr(app.state, "service", None)
    error = getattr(app.state, "service_error", None)
    return service, error


def service_unavailable_response(error: str | None) -> dict[str, Any]:
    return fail(error or "recommendation service is not available")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Return validation errors with the project's unified JSON shape."""

    return JSONResponse(
        status_code=200,
        content=fail(f"request validation failed: {exc.errors()}"),
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Avoid exposing Python tracebacks to the frontend."""

    return JSONResponse(status_code=200, content=fail(f"internal server error: {exc}"))


@app.get("/api/health")
def health() -> dict[str, Any]:
    service, error = get_service_or_error()
    data = {
        "status": "ok" if service is not None else "degraded",
        "service_loaded": service is not None,
        "warning": error,
    }
    if service is None:
        return fail(error or "campus recommender backend is running but service is unavailable", data)
    return ok(data, "campus recommender backend is running")


from backend.api.recommend import router as recommend_router  # noqa: E402
from backend.api.posts import router as posts_router  # noqa: E402
from backend.api.users import router as users_router  # noqa: E402
from backend.api.behavior import router as behavior_router  # noqa: E402


app.include_router(recommend_router)
app.include_router(posts_router)
app.include_router(users_router)
app.include_router(behavior_router)
