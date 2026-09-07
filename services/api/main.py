"""FastAPI application entrypoint.

Responsibilities kept deliberately narrow: process lifecycle (open and close the
three store clients, apply schemas), cross-cutting middleware (request id,
structured access log, safe error mapping), and mounting the routers plus the
static dashboard.

The dashboard is served from this same process at ``/``, so the vanilla-JS
frontend talks to a real, same-origin API. There is no mock server and no CORS
shim, which also means there is no code path where the UI can be pointed at
fixtures by accident.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from services.api.routers import (
    assets,
    compliance,
    documents,
    drawings,
    events,
    feedback,
    graph_router,
    health,
    ingest,
    lessons,
    notifications,
    query,
    rca,
)
from services.common import bus, db, graph, migrate
from services.common.config import get_settings
from services.common.errors import BrainError
from services.common.ids import request_id as new_request_id
from services.common.logging import (
    bind_request_context,
    clear_request_context,
    configure_logging,
    get_logger,
)
from services.retrieval import warmup

log = get_logger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = REPO_ROOT / "web"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    settings = get_settings()
    log.info("api.starting", version=settings.app_version, env=settings.app_env)

    # Store clients are opened independently: a degraded dependency must produce
    # an honest /health report rather than an unbootable process.
    app.state.startup_errors = {}
    for name, opener in (
        ("postgres", db.open_pool),
        ("neo4j", graph.open_driver),
        ("redis", bus.open_client),
    ):
        try:
            await opener()
        except Exception as exc:
            app.state.startup_errors[name] = f"{type(exc).__name__}: {exc}"
            log.error("api.dependency_open_failed", dependency=name, error=str(exc))

    if not app.state.startup_errors:
        try:
            app.state.migration_report = await migrate.run()
            log.info("api.schema_ready", status=app.state.migration_report["status"])
        except Exception as exc:
            app.state.startup_errors["schema"] = f"{type(exc).__name__}: {exc}"
            log.error("api.migration_failed", error=str(exc))
    else:
        app.state.migration_report = {"status": "skipped", "reason": "dependencies unavailable"}

    # Local ONNX models reach steady-state speed only after a few inferences, so
    # they are warmed here rather than on the first engineer's question. Detached
    # deliberately: warm-up on a cold model cache includes the download, and
    # blocking startup on that would fail the container's health check.
    app.state.warmup = asyncio.create_task(warmup.warm_models())

    yield

    app.state.warmup.cancel()
    await db.close_pool()
    await graph.close_driver()
    await bus.close_client()
    log.info("api.stopped")


app = FastAPI(
    title="Unified Asset & Operations Brain",
    version=get_settings().app_version,
    description=(
        "One industrial knowledge substrate. Document ingestion, entity resolution, a "
        "provenance-carrying knowledge graph, hybrid retrieval, and the five capabilities "
        "built as query patterns over it.\n\n"
        "**Truthfulness contract:** any endpoint that cannot complete an operation returns a "
        "structured status naming the missing capability and the environment variables that "
        "would enable it. No endpoint fabricates data."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


@app.middleware("http")
async def request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
    rid = request.headers.get("x-request-id") or new_request_id()
    bind_request_context(request_id=rid, path=request.url.path, method=request.method)
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except BrainError:
        raise
    except Exception as exc:
        elapsed = (time.perf_counter() - started) * 1000
        log.exception("http.unhandled_error", error=str(exc), elapsed_ms=round(elapsed, 2))
        clear_request_context()
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "An internal error occurred.",
                    "detail": {"request_id": rid},
                }
            },
            headers={"x-request-id": rid},
        )
    elapsed = (time.perf_counter() - started) * 1000
    response.headers["x-request-id"] = rid
    if not request.url.path.startswith(("/static", "/assets", "/css", "/js")):
        log.info("http.request", status=response.status_code, elapsed_ms=round(elapsed, 2))
    clear_request_context()
    return response


@app.middleware("http")
async def security_headers(request: Request, call_next):  # type: ignore[no-untyped-def]
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    return response


@app.exception_handler(BrainError)
async def brain_error_handler(request: Request, exc: BrainError) -> JSONResponse:
    log.warning("http.application_error", code=exc.code, message=exc.message)
    return JSONResponse(status_code=exc.status_code, content=exc.to_payload())


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Pydantic's raw errors can echo submitted values; project to field + reason.
    fields = [
        {"field": ".".join(str(p) for p in err.get("loc", [])[1:]), "reason": err.get("msg", "")}
        for err in exc.errors()
    ]
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "code": "validation_error",
                "message": "The request did not match the API contract.",
                "detail": {"fields": fields},
            }
        },
    )


API_PREFIX = "/api/v1"
app.include_router(health.router)
app.include_router(ingest.router, prefix=API_PREFIX)
app.include_router(documents.router, prefix=API_PREFIX)
app.include_router(drawings.router, prefix=API_PREFIX)
app.include_router(query.router, prefix=API_PREFIX)
app.include_router(assets.router, prefix=API_PREFIX)
app.include_router(graph_router.router, prefix=API_PREFIX)
app.include_router(rca.router, prefix=API_PREFIX)
app.include_router(lessons.router, prefix=API_PREFIX)
app.include_router(compliance.router, prefix=API_PREFIX)
app.include_router(notifications.router, prefix=API_PREFIX)
app.include_router(feedback.router, prefix=API_PREFIX)
app.include_router(events.router, prefix=API_PREFIX)


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/ui/index.html")


class RevalidatingStaticFiles(StaticFiles):
    """Serve the dashboard with must-revalidate caching.

    The frontend has no build step and no content hashing -- ``js/api.js`` is
    served under that exact name forever. With Starlette's default headers a
    browser caches it heuristically and keeps using the copy it already has, so
    a code change is simply invisible until someone thinks to hard-refresh. That
    is not a testing annoyance: it means a deployed fix does not reach the
    people who already have the page open, which for a superseded-procedure
    warning is a safety problem.

    ``no-cache`` does not mean "do not cache". It means "cache, but revalidate
    before use": the browser keeps the file and sends an If-None-Match, and
    Starlette answers 304 from the ETag it already computes. The cost is one
    conditional request per asset; the benefit is that what is on screen is what
    is on disk.

    Content-hashed filenames would be better and would allow immutable caching,
    but they need a build step, and adding one to avoid a header would be the
    wrong trade for this project.
    """

    def is_not_modified(self, response_headers, request_headers) -> bool:  # type: ignore[override]
        response_headers["Cache-Control"] = "no-cache"
        return super().is_not_modified(response_headers, request_headers)

    async def get_response(self, path: str, scope):  # type: ignore[override]
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


if WEB_DIR.is_dir():
    app.mount("/ui", RevalidatingStaticFiles(directory=str(WEB_DIR), html=True), name="ui")
else:  # pragma: no cover - only when the image is built without web/
    log.warning("api.web_dir_missing", path=str(WEB_DIR))
