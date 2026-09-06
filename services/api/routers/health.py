"""Health endpoints.

Three levels, because they answer different questions:

``/health/live``
    Is the process running? Used by the container healthcheck; touches nothing.

``/health/ready``
    Can this instance serve traffic? Probes all three stores.

``/health``
    The full, honest report: live dependency state (including real counts from
    each store), which AI providers are configured and which credentials are
    missing, which external connectors are connected, and the effective non-secret
    configuration. This is what the dashboard renders on its system panel, so a
    reader can always tell what is actually running.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Request, Response, status

from services.common import bus, db, graph
from services.common.config import get_settings
from services.common.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health/live", summary="Liveness probe")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready", summary="Readiness probe")
async def ready(response: Response) -> dict[str, Any]:
    checks = {
        "postgres": await db.ping(),
        "neo4j": await graph.ping(),
        "redis": await bus.ping(),
    }
    ok = all(c.get("status") == "up" for c in checks.values())
    if not ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "ready" if ok else "not_ready", "dependencies": checks}


@router.get("/health", response_model=HealthResponse, summary="Full system report")
async def health(request: Request, response: Response) -> HealthResponse:
    settings = get_settings()
    dependencies: dict[str, dict[str, Any]] = {
        "postgres": await db.ping(),
        "neo4j": await graph.ping(),
        "redis": await bus.ping(),
    }

    startup_errors = getattr(request.app.state, "startup_errors", {})
    for name, message in startup_errors.items():
        dependencies.setdefault(name, {})["startup_error"] = message

    migration_report = getattr(request.app.state, "migration_report", None)
    if migration_report is not None:
        dependencies["schema"] = {
            "status": "up" if migration_report.get("status") == "ok" else "degraded",
            "detail": migration_report,
        }

    infra_ok = all(
        dependencies.get(name, {}).get("status") == "up" for name in ("postgres", "neo4j", "redis")
    )
    schema_ok = dependencies.get("schema", {}).get("status") == "up"
    overall = "ok" if infra_ok and schema_ok else ("degraded" if infra_ok else "down")
    if overall == "down":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status=overall,  # type: ignore[arg-type]
        app=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
        checked_at=datetime.now(UTC),
        dependencies=dependencies,
        providers=settings.provider_status(),
        connectors=settings.connector_status(),
        config=settings.public_dict(),
    )
