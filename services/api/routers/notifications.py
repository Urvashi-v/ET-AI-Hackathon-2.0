"""Proactive notifications.

The pull direction -- the user asks, the system answers -- is what every other
part of this API does. This endpoint is the push direction: knowledge reaching
someone who did not know to ask for it.

``GET /api/v1/notifications`` returns real rows from the ``notifications`` table.
Every notification carries its evidence and the reason it fired, because a bare
"warning: similar failure risk" is noise, whereas "in 2019 and 2022 this failure
on sister equipment was caused by dry running during startup -- here are both
reports" is knowledge transfer.

The **matching engine** that produces these rows -- pattern matching a new event
against historical incident clusters -- is not implemented in this build, and the
``engine`` field says so. The endpoint therefore returns an empty list until the
engine exists, rather than sample alerts that would look identical to real ones.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from services.common import db
from services.common.errors import NotFoundError
from services.common.schemas import (
    CapabilityState,
    CapabilityStatus,
    DataClass,
    EvidenceRef,
    Notification,
    NotificationListResponse,
)

router = APIRouter(prefix="/notifications", tags=["proactive"])

_ENGINE_STATUS = CapabilityStatus(
    capability="proactive_pattern_engine",
    state=CapabilityState.NOT_IMPLEMENTED,
    detail=(
        "The event-driven matching engine (new event -> similar historical patterns -> "
        "push with evidence) is not implemented in this build. The event bus that will "
        "trigger it is running: ingestion already publishes 'graph.changed' and "
        "'document.ingested' events, which are visible on /api/v1/events/stream. "
        "Notifications listed here are real rows only; none are generated for display."
    ),
)


@router.get("", response_model=NotificationListResponse, summary="Proactive notifications")
async def list_notifications(
    asset_tag: str | None = Query(None),
    role: str | None = Query(None, description="Filter to a target audience role"),
    unacknowledged_only: bool = Query(False),
    limit: int = Query(50, ge=1, le=200),
) -> NotificationListResponse:
    filters: list[str] = []
    params: dict[str, Any] = {"limit": limit}
    if asset_tag:
        filters.append("a.canonical_tag ILIKE %(asset_tag)s")
        params["asset_tag"] = asset_tag
    if role:
        filters.append("n.audience_role = %(role)s")
        params["role"] = role
    if unacknowledged_only:
        filters.append("n.acknowledged_at IS NULL")
    where = f"WHERE {' AND '.join(filters)}" if filters else ""

    rows = await db.fetch_all(
        f"""
        SELECT n.notification_id, n.severity, n.title, n.message, n.reason,
               n.audience_role, n.evidence, n.pattern_id, n.match_score,
               n.data_class::text AS data_class, n.acknowledged_at, n.created_at,
               a.canonical_tag AS asset_tag
          FROM notifications n
          LEFT JOIN assets a ON a.asset_id = n.asset_id
          {where}
         ORDER BY n.created_at DESC
         LIMIT %(limit)s
        """,
        params,
    )
    total_row = await db.fetch_one("SELECT count(*)::int AS n FROM notifications")

    return NotificationListResponse(
        total=int(total_row["n"]) if total_row else 0,
        items=[
            Notification(
                notification_id=r["notification_id"],
                severity=r["severity"],
                title=r["title"],
                message=r["message"],
                reason=r["reason"],
                asset_tag=r["asset_tag"],
                audience_role=r["audience_role"],
                evidence=[EvidenceRef(**e) for e in (r["evidence"] or [])],
                pattern_id=r["pattern_id"],
                match_score=r["match_score"],
                data_class=DataClass(r["data_class"]),
                acknowledged=r["acknowledged_at"] is not None,
                created_at=r["created_at"],
            )
            for r in rows
        ],
        engine=_ENGINE_STATUS,
    )


@router.post("/{notification_id}/acknowledge", summary="Acknowledge a notification")
async def acknowledge(notification_id: int) -> dict[str, Any]:
    rowcount = await db.execute(
        "UPDATE notifications SET acknowledged_at = now() "
        "WHERE notification_id = %s AND acknowledged_at IS NULL",
        (notification_id,),
    )
    if rowcount == 0:
        exists = await db.fetch_one(
            "SELECT acknowledged_at FROM notifications WHERE notification_id = %s",
            (notification_id,),
        )
        if not exists:
            raise NotFoundError(f"No notification with id {notification_id}.")
        return {"notification_id": notification_id, "acknowledged": True, "already": True}
    return {"notification_id": notification_id, "acknowledged": True, "already": False}
