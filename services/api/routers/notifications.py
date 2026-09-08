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

from services.agents import proactive
from services.common import db
from services.common.errors import NotFoundError
from services.common.schemas import (
    CapabilityState,
    CapabilityStatus,
    DataClass,
    EventEvaluationRequest,
    Notification,
    NotificationListResponse,
)

router = APIRouter(prefix="/notifications", tags=["proactive"])

_ENGINE_STATUS = CapabilityStatus(
    capability="proactive_pattern_engine",
    state=CapabilityState.AVAILABLE,
    detail=(
        "Event-driven matching runs over stored state: an event is matched against "
        "historical incidents (semantic + mechanism + graph signals), against open "
        "corrective actions on the asset and its siblings, against the requirement set, "
        "and against superseded procedures still describing the asset. Notifications are "
        "rows in Postgres raised by that pipeline and pushed on /api/v1/events/stream. "
        "Nothing here is generated for display, and nothing is produced on a timer."
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
                kind=(r["pattern_id"] or "").split(":")[0] or None,
                evidence=r["evidence"] or {},
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


# ---------------------------------------------------------------------------
# The event path (Day 4)
# ---------------------------------------------------------------------------


@router.post("/evaluate", summary="Run the proactive matchers against an event")
async def evaluate_event(request: EventEvaluationRequest) -> dict[str, Any]:
    """Drive the pipeline for one event and raise what it finds.

    This is the same code path ingestion triggers; exposing it directly makes the
    behaviour testable and lets an operator ask "what would you tell me about
    this?" without waiting for the event to occur.

    ``dry_run`` returns the candidates without storing them, which is how the
    matchers are inspected without filling somebody's notification list.
    """
    candidates = await proactive.evaluate_event(
        event_type=request.event_type,
        asset_tag=request.asset_tag,
        description=request.description,
        ref_id=request.ref_id,
    )
    if request.dry_run:
        return {
            "dry_run": True,
            "candidates": [c.to_dict() for c in candidates],
            "raised": [],
            "detail": "Candidates were evaluated but not stored.",
        }

    raised = await proactive.raise_notifications(candidates)
    suppressed = len(candidates) - len(raised)
    return {
        "dry_run": False,
        "candidates": [c.to_dict() for c in candidates],
        "raised": raised,
        "detail": (
            f"{len(raised)} notification(s) raised"
            + (
                f"; {suppressed} suppressed as already outstanding for the same pattern."
                if suppressed
                else "."
            )
        ),
    }
