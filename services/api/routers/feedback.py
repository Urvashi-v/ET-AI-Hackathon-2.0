"""Feedback capture.

A thumbs-down that asks *what* was wrong -- wrong source, outdated, incomplete,
wrong asset -- is worth far more than a bare score, because each reason routes
somewhere different: "wrong asset" is an entity-resolution defect and belongs in
the review queue; "outdated" is a revision-control defect and belongs with the
document owner.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, status

from services.common import db
from services.common.errors import NotFoundError, ValidationError
from services.common.logging import get_logger
from services.common.schemas import FeedbackRequest, FeedbackResponse

log = get_logger(__name__)
router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post(
    "",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit feedback on an answer or a notification",
)
async def submit(request: FeedbackRequest) -> FeedbackResponse:
    if not request.query_id and request.notification_id is None:
        raise ValidationError("Either query_id or notification_id must be provided.")

    if request.query_id:
        exists = await db.fetch_one(
            "SELECT query_id FROM query_log WHERE query_id = %s", (request.query_id,)
        )
        if not exists:
            raise NotFoundError(f"No logged query with id {request.query_id}.")
    if request.notification_id is not None:
        exists = await db.fetch_one(
            "SELECT notification_id FROM notifications WHERE notification_id = %s",
            (request.notification_id,),
        )
        if not exists:
            raise NotFoundError(f"No notification with id {request.notification_id}.")

    row = await db.fetch_one(
        """
        INSERT INTO feedback (query_id, notification_id, verdict, reason, note, submitted_by)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING feedback_id, queued_for_review
        """,
        (
            request.query_id,
            request.notification_id,
            request.verdict,
            request.reason,
            request.note,
            request.submitted_by,
        ),
    )
    assert row is not None

    # A "wrong asset" report is an entity-resolution defect. Route it to the
    # review queue so the correction reaches the layer that caused it, rather
    # than sitting in a feedback table nobody reads.
    if request.verdict == "down" and request.reason == "wrong_asset":
        await db.execute(
            "INSERT INTO review_queue (kind, subject, detail, confidence) VALUES (%s,%s,%s,%s)",
            (
                "entity_resolution",
                f"User reported a wrong asset on query {request.query_id}",
                json.dumps(
                    {
                        "query_id": request.query_id,
                        "note": request.note,
                        "source": "user_feedback",
                    }
                ),
                None,
            ),
        )

    log.info(
        "feedback.received",
        feedback_id=row["feedback_id"],
        verdict=request.verdict,
        reason=request.reason,
    )
    return FeedbackResponse(
        feedback_id=row["feedback_id"], queued_for_review=row["queued_for_review"]
    )


@router.get("", summary="Feedback summary")
async def summary() -> dict:
    rows = await db.fetch_all(
        "SELECT verdict, coalesce(reason, 'unspecified') AS reason, count(*)::int AS n "
        "FROM feedback GROUP BY 1, 2 ORDER BY n DESC"
    )
    open_reviews = await db.fetch_one(
        "SELECT count(*)::int AS n FROM review_queue WHERE resolved_at IS NULL"
    )
    return {
        "by_verdict_and_reason": [dict(r) for r in rows],
        "open_review_items": int(open_reviews["n"]) if open_reviews else 0,
    }
