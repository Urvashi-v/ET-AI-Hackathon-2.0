"""The proactive path: an event arrives, and the system says something useful.

    new work order / incident
        -> graph changed
        -> historical pattern matching   (lessons: has this happened before?)
        -> compliance matching           (does this touch an obligation?)
        -> open-action matching          (did we already decide how to fix it?)
        -> notification candidate
        -> notification

Every step reads real stored state. There is no timer, no simulated arrival, and
no notification that is not the conclusion of a query someone could re-run.

The bar for raising one
-----------------------
A notification interrupts a person, so the threshold is what it costs them to
read it and find nothing. Three rules follow:

* **it must carry its evidence** — the incident it matched, the requirement it
  touches, the action already open. A notification saying "possible issue with
  P-101B" is noise;
* **it must be new** — the same finding about the same asset does not fire twice,
  because a repeated alert is how alerting gets muted;
* **it must clear a real threshold** — precedent below the lessons-learned
  similarity floor is not precedent, and a compliance state the system could not
  decide is not a breach.

The most valuable notification this can produce is not a prediction. It is
"this exact failure happened on the sister pump in 2022, the investigation
raised two corrective actions, and both are still open" — a statement of fact
that was always available and that nobody had assembled.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from services.agents import compliance as compliance_agent
from services.agents import lessons as lessons_agent
from services.common import bus, db, graph
from services.common.logging import get_logger
from services.common.schemas import DataClass

log = get_logger(__name__)

#: Precedent below this is not worth interrupting anyone for. Deliberately above
#: the lessons-learned floor: browsing weak matches is fine when a person went
#: looking, and not fine when the system went looking on their behalf.
MIN_PRECEDENT_SIMILARITY = 0.45

#: Audience per finding kind. A field technician does not need the compliance
#: gap and the compliance officer does not need the strip-down precedent.
_AUDIENCE = {
    "recurrence": "reliability_engineer",
    "open_action": "reliability_engineer",
    "compliance": "hse_officer",
    "procedure_superseded": "field_technician",
}

_SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


@dataclass(slots=True)
class Candidate:
    """A notification the pipeline proposes, before de-duplication."""

    pattern_id: str
    kind: str
    severity: str
    title: str
    message: str
    reason: str
    asset_tag: str | None
    match_score: float
    evidence: dict[str, Any] = field(default_factory=dict)
    audience_role: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "kind": self.kind,
            "severity": self.severity,
            "title": self.title,
            "message": self.message,
            "reason": self.reason,
            "asset_tag": self.asset_tag,
            "match_score": round(self.match_score, 4),
            "evidence": self.evidence,
            "audience_role": self.audience_role or _AUDIENCE.get(self.kind),
        }


async def evaluate_event(
    *,
    event_type: str,
    asset_tag: str | None,
    description: str,
    ref_id: str | None = None,
    doc_id: str | None = None,
) -> list[Candidate]:
    """Run every matcher against one event and return what it found.

    Pure with respect to the notification store: it produces candidates and
    writes nothing, so the matchers can be tested without a side effect and the
    de-duplication policy lives in one place (:func:`raise_notifications`).
    """
    candidates: list[Candidate] = []

    candidates.extend(await _match_precedent(asset_tag, description, ref_id))
    if asset_tag:
        candidates.extend(await _match_open_actions(asset_tag, ref_id))
        candidates.extend(await _match_compliance(asset_tag))
        candidates.extend(await _match_superseded_procedure(asset_tag))

    candidates.sort(key=lambda c: (_SEVERITY_ORDER.get(c.severity, 9), -c.match_score))
    log.info(
        "proactive.evaluated",
        event_type=event_type,
        asset_tag=asset_tag,
        ref_id=ref_id,
        candidates=len(candidates),
    )
    return candidates


async def _match_precedent(
    asset_tag: str | None, description: str, ref_id: str | None
) -> list[Candidate]:
    """Has this happened before? The core lessons-learned trigger."""
    if not description.strip():
        return []
    result = await lessons_agent.find_precedents(
        description=description, asset_tag=asset_tag, limit=3
    )
    strong = [m for m in result.matches if m.similarity >= MIN_PRECEDENT_SIMILARITY]
    if not strong:
        return []

    best = strong[0]
    open_actions = [a for a in best.corrective_actions if a.get("is_open")]
    severity = "high" if open_actions else "medium"
    action_note = (
        f" That investigation raised {len(open_actions)} corrective action(s) that are "
        f"still open: {', '.join(a['action_id'] for a in open_actions)}."
        if open_actions
        else " Its corrective actions are all closed."
    )
    return [
        Candidate(
            pattern_id=f"recurrence:{best.incident_id}",
            kind="recurrence",
            severity=severity,
            title=f"Recurrence of {best.incident_id} on {asset_tag or 'this asset'}",
            message=(
                f"{best.title} ({best.occurred_on}) describes the same failure mechanism."
                f"{action_note}"
            ),
            reason="; ".join(best.match_reasons),
            asset_tag=asset_tag,
            match_score=best.similarity,
            evidence={
                "matched_incident": best.to_dict(),
                "triggering_ref": ref_id,
                "other_matches": [m.incident_id for m in strong[1:]],
            },
        )
    ]


_OPEN_ACTIONS = """
MATCH (e:Equipment {canonical_tag: $tag})
OPTIONAL MATCH (e)-[:SIBLING_OF]-(sib:Equipment)
WITH collect(DISTINCT e) + collect(DISTINCT sib) AS pumps, e
UNWIND pumps AS pump
MATCH (pump)<-[:INVOLVED]-(i:Incident)-[:GENERATED]->(c:CAPA)
WHERE coalesce(c.status, 'OPEN') <> 'CLOSED'
RETURN DISTINCT c.capa_id AS capa_id, c.action AS action, c.owner AS owner,
       c.due_date AS due_date, i.incident_id AS from_incident,
       pump.canonical_tag AS raised_against,
       pump.canonical_tag <> e.canonical_tag AS is_sibling
"""


async def _match_open_actions(asset_tag: str, ref_id: str | None) -> list[Candidate]:
    """An action already exists for this. Do not raise a second investigation.

    The single most useful thing to tell someone opening a work order: the plant
    already worked out what to do, wrote it down, and has not done it.
    """
    rows = await graph.read(_OPEN_ACTIONS, tag=asset_tag)
    if not rows:
        return []
    actions = [
        {
            "capa_id": r["capa_id"],
            "action": r["action"],
            "owner": r["owner"],
            "due_date": _iso(r["due_date"]),
            "from_incident": r["from_incident"],
            "raised_against": r["raised_against"],
            "is_sibling": bool(r["is_sibling"]),
        }
        for r in rows
    ]
    sibling_only = all(a["is_sibling"] for a in actions)
    return [
        Candidate(
            pattern_id=f"open_action:{asset_tag}:{'|'.join(sorted(a['capa_id'] for a in actions))}",
            kind="open_action",
            severity="high",
            title=f"{len(actions)} open corrective action(s) already cover {asset_tag}",
            message=(
                "; ".join(f"{a['capa_id']} ({a['owner'] or 'unassigned'}): {a['action']}" for a in actions[:3])
                + (
                    " — raised against identical equipment, not this asset."
                    if sibling_only
                    else ""
                )
            ),
            reason=(
                "Open CAPAs were found on this asset or its siblings via "
                "Equipment<-[:INVOLVED]-Incident-[:GENERATED]->CAPA."
            ),
            asset_tag=asset_tag,
            match_score=1.0,
            evidence={"open_actions": actions, "triggering_ref": ref_id},
        )
    ]


async def _match_compliance(asset_tag: str) -> list[Candidate]:
    """Does this event touch an obligation that is currently unmet?"""
    result = await compliance_agent.evaluate(asset_tag=asset_tag)
    gaps = [f for f in result["findings"] if f["state"] == "gap"]
    if not gaps:
        return []
    # Overdue records first: a missed statutory interval is a harder finding than
    # a procedure that could not be matched by retrieval.
    gaps.sort(key=lambda f: (f["testable_by"] != "evidence_document", f["req_id"]))
    top = gaps[0]
    return [
        Candidate(
            pattern_id=f"compliance:{asset_tag}:{top['req_id']}",
            kind="compliance",
            severity="high" if top["testable_by"] == "evidence_document" else "medium",
            title=f"{len(gaps)} unmet obligation(s) affect {asset_tag}",
            message=f"{top['req_id']} ({top['source_standard']} {top['clause'] or ''}): {top['reason']}",
            reason=(
                f"Evaluated against stored evidence. {result['decidable_count']} of "
                f"{result['findings_total']} requirement(s) in scope could be decided."
            ),
            asset_tag=asset_tag,
            match_score=1.0,
            evidence={
                "gaps": gaps[:5],
                "requirement_provenance": result["requirement_provenance"],
                "coverage_pct_of_decidable": result["coverage_pct_of_decidable"],
            },
        )
    ]


_SUPERSEDED_PROCEDURE = """
MATCH (d:Document)-[:DESCRIBES]->(e:Equipment {canonical_tag: $tag})
WHERE d.doc_type IN ['sop', 'manual'] AND d.is_current = false
OPTIONAL MATCH (newer:Document)-[:SUPERSEDES]->(d)
RETURN d.doc_id AS doc_id, d.title AS title, d.revision AS revision,
       d.doc_number AS doc_number,
       collect({doc_id: newer.doc_id, title: newer.title, revision: newer.revision})
         AS replacements
"""


async def _match_superseded_procedure(asset_tag: str) -> list[Candidate]:
    """A superseded procedure still describes this asset.

    Aimed at the field technician about to work to it. This is the finding with
    the shortest path from "system noticed" to "someone does not get hurt", which
    is why it exists even though it is the simplest query here.
    """
    rows = await graph.read(_SUPERSEDED_PROCEDURE, tag=asset_tag)
    if not rows:
        return []
    stale = [
        {
            "doc_id": r["doc_id"],
            "title": r["title"],
            "revision": r["revision"],
            "doc_number": r["doc_number"],
            "replacements": [x for x in r["replacements"] if x.get("doc_id")],
        }
        for r in rows
    ]
    first = stale[0]
    replacement = first["replacements"][0] if first["replacements"] else None
    return [
        Candidate(
            pattern_id=f"procedure_superseded:{asset_tag}:{first['doc_id']}",
            kind="procedure_superseded",
            severity="medium",
            title=f"Superseded procedure still on file for {asset_tag}",
            message=(
                f"{first['title']} (rev {first['revision']}) has been superseded"
                + (
                    f" by rev {replacement['revision']}."
                    if replacement and replacement.get("revision")
                    else "."
                )
                + " Confirm the current revision before working to it."
            ),
            reason="Document is linked is_current=false with a SUPERSEDES edge from a newer revision.",
            asset_tag=asset_tag,
            match_score=1.0,
            evidence={"superseded_documents": stale},
        )
    ]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


async def raise_notifications(
    candidates: list[Candidate], *, event_id: int | None = None
) -> list[dict[str, Any]]:
    """Store candidates that are not already outstanding, and publish them.

    De-duplication is on ``pattern_id`` among *unacknowledged* notifications: the
    same finding does not fire twice while it is still on someone's list, but it
    can fire again after they acknowledge it and the condition persists. Muting
    it forever would be worse than repeating it.
    """
    raised: list[dict[str, Any]] = []
    for candidate in candidates:
        payload = candidate.to_dict()
        asset_id = await _asset_id(candidate.asset_tag)

        existing = await db.fetch_one(
            "SELECT notification_id FROM notifications "
            "WHERE pattern_id = %s AND acknowledged_at IS NULL",
            (candidate.pattern_id,),
        )
        if existing:
            continue

        row = await db.fetch_one(
            """
            INSERT INTO notifications (
                event_id, asset_id, severity, audience_role, title, message, reason,
                evidence, pattern_id, match_score, data_class
            ) VALUES (
                %(event_id)s, %(asset_id)s, %(severity)s, %(audience)s, %(title)s,
                %(message)s, %(reason)s, %(evidence)s, %(pattern_id)s, %(score)s, %(data_class)s
            )
            RETURNING notification_id, created_at
            """,
            {
                "event_id": event_id,
                "asset_id": asset_id,
                "severity": candidate.severity,
                "audience": payload["audience_role"],
                "title": candidate.title,
                "message": candidate.message,
                "reason": candidate.reason,
                "evidence": json.dumps(candidate.evidence, default=str),
                "pattern_id": candidate.pattern_id,
                "score": candidate.match_score,
                # A notification is a conclusion drawn from records, not a record.
                "data_class": DataClass.MODEL_DERIVED.value,
            },
        )
        if not row:
            continue
        stored = {
            "notification_id": int(row["notification_id"]),
            "created_at": row["created_at"].isoformat(),
            **payload,
        }
        raised.append(stored)

        # Live delivery to anything listening on the SSE stream. The store is the
        # source of truth; this is the push half of the same fact.
        try:
            await bus.publish_event("notification.raised", stored)
        except Exception as exc:  # delivery must not lose the stored notification
            log.warning("proactive.publish_failed", error=str(exc))

    if raised:
        log.info(
            "proactive.notifications_raised",
            count=len(raised),
            patterns=[r["pattern_id"] for r in raised],
        )
    return raised


async def _asset_id(asset_tag: str | None) -> str | None:
    if not asset_tag:
        return None
    row = await db.fetch_one(
        "SELECT asset_id FROM assets WHERE upper(canonical_tag) = upper(%s)", (asset_tag,)
    )
    return str(row["asset_id"]) if row else None


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "iso_format"):
        return value.iso_format()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)
