"""Root cause analysis endpoint.

The parts that are deterministic run today and return real results:

* **evidence gathering** -- failure history on this asset and on its siblings,
  incidents, inspections, recent MOCs touching the asset, and open CAPAs, all by
  real graph and SQL queries;
* **reliability metrics** -- MTBF, mean downtime and event counts, computed from
  the stored work orders, labelled ``calculated_metric``;
* **similar historical events** -- ranked by a composite of structural signals
  (same equipment class, same coded failure mode, sibling relationship,
  graph distance), each returned with the reason it matched, because an
  unexplained similarity score is dismissed and an explained one is trusted.

The part that requires reasoning -- building the causal tree down to a systemic
cause, and listing the hypotheses it ruled out -- is capability-gated. Without a
generation provider this endpoint returns the evidence and says plainly that the
causal analysis was not produced. It does not emit a template tree dressed up as
an inference.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from fastapi import APIRouter

from services.agents import rca as rca_agent
from services.common import db, graph
from services.common.logging import get_logger
from services.common.schemas import (
    CandidateCauseOut,
    CapabilityState,
    CapabilityStatus,
    DataClass,
    RCARequest,
    RCAResponse,
)
from services.common.tags import parse
from services.retrieval.generate import generation_capability

log = get_logger(__name__)
router = APIRouter(prefix="/rca", tags=["reliability"])


@router.post("", response_model=RCAResponse, summary="Root cause analysis for a failure event")
async def run_rca(request: RCARequest) -> RCAResponse:
    parsed = parse(request.asset_tag)
    canonical = parsed.canonical if parsed.parsed else request.asset_tag.upper()

    asset = await db.fetch_one(
        "SELECT asset_id, canonical_tag, class_code, class_label FROM assets "
        "WHERE upper(canonical_tag) = upper(%s)",
        (canonical,),
    )
    if not asset:
        return RCAResponse(
            asset_tag=canonical,
            asset_found=False,
            event=request.failure_description,
            status=CapabilityStatus(
                capability="rca",
                state=CapabilityState.NOT_CONFIGURED,
                detail=(
                    f"No asset matching '{request.asset_tag}' exists in the corpus, so there "
                    "is no history to analyse. Ingest documents mentioning this tag first."
                ),
            ),
        )

    asset_id = asset["asset_id"]

    work_orders = await db.fetch_all(
        "SELECT wo_id, wo_type, description, long_text, as_found, as_left, "
        "coded_failure_mode, extracted_failure_mode, opened_on, closed_on, "
        "downtime_hours, cost, source_system, data_class::text AS data_class "
        "FROM work_orders WHERE asset_id = %s ORDER BY opened_on DESC NULLS LAST",
        (asset_id,),
    )
    incidents = await db.fetch_all(
        "SELECT incident_id, title, occurred_on, severity, immediate_cause, root_cause, "
        "investigation_status, source_system, data_class::text AS data_class "
        "FROM incidents WHERE asset_id = %s ORDER BY occurred_on DESC NULLS LAST",
        (asset_id,),
    )
    inspections = await db.fetch_all(
        "SELECT inspection_id, cml_id, method, inspected_on, thickness_mm, min_required_mm, "
        "finding, data_class::text AS data_class FROM inspections WHERE asset_id = %s "
        "ORDER BY inspected_on DESC NULLS LAST",
        (asset_id,),
    )

    siblings = await _siblings(asset["canonical_tag"])
    sibling_history = await _sibling_history([s["canonical_tag"] for s in siblings])
    open_capas = await _open_capas(asset["canonical_tag"])
    mocs = await _recent_mocs(asset["canonical_tag"])

    similar = _rank_similar_events(
        target_description=request.failure_description,
        target_class=asset["class_code"],
        own_history=work_orders,
        own_incidents=incidents,
        sibling_history=sibling_history,
        sibling_tags={s["canonical_tag"] for s in siblings},
    )

    metrics = _reliability_metrics(work_orders)

    evidence_gathered = {
        "work_orders": len(work_orders),
        "incidents": len(incidents),
        "inspections": len(inspections),
        "siblings": len(siblings),
        "sibling_events": len(sibling_history),
        "open_capas": len(open_capas),
        "management_of_change_records": len(mocs),
    }

    # --- causal analysis over recorded evidence -----------------------------
    # Deterministic and credential-free. Candidate causes are aggregated from
    # cause statements the plant already wrote down -- incident root causes,
    # work-order as-found notes -- grouped by mechanism and ranked by how much
    # independent evidence supports each. Nothing is generated: an LLM would
    # produce a fluent RCA for any pump on earth, and that is precisely the
    # failure mode this replaces.
    analysis = rca_agent.analyse(
        asset_tag=asset["canonical_tag"],
        incidents=incidents,
        work_orders=work_orders,
        sibling_events=sibling_history,
        inspections=inspections,
    )

    generation = generation_capability()
    if analysis.abstained:
        status = CapabilityStatus(
            capability="rca_causal_analysis",
            state=CapabilityState.AVAILABLE,
            detail=(
                "Evidence gathered and returned below, but no causal ranking was produced. "
                + (analysis.abstain_reason or "")
            ),
        )
    else:
        detail = (
            f"{len(analysis.candidates)} candidate cause(s) ranked from "
            f"{analysis.evidence_count} recorded statement(s) across incidents, work orders "
            "and identical equipment. Aggregated from stored records, not generated."
        )
        if generation.state is not CapabilityState.AVAILABLE:
            detail += (
                " A narrative causal tree would additionally require a generation provider; "
                "the ranking below does not."
            )
        status = CapabilityStatus(
            capability="rca_causal_analysis",
            state=CapabilityState.AVAILABLE,
            detail=detail,
            required_env=(
                [] if generation.state is CapabilityState.AVAILABLE else generation.required_env
            ),
        )

    return RCAResponse(
        asset_tag=asset["canonical_tag"],
        asset_found=True,
        event=request.failure_description,
        observed_symptom=rca_agent.summarise_symptom(request.failure_description),
        status=status,
        evidence_gathered=evidence_gathered,
        reliability_metrics=metrics,
        candidate_causes=[CandidateCauseOut(**c.to_dict()) for c in analysis.candidates],
        causal_analysis_abstained=analysis.abstained,
        causal_analysis_reason=analysis.abstain_reason,
        historical_occurrences=_timeline(work_orders, incidents, sibling_history),
        related_work_orders=[_wo_summary(w) for w in work_orders],
        related_incidents=[_incident_summary(i) for i in incidents],
        sibling_history=sibling_history,
        open_corrective_actions=open_capas,
        management_of_change=mocs,
        similar_events=similar,
        duplicate_of_open_capa=open_capas[0]["capa_id"] if open_capas else None,
        overall_confidence=_overall_confidence(analysis, evidence_gathered),
        discriminating_evidence_needed=_missing_evidence(work_orders, incidents, inspections, mocs),
    )


def _timeline(
    work_orders: list[dict[str, Any]],
    incidents: list[dict[str, Any]],
    sibling_history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Every recorded event on one axis, oldest first.

    Siblings are included and flagged rather than kept separate: the point of the
    timeline is to make a pattern across a duty/standby pair visible, and two
    separate timelines hide exactly that.
    """
    events: list[dict[str, Any]] = []
    for wo in work_orders:
        events.append(
            {
                "kind": "work_order",
                "ref_id": wo["wo_id"],
                "date": _iso(wo.get("opened_on")),
                "label": wo.get("description") or wo["wo_id"],
                "detail": wo.get("as_found"),
                "wo_type": wo.get("wo_type"),
                "downtime_hours": float(wo["downtime_hours"]) if wo.get("downtime_hours") else None,
                "is_sibling": False,
                "data_class": wo.get("data_class"),
            }
        )
    for inc in incidents:
        events.append(
            {
                "kind": "incident",
                "ref_id": inc["incident_id"],
                "date": _iso(inc.get("occurred_on")),
                "label": inc.get("title") or inc["incident_id"],
                "detail": inc.get("root_cause") or inc.get("immediate_cause"),
                "severity": inc.get("severity"),
                "status": inc.get("investigation_status"),
                "is_sibling": False,
                "data_class": inc.get("data_class"),
            }
        )
    for event in sibling_history:
        events.append(
            {
                "kind": str(event.get("kind") or "work_order"),
                "ref_id": event.get("id") or event.get("wo_id"),
                "date": _iso(event.get("date") or event.get("opened_on")),
                "label": event.get("summary") or event.get("id"),
                "detail": event.get("as_found") or event.get("root_cause"),
                "asset_tag": event.get("asset_tag"),
                "is_sibling": True,
                "data_class": event.get("data_class"),
            }
        )
    # Undated events sort last rather than being dropped: an event with no date
    # is still an event, and hiding it would understate the history.
    return sorted(events, key=lambda e: (e["date"] is None, e["date"] or ""))


def _wo_summary(wo: dict[str, Any]) -> dict[str, Any]:
    return {
        "wo_id": wo["wo_id"],
        "wo_type": wo.get("wo_type"),
        "description": wo.get("description"),
        "as_found": wo.get("as_found"),
        "opened_on": _iso(wo.get("opened_on")),
        "closed_on": _iso(wo.get("closed_on")),
        "downtime_hours": float(wo["downtime_hours"]) if wo.get("downtime_hours") else None,
        "coded_failure_mode": wo.get("coded_failure_mode"),
        "extracted_failure_mode": wo.get("extracted_failure_mode"),
        "data_class": wo.get("data_class"),
    }


def _incident_summary(inc: dict[str, Any]) -> dict[str, Any]:
    return {
        "incident_id": inc["incident_id"],
        "title": inc.get("title"),
        "occurred_on": _iso(inc.get("occurred_on")),
        "severity": inc.get("severity"),
        "immediate_cause": inc.get("immediate_cause"),
        "root_cause": inc.get("root_cause"),
        "investigation_status": inc.get("investigation_status"),
        "data_class": inc.get("data_class"),
    }


def _overall_confidence(analysis: Any, evidence: dict[str, int]) -> float | None:
    """How much of the recorded evidence points at the leading candidate.

    Deliberately not the raw ranking score, which mixes weights and decay into a
    number with no natural scale. This answers a question the data can actually
    answer -- what share of cause statements name this mechanism -- and discounts
    it when the support is thin in ways that matter: only siblings, or only one
    document. An abstained analysis returns None rather than a low number,
    because there is no candidate to be confident about.
    """
    if analysis.abstained or not analysis.candidates:
        return None
    leading = analysis.candidates[0]
    share = leading.occurrences / max(analysis.evidence_count, 1)
    if leading.on_this_asset == 0:
        share *= 0.7
    if len({e.ref_id for e in leading.evidence}) < 2:
        share *= 0.6
    return round(min(share, 0.95), 4)


# ---------------------------------------------------------------------------
# Evidence gathering
# ---------------------------------------------------------------------------


async def _siblings(canonical_tag: str) -> list[dict[str, Any]]:
    rows = await graph.read(
        "MATCH (e:Equipment {canonical_tag: $tag})-[:SIBLING_OF]->(s:Equipment) "
        "RETURN s.canonical_tag AS canonical_tag, s.class_label AS class_label",
        tag=canonical_tag,
    )
    return [dict(r) for r in rows]


async def _sibling_history(sibling_tags: list[str]) -> list[dict[str, Any]]:
    """Sibling history is the evidence a flat document system cannot produce.

    "Check the sister pump" is the first thing an experienced engineer does, and
    it is only possible because siblings were linked rather than merged.
    """
    if not sibling_tags:
        return []

    # Both event types, unioned. Work orders alone miss the thing that matters
    # most: the sibling's *investigated* failure, which is where a root cause and
    # an open corrective action actually live. In this corpus the duty pump's
    # 2022 investigation concluded dry running and raised two CAPAs that are
    # still open -- evidence about the standby pump's identical failure that a
    # work-order-only query cannot see.
    rows = await db.fetch_all(
        """
        SELECT w.wo_id                     AS id,
               'work_order'                AS kind,
               w.description               AS summary,
               w.as_found                  AS as_found,
               NULL::text                  AS root_cause,
               w.coded_failure_mode        AS coded_failure_mode,
               w.opened_on                 AS date,
               w.downtime_hours            AS downtime_hours,
               NULL::text                  AS status,
               w.source_system, w.data_class::text AS data_class,
               a.canonical_tag             AS asset_tag
          FROM work_orders w
          JOIN assets a ON a.asset_id = w.asset_id
         WHERE a.canonical_tag = ANY(%(tags)s)
        UNION ALL
        SELECT i.incident_id               AS id,
               'incident'                  AS kind,
               i.title                     AS summary,
               i.immediate_cause           AS as_found,
               i.root_cause                AS root_cause,
               NULL::text                  AS coded_failure_mode,
               i.occurred_on               AS date,
               NULL::numeric               AS downtime_hours,
               i.investigation_status      AS status,
               i.source_system, i.data_class::text AS data_class,
               a.canonical_tag             AS asset_tag
          FROM incidents i
          JOIN assets a ON a.asset_id = i.asset_id
         WHERE a.canonical_tag = ANY(%(tags)s)
         ORDER BY date DESC NULLS LAST
         LIMIT 100
        """,
        {"tags": sibling_tags},
    )
    return [
        {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in dict(r).items()}
        for r in rows
    ]


async def _open_capas(canonical_tag: str) -> list[dict[str, Any]]:
    """An open CAPA that already addresses this failure is the single most
    valuable thing to surface: the plant already decided how to fix it."""
    # Siblings are included, and flagged. An open action raised against the duty
    # pump after its seal ran dry is the most useful thing this endpoint can put
    # in front of an engineer investigating the standby pump doing the same
    # thing -- and restricting the query to one tag hides it. The corpus this was
    # built against contains exactly that: CAPA-88 and CAPA-89, raised in 2022
    # against P-101A, still open, and directly about the mechanism that later
    # took out P-101B.
    rows = await graph.read(
        """
        MATCH (e:Equipment {canonical_tag: $tag})
        OPTIONAL MATCH (e)-[:SIBLING_OF]-(sib:Equipment)
        WITH collect(DISTINCT e) + collect(DISTINCT sib) AS pumps, e
        UNWIND pumps AS pump
        MATCH (pump)<-[:INVOLVED]-(i:Incident)-[:GENERATED]->(c:CAPA)
        WHERE coalesce(c.status, 'OPEN') <> 'CLOSED'
        RETURN DISTINCT
               c.capa_id AS capa_id, c.action AS action, c.owner AS owner,
               c.due_date AS due_date, c.status AS status,
               i.incident_id AS from_incident,
               pump.canonical_tag AS raised_against,
               pump.canonical_tag <> e.canonical_tag AS is_sibling
        ORDER BY is_sibling, due_date
        """,
        tag=canonical_tag,
    )
    return [
        {k: (v.iso_format() if hasattr(v, "iso_format") else v) for k, v in dict(r).items()}
        for r in rows
    ]


async def _recent_mocs(canonical_tag: str) -> list[dict[str, Any]]:
    """Management of Change records are the most frequently missed causal factor:
    the plant no longer matches its own drawings and nothing looks wrong."""
    rows = await graph.read(
        "MATCH (m:MOC)-[c:CHANGED]->(e:Equipment {canonical_tag: $tag}) "
        "RETURN m.moc_id AS moc_id, m.change_desc AS change_desc, "
        "m.approved_on AS approved_on, c.date AS changed_on ORDER BY c.date DESC",
        tag=canonical_tag,
    )
    return [
        {k: (v.iso_format() if hasattr(v, "iso_format") else v) for k, v in dict(r).items()}
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Deterministic analysis
# ---------------------------------------------------------------------------


def _reliability_metrics(work_orders: list[dict[str, Any]]) -> dict[str, Any]:
    """MTBF, MTTR-proxy and downtime, computed from stored work orders only.

    Returns ``insufficient_data`` rather than a number when there are too few
    events. Three failures is the minimum from which a between-failure interval
    means anything; below that a computed MTBF is decoration.
    """
    corrective = [
        w for w in work_orders if (w.get("wo_type") or "").upper().startswith("CORR")
    ] or work_orders
    dates = sorted(w["opened_on"] for w in corrective if w.get("opened_on"))
    downtimes = [float(w["downtime_hours"]) for w in corrective if w.get("downtime_hours")]

    metrics: dict[str, Any] = {
        "corrective_events": len(corrective),
        "total_downtime_hours": round(sum(downtimes), 1) if downtimes else 0.0,
        "mean_downtime_hours": round(sum(downtimes) / len(downtimes), 2) if downtimes else None,
        "metrics_data_class": DataClass.CALCULATED_METRIC.value,
    }

    if len(dates) < 3:
        metrics["mtbf_days"] = None
        metrics["mtbf_status"] = f"insufficient_data ({len(dates)} dated events; 3 required)"
        return metrics

    intervals = [(b - a).days for a, b in zip(dates, dates[1:], strict=False) if (b - a).days > 0]
    metrics["mtbf_days"] = round(sum(intervals) / len(intervals), 1) if intervals else None
    metrics["mtbf_status"] = (
        "computed" if intervals else "insufficient_data (no positive intervals)"
    )
    metrics["first_event"] = dates[0].isoformat()
    metrics["last_event"] = dates[-1].isoformat()
    return metrics


def _rank_similar_events(
    *,
    target_description: str,
    target_class: str | None,
    own_history: list[dict[str, Any]],
    own_incidents: list[dict[str, Any]],
    sibling_history: list[dict[str, Any]],
    sibling_tags: set[str],
) -> list[dict[str, Any]]:
    """Rank historical events by structural similarity, with an explanation.

    Deliberately *not* text-embedding similarity: two reports of the same failure
    written by different authors look dissimilar, and two unrelated failures
    described in the same house style look similar. Structural signals -- same
    failure vocabulary, same equipment class, sibling relationship -- are what
    actually predicts recurrence, and they need no model to compute.
    """
    target_terms = _significant_terms(target_description)
    scored: list[dict[str, Any]] = []

    def consider(record: dict[str, Any], origin: str, tag: str | None) -> None:
        text = " ".join(
            str(record.get(k) or "")
            for k in (
                "description",
                "long_text",
                "as_found",
                "title",
                "root_cause",
                "immediate_cause",
                "coded_failure_mode",
            )
        )
        terms = _significant_terms(text)
        overlap = target_terms & terms
        if not overlap:
            return
        score = len(overlap) / max(len(target_terms), 1)
        reasons = [f"shared failure vocabulary: {', '.join(sorted(overlap)[:5])}"]
        if origin == "sibling":
            score += 0.25
            reasons.append(
                f"occurred on sibling asset {tag} (same class and sequence, other train)"
            )
        if target_class and origin != "sibling":
            reasons.append(f"same equipment class ({target_class})")
        scored.append(
            {
                "id": record.get("wo_id") or record.get("incident_id"),
                "kind": "work_order" if record.get("wo_id") else "incident",
                "asset_tag": tag,
                "date": _iso(record.get("opened_on") or record.get("occurred_on")),
                "summary": (record.get("description") or record.get("title") or "")[:300],
                "as_found": record.get("as_found"),
                "root_cause": record.get("root_cause"),
                "score": round(min(score, 1.0), 3),
                "explanation": "; ".join(reasons),
                "source_system": record.get("source_system"),
                "data_class": record.get("data_class"),
            }
        )

    for record in own_history:
        consider(record, "self", None)
    for record in own_incidents:
        consider(record, "self", None)
    for record in sibling_history:
        consider(record, "sibling", record.get("canonical_tag"))

    scored.sort(key=lambda r: -r["score"])
    return scored[:10]


_STOPWORDS = frozenset(
    [
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "at",
        "for",
        "from",
        "with",
        "by",
        "as",
        "is",
        "was",
        "were",
        "be",
        "been",
        "this",
        "that",
        "it",
        "its",
        "after",
        "before",
        "during",
        "due",
        "found",
        "noted",
        "observed",
        "replaced",
    ]
)


def _significant_terms(text: str) -> set[str]:
    return {
        word
        for word in "".join(c.lower() if c.isalnum() else " " for c in (text or "")).split()
        if len(word) > 3 and word not in _STOPWORDS
    }


def _missing_evidence(
    work_orders: list[dict[str, Any]],
    incidents: list[dict[str, Any]],
    inspections: list[dict[str, Any]],
    mocs: list[dict[str, Any]],
) -> list[str]:
    """What evidence would discriminate between competing explanations.

    Naming the gap is itself a recommendation: if the discriminating evidence
    does not exist, obtaining it is the action.
    """
    missing: list[str] = []
    if not any(w.get("as_found") for w in work_orders):
        missing.append(
            "As-found condition is not recorded on any work order. Only the as-found "
            "condition is evidence about the failure mechanism; as-left is not."
        )
    if not inspections:
        missing.append(
            "No inspection or condition-monitoring readings are linked to this asset, so "
            "degradation cannot be trended."
        )
    if not mocs:
        missing.append(
            "No Management of Change record is linked to this asset. If the machine has been "
            "modified, the change is not visible to the system and cannot be ruled in or out."
        )
    if not incidents:
        missing.append(
            "No incident investigation is linked to this asset, so no stated causal chain "
            "exists to corroborate or contradict."
        )
    return missing


def _iso(value: Any) -> str | None:
    if isinstance(value, date):
        return value.isoformat()
    return str(value) if value else None
