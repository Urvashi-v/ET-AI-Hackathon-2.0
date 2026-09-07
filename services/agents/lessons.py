"""Lessons learned: has this happened before, and what did we decide last time?

The failure this addresses is organisational rather than technical. A plant files
incident reports by date. Two identical failures three years apart are
investigated by different people who never see each other's report, each raises
corrective actions, and the actions from the first one are still open when the
second happens. The knowledge existed; nothing connected it.

Matching, and why it is not one number
--------------------------------------
Similarity is computed from four independent signals, each returned with the
evidence that produced it, because an unexplained "87% similar" is dismissed by
the engineer reading it and an explained one is checked:

``semantic``
    Cosine similarity between the cause statements, using the same real
    embedding model the copilot retrieves with. Catches "ran without liquid
    film" against "lost seal flush" — different words, same event.

``mechanism``
    Overlap of failure mechanisms recognised in the two cause statements, using
    the same vocabulary the RCA agent ranks with. Two incidents that both name
    dry running share a mechanism whatever else differs.

``structural``
    Graph proximity: the same equipment, a sibling, the same functional location,
    the same equipment class. An identical pump in identical service failing the
    same way is the strongest structural signal a plant produces.

``documentary``
    One report explicitly citing the other, or both citing the same procedure.
    Rare and decisive when present — it is a human having already made the link.

No match is returned below :data:`MIN_SIMILARITY`. An empty result is a real
answer: this failure has no precedent on file, which is worth knowing precisely
because it means the corrective actions have to be thought through rather than
looked up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from services.common import db
from services.common.logging import get_logger
from services.common.schemas import CapabilityState
from services.agents.rca import _mechanisms_in
from services.ingest.embeddings import _to_pgvector, embedding_capability, get_embedding_provider

log = get_logger(__name__)

#: Below this combined score a match is not reported. Set so that two incidents
#: sharing only a generic word ("pump", "failure") do not surface as precedent --
#: a false precedent is worse than none, because it sends an engineer to read a
#: report that will not help and teaches them to ignore the panel.
MIN_SIMILARITY = 0.35

#: Signal weights. Semantic similarity leads because it is the only signal that
#: works when two reports describe one event in completely different words, which
#: is the normal case across authors and decades. Structural is weighted below it
#: deliberately: two failures on the same pump are *related*, but relatedness is
#: not similarity, and over-weighting it turns the panel into an asset history.
WEIGHTS = {
    "semantic": 0.40,
    "mechanism": 0.30,
    "structural": 0.20,
    "documentary": 0.10,
}


@dataclass(slots=True)
class SimilarIncident:
    incident_id: str
    title: str
    occurred_on: str | None
    asset_tag: str | None
    severity: str | None
    status: str | None
    root_cause: str | None
    immediate_cause: str | None
    similarity: float
    signals: dict[str, float] = field(default_factory=dict)
    #: Why it matched, in words. One line per contributing signal.
    match_reasons: list[str] = field(default_factory=list)
    shared_mechanisms: list[str] = field(default_factory=list)
    corrective_actions: list[dict[str, Any]] = field(default_factory=list)
    doc_id: str | None = None
    data_class: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "title": self.title,
            "occurred_on": self.occurred_on,
            "asset_tag": self.asset_tag,
            "severity": self.severity,
            "status": self.status,
            "root_cause": self.root_cause,
            "immediate_cause": self.immediate_cause,
            "similarity": round(self.similarity, 4),
            "signals": {k: round(v, 4) for k, v in self.signals.items()},
            "match_reasons": self.match_reasons,
            "shared_mechanisms": self.shared_mechanisms,
            "corrective_actions": self.corrective_actions,
            "doc_id": self.doc_id,
            "data_class": self.data_class,
        }


@dataclass(slots=True)
class LessonsResult:
    matches: list[SimilarIncident] = field(default_factory=list)
    considered: int = 0
    semantic_state: CapabilityState = CapabilityState.AVAILABLE
    semantic_detail: str | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "matches": [m.to_dict() for m in self.matches],
            "incidents_considered": self.considered,
            "semantic_matching": {
                "state": self.semantic_state.value,
                "detail": self.semantic_detail,
            },
            "detail": self.detail,
        }


async def find_precedents(
    *,
    description: str,
    asset_tag: str | None = None,
    exclude_incident_id: str | None = None,
    limit: int = 5,
) -> LessonsResult:
    """Find historical incidents resembling a described event.

    ``description`` is whatever text describes the new event — a work-order
    as-found note, an incident narrative, an operator's words. It is compared
    against every incident on file; the corpus is small enough that an exhaustive
    comparison is cheaper and more predictable than an approximate index, and the
    ranking is then fully explainable.
    """
    history = await _load_incidents(exclude_incident_id)
    if not history:
        return LessonsResult(
            considered=0,
            detail=(
                "No incident records exist yet, so there is no precedent to compare against. "
                "Incidents are created from ingested investigation reports."
            ),
        )

    context = await _asset_context(asset_tag)
    query_mechanisms = {key for key, _ in _mechanisms_in(description)}
    semantic_scores, semantic_state, semantic_detail = await _semantic_scores(description, history)

    matches: list[SimilarIncident] = []
    for incident in history:
        signals: dict[str, float] = {}
        reasons: list[str] = []

        semantic = semantic_scores.get(incident["incident_id"], 0.0)
        if semantic > 0:
            signals["semantic"] = semantic
            if semantic >= 0.5:
                reasons.append(
                    f"cause statements are semantically close ({semantic:.2f} cosine, "
                    "same embedding model the copilot retrieves with)"
                )

        cause_text = " ".join(
            filter(None, [incident.get("root_cause"), incident.get("immediate_cause")])
        )
        incident_mechanisms = {key for key, _ in _mechanisms_in(cause_text)}
        shared = sorted(query_mechanisms & incident_mechanisms)
        if query_mechanisms and shared:
            signals["mechanism"] = len(shared) / len(query_mechanisms | incident_mechanisms)
            reasons.append(f"shares failure mechanism: {', '.join(shared)}")

        structural, structural_reason = _structural(incident, context)
        if structural:
            signals["structural"] = structural
            reasons.append(structural_reason)

        documentary, documentary_reason = _documentary(incident, context)
        if documentary:
            signals["documentary"] = documentary
            reasons.append(documentary_reason)

        score = sum(WEIGHTS[name] * value for name, value in signals.items())
        if score < MIN_SIMILARITY:
            continue

        matches.append(
            SimilarIncident(
                incident_id=incident["incident_id"],
                title=incident["title"],
                occurred_on=_iso(incident.get("occurred_on")),
                asset_tag=incident.get("asset_tag") or incident.get("raw_asset_tag"),
                severity=incident.get("severity"),
                status=incident.get("investigation_status"),
                root_cause=incident.get("root_cause"),
                immediate_cause=incident.get("immediate_cause"),
                similarity=score,
                signals=signals,
                match_reasons=reasons,
                shared_mechanisms=shared,
                corrective_actions=_actions_of(incident),
                doc_id=incident.get("doc_id"),
                data_class=incident.get("data_class"),
            )
        )

    matches.sort(key=lambda m: -m.similarity)
    log.info(
        "lessons.matched",
        considered=len(history),
        matched=len(matches),
        semantic=semantic_state.value,
    )
    return LessonsResult(
        matches=matches[:limit],
        considered=len(history),
        semantic_state=semantic_state,
        semantic_detail=semantic_detail,
        detail=(
            None
            if matches
            else (
                f"{len(history)} incident(s) were compared and none scored above the "
                f"{MIN_SIMILARITY} similarity floor. This failure has no precedent on file — "
                "which is itself worth knowing, because the corrective actions have to be "
                "worked out rather than looked up."
            )
        ),
    )


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------


async def _semantic_scores(
    description: str, history: list[dict[str, Any]]
) -> tuple[dict[str, float], CapabilityState, str | None]:
    """Cosine similarity between the new description and each incident's causes.

    Computed in Postgres against the *already stored* chunk vectors for each
    incident's evidence chunks — the same vectors dense retrieval uses. No second
    embedding of the corpus, and no separate index that could drift out of step
    with the first one.

    With no embedding provider this returns no scores and says so; the other
    three signals still work, so precedent matching degrades rather than
    disappearing.
    """
    state, detail, _ = embedding_capability()
    if state is not CapabilityState.AVAILABLE:
        return {}, state, detail

    try:
        vector = await get_embedding_provider().embed_query(description)
    except Exception as exc:
        log.error("lessons.embed_failed", error=str(exc))
        return {}, CapabilityState.ERROR, f"{type(exc).__name__}: {str(exc)[:160]}"
    if not vector:
        return {}, CapabilityState.NOT_CONFIGURED, "Embedding provider returned no vector."

    chunk_ids: list[str] = []
    owner: dict[str, str] = {}
    for incident in history:
        for chunk_id in incident.get("evidence_chunks") or []:
            chunk_ids.append(chunk_id)
            owner.setdefault(chunk_id, incident["incident_id"])
    if not chunk_ids:
        return {}, CapabilityState.AVAILABLE, "No incident evidence chunks are embedded yet."

    rows = await db.fetch_all(
        """
        SELECT e.chunk_id, 1 - (e.embedding <=> %(vec)s::vector) AS score
          FROM chunk_embeddings e
         WHERE e.chunk_id = ANY(%(ids)s)
        """,
        {"vec": _to_pgvector(vector), "ids": chunk_ids},
    )

    # An incident scores as its *best* matching chunk, not its mean. A report's
    # root-cause paragraph is the part that should match; averaging it with the
    # corrective-action table dilutes the signal that matters.
    best: dict[str, float] = {}
    for row in rows:
        incident_id = owner.get(row["chunk_id"])
        if not incident_id:
            continue
        score = float(row["score"])
        if score > best.get(incident_id, 0.0):
            best[incident_id] = score
    return best, CapabilityState.AVAILABLE, None


def _structural(incident: dict[str, Any], context: dict[str, Any]) -> tuple[float, str]:
    """Graph proximity between the incident's asset and the event's asset."""
    tag = (incident.get("asset_tag") or incident.get("raw_asset_tag") or "").upper()
    if not tag or not context.get("tag"):
        return 0.0, ""
    if tag == context["tag"]:
        return 1.0, "same equipment"
    if tag in context.get("siblings", set()):
        return 0.8, f"identical equipment in the same service ({tag} is a sibling)"
    if incident.get("functional_location") and incident["functional_location"] == context.get(
        "functional_location"
    ):
        return 0.6, f"same functional location ({context['functional_location']})"
    if context.get("class_code") and incident.get("class_code") == context["class_code"]:
        return 0.4, f"same equipment class ({context['class_code']})"
    return 0.0, ""


def _documentary(incident: dict[str, Any], context: dict[str, Any]) -> tuple[float, str]:
    """An explicit human-made link between the two reports."""
    referenced = set(incident.get("referenced_incidents") or [])
    if context.get("incident_id") and context["incident_id"] in referenced:
        return 1.0, f"{incident['incident_id']} explicitly references this incident"
    shared_procedures = set(incident.get("referenced_procedures") or []) & set(
        context.get("procedures") or []
    )
    if shared_procedures:
        return 0.6, f"both cite {', '.join(sorted(shared_procedures))}"
    return 0.0, ""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


async def _load_incidents(exclude: str | None) -> list[dict[str, Any]]:
    rows = await db.fetch_all(
        """
        SELECT i.incident_id, i.doc_id, i.title, i.occurred_on, i.severity,
               i.immediate_cause, i.root_cause, i.investigation_status,
               i.raw_asset_tag, i.metadata, i.data_class::text AS data_class,
               a.canonical_tag AS asset_tag, a.class_code
          FROM incidents i
          LEFT JOIN assets a ON a.asset_id = i.asset_id
         WHERE (%(exclude)s::text IS NULL OR i.incident_id <> %(exclude)s::text)
         ORDER BY i.occurred_on DESC NULLS LAST
        """,
        {"exclude": exclude},
    )
    for row in rows:
        meta = row.get("metadata") or {}
        row["evidence_chunks"] = meta.get("evidence_chunks") or []
        row["referenced_incidents"] = meta.get("referenced_incidents") or []
        row["referenced_procedures"] = meta.get("referenced_procedures") or []
        row["functional_location"] = meta.get("functional_location")
        row["corrective_actions"] = meta.get("corrective_actions") or []
    return rows


async def _asset_context(asset_tag: str | None) -> dict[str, Any]:
    """What the new event's asset is, and what it is related to."""
    if not asset_tag:
        return {}
    row = await db.fetch_one(
        "SELECT asset_id, canonical_tag, class_code, functional_location FROM assets "
        "WHERE upper(canonical_tag) = upper(%s)",
        (asset_tag,),
    )
    if not row:
        return {"tag": asset_tag.upper(), "siblings": set()}

    siblings = await db.fetch_all(
        """
        SELECT b.canonical_tag
          FROM assets a
          JOIN assets b ON b.class_code = a.class_code
                       AND b.functional_location IS NOT DISTINCT FROM a.functional_location
                       AND b.asset_id <> a.asset_id
         WHERE a.asset_id = %s
        """,
        (row["asset_id"],),
    )
    return {
        "tag": row["canonical_tag"].upper(),
        "class_code": row.get("class_code"),
        "functional_location": row.get("functional_location"),
        "siblings": {s["canonical_tag"].upper() for s in siblings},
        "procedures": [],
    }


def _actions_of(incident: dict[str, Any]) -> list[dict[str, Any]]:
    """The corrective actions, with open ones first.

    Ordering matters more than it looks: the point of showing precedent is
    usually "we already decided what to do and never did it", and an open action
    buried under three closed ones does not deliver that.
    """
    actions = list(incident.get("corrective_actions") or [])
    return sorted(actions, key=lambda a: (not a.get("is_open", False), a.get("action_id") or ""))


def _iso(value: Any) -> str | None:
    return value.isoformat() if hasattr(value, "isoformat") else (str(value) if value else None)
