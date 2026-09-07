"""Root cause analysis from recorded evidence.

The temptation this module exists to resist: a language model will produce a
fluent, plausible, well-structured RCA for any pump you name, drawing on how
centrifugal pumps fail *in general*. It reads like expertise. It is not evidence
about **this** pump, and an engineer who acts on it is acting on a plausible
story rather than on their own plant's history.

So candidate causes here are not generated. They are **aggregated from cause
statements the plant already wrote down** — the root cause and immediate cause
fields of investigated incidents, and the as-found condition recorded on work
orders — grouped by failure mechanism and ranked by how much independent
evidence supports each. Every candidate carries the documents that assert it. If
you disagree with a ranking you can open the reports and see why it ranked.

Ranking signals, all measurable
-------------------------------
* **independent occurrences** — how many distinct documents record this
  mechanism. Two investigations reaching the same conclusion separately is the
  strongest evidence a plant produces.
* **subject weight** — evidence from this asset counts more than evidence from a
  sibling, but sibling evidence counts. Identical pumps in identical service fail
  identically, and the corpus this was built against contains exactly that case:
  two dry-running seal failures three years apart on a duty/standby pair,
  investigated independently because reports are filed by date rather than cause.
* **recency** — a mechanism seen last year is more likely live than one seen a
  decade ago and since designed out.
* **corroboration across evidence types** — an incident investigation *and* a
  work order as-found note agreeing is worth more than two of either.

When it abstains
----------------
Below :data:`MIN_EVIDENCE_FOR_RANKING` distinct pieces of evidence, no ranking is
returned at all. One recorded failure tells you what happened once; it does not
establish a cause, and presenting a single occurrence as "the leading candidate,
confidence 0.9" would be the same fabrication as the LLM version wearing a
different hat.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from services.common.logging import get_logger
from services.ingest.extract import extract_failure_terms

log = get_logger(__name__)

#: Fewer distinct pieces of evidence than this and no causes are ranked. Two is
#: the minimum at which "this happens repeatedly" can be said at all; below it
#: the honest output is the event list itself.
MIN_EVIDENCE_FOR_RANKING = 2

#: Evidence from the asset itself versus from a sibling. Sibling evidence is
#: real -- identical machines in identical service fail identically -- but it is
#: about a different machine, and the ranking should say so.
_WEIGHT_SELF = 1.0
_WEIGHT_SIBLING = 0.6

#: An investigated incident carries more weight than a work-order note: someone
#: was assigned to determine the cause and wrote it under a heading, rather than
#: recording what they saw on strip-down.
_WEIGHT_BY_KIND = {"incident": 1.0, "work_order": 0.7, "inspection": 0.5}

#: Half-life for recency, in years. A mechanism last seen this long ago counts
#: half as much as one seen today. Five years is roughly a turnaround cycle --
#: long enough that a design change or procedure revision has plausibly
#: intervened.
_RECENCY_HALF_LIFE_YEARS = 5.0

#: Mechanism vocabulary beyond the ISO 14224 codes. These are the *conditions*
#: that produce a failure mode rather than the mode itself, and they are what an
#: engineer actually wants ranked: "the seal failed" is the mode, "it ran dry
#: because the suction was throttled" is the cause.
_CONDITION_PATTERNS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "dry_running",
        "Dry running / loss of seal flush",
        ("dry run", "without liquid film", "no liquid film", "loss of flush", "ran dry",
         "lost flush", "without flush", "starved"),
    ),
    (
        "throttled_suction",
        "Startup against a throttled or closed suction",
        ("suction valve throttled", "throttled suction", "closed suction",
         "suction valve closed", "suction throttled", "against a closed"),
    ),
    (
        "misalignment",
        "Shaft misalignment",
        ("misalign", "out of alignment", "alignment out", "coupling alignment"),
    ),
    (
        "cavitation",
        "Cavitation / insufficient NPSH",
        ("cavitat", "npsh", "insufficient suction head", "vapour lock"),
    ),
    (
        "vibration",
        "Excessive vibration",
        ("high vibration", "excessive vibration", "vibration alarm", "vibration exceeded"),
    ),
    (
        "bearing_distress",
        "Bearing distress",
        ("bearing fail", "bearing damage", "bearing wear", "spalling", "bearing temperature"),
    ),
    (
        "corrosion_erosion",
        "Corrosion or erosion",
        ("corrosion", "erosion", "wall loss", "thinning", "pitting"),
    ),
    (
        "procedure_gap",
        "Procedure does not require the check that would have prevented it",
        ("does not require", "no interlock", "not required by", "procedure did not",
         "no requirement", "still does not"),
    ),
    (
        "operator_action",
        "Operating outside the procedure",
        ("operator error", "not followed", "deviation from procedure", "failure to follow"),
    ),
)


@dataclass(slots=True)
class CauseEvidence:
    """One recorded statement supporting a candidate cause."""

    kind: str  # incident | work_order | inspection
    ref_id: str
    asset_tag: str | None
    occurred_on: date | None
    quote: str
    field: str
    is_sibling: bool = False
    doc_id: str | None = None
    chunk_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "ref_id": self.ref_id,
            "asset_tag": self.asset_tag,
            "occurred_on": self.occurred_on.isoformat() if self.occurred_on else None,
            "quote": self.quote,
            "field": self.field,
            "is_sibling": self.is_sibling,
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
        }


@dataclass(slots=True)
class CandidateCause:
    key: str
    label: str
    evidence: list[CauseEvidence] = field(default_factory=list)
    score: float = 0.0
    #: Plain-language statement of *why* this ranked where it did. An unexplained
    #: score is dismissed by the engineer reading it; an explained one is checked.
    rationale: str = ""

    @property
    def occurrences(self) -> int:
        return len(self.evidence)

    @property
    def on_this_asset(self) -> int:
        return sum(1 for e in self.evidence if not e.is_sibling)

    @property
    def latest(self) -> date | None:
        dates = [e.occurred_on for e in self.evidence if e.occurred_on]
        return max(dates) if dates else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "score": round(self.score, 4),
            "occurrences": self.occurrences,
            "on_this_asset": self.on_this_asset,
            "on_siblings": self.occurrences - self.on_this_asset,
            "latest_occurrence": self.latest.isoformat() if self.latest else None,
            "rationale": self.rationale,
            "evidence": [e.to_dict() for e in self.evidence],
        }


@dataclass(slots=True)
class CausalAnalysis:
    candidates: list[CandidateCause] = field(default_factory=list)
    evidence_count: int = 0
    abstained: bool = False
    abstain_reason: str | None = None
    method: str = "evidence_aggregation"

    @property
    def leading(self) -> CandidateCause | None:
        return self.candidates[0] if self.candidates else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "abstained": self.abstained,
            "abstain_reason": self.abstain_reason,
            "evidence_count": self.evidence_count,
            "candidates": [c.to_dict() for c in self.candidates],
        }


def analyse(
    *,
    asset_tag: str,
    incidents: list[dict[str, Any]],
    work_orders: list[dict[str, Any]],
    sibling_events: list[dict[str, Any]],
    inspections: list[dict[str, Any]] | None = None,
    today: date | None = None,
) -> CausalAnalysis:
    """Rank candidate causes from recorded evidence, or decline to.

    ``incidents`` and ``work_orders`` are this asset's; ``sibling_events`` are the
    same shapes for identical equipment. Every input is a stored record — nothing
    here reads a document at analysis time, so the ranking is reproducible from
    the database alone.
    """
    today = today or date.today()
    statements = _collect_statements(asset_tag, incidents, work_orders, sibling_events, inspections)

    if len(statements) < MIN_EVIDENCE_FOR_RANKING:
        return CausalAnalysis(
            evidence_count=len(statements),
            abstained=True,
            abstain_reason=(
                f"Only {len(statements)} recorded cause statement(s) were found for {asset_tag} "
                f"and its siblings. At least {MIN_EVIDENCE_FOR_RANKING} are needed before a "
                "mechanism can be said to recur. The events themselves are listed; the causal "
                "ranking is withheld rather than computed from a single data point."
            ),
        )

    grouped: dict[str, CandidateCause] = {}
    for statement in statements:
        for key, label in _mechanisms_in(statement.quote):
            candidate = grouped.get(key)
            if candidate is None:
                candidate = CandidateCause(key=key, label=label)
                grouped[key] = candidate
            candidate.evidence.append(statement)

    if not grouped:
        return CausalAnalysis(
            evidence_count=len(statements),
            abstained=True,
            abstain_reason=(
                f"{len(statements)} cause statement(s) were found, but none names a failure "
                "mechanism this system recognises. The statements are returned as evidence; "
                "no mechanism is inferred from wording it cannot interpret."
            ),
        )

    for candidate in grouped.values():
        candidate.score = _score(candidate, today)
        candidate.rationale = _rationale(candidate)

    ranked = sorted(grouped.values(), key=lambda c: (-c.score, -c.occurrences, c.key))
    log.info(
        "rca.causes_ranked",
        asset_tag=asset_tag,
        evidence=len(statements),
        candidates=len(ranked),
        leading=ranked[0].key if ranked else None,
    )
    return CausalAnalysis(candidates=ranked, evidence_count=len(statements))


# ---------------------------------------------------------------------------
# Evidence collection
# ---------------------------------------------------------------------------


def _collect_statements(
    asset_tag: str,
    incidents: list[dict[str, Any]],
    work_orders: list[dict[str, Any]],
    sibling_events: list[dict[str, Any]],
    inspections: list[dict[str, Any]] | None,
) -> list[CauseEvidence]:
    """Pull every recorded cause statement into one comparable shape."""
    out: list[CauseEvidence] = []

    for incident in incidents:
        for field_name in ("root_cause", "immediate_cause"):
            text = (incident.get(field_name) or "").strip()
            if not text:
                continue
            out.append(
                CauseEvidence(
                    kind="incident",
                    ref_id=str(incident.get("incident_id") or incident.get("id") or "incident"),
                    asset_tag=incident.get("raw_asset_tag") or asset_tag,
                    occurred_on=_as_date(incident.get("occurred_on")),
                    quote=text,
                    field=field_name,
                    doc_id=incident.get("doc_id"),
                )
            )

    for order in work_orders:
        # as_found is the observation made on strip-down. as_left is what was
        # done about it, which is a remedy rather than a cause, so it is not
        # treated as evidence of mechanism.
        text = (order.get("as_found") or "").strip()
        if not text:
            continue
        out.append(
            CauseEvidence(
                kind="work_order",
                ref_id=str(order.get("wo_id") or "work order"),
                asset_tag=order.get("raw_asset_tag") or asset_tag,
                occurred_on=_as_date(order.get("opened_on")),
                quote=text,
                field="as_found",
                doc_id=order.get("doc_id"),
            )
        )

    for event in sibling_events or []:
        text = (event.get("as_found") or event.get("root_cause") or event.get("summary") or "").strip()
        if not text:
            continue
        out.append(
            CauseEvidence(
                kind=str(event.get("kind") or "work_order"),
                ref_id=str(event.get("id") or event.get("wo_id") or "sibling event"),
                asset_tag=event.get("asset_tag"),
                occurred_on=_as_date(event.get("date") or event.get("opened_on")),
                quote=text,
                field=event.get("field") or "as_found",
                is_sibling=True,
                doc_id=event.get("doc_id"),
            )
        )

    for inspection in inspections or []:
        text = (inspection.get("finding") or inspection.get("remarks") or "").strip()
        if not text:
            continue
        out.append(
            CauseEvidence(
                kind="inspection",
                ref_id=str(inspection.get("inspection_id") or "inspection"),
                asset_tag=inspection.get("raw_asset_tag") or asset_tag,
                occurred_on=_as_date(inspection.get("inspected_on")),
                quote=text,
                field="finding",
            )
        )
    return out


def _mechanisms_in(text: str) -> list[tuple[str, str]]:
    """Which mechanisms a statement names.

    Two vocabularies, both already in the codebase. The ISO 14224 extractor
    recognises failure *modes* ("mechanical seal failure" -> ELP); the condition
    patterns above recognise the *circumstances* that produce them. A statement
    can name both, and usually the circumstance is the one worth ranking.

    A statement naming no recognised mechanism contributes nothing rather than
    being filed under "other" -- an "other" bucket accumulates unrelated
    statements and then ranks first on volume alone.
    """
    lowered = text.lower()
    found: list[tuple[str, str]] = []

    for key, label, phrases in _CONDITION_PATTERNS:
        if any(phrase in lowered for phrase in phrases):
            found.append((key, label))

    for term in extract_failure_terms(text):
        code = term.get("failure_mode_code")
        if not code:
            continue
        key = f"mode:{code}"
        if all(k != key for k, _ in found):
            # The extractor returns the matched wording under "phrase"; asking
            # for "surface_form" produced an empty parenthesis on every label,
            # which read as a missing value rather than a naming mistake.
            phrase = (term.get("phrase") or term.get("quote") or "").strip()
            label = f"Failure mode {code}" + (f" — {phrase}" if phrase else "")
            found.append((key, label))
    return found


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def _score(candidate: CandidateCause, today: date) -> float:
    """Combine the four signals into one comparable number.

    Multiplicative in nothing: each contribution is additive so that a single
    strong signal cannot dominate, and the components stay separable in the
    rationale. Deliberately not normalised to look like a probability — it is an
    ordering, and calling it 0.87 confident would overstate what four heuristics
    can support.
    """
    total = 0.0
    for evidence in candidate.evidence:
        weight = _WEIGHT_BY_KIND.get(evidence.kind, 0.5)
        weight *= _WEIGHT_SIBLING if evidence.is_sibling else _WEIGHT_SELF
        total += weight * _recency_factor(evidence.occurred_on, today)

    # Independent corroboration across *documents*, not statements: an incident
    # whose root cause and immediate cause both mention dry running is one
    # investigation agreeing with itself.
    distinct_refs = len({e.ref_id for e in candidate.evidence})
    if distinct_refs > 1:
        total *= 1.0 + 0.25 * (distinct_refs - 1)

    # Agreement across evidence *types* is worth more than repetition within one.
    if len({e.kind for e in candidate.evidence}) > 1:
        total *= 1.15
    return total


def _recency_factor(occurred: date | None, today: date) -> float:
    """Exponential decay with a five-year half-life; undated evidence counts less."""
    if occurred is None:
        return 0.7
    years = max(0.0, (today - occurred).days / 365.25)
    return float(0.5 ** (years / _RECENCY_HALF_LIFE_YEARS))


def _rationale(candidate: CandidateCause) -> str:
    refs = sorted({e.ref_id for e in candidate.evidence})
    parts = [
        f"Recorded in {len(refs)} document(s): {', '.join(refs[:6])}"
        + ("…" if len(refs) > 6 else "")
    ]
    if candidate.on_this_asset and candidate.occurrences > candidate.on_this_asset:
        parts.append(
            f"{candidate.on_this_asset} on this asset and "
            f"{candidate.occurrences - candidate.on_this_asset} on identical equipment"
        )
    elif not candidate.on_this_asset:
        parts.append("recorded only on identical equipment, not on this asset")
    kinds = sorted({e.kind for e in candidate.evidence})
    if len(kinds) > 1:
        parts.append(f"corroborated across {' and '.join(kinds).replace('_', ' ')} records")
    if candidate.latest:
        parts.append(f"most recently {candidate.latest.isoformat()}")
    return "; ".join(parts) + "."


def _as_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    text = str(value)[:10]
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


_WHITESPACE = re.compile(r"\s+")


def summarise_symptom(description: str) -> str:
    """Normalise the reported symptom for display. Not interpreted, just tidied."""
    return _WHITESPACE.sub(" ", description or "").strip()
