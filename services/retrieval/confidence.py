"""Confidence scoring and the discipline of abstaining.

In a safety-critical domain a system that never says "I don't know" is a
liability: a wrong torque figure or a wrong isolation sequence can injure
someone. So abstention is a designed, measured feature rather than an accident.

The score combines **independent signals** rather than trusting a model's own
report of how sure it is, because self-reported confidence is exactly the signal
that fails when the model is confidently wrong:

=====================  ======================================================
retrieval_strength     how good is the best evidence, on the retriever's own
                       scale, normalised
source_agreement       do independent documents corroborate each other? a
                       single-source answer is fragile
claim_coverage         is every claim in the answer entailed by a cited
                       passage (verified by the generator's own citation
                       binding)
currency               is the evidence from a current revision, or a
                       superseded one
graph_support          did the knowledge graph corroborate the text
=====================  ======================================================

Bands: ANSWER, ANSWER_WITH_CAVEAT, ABSTAIN_AND_ROUTE. Abstention never returns a
bare refusal -- it returns what *was* found, what specifically is missing, and
who owns the asset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from services.common.config import get_settings
from services.common.schemas import ConfidenceMode, ConfidenceReport

#: Weights sum to 1.0. They are configuration, and the evaluation harness is what
#: justifies them: changing a weight changes the abstention curve, which the
#: golden set measures directly.
WEIGHTS: dict[str, float] = {
    "retrieval_strength": 0.30,
    "claim_coverage": 0.30,
    "source_agreement": 0.20,
    "currency": 0.10,
    "graph_support": 0.10,
}


@dataclass(slots=True)
class ConfidenceInputs:
    top_score: float = 0.0
    scores: list[float] = field(default_factory=list)
    distinct_documents: int = 0
    distinct_source_systems: int = 0
    current_documents: int = 0
    total_documents: int = 0
    graph_facts: int = 0
    total_claims: int = 0
    verified_claims: int = 0
    anchors_missing: list[str] = field(default_factory=list)
    generator_available: bool = True


def score(inputs: ConfidenceInputs) -> ConfidenceReport:
    settings = get_settings()

    retrieval_strength = _normalise_top_score(inputs.top_score)
    source_agreement = _source_agreement(inputs)
    claim_coverage = inputs.verified_claims / inputs.total_claims if inputs.total_claims else 0.0
    currency = inputs.current_documents / inputs.total_documents if inputs.total_documents else 0.0
    graph_support = 1.0 if inputs.graph_facts else 0.4

    signals = {
        "retrieval_strength": round(retrieval_strength, 4),
        "source_agreement": round(source_agreement, 4),
        "claim_coverage": round(claim_coverage, 4),
        "currency": round(currency, 4),
        "graph_support": round(graph_support, 4),
    }
    total = sum(WEIGHTS[name] * value for name, value in signals.items())

    # Naming an asset the corpus has never seen is decisive: no amount of
    # loosely-related evidence makes an answer about it safe.
    if inputs.anchors_missing:
        total = min(total, 0.35)

    if not inputs.generator_available:
        mode = ConfidenceMode.ABSTAIN_NO_GENERATOR
        explanation = (
            "Retrieval completed and the evidence below is real, but no generation provider "
            "is configured, so no prose answer was synthesised."
        )
    elif total >= settings.confidence_answer_threshold:
        mode = ConfidenceMode.ANSWER
        explanation = "Evidence is strong, current and corroborated across sources."
    elif total >= settings.confidence_caveat_threshold:
        mode = ConfidenceMode.ANSWER_WITH_CAVEAT
        explanation = _weakest_signal_explanation(signals)
    else:
        mode = ConfidenceMode.ABSTAIN_AND_ROUTE
        explanation = (
            f"Insufficient evidence to answer safely ({_weakest_signal_explanation(signals)})."
            if not inputs.anchors_missing
            else (
                "The asset named in the question is not present in the corpus: "
                + ", ".join(inputs.anchors_missing)
            )
        )

    return ConfidenceReport(
        score=round(total, 4), mode=mode, signals=signals, explanation=explanation
    )


def _normalise_top_score(top_score: float) -> float:
    """Map an unbounded BM25-family score onto [0, 1].

    A saturating transform, not a calibration: it is monotone in the raw score
    and the evaluation harness is what makes the resulting thresholds meaningful.
    A proper calibration needs a cross-encoder, which is the reranker slot.
    """
    if top_score <= 0:
        return 0.0
    return min(1.0, top_score / (top_score + 4.0) * 2.0)


def _source_agreement(inputs: ConfidenceInputs) -> float:
    if inputs.distinct_documents <= 1:
        return 0.25 if inputs.distinct_documents == 1 else 0.0
    base = min(1.0, 0.4 + 0.2 * (inputs.distinct_documents - 1))
    # Corroboration across *different source systems* is the strongest form:
    # it is the thing the whole platform claims to make possible.
    if inputs.distinct_source_systems > 1:
        base = min(1.0, base + 0.2)
    return base


def _weakest_signal_explanation(signals: dict[str, float]) -> str:
    weakest = min(signals, key=lambda k: signals[k])
    return {
        "retrieval_strength": "the best matching passage is only weakly relevant",
        "source_agreement": "only one source supports this; nothing corroborates it",
        "claim_coverage": "not every statement could be tied to a cited passage",
        "currency": "the supporting evidence is from a superseded revision",
        "graph_support": "the knowledge graph does not corroborate the retrieved text",
    }[weakest]


def build_referral(
    *, anchors_missing: list[str], intent: str, owner_role: str | None = None
) -> dict[str, Any]:
    """What to return instead of a guess.

    A bare refusal is not useful. This names what is missing, which document
    class should contain it, and which role owns it -- so the abstention still
    moves the technician forward.
    """
    if anchors_missing:
        reason = (
            f"No asset matching {', '.join(anchors_missing)} exists in the ingested corpus. "
            "It may be tagged differently in the source systems, or its documents may not "
            "have been ingested."
        )
        expected = "asset register, P&ID, or CMMS equipment export"
    else:
        reason = "The corpus does not contain evidence sufficient to answer this question."
        expected = {
            "procedural": "the governing SOP or operating instruction",
            "diagnostic": "work order history and incident reports for this asset",
            "aggregate": "a complete CMMS work-order export for the period",
            "lookup": "the equipment datasheet or OEM manual",
        }.get(intent, "the source document for this topic")

    return {
        "reason": reason,
        "expected_document": expected,
        "owner_role": owner_role or "reliability_engineer",
        "next_step": "Ingest the document above, or route the question to the owning role.",
    }
