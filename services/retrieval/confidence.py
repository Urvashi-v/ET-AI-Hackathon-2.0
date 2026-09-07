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
    # Was 0.30 when the only answerer was an LLM, where "does every claim carry a
    # resolvable citation" is a real question. Extraction answers it 1.0 by
    # construction, so at that weight it was 0.30 of free confidence on every
    # query -- which is exactly how three unanswerable questions started getting
    # answered. Kept, because it still discriminates for abstractive answers, but
    # weighted for what it can actually tell apart.
    "claim_coverage": 0.15,
    "answer_relevance": 0.25,
    "source_agreement": 0.10,
    "currency": 0.10,
    "graph_support": 0.10,
}

#: Below this, the answer covers so little of the question that it is answering
#: something else. Chosen from measured separation on the golden set, not picked
#: for roundness: answerable questions score 0.5-1.0 here, and the unanswerable
#: ones that survived every other check score 0.0-0.25.
RELEVANCE_FLOOR = 0.34

#: What a hard gate caps confidence at -- below the abstain threshold, but not
#: zero, because the retrieved evidence is still real and worth showing.
_GATE_CEILING = 0.35


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
    #: Whether *any* answer text was produced -- by extraction or by an LLM. False
    #: means the question could not be answered from the corpus at all, which is a
    #: different failure from weak evidence and is reported as such.
    answerer_available: bool = True
    #: Fraction of the question's content terms the answer covers. See
    #: ``compose.ComposedAnswer.relevance``.
    answer_relevance: float = 1.0
    #: Question terms absent from the answer, used to explain an abstention.
    missing_terms: list[str] = field(default_factory=list)
    #: Proper nouns in the question that appear nowhere in the corpus.
    unknown_terms: list[str] = field(default_factory=list)
    #: The question asks for the present value of a live measurement.
    wants_live_state: bool = False
    #: How the answer was produced, for the explanation text only. Extraction and
    #: synthesis fail for different reasons and the operator should be told which.
    answer_method: str | None = None
    #: Which scale ``top_score`` is on. The cross-encoder emits a bounded
    #: relevance score; RRF emits an unbounded rank-sum. Squashing the former as
    #: though it were the latter compressed every query into 0.07-0.33 and threw
    #: away the discrimination the reranker exists to provide.
    score_scale: str = "fusion"


def score(inputs: ConfidenceInputs) -> ConfidenceReport:
    settings = get_settings()

    retrieval_strength = _normalise_top_score(inputs.top_score, inputs.score_scale)
    source_agreement = _source_agreement(inputs)
    claim_coverage = inputs.verified_claims / inputs.total_claims if inputs.total_claims else 0.0
    currency = inputs.current_documents / inputs.total_documents if inputs.total_documents else 0.0
    graph_support = 1.0 if inputs.graph_facts else 0.4

    signals = {
        "retrieval_strength": round(retrieval_strength, 4),
        "source_agreement": round(source_agreement, 4),
        "claim_coverage": round(claim_coverage, 4),
        "answer_relevance": round(inputs.answer_relevance, 4),
        "currency": round(currency, 4),
        "graph_support": round(graph_support, 4),
    }
    total = sum(WEIGHTS[name] * value for name, value in signals.items())

    # --- hard gates -------------------------------------------------------
    # Two conditions that no amount of corroborating evidence should be able to
    # outvote. Both are about the question being unanswerable rather than the
    # evidence being weak, so they cap rather than subtract: a weighted sum can
    # always be dragged back up by unrelated signals, and these must not be.
    gate: str | None = None

    # Naming an asset the corpus has never seen. No amount of loosely-related
    # evidence makes an answer about it safe.
    if inputs.anchors_missing:
        total = min(total, _GATE_CEILING)
        gate = "unknown_asset"

    # The answer does not address the question. Retrieval can return excellent
    # passages about the right equipment that say nothing about what was asked --
    # the NPSH of a pump whose datasheet the corpus does not hold. Every other
    # signal looks healthy in that case, which is what makes the gate necessary.
    if inputs.answer_relevance < RELEVANCE_FLOOR:
        total = min(total, _GATE_CEILING)
        gate = gate or "answer_off_topic"

    # The question names a place, plant or system the corpus has never recorded.
    # Retrieval will still return good passages -- about somewhere else. Asked
    # about seal failures at Barauni, a corpus of Haldia documents answers with
    # Haldia's failures and every signal looks healthy, which is a worse outcome
    # than admitting the site is unknown.
    if inputs.unknown_terms:
        total = min(total, _GATE_CEILING)
        gate = gate or "unknown_term"

    # The question asks what a sensor reads right now. This system holds
    # documents, and the newest thing in them is historical -- so retrieval will
    # return a real, well-ranked, correctly cited *past* reading and it will look
    # like an answer. That is the most dangerous shape of wrong answer available
    # here, because everything about it is right except that it is out of date.
    if inputs.wants_live_state:
        total = min(total, _GATE_CEILING)
        gate = gate or "live_state_unavailable"

    if not inputs.answerer_available:
        mode = ConfidenceMode.ABSTAIN_NO_ANSWER
        explanation = (
            "Retrieval completed and the evidence below is real, but no sentence in it "
            "addresses the question, so no answer was composed. The passages are shown "
            "rather than the nearest-looking paragraph being presented as an answer."
        )
    elif total >= settings.confidence_answer_threshold:
        mode = ConfidenceMode.ANSWER
        explanation = "Evidence is strong, current and corroborated across sources."
    elif total >= settings.confidence_caveat_threshold:
        mode = ConfidenceMode.ANSWER_WITH_CAVEAT
        explanation = _weakest_signal_explanation(signals)
    else:
        mode = ConfidenceMode.ABSTAIN_AND_ROUTE
        explanation = _abstention_reason(inputs, signals, gate)

    return ConfidenceReport(
        score=round(total, 4), mode=mode, signals=signals, explanation=explanation
    )


def _normalise_top_score(top_score: float, scale: str = "fusion") -> float:
    """Put the best passage's score onto [0, 1].

    Two scales reach this function and they need different treatment.

    ``reranker`` -- a cross-encoder relevance score, already bounded and already
    monotone in relevance. It is used as-is. Passing it through the saturating
    transform below squeezed 0.33-0.74 down to 0.07-0.33 and destroyed the
    separation between an answerable and an unanswerable question, which is the
    single thing this signal is for.

    ``fusion`` -- an unbounded RRF rank-sum, which needs the saturating
    transform. Monotone, not calibrated; the evaluation harness is what makes the
    resulting thresholds mean anything.
    """
    if top_score <= 0:
        return 0.0
    if scale == "reranker":
        return min(1.0, top_score)
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
    # Keyed by signal name, with a fallback: a new signal must not be able to
    # turn an explanation into a 500. Losing a sentence of detail is a far
    # smaller failure than losing the answer.
    return {
        "retrieval_strength": "the best matching passage is only weakly relevant",
        "source_agreement": "only one source supports this; nothing corroborates it",
        "claim_coverage": "not every statement could be tied to a cited passage",
        "answer_relevance": "the answer covers only part of what was asked",
        "currency": "the supporting evidence is from a superseded revision",
        "graph_support": "the knowledge graph does not corroborate the retrieved text",
    }.get(weakest, f"the {weakest.replace('_', ' ')} signal is weak")


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


def _abstention_reason(
    inputs: ConfidenceInputs, signals: dict[str, float], gate: str | None
) -> str:
    """Say what was missing, not merely that something was.

    "Insufficient evidence" tells an engineer nothing they can act on. Naming the
    asset the corpus has never heard of, or the words the question turned on that
    no source contains, tells them where to look next -- which is the whole
    difference between an abstention and a shrug.
    """
    if gate == "unknown_asset":
        return (
            "The asset named in the question is not present in the corpus: "
            + ", ".join(inputs.anchors_missing)
        )
    if gate == "live_state_unavailable":
        return (
            "This asks for the present value of a measurement. The system holds documents, "
            "not live process data -- no historian or DCS is connected -- so the most it can "
            "offer is the last recorded value, which is shown below. Presenting a historical "
            "reading as a current one is the wrong answer this refusal exists to prevent."
        )
    if gate == "unknown_term":
        return (
            "The question names something the corpus has no record of: "
            + ", ".join(inputs.unknown_terms)
            + ". Evidence about other sites or systems was found, but answering from it "
            "would answer about the wrong one."
        )
    if gate == "answer_off_topic":
        missing = ", ".join(inputs.missing_terms[:6])
        return (
            "Passages about the right equipment were retrieved, but none of them addresses "
            f"what was asked{f': no source mentions {missing}' if missing else ''}. "
            "Answering from what was found would be answering a different question."
        )
    return f"Insufficient evidence to answer safely ({_weakest_signal_explanation(signals)})."
