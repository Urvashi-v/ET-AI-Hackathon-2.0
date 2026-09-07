"""Reciprocal Rank Fusion.

Cosine similarity and BM25 scores are not comparable -- they live on different
scales and neither is calibrated. RRF sidesteps the problem entirely by fusing
on *rank* rather than score::

    score(d) = sum over retrievers i of  w_i / (k + rank_i(d)),   k = 60

Three lines of arithmetic, no normalisation to tune, and it reliably beats any
single retriever because a document that several independent strategies rank
highly is genuinely more likely to be relevant than one that a single strategy
loves.

Weights vary by intent, which is the part that matters in this domain: graph
evidence should dominate a diagnostic question, exact lexical matching should
dominate a tag lookup, and dense similarity should dominate a procedural
"how do I..." question where the wording differs from the document's.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from services.common.schemas import QueryIntent

DEFAULT_K = 60

#: Per-intent retriever weights. Missing retrievers simply contribute nothing,
#: so an unconfigured dense leg degrades the ranking rather than breaking it.
INTENT_WEIGHTS: dict[QueryIntent, dict[str, float]] = {
    QueryIntent.LOOKUP: {"lexical": 1.5, "dense": 1.0, "graph": 0.6},
    QueryIntent.DIAGNOSTIC: {"graph": 1.5, "dense": 1.0, "lexical": 0.8},
    QueryIntent.MULTI_HOP: {"graph": 1.8, "lexical": 0.8, "dense": 0.8},
    QueryIntent.PROCEDURAL: {"dense": 1.3, "lexical": 1.0, "graph": 0.5},
    QueryIntent.AGGREGATE: {"graph": 1.8, "lexical": 0.6, "dense": 0.4},
    QueryIntent.COMPARATIVE: {"graph": 1.2, "dense": 1.0, "lexical": 1.0},
    QueryIntent.UNANSWERABLE: {"lexical": 1.0, "dense": 1.0, "graph": 1.0},
}


#: A leg named ``<retriever>:sub<n>`` ran the retriever over a decomposed
#: sub-question rather than the whole question. It inherits its base retriever's
#: intent weight, discounted: a passage found only by a fragment of the question
#: is real evidence, but weaker than one found by the question entire.
SUB_QUESTION_DISCOUNT = 0.6


@dataclass(slots=True)
class FusedResult:
    chunk_id: str
    score: float
    contributions: dict[str, int] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def retriever_summary(self) -> str:
        """Base retriever names, deduped -- what the citation displays.

        ``lexical`` and ``lexical:sub2`` are the same strategy; showing both
        would suggest two independent retrievers agreed when only one did.
        """
        return "+".join(sorted({name.partition(":")[0] for name in self.contributions})) or "none"

    @property
    def base_retrievers(self) -> set[str]:
        return {name.partition(":")[0] for name in self.contributions}


def weights_for(intent: QueryIntent) -> dict[str, float]:
    return INTENT_WEIGHTS.get(intent, {"lexical": 1.0, "dense": 1.0, "graph": 1.0})


def leg_weight(name: str, weights: dict[str, float]) -> float:
    base, separator, _ = name.partition(":")
    weight = weights.get(base, 1.0)
    return weight * SUB_QUESTION_DISCOUNT if separator else weight


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[dict[str, Any]]],
    *,
    k: int = DEFAULT_K,
    weights: dict[str, float] | None = None,
    id_field: str = "chunk_id",
) -> list[FusedResult]:
    """Fuse ranked lists from heterogeneous retrievers.

    ``ranked_lists`` maps a retriever name to its results in rank order. Each
    result must carry ``id_field``; whatever else it carries is preserved on the
    fused result, with the first retriever to contribute a document supplying
    the payload.
    """
    weights = weights or {}
    scores: dict[str, float] = defaultdict(float)
    contributions: dict[str, dict[str, int]] = defaultdict(dict)
    payloads: dict[str, dict[str, Any]] = {}

    for retriever, results in ranked_lists.items():
        weight = leg_weight(retriever, weights)
        for rank, item in enumerate(results, start=1):
            key = item.get(id_field)
            if not key:
                continue
            scores[key] += weight / (k + rank)
            contributions[key][retriever] = rank
            if key not in payloads:
                payloads[key] = item

    fused = [
        FusedResult(
            chunk_id=key,
            score=score,
            contributions=contributions[key],
            payload=payloads.get(key, {}),
        )
        for key, score in scores.items()
    ]
    # Ties broken by breadth of agreement, then by best single rank: a document
    # two retrievers found should outrank one that only one retriever found.
    fused.sort(
        key=lambda r: (-r.score, -len(r.contributions), min(r.contributions.values(), default=999))
    )
    return fused
