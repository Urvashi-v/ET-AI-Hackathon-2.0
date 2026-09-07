"""Cross-encoder reranking.

Reranking the fused top ~50 down to a top ~8 is consistently the highest-return
quality intervention available in a retrieval pipeline, and the reason is
structural rather than incidental. A bi-encoder embeds query and passage
*separately*, so it can only compare two summaries of meaning. A cross-encoder
reads them **together**, one forward pass per pair, and can therefore see that a
passage answers a question rather than merely resembling it.

The model is real: ``Xenova/ms-marco-MiniLM-L-6-v2``, the published MS MARCO
cross-encoder, run through ONNX Runtime. There is no hand-rolled similarity
score anywhere in this module — a fabricated one would reorder results
plausibly and make every retrieval metric describe something other than what
was measured.

Cost is the reason it is a separate, bounded stage: one model pass per candidate
rather than one per corpus chunk. Reranking 50 candidates is a few hundred
milliseconds; reranking the corpus would be minutes.

With ``RERANKER_PROVIDER=none`` the fused RRF order is used unchanged and the
pipeline reports that, rather than claiming a rerank happened.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Protocol

from services.common.config import get_settings
from services.common.logging import get_logger
from services.common.schemas import CapabilityState

log = get_logger(__name__)

#: Cross-encoder scores are unbounded logits, not probabilities. They are
#: comparable *within* one query's candidate set and meaningless across queries,
#: so they are used for ordering and then normalised for display.
_LOGIT_FLOOR = -12.0
_LOGIT_CEILING = 12.0

#: Passage text handed to the cross-encoder. Two reasons it is this small.
#: Cost first: attention is quadratic in sequence length, and measured on this
#: corpus a 2000-character passage costs ~870 ms to score against ~30 ms for a
#: 400-character one -- a 25-candidate rerank goes from 22 s to 0.8 s. Fit
#: second: MS MARCO cross-encoders are trained on passages of roughly this
#: length, so longer input is out of distribution as well as slow. 90 of the 91
#: chunks in the current corpus are already under this, so it truncates almost
#: nothing -- and only the reranker's *view*: the full passage still reaches the
#: answer composer and the citation.
MAX_PASSAGE_CHARS = 1000


@dataclass(slots=True)
class RerankResult:
    state: CapabilityState
    #: Candidate indices in their new order, best first.
    order: list[int] = field(default_factory=list)
    #: Raw cross-encoder logit per candidate, in the original input order.
    scores: list[float] = field(default_factory=list)
    model: str | None = None
    detail: str | None = None
    required_env: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0

    def normalised(self, index: int) -> float:
        """Map a logit to [0, 1] for display only.

        Deliberately not called a probability: it is a monotone squash of an
        unbounded score, and the thresholds that matter are set by the evaluation
        harness rather than read off this number.
        """
        if index >= len(self.scores):
            return 0.0
        clamped = max(_LOGIT_FLOOR, min(_LOGIT_CEILING, self.scores[index]))
        return round((clamped - _LOGIT_FLOOR) / (_LOGIT_CEILING - _LOGIT_FLOOR), 4)


class Reranker(Protocol):
    name: str

    async def rerank(self, query: str, passages: list[str]) -> RerankResult: ...


class DisabledReranker:
    """The honest no-op. Preserves the input order and says it did nothing."""

    name = "none"

    async def rerank(self, query: str, passages: list[str]) -> RerankResult:
        return RerankResult(
            state=CapabilityState.NOT_CONFIGURED,
            order=list(range(len(passages))),
            detail=(
                "Cross-encoder reranking is not configured, so the fused RRF order is used "
                "unchanged. This is reported rather than silently skipped."
            ),
            required_env=["RERANKER_PROVIDER"],
        )


#: Reranker results, keyed by the exact question and the exact candidate set.
#:
#: Reranking is the slowest stage in the pipeline by an order of magnitude, and
#: it is *pure*: the same query against the same passages in the same order
#: always produces the same scores. That makes it the one place in this system
#: where caching is unambiguously safe -- there is no staleness question, because
#: the key contains everything the computation reads.
#:
#: Worth doing because repetition is real rather than hypothetical: a shift
#: handover asks the same question twice, two technicians on the same job ask it
#: separately, and someone re-runs a query after reading the evidence. The
#: benchmark repeats it fifty-five times.
#:
#: Deliberately NOT cached anywhere else. Retrieval results depend on corpus
#: state that changes under ingestion, and a cache there would serve answers from
#: a corpus that no longer exists -- which in this domain is the failure the whole
#: project is built to avoid.
_SCORE_CACHE: OrderedDict[str, list[float]] = OrderedDict()

#: Small on purpose. Each entry is a handful of floats, but an unbounded cache in
#: a long-running process is a memory leak with a friendly name.
_CACHE_MAX_ENTRIES = 256


def _cache_key(model: str, query: str, passages: list[str]) -> str:
    """Everything the scoring reads, hashed.

    The passage *contents* are hashed rather than their ids: two retrievals can
    return the same chunk ids after the chunk text has been re-ingested, and
    scoring stale text would be worse than not caching at all.
    """
    # A NUL separator, because it cannot occur in the text being hashed. A space
    # would let "a b" + "c" and "a" + "b c" produce the same digest, and a cache
    # collision here returns another question's ranking under this question's
    # citations -- silently, and with full confidence.
    separator = bytes([0])
    digest = hashlib.sha256()
    digest.update(model.encode("utf-8"))
    for part in (query, *passages):
        digest.update(separator)
        digest.update(part.encode("utf-8"))
    return digest.hexdigest()


def cache_stats() -> dict[str, Any]:
    """Hit rate, for the health endpoint. A cache nobody can see is a cache
    nobody can tell is broken."""
    total = _CACHE_HITS + _CACHE_MISSES
    return {
        "entries": len(_SCORE_CACHE),
        "capacity": _CACHE_MAX_ENTRIES,
        "hits": _CACHE_HITS,
        "misses": _CACHE_MISSES,
        "hit_rate": round(_CACHE_HITS / total, 4) if total else None,
    }


def clear_cache() -> None:
    """Drop everything. Used by tests, and available if a model is swapped."""
    global _CACHE_HITS, _CACHE_MISSES
    _SCORE_CACHE.clear()
    _CACHE_HITS = _CACHE_MISSES = 0


_CACHE_HITS = 0
_CACHE_MISSES = 0


#: Loaded ONNX cross-encoder sessions, by model name. Process-global for the same
#: reason as the embedders: loading costs seconds, the session is reusable.
_LOADED_RERANKERS: dict[str, Any] = {}


class LocalCrossEncoderReranker:
    """A real cross-encoder, run locally through ONNX Runtime.

    Loaded once per process and reused. The first call pays the model load; the
    rest are a few milliseconds per candidate.
    """

    name = "local"

    def __init__(self) -> None:
        self._model_name = get_settings().reranker_local_model
        self._model: Any = None

    def _load(self) -> Any:
        # Cached on the class, not the instance: get_reranker() hands back a new
        # object per request, so an instance cache would reload the ONNX session
        # from disk every query and spend seconds re-reading identical weights.
        model = _LOADED_RERANKERS.get(self._model_name)
        if model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            started = time.perf_counter()
            model = TextCrossEncoder(self._model_name, threads=get_settings().onnx_threads)
            _LOADED_RERANKERS[self._model_name] = model
            log.info(
                "rerank.model_loaded",
                model=self._model_name,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
            )
        return model

    def _score(self, query: str, passages: list[str]) -> list[float]:
        # Truncated to the model's actual input window. MiniLM-L-6 is a 512-token
        # encoder: text beyond that is discarded by the tokenizer regardless, so
        # sending a 4 kB passage buys nothing and costs real time. Cutting it here
        # makes the cost per candidate predictable instead of proportional to
        # whatever the chunker happened to produce.
        clipped = [p[:MAX_PASSAGE_CHARS] for p in passages]
        return [float(s) for s in self._load().rerank(query, clipped)]

    async def rerank(self, query: str, passages: list[str]) -> RerankResult:
        global _CACHE_HITS, _CACHE_MISSES

        if not passages:
            return RerankResult(state=CapabilityState.AVAILABLE, model=self._model_name)

        started = time.perf_counter()
        key = _cache_key(self._model_name, query, passages)
        cached = _SCORE_CACHE.get(key)
        if cached is not None:
            _CACHE_HITS += 1
            _SCORE_CACHE.move_to_end(key)
            order = sorted(range(len(cached)), key=lambda i: -cached[i])
            return RerankResult(
                state=CapabilityState.AVAILABLE,
                order=order,
                scores=cached,
                model=self._model_name,
                detail=f"model={self._model_name} candidates={len(passages)} (cached)",
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )
        _CACHE_MISSES += 1

        try:
            # CPU-bound ONNX inference: off the event loop, or every concurrent
            # request stalls behind it.
            scores = await asyncio.to_thread(self._score, query, passages)
        except ImportError:
            return RerankResult(
                state=CapabilityState.NOT_CONFIGURED,
                order=list(range(len(passages))),
                detail="RERANKER_PROVIDER=local requires 'fastembed', which is not installed.",
                required_env=["RERANKER_PROVIDER"],
            )
        except Exception as exc:
            log.error("rerank.failed", model=self._model_name, error=str(exc))
            return RerankResult(
                state=CapabilityState.ERROR,
                # Degrade to the fused order rather than losing the results.
                order=list(range(len(passages))),
                model=self._model_name,
                detail=f"Reranking failed: {type(exc).__name__}: {str(exc)[:200]}",
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )

        _SCORE_CACHE[key] = scores
        while len(_SCORE_CACHE) > _CACHE_MAX_ENTRIES:
            _SCORE_CACHE.popitem(last=False)

        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        elapsed = (time.perf_counter() - started) * 1000
        log.info(
            "rerank.done",
            model=self._model_name,
            candidates=len(passages),
            total_chars=sum(len(p) for p in passages),
            max_chars=max(len(p) for p in passages),
            elapsed_ms=round(elapsed, 1),
        )
        return RerankResult(
            state=CapabilityState.AVAILABLE,
            order=order,
            scores=scores,
            model=self._model_name,
            detail=f"model={self._model_name} candidates={len(passages)}",
            elapsed_ms=elapsed,
        )


def get_reranker() -> Reranker:
    if get_settings().reranker_provider == "local":
        return LocalCrossEncoderReranker()
    return DisabledReranker()


def reranker_capability() -> tuple[CapabilityState, str, list[str]]:
    """What the pipeline reports for its rerank stage."""
    settings = get_settings()
    if settings.reranker_provider == "none":
        return (
            CapabilityState.NOT_CONFIGURED,
            "Cross-encoder reranking is not configured; the fused RRF order is used unchanged.",
            ["RERANKER_PROVIDER"],
        )
    return (
        CapabilityState.AVAILABLE,
        f"provider=local model={settings.reranker_local_model}",
        [],
    )
