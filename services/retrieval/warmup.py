"""Pre-warm the local ONNX models at startup.

ONNX Runtime does not reach steady-state throughput on its first inference. On
this deployment the cross-encoder was measured at 15.6 s, then 4.7 s, then 1.8 s
for the same 25 passages — session setup, memory-arena growth and kernel
selection all resolve over the first few calls. Left alone, the first engineer to
ask a question pays all of it and concludes the system is unusable.

So it is paid at startup instead, on synthetic strings that never touch the
corpus or the database. This is warm-up, not caching: no result is stored and no
answer is precomputed. Two consequences worth stating plainly:

* It runs in the background. Blocking startup on it would make the container fail
  its health check on a cold model cache, and a slow first query is a better
  failure than a container that will not start.
* On the very first run of a fresh deployment this also performs the model
  *download*, which is why it can take a minute and why the log line reports
  elapsed time. Subsequent starts read from the cache volume.

If a model is unavailable, warm-up logs and returns. It must never be the reason
the API fails to serve — the capability layer reports unavailability truthfully
at query time.
"""

from __future__ import annotations

import time
from typing import Any

from services.common.config import get_settings
from services.common.logging import get_logger
from services.common.schemas import CapabilityState
from services.ingest.embeddings import embedding_capability, get_embedding_provider
from services.retrieval.rerank import get_reranker, reranker_capability

log = get_logger(__name__)

#: Deliberately mundane, deliberately not from the corpus. Warming must not put
#: plant text through a model before a user has asked for anything, and using a
#: real passage here would make it look as though it had.
_WARM_QUERY = "warm up the retrieval models"
_WARM_PASSAGES = ["A short passage used only to warm the inference session."] * 4

#: Three passes: the first is session setup, the second still pays arena growth,
#: the third is close to steady state. Measured, not guessed.
_PASSES = 3


async def warm_models() -> dict[str, Any]:
    """Run a few throwaway inferences so the first real query is not the slowest."""
    report: dict[str, Any] = {}
    started = time.perf_counter()

    embed_state, embed_detail, _ = embedding_capability()
    if embed_state is CapabilityState.AVAILABLE:
        report["embedding"] = await _warm(
            "embedding",
            get_settings().active_embedding_model,
            lambda: get_embedding_provider().embed_query(_WARM_QUERY),
        )
    else:
        report["embedding"] = {"state": embed_state.value, "detail": embed_detail}

    rerank_state, rerank_detail, _ = reranker_capability()
    if rerank_state is CapabilityState.AVAILABLE:
        report["reranker"] = await _warm(
            "reranker",
            get_settings().reranker_local_model,
            lambda: get_reranker().rerank(_WARM_QUERY, _WARM_PASSAGES),
        )
    else:
        report["reranker"] = {"state": rerank_state.value, "detail": rerank_detail}

    report["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
    log.info("warmup.complete", **{k: v for k, v in report.items() if k == "elapsed_ms"})
    return report


async def _warm(kind: str, model: str, call: Any) -> dict[str, Any]:
    timings: list[float] = []
    try:
        for _ in range(_PASSES):
            started = time.perf_counter()
            await call()
            timings.append(round((time.perf_counter() - started) * 1000, 1))
    except Exception as exc:
        # A failed warm-up is not a failed startup. The capability layer will
        # report the same problem at query time, with the same detail.
        log.warning("warmup.failed", kind=kind, model=model, error=str(exc)[:200])
        return {"state": "error", "model": model, "detail": f"{type(exc).__name__}: {exc}"[:200]}

    log.info("warmup.model_warm", kind=kind, model=model, passes_ms=timings)
    return {"state": "warm", "model": model, "passes_ms": timings}
