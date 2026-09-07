"""Embedding providers.

The interface exists; the implementations are gated on real credentials. There
is deliberately no fake provider and no random-vector fallback: a dense index
built from noise would return plausible-looking neighbours and silently poison
every retrieval metric, which is exactly the failure this project is meant to
avoid.

Providers
---------
``none`` (default)
    Dense retrieval is disabled. Ingestion reports the ``embed`` stage as
    ``provider_not_configured`` with the variables needed to enable it, and the
    retrieval pipeline runs lexical + graph, which need no credentials.

``local`` (recommended)
    A real published model — ``BAAI/bge-small-en-v1.5``, 384 dimensions — run
    through ONNX Runtime by fastembed. No API key, and after the model is cached
    on first use, no network call. This is what makes the air-gapped deployment
    story true for retrieval as well as for OCR.

``openai``
    Requires ``OPENAI_API_KEY``. Set ``EMBEDDING_MODEL`` and ``EMBEDDING_DIM`` to
    match.

Changing ``EMBEDDING_DIM`` rebuilds the pgvector column and clears the stored
vectors, because vectors from one model are meaningless to another. The
reconciliation runs at startup and says loudly what it did — see
``services/common/migrate.py``.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from services.common import db
from services.common.config import get_settings
from services.common.logging import get_logger
from services.common.schemas import CapabilityState

log = get_logger(__name__)


@dataclass(slots=True)
class EmbedOutcome:
    state: CapabilityState
    count: int = 0
    detail: str | None = None
    required_env: list[str] = field(default_factory=list)


class EmbeddingProvider(Protocol):
    name: str

    async def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_chunks(self, *, doc_id: str, chunks: list[dict[str, Any]]) -> EmbedOutcome: ...

    async def embed_query(self, text: str) -> list[float] | None: ...


class DisabledEmbeddingProvider:
    """The honest no-op. Reports why it did nothing; never writes a vector."""

    name = "none"

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return []

    async def embed_chunks(self, *, doc_id: str, chunks: list[dict[str, Any]]) -> EmbedOutcome:
        return EmbedOutcome(
            state=CapabilityState.NOT_CONFIGURED,
            count=0,
            detail=(
                "No embedding provider is configured, so no vectors were written and dense "
                "retrieval is unavailable. Lexical (BM25) and graph retrieval are unaffected."
            ),
            required_env=["EMBEDDING_PROVIDER", "EMBEDDING_MODEL", "EMBEDDING_DIM"],
        )

    async def embed_query(self, text: str) -> list[float] | None:
        return None


class OpenAIEmbeddingProvider:
    """OpenAI embeddings over the REST API.

    Batched, retried on transient failures, and hard-failed on a dimension
    mismatch -- writing 1536-dim vectors into a 1024-dim column would be caught
    by Postgres anyway, but failing here gives a message an operator can act on.
    """

    name = "openai"
    _BATCH = 96

    def __init__(self) -> None:
        settings = get_settings()
        self._api_key = settings.openai_api_key.get_secret_value()
        self._base_url = (settings.openai_base_url or "https://api.openai.com/v1").rstrip("/")
        self._model = settings.embedding_model
        self._dim = settings.embedding_dim

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        async with httpx.AsyncClient(timeout=60) as client:
            for start in range(0, len(texts), self._BATCH):
                batch = texts[start : start + self._BATCH]
                response = await client.post(
                    f"{self._base_url}/embeddings",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={"model": self._model, "input": batch},
                )
                response.raise_for_status()
                payload = response.json()
                for item in sorted(payload["data"], key=lambda d: d["index"]):
                    vector = item["embedding"]
                    if len(vector) != self._dim:
                        raise ValueError(
                            f"Embedding provider returned {len(vector)} dimensions but "
                            f"EMBEDDING_DIM is {self._dim}. Set EMBEDDING_DIM to "
                            f"{len(vector)} and re-run migrations."
                        )
                    vectors.append(vector)
        return vectors

    async def embed_chunks(self, *, doc_id: str, chunks: list[dict[str, Any]]) -> EmbedOutcome:
        if not chunks:
            return EmbedOutcome(CapabilityState.AVAILABLE, 0, "no chunks to embed")
        started = time.perf_counter()
        try:
            vectors = await self.embed_texts([c["text"] for c in chunks])
        except Exception as exc:
            log.error("embed.failed", provider=self.name, error=str(exc))
            return EmbedOutcome(
                CapabilityState.ERROR,
                0,
                f"Embedding request failed: {type(exc).__name__}: {str(exc)[:200]}",
            )

        async with db.connection() as conn, conn.cursor() as cur:
            await cur.executemany(
                "INSERT INTO chunk_embeddings (chunk_id, doc_id, embedding, model, dim) "
                "VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (chunk_id) DO UPDATE SET embedding = EXCLUDED.embedding, "
                "model = EXCLUDED.model, dim = EXCLUDED.dim",
                [
                    (c["chunk_id"], doc_id, _to_pgvector(v), self._model, self._dim)
                    for c, v in zip(chunks, vectors, strict=False)
                ],
            )
        elapsed = (time.perf_counter() - started) * 1000
        return EmbedOutcome(
            CapabilityState.AVAILABLE,
            len(vectors),
            f"model={self._model} dim={self._dim} elapsed_ms={elapsed:.0f}",
        )

    async def embed_query(self, text: str) -> list[float] | None:
        vectors = await self.embed_texts([text])
        return vectors[0] if vectors else None


#: Loaded ONNX embedding sessions, by model name. Process-global because loading
#: one costs seconds and the session is stateless and thread-safe to share.
_LOADED_EMBEDDERS: dict[str, Any] = {}


class LocalEmbeddingProvider:
    """A real embedding model, run locally through ONNX Runtime.

    Default: ``BAAI/bge-small-en-v1.5`` — a published 384-dimension model, not a
    stand-in. fastembed rather than sentence-transformers because it needs no
    torch: ~90 MB of dependency instead of ~2.5 GB.

    Two properties matter beyond convenience. The model is downloaded once and
    cached, so after the first run there is **no network call and no API key** —
    which is what makes the air-gapped deployment story true for retrieval as
    well as for OCR. And it is deterministic: the same text always produces the
    same vector, so the evaluation harness measures the system rather than the
    weather.

    Query and passage are embedded asymmetrically. BGE models are trained with a
    query instruction prefix, and ``query_embed`` applies it; using the passage
    encoder for queries measurably degrades recall.
    """

    name = "local"

    def __init__(self) -> None:
        settings = get_settings()
        self._model_name = settings.embedding_local_model
        self._dim = settings.embedding_dim
        self._model: Any = None

    def _load(self) -> Any:
        # Cached on the class, not the instance. get_embedding_provider() returns a
        # fresh object per call, so an instance-level cache reloads the ONNX
        # session from disk on every request -- several seconds each, for a model
        # that is identical every time. Keyed by model name so switching
        # EMBEDDING_LOCAL_MODEL still loads the right one.
        model = _LOADED_EMBEDDERS.get(self._model_name)
        if model is None:
            from fastembed import TextEmbedding

            started = time.perf_counter()
            model = TextEmbedding(self._model_name, threads=get_settings().onnx_threads)
            _LOADED_EMBEDDERS[self._model_name] = model
            log.info(
                "embed.model_loaded",
                model=self._model_name,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
            )
        return model

    def _encode(self, texts: list[str], *, as_query: bool) -> list[list[float]]:
        model = self._load()
        vectors = model.query_embed(texts) if as_query else model.embed(texts)
        return [[float(x) for x in vector] for vector in vectors]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        # ONNX inference is CPU-bound and releases no event loop time, so it runs
        # on a worker thread rather than stalling every other request.
        return await asyncio.to_thread(self._encode, texts, as_query=False)

    async def embed_chunks(self, *, doc_id: str, chunks: list[dict[str, Any]]) -> EmbedOutcome:
        if not chunks:
            return EmbedOutcome(CapabilityState.AVAILABLE, 0, "no chunks to embed")
        started = time.perf_counter()
        try:
            vectors = await self.embed_texts([c["text"] for c in chunks])
        except ImportError:
            return EmbedOutcome(
                CapabilityState.NOT_CONFIGURED,
                0,
                "EMBEDDING_PROVIDER=local requires 'fastembed', which is not installed.",
                ["EMBEDDING_PROVIDER"],
            )
        except Exception as exc:
            log.error("embed.local_failed", model=self._model_name, error=str(exc))
            return EmbedOutcome(CapabilityState.ERROR, 0, f"{type(exc).__name__}: {str(exc)[:200]}")

        if vectors and len(vectors[0]) != self._dim:
            # Writing the wrong width would be caught by the vector column, but
            # failing here names the fix instead of surfacing a driver error.
            return EmbedOutcome(
                CapabilityState.ERROR,
                0,
                f"{self._model_name} produced {len(vectors[0])}-dimension vectors but "
                f"EMBEDDING_DIM is {self._dim}. Set EMBEDDING_DIM={len(vectors[0])} and "
                "restart; the vector column is rebuilt automatically.",
            )

        async with db.connection() as conn, conn.cursor() as cur:
            await cur.executemany(
                "INSERT INTO chunk_embeddings (chunk_id, doc_id, embedding, model, dim) "
                "VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (chunk_id) DO UPDATE SET embedding = EXCLUDED.embedding",
                [
                    (c["chunk_id"], doc_id, _to_pgvector(v), self._model_name, len(v))
                    for c, v in zip(chunks, vectors, strict=False)
                ],
            )
        elapsed = (time.perf_counter() - started) * 1000
        return EmbedOutcome(
            CapabilityState.AVAILABLE,
            len(vectors),
            f"model={self._model_name} dim={self._dim} elapsed_ms={elapsed:.0f}",
        )

    async def embed_query(self, text: str) -> list[float] | None:
        vectors = await asyncio.to_thread(self._encode, [text], as_query=True)
        return vectors[0] if vectors else None


def _to_pgvector(vector: list[float]) -> str:
    return "[" + ",".join(f"{v:.7f}" for v in vector) + "]"


def get_embedding_provider() -> EmbeddingProvider:
    settings = get_settings()
    if settings.embedding_provider == "openai":
        if not settings.openai_api_key.get_secret_value():
            log.warning("embed.provider_selected_without_credential", provider="openai")
            return DisabledEmbeddingProvider()
        return OpenAIEmbeddingProvider()
    if settings.embedding_provider == "local":
        return LocalEmbeddingProvider()
    return DisabledEmbeddingProvider()


def embedding_capability() -> tuple[CapabilityState, str, list[str]]:
    """What the retrieval layer reports for its dense leg."""
    settings = get_settings()
    if settings.embedding_provider == "none":
        return (
            CapabilityState.NOT_CONFIGURED,
            "Dense retrieval is disabled: no embedding provider is configured.",
            ["EMBEDDING_PROVIDER", "EMBEDDING_MODEL", "EMBEDDING_DIM"],
        )
    if settings.embedding_provider == "openai" and not settings.openai_api_key.get_secret_value():
        return (
            CapabilityState.NOT_CONFIGURED,
            "EMBEDDING_PROVIDER=openai but OPENAI_API_KEY is empty.",
            ["OPENAI_API_KEY"],
        )
    return (CapabilityState.AVAILABLE, f"provider={settings.embedding_provider}", [])
