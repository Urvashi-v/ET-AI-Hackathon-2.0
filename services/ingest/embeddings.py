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

``openai``
    Requires ``OPENAI_API_KEY``. Set ``EMBEDDING_MODEL`` and ``EMBEDDING_DIM`` to
    match; the dimension must equal the ``vector(n)`` column created by
    migration 003.

``local``
    Requires ``sentence-transformers`` (not in ``requirements.txt`` -- it pulls
    in torch) and a model available offline. Supports the air-gapped deployment
    story: no external network call anywhere in the pipeline.
"""

from __future__ import annotations

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


class LocalEmbeddingProvider:
    """sentence-transformers, loaded lazily.

    Not installed by default: torch is a heavy dependency and most of the
    pipeline does not need it. Selecting ``EMBEDDING_PROVIDER=local`` without
    installing it produces a clear message, not an import traceback.
    """

    name = "local"

    def __init__(self) -> None:
        self._model_name = get_settings().embedding_local_model
        self._dim = get_settings().embedding_dim
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        return [list(map(float, v)) for v in model.encode(texts, normalize_embeddings=True)]

    async def embed_chunks(self, *, doc_id: str, chunks: list[dict[str, Any]]) -> EmbedOutcome:
        if not chunks:
            return EmbedOutcome(CapabilityState.AVAILABLE, 0, "no chunks to embed")
        try:
            vectors = await self.embed_texts([c["text"] for c in chunks])
        except ImportError:
            return EmbedOutcome(
                CapabilityState.NOT_CONFIGURED,
                0,
                "EMBEDDING_PROVIDER=local requires 'sentence-transformers', which is not "
                "installed. Install it in the image, or switch provider.",
                ["EMBEDDING_PROVIDER"],
            )
        except Exception as exc:
            return EmbedOutcome(CapabilityState.ERROR, 0, f"{type(exc).__name__}: {str(exc)[:200]}")

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
        return EmbedOutcome(CapabilityState.AVAILABLE, len(vectors), f"model={self._model_name}")

    async def embed_query(self, text: str) -> list[float] | None:
        vectors = await self.embed_texts([text])
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
