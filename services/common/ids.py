"""Deterministic identifiers.

Ingestion must be idempotent: re-submitting the same bytes must not create a
second document, and re-processing a document must not duplicate its chunks or
its mentions. That is achieved by deriving every identifier from content rather
than from a random UUID, so the same input always produces the same keys and
``ON CONFLICT DO NOTHING`` / ``MERGE`` become sufficient.
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

_NAMESPACE = uuid.UUID("6f1d3b6a-0f1f-4a4a-9a1a-5b1a2c3d4e5f")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def deterministic_uuid(*parts: str) -> uuid.UUID:
    """UUIDv5 over the joined parts. Stable across processes and runs."""
    return uuid.uuid5(_NAMESPACE, "\x1f".join(parts))


def document_id(content_hash: str) -> str:
    """A document is identified by its content. Same bytes -> same document."""
    return f"doc_{content_hash[:24]}"


def chunk_id(doc_id: str, ordinal: int) -> str:
    return f"chk_{deterministic_uuid(doc_id, str(ordinal)).hex[:24]}"


def mention_id(chunk: str, surface_form: str, char_start: int) -> str:
    return f"men_{deterministic_uuid(chunk, surface_form, str(char_start)).hex[:24]}"


def job_id() -> str:
    """Ingestion jobs are events in time, not content -- random is correct."""
    return f"job_{uuid.uuid4().hex[:24]}"


def request_id() -> str:
    return uuid.uuid4().hex


def asset_id(canonical_tag: str) -> str:
    return f"ast_{deterministic_uuid('asset', canonical_tag).hex[:24]}"
