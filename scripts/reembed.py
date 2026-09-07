#!/usr/bin/env python3
"""Populate the dense index for chunks that have no vector under the active model.

Embeddings are stored with the model that produced them, which makes model
migration incremental rather than all-or-nothing: switching models does not
require re-ingesting the corpus, only re-embedding it. Parsing, chunking, entity
resolution and the graph are untouched.

Run inside the API container, where the ONNX model cache lives::

    docker compose exec api python scripts/reembed.py
    docker compose exec api python scripts/reembed.py --all     # re-embed everything

Idempotent: chunks that already have a vector for the active model are skipped,
so it is safe to re-run after an interrupted pass.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from services.common import db  # noqa: E402
from services.common.config import get_settings  # noqa: E402
from services.common.logging import configure_logging, get_logger  # noqa: E402
from services.common.schemas import CapabilityState  # noqa: E402
from services.ingest.embeddings import embedding_capability, get_embedding_provider  # noqa: E402

log = get_logger(__name__)

#: Embedded in batches so a large corpus does not hold the whole vector set in
#: memory, and so progress is visible on a long run.
BATCH_SIZE = 64


async def reembed(*, force: bool, batch_size: int) -> int:
    settings = get_settings()
    model = settings.active_embedding_model

    state, detail, required_env = embedding_capability()
    if state is not CapabilityState.AVAILABLE:
        print(f"Embedding provider is not available: {detail}", file=sys.stderr)
        if required_env:
            print(f"Set: {', '.join(required_env)}", file=sys.stderr)
        return 1

    if force:
        removed = await db.execute("DELETE FROM chunk_embeddings WHERE model = %s", (model,))
        print(f"--all: dropped {removed} existing vector(s) for {model}")

    pending = await db.fetch_all(
        """
        SELECT c.chunk_id, c.doc_id, c.text, c.context_header
          FROM document_chunks c
          LEFT JOIN chunk_embeddings e
                 ON e.chunk_id = c.chunk_id AND e.model = %s
         WHERE e.chunk_id IS NULL
         ORDER BY c.doc_id, c.ordinal
        """,
        (model,),
    )
    if not pending:
        print(f"Nothing to do: every chunk already has a vector for {model}.")
        return 0

    print(f"Embedding {len(pending)} chunk(s) with {model} (dim {settings.embedding_dim})…")
    provider = get_embedding_provider()
    started = time.perf_counter()
    embedded = 0

    # Grouped by document because embed_chunks writes per document, and because
    # a failure part-way through then leaves whole documents done rather than a
    # document half-embedded.
    by_doc: dict[str, list[dict]] = {}
    for row in pending:
        by_doc.setdefault(row["doc_id"], []).append(row)

    for doc_id, rows in by_doc.items():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            outcome = await provider.embed_chunks(
                doc_id=doc_id,
                chunks=[
                    {
                        "chunk_id": r["chunk_id"],
                        # The same text the ingest path embeds: contextual header
                        # plus body. Embedding the body alone here would make the
                        # index inconsistent with itself.
                        "text": (
                            f"{r['context_header']}\n{r['text']}"
                            if r["context_header"]
                            else r["text"]
                        ),
                    }
                    for r in batch
                ],
            )
            if outcome.state is not CapabilityState.AVAILABLE:
                print(f"\nFailed on {doc_id}: {outcome.detail}", file=sys.stderr)
                return 1
            embedded += outcome.count
            print(f"  {embedded}/{len(pending)}", end="\r", flush=True)

    elapsed = time.perf_counter() - started
    total = await db.fetch_one(
        "SELECT count(*)::int AS n FROM chunk_embeddings WHERE model = %s", (model,)
    )
    print(
        f"\nEmbedded {embedded} chunk(s) in {elapsed:.1f}s "
        f"({embedded / elapsed:.1f}/s). Dense index now holds "
        f"{int(total['n']) if total else 0} vector(s) for {model}."
    )
    return 0


async def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all", action="store_true", help="re-embed every chunk, not just the missing ones"
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    configure_logging()
    await db.open_pool()
    try:
        return await reembed(force=args.all, batch_size=args.batch_size)
    finally:
        await db.close_pool()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
