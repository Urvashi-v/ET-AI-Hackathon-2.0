#!/usr/bin/env python3
"""Derive document numbers and revision lineage for already-ingested documents.

Revision tracking arrived after the corpus did. Re-ingesting to pick it up would
work but is the wrong tool: parsing, OCR, chunking, entity resolution and the
graph are all unchanged, and re-running them risks disturbing data that is
correct. This reads the stored title, filename and first chunk, derives the
document number the same way the ingest path now does, and reconciles each
series.

Run inside the API container::

    docker compose exec api python scripts/backfill_revisions.py
    docker compose exec api python scripts/backfill_revisions.py --dry-run

Idempotent: deriving the same number twice is a no-op, and reconciliation always
converges to the same chain regardless of how many times it runs.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from services.common import db, graph  # noqa: E402
from services.common.logging import configure_logging  # noqa: E402
from services.common.schemas import DocumentType  # noqa: E402
from services.ingest import graph_writer, revisions  # noqa: E402


async def backfill(*, dry_run: bool) -> int:
    rows = await db.fetch_all(
        """
        SELECT d.doc_id, d.title, d.original_filename, d.doc_type::text AS doc_type,
               d.doc_number, d.revision,
               (SELECT c.text FROM document_chunks c
                 WHERE c.doc_id = d.doc_id ORDER BY c.ordinal LIMIT 1) AS head_text
          FROM documents d
         ORDER BY d.created_at
        """
    )
    if not rows:
        print("No documents to backfill.")
        return 0

    derived = 0
    print(f"Scanning {len(rows)} document(s)…\n")
    for row in rows:
        number = revisions.derive_doc_number(
            title=row["title"] or "",
            filename=row["original_filename"] or "",
            head_text=row["head_text"] or "",
            doc_type=DocumentType(row["doc_type"]),
        )
        marker = "=" if row["doc_number"] == (number.value if number else None) else ">"
        label = f"{number.value} ({number.method})" if number else "— no document number"
        print(f"  {marker} {row['title'][:46]:48} {label}")
        if number and not dry_run:
            await db.execute(
                "UPDATE documents SET doc_number = %s, doc_number_method = %s WHERE doc_id = %s",
                (number.value, number.method, row["doc_id"]),
            )
        if number:
            derived += 1

    if dry_run:
        print(f"\n--dry-run: {derived} document number(s) would be set. Nothing written.")
        return 0

    numbers = await db.fetch_all(
        "SELECT DISTINCT doc_number FROM documents WHERE doc_number IS NOT NULL ORDER BY doc_number"
    )
    print(f"\nReconciling {len(numbers)} document series…\n")
    conflicts = 0
    for entry in numbers:
        report = await revisions.reconcile(entry["doc_number"])
        await graph_writer.link_revision_chain(report)
        if report["status"] == "conflict":
            conflicts += 1
            print(f"  ! {entry['doc_number']:14} CONFLICT — {report['note']}")
        elif report.get("superseded"):
            chain = " -> ".join(
                f"rev {c['revision'] or '?'}"
                + (f" ({len(c['documents'])} formats)" if len(c["documents"]) > 1 else "")
                for c in report["chain"]
            )
            print(
                f"  * {entry['doc_number']:14} {report['documents']} documents in "
                f"{report['levels']} revisions (by {report['basis']}): {chain}"
            )
        elif report["documents"] > 1:
            print(
                f"  ~ {entry['doc_number']:14} {report['documents']} formats of one revision, "
                "all current"
            )
        else:
            print(f"    {entry['doc_number']:14} single revision")

    print(
        f"\nDone. {derived} numbered, {len(numbers)} series reconciled, "
        f"{conflicts} needing human resolution."
    )
    return 0


async def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="show what would be derived, write nothing"
    )
    args = parser.parse_args()

    configure_logging()
    await db.open_pool()
    await graph.open_driver()
    try:
        return await backfill(dry_run=args.dry_run)
    finally:
        await db.close_pool()
        await graph.close_driver()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
