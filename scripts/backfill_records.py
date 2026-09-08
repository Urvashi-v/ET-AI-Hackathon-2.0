#!/usr/bin/env python3
"""Extract structured records from documents already in the corpus.

Record extraction arrived after the documents did. Re-ingesting would work but is
the wrong tool: parsing, OCR, chunking, entity resolution and the graph are all
unchanged, and re-running them risks disturbing data that is correct. This reads
the stored chunks — exactly what a citation would open — and writes the incident
and change records they contain.

Run inside the API container::

    docker compose exec api python scripts/backfill_records.py
    docker compose exec api python scripts/backfill_records.py --dry-run

Idempotent: records key on the identifier printed on the source document, so
re-running updates rather than duplicating. Two renditions of one report (a
Markdown source and a scan of the signed copy) converge on a single record.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from itertools import groupby
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from services.common import db, graph  # noqa: E402
from services.common.logging import configure_logging  # noqa: E402
from services.common.schemas import DataClass, DocumentType  # noqa: E402
from services.ingest import record_writer, records  # noqa: E402


async def backfill(*, dry_run: bool) -> int:
    rows = await db.fetch_all(
        """
        SELECT d.doc_id, d.title, d.doc_number, d.doc_type::text AS doc_type,
               d.data_class::text AS data_class,
               c.chunk_id, c.text, c.section_path, c.page_from, c.ordinal
          FROM documents d
          JOIN document_chunks c ON c.doc_id = d.doc_id
         WHERE d.doc_type IN ('incident_report', 'moc')
         ORDER BY d.doc_id, c.ordinal
        """
    )
    if not rows:
        print("No incident or MOC documents in the corpus.")
        return 0

    incidents = changes = rejected = 0
    for doc_id, group in groupby(rows, key=lambda r: r["doc_id"]):
        chunks = list(group)
        head = chunks[0]
        doc_type = DocumentType(head["doc_type"])
        data_class = DataClass(head["data_class"])

        if doc_type is DocumentType.INCIDENT_REPORT:
            record = records.extract_incident(
                doc_id=doc_id,
                title=head["title"],
                doc_number=head["doc_number"],
                chunks=chunks,
            )
            if record is None:
                rejected += 1
                print(f"  - {head['title'][:44]:46} no incident structure found")
                continue
            missing = (
                f" missing: {', '.join(record.fields_missing)}" if record.fields_missing else ""
            )
            print(
                f"  * {record.incident_id:12} {head['title'][:32]:34} "
                f"{len(record.corrective_actions)} CAPA(s){missing}"
            )
            if not dry_run:
                await record_writer.write_incident(record, data_class=data_class)
            incidents += 1
            continue

        change = records.extract_change(
            doc_id=doc_id, title=head["title"], doc_number=head["doc_number"], chunks=chunks
        )
        if change is None:
            rejected += 1
            print(f"  - {head['title'][:44]:46} no MOC number found")
            continue
        print(f"  * {change.moc_id:12} {head['title'][:32]:34} affects {change.asset_tags}")
        if not dry_run:
            await record_writer.write_change(change, data_class=data_class)
        changes += 1

    verb = "would be written" if dry_run else "written"
    print(f"\n{incidents} incident(s) and {changes} change record(s) {verb}; {rejected} rejected.")
    if not dry_run:
        counts = await db.fetch_one("SELECT count(*)::int AS n FROM incidents")
        graph_counts = await graph.read(
            "MATCH (i:Incident) WITH count(i) AS incidents "
            "CALL { MATCH (c:CorrectiveAction) RETURN count(c) AS actions } "
            "RETURN incidents, actions"
        )
        print(
            f"Postgres now holds {counts['n'] if counts else 0} incident(s); "
            f"Neo4j holds {graph_counts[0]['incidents']} Incident and "
            f"{graph_counts[0]['actions']} CorrectiveAction node(s)."
        )
    return 0


async def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="show what would be extracted")
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
