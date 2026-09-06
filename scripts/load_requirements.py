#!/usr/bin/env python3
"""Load atomised regulatory requirements into Postgres and Neo4j.

Requirements are reference data, not ingested documents: they are curated by
hand, reviewed, and loaded deterministically. Re-running is safe -- rows are
upserted on ``req_id``.

The loader refuses to load a requirement without a ``provenance_note``, and
refuses ``text_status`` values it does not recognise. A compliance finding is
only as good as the traceability of the requirement behind it, so an
untraceable requirement must not reach the database at all.

Usage::

    python scripts/load_requirements.py                       # default file
    python scripts/load_requirements.py --file path/to.json
    python scripts/load_requirements.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from services.common import db, graph  # noqa: E402
from services.common.logging import configure_logging, get_logger  # noqa: E402

log = get_logger(__name__)

DEFAULT_FILE = REPO_ROOT / "data" / "requirements" / "atomised_requirements.json"
VALID_TEXT_STATUS = {"verbatim", "paraphrase_for_demo"}
VALID_MODALITY = {"shall", "should", "may"}
REQUIRED_FIELDS = (
    "req_id",
    "source_standard",
    "clause",
    "obligation_text",
    "modality",
    "text_status",
    "provenance_note",
)

_UPSERT_GRAPH = """
UNWIND $requirements AS r
MERGE (req:Requirement {req_id: r.req_id})
  ON CREATE SET req.created_at = datetime()
SET req.source_standard   = r.source_standard,
    req.clause            = r.clause,
    req.obligation_text   = r.obligation_text,
    req.modality          = r.modality,
    req.applies_to_class  = r.applies_to_class,
    req.frequency_months  = r.frequency_months,
    req.testable_by       = r.testable_by,
    req.text_status       = r.text_status,
    req.provenance_note   = r.provenance_note,
    req.effective_from    = CASE WHEN r.effective_from IS NULL
                                 THEN NULL ELSE date(r.effective_from) END,
    req.data_class        = 'reference_taxonomy',
    req.updated_at        = datetime()
WITH req, r
OPTIONAL MATCH (cls:EquipmentClass {code: r.applies_to_class})
FOREACH (_ IN CASE WHEN cls IS NULL THEN [] ELSE [1] END |
    MERGE (req)-[:APPLIES_TO]->(cls))
RETURN count(req) AS loaded
"""


def validate(entry: dict[str, Any], index: int) -> list[str]:
    problems: list[str] = []
    for field in REQUIRED_FIELDS:
        if not entry.get(field):
            problems.append(f"[{index}] missing required field '{field}'")
    if entry.get("modality") and entry["modality"] not in VALID_MODALITY:
        problems.append(
            f"[{index}] modality '{entry['modality']}' is not one of {sorted(VALID_MODALITY)}; "
            "shall/should/may carry different legal weight and must not be flattened"
        )
    if entry.get("text_status") and entry["text_status"] not in VALID_TEXT_STATUS:
        problems.append(
            f"[{index}] text_status '{entry['text_status']}' is not one of "
            f"{sorted(VALID_TEXT_STATUS)}"
        )
    if entry.get("text_status") == "verbatim" and "retriev" not in (
        entry.get("provenance_note", "").lower()
    ):
        problems.append(
            f"[{index}] req_id={entry.get('req_id')}: text_status is 'verbatim' but the "
            "provenance_note does not state where and when the text was retrieved. "
            "Verbatim requires a citation."
        )
    return problems


async def load(path: Path, *, dry_run: bool) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries: list[dict[str, Any]] = payload.get("requirements", [])
    if not entries:
        log.error("requirements.empty_file", path=str(path))
        return 1

    problems: list[str] = []
    for index, entry in enumerate(entries):
        problems.extend(validate(entry, index))
    if problems:
        for problem in problems:
            print(f"REJECTED  {problem}", file=sys.stderr)
        print(
            f"\n{len(problems)} validation problem(s). Nothing was loaded.",
            file=sys.stderr,
        )
        return 1

    by_status: dict[str, int] = {}
    for entry in entries:
        by_status[entry["text_status"]] = by_status.get(entry["text_status"], 0) + 1

    print(f"Validated {len(entries)} requirements from {path.name}")
    for status, count in sorted(by_status.items()):
        print(f"  {status:22} {count}")
    if by_status.get("paraphrase_for_demo"):
        print(
            "\n  NOTE: paraphrased requirements must not be used to assert a regulatory\n"
            "  position. The API reports this breakdown on every compliance response."
        )
    if dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    async with db.connection() as conn, conn.cursor() as cur:
        for entry in entries:
            await cur.execute(
                """
                INSERT INTO requirements (
                    req_id, source_standard, clause, obligation_text, modality,
                    applies_to_class, frequency_months, limit_json, trigger_text,
                    testable_by, effective_from, text_status, provenance_note, data_class
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'reference_taxonomy')
                ON CONFLICT (req_id) DO UPDATE SET
                    obligation_text = EXCLUDED.obligation_text,
                    modality        = EXCLUDED.modality,
                    frequency_months= EXCLUDED.frequency_months,
                    limit_json      = EXCLUDED.limit_json,
                    trigger_text    = EXCLUDED.trigger_text,
                    testable_by     = EXCLUDED.testable_by,
                    text_status     = EXCLUDED.text_status,
                    provenance_note = EXCLUDED.provenance_note
                """,
                (
                    entry["req_id"],
                    entry["source_standard"],
                    entry["clause"],
                    entry["obligation_text"],
                    entry["modality"],
                    entry.get("applies_to_class"),
                    entry.get("frequency_months"),
                    json.dumps(entry["limit_json"]) if entry.get("limit_json") else None,
                    entry.get("trigger_text"),
                    entry.get("testable_by"),
                    entry.get("effective_from"),
                    entry["text_status"],
                    entry["provenance_note"],
                ),
            )

    await graph.write(_UPSERT_GRAPH, requirements=entries)
    print(f"\nLoaded {len(entries)} requirements into Postgres and Neo4j.")
    return 0


async def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default=str(DEFAULT_FILE))
    parser.add_argument("--dry-run", action="store_true", help="validate without writing")
    args = parser.parse_args()

    configure_logging()
    path = Path(args.file)
    if not path.is_file():
        print(f"No such file: {path}", file=sys.stderr)
        return 1

    if args.dry_run:
        payload = json.loads(path.read_text(encoding="utf-8"))
        problems = [
            p for i, e in enumerate(payload.get("requirements", [])) for p in validate(e, i)
        ]
        for problem in problems:
            print(f"REJECTED  {problem}", file=sys.stderr)
        print(f"{len(payload.get('requirements', []))} entries, {len(problems)} problem(s).")
        return 1 if problems else 0

    await db.open_pool()
    await graph.open_driver()
    try:
        return await load(path, dry_run=False)
    finally:
        await db.close_pool()
        await graph.close_driver()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
