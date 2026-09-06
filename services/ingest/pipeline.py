"""The write path: bytes in, resolved knowledge out.

One document at a time, through these stages::

    classify -> parse -> chunk -> index (lexical) -> embed -> extract
             -> resolve -> persist (Postgres) -> upsert (Neo4j) -> emit event

Two properties hold throughout:

**Idempotence.** Document ids are content hashes and every write is an upsert on
a natural key, so re-running the pipeline over the same bytes converges rather
than duplicating. Re-ingesting a corpus is therefore safe, which matters because
the alternative -- double-counted work orders -- corrupts every statistic
silently.

**Truthful stage reporting.** Each stage records what it did or why it could
not, in ``stage_report``. A stage that needs a provider that is not configured
reports ``provider_not_configured`` together with the environment variables
required. Nothing is skipped quietly, and no stage substitutes invented output
for a real one.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from services.common import bus, db
from services.common.config import get_settings
from services.common.ids import chunk_id as make_chunk_id
from services.common.ids import document_id as make_document_id
from services.common.ids import mention_id as make_mention_id
from services.common.logging import get_logger
from services.common.schemas import CapabilityState, DataClass, DocumentType
from services.ingest import graph_writer, storage
from services.ingest.chunk import Chunk, chunk_document
from services.ingest.classify import classify
from services.ingest.embeddings import get_embedding_provider
from services.ingest.extract import extract_all
from services.ingest.parsers import parse_document
from services.ingest.resolve import AssetIndex, resolve_mentions
from services.retrieval.lexical import index_chunk_terms

log = get_logger(__name__)


@dataclass(slots=True)
class StageOutcome:
    stage: str
    state: CapabilityState
    items: int = 0
    detail: str | None = None
    required_env: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "state": self.state.value,
            "items": self.items,
            "detail": self.detail,
            "required_env": self.required_env,
        }


@dataclass(slots=True)
class DocumentResult:
    doc_id: str | None
    filename: str
    status: str  # ingested | duplicate | failed
    doc_type: str | None = None
    chunks: int = 0
    mentions: int = 0
    assets: int = 0
    edges: int = 0
    stages: list[StageOutcome] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "status": self.status,
            "doc_type": self.doc_type,
            "chunks": self.chunks,
            "mentions": self.mentions,
            "assets": self.assets,
            "edges": self.edges,
            "stages": [s.to_dict() for s in self.stages],
            "warnings": self.warnings,
            "error": self.error,
        }


async def ingest_file(
    *,
    job_id: str,
    blob: storage.StoredBlob,
    data_class: DataClass,
    source_system: str,
    source_path: str | None = None,
) -> DocumentResult:
    """Run the full write path for one stored blob."""
    started = time.perf_counter()
    result = DocumentResult(doc_id=None, filename=blob.original_filename, status="failed")
    doc_id = make_document_id(blob.content_hash)

    existing = await db.fetch_one(
        "SELECT doc_id, title FROM documents WHERE content_hash = %s", (blob.content_hash,)
    )
    if existing:
        result.doc_id = existing["doc_id"]
        result.status = "duplicate"
        result.stages.append(
            StageOutcome(
                "dedup",
                CapabilityState.AVAILABLE,
                detail=f"content hash already ingested as {existing['doc_id']}",
            )
        )
        log.info("ingest.duplicate", doc_id=existing["doc_id"], filename=blob.original_filename)
        return result

    data = storage.read_blob(blob.path)

    # --- parse -------------------------------------------------------------
    parsed = parse_document(data, filename=blob.original_filename, extension=blob.extension)
    result.warnings.extend(parsed.warnings)
    if parsed.blocks:
        result.stages.append(
            StageOutcome(
                "parse",
                CapabilityState.AVAILABLE,
                items=len(parsed.blocks),
                detail=f"parser={parsed.parser}",
            )
        )
    else:
        result.stages.append(
            StageOutcome(
                "parse",
                CapabilityState.ERROR,
                detail="; ".join(parsed.warnings) or "no text extracted",
            )
        )

    # --- classify ----------------------------------------------------------
    classification = classify(
        filename=blob.original_filename,
        extension=blob.extension,
        head_text=parsed.head_text,
        page_count=parsed.page_count,
        has_text_layer=parsed.has_text_layer,
    )
    result.doc_type = classification.doc_type.value
    result.stages.append(
        StageOutcome(
            "classify",
            CapabilityState.AVAILABLE,
            detail=f"{classification.doc_type.value} "
            f"(confidence {classification.confidence:.2f}, via {classification.method})",
        )
    )

    # --- OCR (capability boundary) ----------------------------------------
    settings = get_settings()
    if classification.needs_ocr or not parsed.blocks:
        if settings.ocr_provider == "none":
            result.stages.append(
                StageOutcome(
                    "ocr",
                    CapabilityState.NOT_CONFIGURED,
                    detail=(
                        "This document has no usable text layer. No OCR provider is "
                        "configured, so its text was not recovered. It is recorded with "
                        "its metadata and queued for review."
                    ),
                    required_env=["OCR_PROVIDER"],
                )
            )
        else:
            result.stages.append(
                StageOutcome(
                    "ocr",
                    CapabilityState.NOT_IMPLEMENTED,
                    detail=f"OCR_PROVIDER={settings.ocr_provider} is selected but the "
                    "provider adapter is not implemented yet.",
                )
            )

    # --- persist the document row -----------------------------------------
    title = _derive_title(parsed.metadata, blob.original_filename)
    revision = _derive_revision(parsed.head_text)
    issued_on = _derive_issue_date(parsed.head_text)

    await db.execute(
        """
        INSERT INTO documents (
            doc_id, content_hash, title, doc_type, doc_type_confidence, doc_type_method,
            data_class, source_system, source_path, original_filename, mime_type,
            byte_size, page_count, has_text_layer, revision, issued_on, blob_path,
            ingest_job_id, metadata
        ) VALUES (
            %(doc_id)s, %(content_hash)s, %(title)s, %(doc_type)s, %(conf)s, %(method)s,
            %(data_class)s, %(source_system)s, %(source_path)s, %(filename)s, %(mime)s,
            %(byte_size)s, %(page_count)s, %(has_text)s, %(revision)s, %(issued_on)s,
            %(blob_path)s, %(job_id)s, %(metadata)s
        )
        ON CONFLICT (doc_id) DO NOTHING
        """,
        {
            "doc_id": doc_id,
            "content_hash": blob.content_hash,
            "title": title,
            "doc_type": classification.doc_type.value,
            "conf": classification.confidence,
            "method": classification.method,
            "data_class": data_class.value,
            "source_system": source_system,
            "source_path": source_path,
            "filename": blob.original_filename,
            "mime": blob.mime_type,
            "byte_size": blob.byte_size,
            "page_count": parsed.page_count,
            "has_text": parsed.has_text_layer,
            "revision": revision,
            "issued_on": issued_on,
            "blob_path": blob.path,
            "job_id": job_id,
            "metadata": json.dumps(
                {
                    **parsed.metadata,
                    "classification": classification.to_dict(),
                    "parser_warnings": parsed.warnings[:20],
                }
            ),
        },
    )
    result.doc_id = doc_id

    await graph_writer.upsert_document(
        {
            "doc_id": doc_id,
            "title": title,
            "doc_type": classification.doc_type.value,
            "data_class": data_class.value,
            "source_system": source_system,
            "content_hash": blob.content_hash,
            "revision": revision,
            "issued_on": issued_on.isoformat() if issued_on else None,
            "page_count": parsed.page_count,
            "is_current": True,
        }
    )

    # --- chunk -------------------------------------------------------------
    chunks = chunk_document(
        parsed, doc_type=classification.doc_type, doc_title=title, revision=revision
    )
    await _persist_chunks(doc_id, chunks, data_class)
    result.chunks = len(chunks)
    result.stages.append(
        StageOutcome(
            "chunk",
            CapabilityState.AVAILABLE,
            items=len(chunks),
            detail=f"strategy={_strategy_name(classification.doc_type)}",
        )
    )

    # --- lexical index (always available: no provider required) ------------
    indexed_terms = await index_chunk_terms(
        [
            {
                "chunk_id": make_chunk_id(doc_id, c.ordinal),
                "doc_id": doc_id,
                "text": c.embedding_text,
                "data_class": data_class.value,
            }
            for c in chunks
        ]
    )
    result.stages.append(
        StageOutcome(
            "lexical_index",
            CapabilityState.AVAILABLE,
            items=indexed_terms,
            detail="Okapi BM25 postings written",
        )
    )

    # --- dense index (capability boundary) ---------------------------------
    provider = get_embedding_provider()
    embed_outcome = await provider.embed_chunks(
        doc_id=doc_id,
        chunks=[
            {"chunk_id": make_chunk_id(doc_id, c.ordinal), "text": c.embedding_text} for c in chunks
        ],
    )
    result.stages.append(
        StageOutcome(
            "embed",
            embed_outcome.state,
            items=embed_outcome.count,
            detail=embed_outcome.detail,
            required_env=embed_outcome.required_env,
        )
    )

    # --- extract + resolve -------------------------------------------------
    index = AssetIndex(await _load_asset_index())
    all_mentions: list[dict[str, Any]] = []
    asset_links: dict[str, dict[str, Any]] = {}
    sibling_pairs: set[tuple[str, str]] = set()
    review_items: list[dict[str, Any]] = []
    new_assets: dict[str, dict[str, Any]] = {}
    degradation_cues = 0

    for chunk in chunks:
        cid = make_chunk_id(doc_id, chunk.ordinal)
        extraction = extract_all(chunk.text)
        degradation_cues += len(extraction.degradation_cues)
        if not extraction.tags:
            continue

        outcome = resolve_mentions(
            extraction.tags,
            index,
            doc_id=doc_id,
            data_class=data_class.value,
        )
        review_items.extend(outcome.review_items)
        sibling_pairs |= outcome.sibling_pairs
        new_assets.update({k: v for k, v in outcome.new_assets.items() if v})

        for decision in outcome.decisions:
            mention = decision.mention
            all_mentions.append(
                {
                    "mention_id": make_mention_id(cid, mention.surface_form, mention.char_start),
                    "chunk_id": cid,
                    "doc_id": doc_id,
                    "surface_form": mention.surface_form,
                    "normalised": mention.normalised,
                    "canonical_tag": decision.canonical_tag,
                    "tag_kind": mention.tag_kind,
                    "char_start": mention.char_start,
                    "char_end": mention.char_end,
                    "page": chunk.page_from,
                    "extractor": mention.extractor,
                    "extractor_confidence": mention.confidence,
                    "resolved_asset_id": decision.asset_id,
                    "resolution_score": decision.score,
                    "resolution_method": decision.method,
                    "resolution_reason": decision.reason,
                    "resolution_action": decision.action,
                    "needs_review": decision.needs_review,
                    "data_class": data_class.value,
                }
            )
            link = asset_links.setdefault(
                decision.canonical_tag,
                {
                    "canonical_tag": decision.canonical_tag,
                    "page": chunk.page_from,
                    "confidence": decision.score,
                    "method": decision.method,
                    "mention_count": 0,
                    "evidence_chunks": [],
                },
            )
            link["mention_count"] += 1
            if cid not in link["evidence_chunks"]:
                link["evidence_chunks"].append(cid)

    await _persist_assets(new_assets.values())
    await _persist_mentions(all_mentions)
    await _persist_review_items(doc_id, review_items)

    result.mentions = len(all_mentions)
    result.assets = len(new_assets)
    result.stages.append(
        StageOutcome(
            "extract",
            CapabilityState.AVAILABLE,
            items=len(all_mentions),
            detail=f"deterministic extractors (regex grammar + gazetteer); "
            f"{degradation_cues} degradation cues found",
        )
    )
    if settings.llm_provider == "none":
        result.stages.append(
            StageOutcome(
                "extract_llm",
                CapabilityState.NOT_CONFIGURED,
                detail=(
                    "Failure modes, causes and obligations expressed in prose require a "
                    "generation provider. Deterministic tag/date/quantity extraction ran and "
                    "is unaffected."
                ),
                required_env=["LLM_PROVIDER", "LLM_MODEL", "OPENAI_API_KEY or ANTHROPIC_API_KEY"],
            )
        )
    result.stages.append(
        StageOutcome(
            "resolve",
            CapabilityState.AVAILABLE,
            items=len(new_assets),
            detail=f"{len(sibling_pairs)} sibling links, {len(review_items)} flagged for review",
        )
    )

    # --- graph upsert ------------------------------------------------------
    await graph_writer.upsert_chunks(
        doc_id,
        [
            {
                "chunk_id": make_chunk_id(doc_id, c.ordinal),
                "ordinal": c.ordinal,
                "page_from": c.page_from,
                "section_path": c.section_path,
                "kind": c.kind,
            }
            for c in chunks
        ],
        data_class.value,
    )
    await graph_writer.upsert_assets(list(new_assets.values()))
    await graph_writer.upsert_mentions(all_mentions)
    edges = await graph_writer.link_document_to_assets(
        doc_id=doc_id,
        links=list(asset_links.values()),
        data_class=data_class.value,
        source_system=source_system,
        issued_on=issued_on.isoformat() if issued_on else None,
    )
    edges += await graph_writer.link_siblings(
        [
            {
                "a": a,
                "b": b,
                "reason": "identical class and sequence, different item suffix "
                "(parallel train / duty-standby pair)",
                "confidence": 0.9,
            }
            for a, b in sorted(sibling_pairs)
        ]
    )

    # --- structured records ------------------------------------------------
    records_written = await _persist_records(
        doc_id=doc_id,
        parsed_records=parsed.records,
        doc_type=classification.doc_type,
        data_class=data_class,
        source_system=source_system,
        index=index,
    )
    if parsed.records:
        result.stages.append(
            StageOutcome(
                "records",
                CapabilityState.AVAILABLE,
                items=records_written,
                detail=f"{records_written} structured records persisted",
            )
        )

    result.edges = edges
    result.status = "ingested"
    result.stages.append(
        StageOutcome(
            "graph_upsert",
            CapabilityState.AVAILABLE,
            items=edges,
            detail="idempotent MERGE with provenance on every asserted fact",
        )
    )

    await db.execute(
        "INSERT INTO events (event_type, doc_id, job_id, payload) VALUES (%s, %s, %s, %s)",
        (
            "document.ingested",
            doc_id,
            job_id,
            json.dumps(
                {
                    "doc_type": classification.doc_type.value,
                    "chunks": len(chunks),
                    "assets": sorted(asset_links.keys()),
                    "data_class": data_class.value,
                }
            ),
        ),
    )
    await bus.publish_event(
        "graph.changed",
        {
            "doc_id": doc_id,
            "job_id": job_id,
            "title": title,
            "doc_type": classification.doc_type.value,
            "assets": sorted(asset_links.keys()),
            "chunks": len(chunks),
            "data_class": data_class.value,
        },
    )

    log.info(
        "ingest.completed",
        doc_id=doc_id,
        doc_type=classification.doc_type.value,
        chunks=len(chunks),
        mentions=len(all_mentions),
        new_assets=len(new_assets),
        edges=edges,
        elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
    )
    return result


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


async def _persist_chunks(doc_id: str, chunks: list[Chunk], data_class: DataClass) -> None:
    if not chunks:
        return
    async with db.connection() as conn, conn.cursor() as cur:
        for chunk in chunks:
            await cur.execute(
                """
                INSERT INTO document_chunks (
                    chunk_id, doc_id, ordinal, text, context_header, section_path,
                    page_from, page_to, char_start, char_end, bbox, chunk_kind,
                    token_count, data_class
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    text = EXCLUDED.text,
                    context_header = EXCLUDED.context_header,
                    section_path = EXCLUDED.section_path,
                    token_count = EXCLUDED.token_count
                """,
                (
                    make_chunk_id(doc_id, chunk.ordinal),
                    doc_id,
                    chunk.ordinal,
                    chunk.text,
                    chunk.context_header,
                    chunk.section_path,
                    chunk.page_from,
                    chunk.page_to,
                    chunk.char_start,
                    chunk.char_end,
                    json.dumps(chunk.bbox) if chunk.bbox else None,
                    chunk.kind,
                    chunk.token_estimate,
                    data_class.value,
                ),
            )


async def _persist_assets(assets: Any) -> None:
    rows = [a for a in assets if a.get("canonical_tag")]
    if not rows:
        return
    async with db.connection() as conn, conn.cursor() as cur:
        for asset in rows:
            await cur.execute(
                """
                INSERT INTO assets (
                    asset_id, canonical_tag, tag_kind, class_code, class_label,
                    unit_prefix, sequence_no, item_suffix, site, data_class, first_seen_doc
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (canonical_tag) DO UPDATE SET updated_at = now()
                """,
                (
                    asset["asset_id"],
                    asset["canonical_tag"],
                    asset["tag_kind"],
                    asset.get("class_code"),
                    asset.get("class_label"),
                    asset.get("unit_prefix"),
                    asset.get("sequence_no"),
                    asset.get("item_suffix"),
                    asset.get("site"),
                    asset["data_class"],
                    asset.get("first_seen_doc"),
                ),
            )


async def _persist_mentions(mentions: list[dict[str, Any]]) -> None:
    if not mentions:
        return
    async with db.connection() as conn, conn.cursor() as cur:
        for m in mentions:
            await cur.execute(
                """
                INSERT INTO mentions (
                    mention_id, chunk_id, doc_id, surface_form, normalised, tag_kind,
                    char_start, char_end, page, extractor, extractor_confidence,
                    resolved_asset_id, resolution_score, resolution_method,
                    resolution_action, needs_review, data_class
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (mention_id) DO NOTHING
                """,
                (
                    m["mention_id"],
                    m["chunk_id"],
                    m["doc_id"],
                    m["surface_form"],
                    m["normalised"],
                    m["tag_kind"],
                    m["char_start"],
                    m["char_end"],
                    m["page"],
                    m["extractor"],
                    m["extractor_confidence"],
                    m["resolved_asset_id"],
                    m["resolution_score"],
                    m["resolution_method"],
                    m["resolution_action"],
                    m["needs_review"],
                    m["data_class"],
                ),
            )
        # Recompute the linkage counters that the dashboard reports. These are
        # calculated metrics, derived from the mentions actually stored.
        await cur.execute(
            """
            UPDATE assets a SET
                mention_count = s.mentions,
                document_count = s.docs,
                source_system_count = s.systems,
                updated_at = now()
            FROM (
                SELECT m.resolved_asset_id AS asset_id,
                       count(*)::int AS mentions,
                       count(DISTINCT m.doc_id)::int AS docs,
                       count(DISTINCT d.source_system)::int AS systems
                  FROM mentions m
                  JOIN documents d ON d.doc_id = m.doc_id
                 WHERE m.resolved_asset_id IS NOT NULL
                 GROUP BY m.resolved_asset_id
            ) s
            WHERE a.asset_id = s.asset_id
            """
        )


async def _persist_review_items(doc_id: str, items: list[dict[str, Any]]) -> None:
    if not items:
        return
    async with db.connection() as conn, conn.cursor() as cur:
        for item in items:
            await cur.execute(
                """
                INSERT INTO review_queue (kind, doc_id, subject, detail, confidence)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    item["kind"],
                    doc_id,
                    item["subject"],
                    json.dumps(item["detail"]),
                    item.get("confidence"),
                ),
            )


async def _persist_records(
    *,
    doc_id: str,
    parsed_records: list[dict[str, Any]],
    doc_type: DocumentType,
    data_class: DataClass,
    source_system: str,
    index: AssetIndex,
) -> int:
    """Write structured records to their relational tables and to the graph."""
    if not parsed_records:
        return 0

    work_orders: list[dict[str, Any]] = []
    incidents: list[dict[str, Any]] = []
    inspections: list[dict[str, Any]] = []
    # Functional location -> the equipment tags observed occupying it.
    locations: dict[str, set[str]] = {}

    for record in parsed_records:
        raw_tag = record.get("asset_tag") or record.get("functional_location")
        canonical = None
        if raw_tag:
            from services.common.tags import parse as parse_tag

            parsed_tag = parse_tag(str(raw_tag))
            canonical = parsed_tag.canonical if parsed_tag.parsed else None
        asset = index.get(canonical) if canonical else None
        asset_id = asset["asset_id"] if asset else None

        fl_tag = record.get("functional_location")
        if fl_tag and canonical:
            locations.setdefault(str(fl_tag).strip().upper(), set()).add(canonical)

        enriched = {**record, "canonical_tag": canonical, "asset_id": asset_id}
        # Ordered by how specific the evidence is. A thickness reading against a
        # condition-monitoring location is unambiguously an inspection, so it is
        # tested before the more generic identifier fields.
        if record.get("cml_id") or record.get("thickness_mm") is not None:
            inspections.append(enriched)
        elif record.get("wo_id"):
            work_orders.append(enriched)
        elif record.get("incident_id"):
            incidents.append(enriched)

    written = 0
    async with db.connection() as conn, conn.cursor() as cur:
        for w in work_orders:
            await cur.execute(
                """
                INSERT INTO work_orders (
                    wo_id, doc_id, asset_id, raw_asset_tag, functional_location, wo_type,
                    status, priority, description, long_text, as_found, as_left,
                    coded_failure_mode, opened_on, closed_on, downtime_hours, cost,
                    source_system, data_class, metadata
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (wo_id) DO NOTHING
                """,
                (
                    str(w["wo_id"]),
                    doc_id,
                    w.get("asset_id"),
                    w.get("asset_tag"),
                    w.get("functional_location"),
                    w.get("wo_type"),
                    w.get("status"),
                    w.get("priority"),
                    w.get("description"),
                    w.get("long_text"),
                    w.get("as_found"),
                    w.get("as_left"),
                    w.get("coded_failure_mode"),
                    w.get("opened_on"),
                    w.get("closed_on"),
                    w.get("downtime_hours"),
                    w.get("cost"),
                    source_system,
                    data_class.value,
                    json.dumps(w.get("extra", {})),
                ),
            )
            written += 1
        for i in incidents:
            await cur.execute(
                """
                INSERT INTO incidents (
                    incident_id, doc_id, asset_id, raw_asset_tag, title, occurred_on,
                    severity, event_type, narrative, immediate_cause, root_cause,
                    investigation_status, source_system, data_class, metadata
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (incident_id) DO NOTHING
                """,
                (
                    str(i["incident_id"]),
                    doc_id,
                    i.get("asset_id"),
                    i.get("asset_tag"),
                    i.get("description") or i.get("title"),
                    i.get("occurred_on"),
                    i.get("severity"),
                    i.get("event_type", "incident"),
                    i.get("narrative"),
                    i.get("immediate_cause"),
                    i.get("root_cause"),
                    i.get("investigation_status"),
                    source_system,
                    data_class.value,
                    json.dumps(i.get("extra", {})),
                ),
            )
            written += 1
        for s in inspections:
            inspection_id = str(
                s.get("inspection_id") or f"{doc_id}:{s.get('cml_id')}:{s.get('inspected_on')}"
            )
            s["inspection_id"] = inspection_id
            await cur.execute(
                """
                INSERT INTO inspections (
                    inspection_id, doc_id, asset_id, raw_asset_tag, cml_id, method,
                    inspected_on, thickness_mm, min_required_mm, inspector, finding,
                    source_system, data_class, metadata
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (inspection_id) DO NOTHING
                """,
                (
                    inspection_id,
                    doc_id,
                    s.get("asset_id"),
                    s.get("asset_tag"),
                    s.get("cml_id"),
                    s.get("method"),
                    s.get("inspected_on"),
                    s.get("thickness_mm"),
                    s.get("min_required_mm"),
                    s.get("inspector"),
                    s.get("finding"),
                    source_system,
                    data_class.value,
                    json.dumps(s.get("extra", {})),
                ),
            )
            written += 1

    if locations:
        await graph_writer.upsert_functional_locations(
            [
                {
                    "fl_tag": fl_tag,
                    "description": None,
                    "equipment": sorted(tags),
                    "source_doc": doc_id,
                    "data_class": data_class.value,
                }
                for fl_tag, tags in sorted(locations.items())
            ]
        )
        async with db.connection() as conn, conn.cursor() as cur:
            for fl_tag, tags in locations.items():
                await cur.execute(
                    "UPDATE assets SET functional_location = %s, updated_at = now() "
                    "WHERE canonical_tag = ANY(%s)",
                    (fl_tag, sorted(tags)),
                )

    for record_set, writer in (
        (work_orders, graph_writer.upsert_work_orders),
        (incidents, graph_writer.upsert_incidents),
        (inspections, graph_writer.upsert_inspections),
    ):
        if record_set:
            payload = [
                {**r, "source_system": source_system, "data_class": data_class.value}
                for r in record_set
            ]
            await writer(doc_id, payload)  # type: ignore[operator]

    return written


async def _load_asset_index() -> list[dict[str, Any]]:
    return await db.fetch_all(
        "SELECT asset_id, canonical_tag, tag_kind, class_code, class_label, unit_prefix, "
        "sequence_no, item_suffix, site, data_class::text AS data_class FROM assets"
    )


# ---------------------------------------------------------------------------
# Small derivations
# ---------------------------------------------------------------------------


def _strategy_name(doc_type: DocumentType) -> str:
    return {
        DocumentType.SOP: "one chunk per step, precondition carried",
        DocumentType.WORK_ORDER: "one chunk per record",
        DocumentType.INSPECTION_REPORT: "one chunk per reading row",
        DocumentType.INCIDENT_REPORT: "semantic sections",
        DocumentType.PID: "no prose chunking (topology, not text)",
    }.get(doc_type, "heading-aware prose")


def _derive_title(metadata: dict[str, Any], filename: str) -> str:
    title = str(metadata.get("title") or "").strip()
    if title and len(title) > 3:
        return title[:300]
    return Path(filename).stem.replace("_", " ").replace("-", " ").strip()[:300] or filename


_REVISION_MARKERS = ("revision", "rev.", "rev ", "issue ")


def _derive_revision(head_text: str) -> str | None:
    import re

    for marker in _REVISION_MARKERS:
        idx = head_text.lower().find(marker)
        if idx == -1:
            continue
        window = head_text[idx : idx + 40]
        match = re.search(r"(?:rev(?:ision)?|issue)\.?\s*[:#]?\s*([A-Z0-9]{1,4})", window, re.I)
        if match:
            return match.group(1)
    return None


def _derive_issue_date(head_text: str) -> date | None:
    from services.ingest.extract import extract_all as _extract

    dates = _extract(head_text[:2000]).dates
    if not dates:
        return None
    return date.fromisoformat(dates[0]["date"])
