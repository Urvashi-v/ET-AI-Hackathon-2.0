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

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from services.agents import proactive
from services.common import bus, db
from services.common.ids import chunk_id as make_chunk_id
from services.common.ids import document_id as make_document_id
from services.common.ids import mention_id as make_mention_id
from services.common.logging import get_logger
from services.common.schemas import CapabilityState, DataClass, DocumentType
from services.ingest import (
    graph_writer,
    pid,
    pid_writer,
    record_writer,
    records,
    revisions,
    storage,
)
from services.ingest.chunk import Chunk, chunk_document
from services.ingest.classify import classify
from services.ingest.embeddings import get_embedding_provider
from services.ingest.extract import extract_all
from services.ingest.llm_extract import extract_facts as llm_extract_facts
from services.ingest.llm_extract import extraction_capability
from services.ingest.ocr import get_ocr_provider, ocr_capability
from services.ingest.parsers import ParsedDocument, parse_document
from services.ingest.parsers.image_parser import blocks_from_ocr, ocr_quality_warnings
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
    processing_ms: int = 0
    pages: int | None = None
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
            "processing_ms": self.processing_ms,
            "pages": self.pages,
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
    elif parsed.page_count and parsed.has_text_layer is False:
        # No text layer is a routing decision, not a failure. The document is a
        # scan; the OCR stage below reports what actually became of it. Calling
        # this an error would make every scanned document look broken and would
        # hide the real signal when a file genuinely cannot be read.
        result.stages.append(
            StageOutcome(
                "parse",
                CapabilityState.AVAILABLE,
                items=0,
                detail=(
                    f"{parsed.page_count} page(s) with no embedded text layer: this is a "
                    "scan, routed to OCR."
                ),
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

    # --- OCR ---------------------------------------------------------------
    # A PDF whose pages carry no text layer is a scan. The parser has already
    # said so; this is where the text is actually recovered, before
    # classification, because a document nobody can read cannot be classified by
    # its content either.
    needs_ocr = blob.extension.lower() == ".pdf" and not parsed.has_text_layer
    if needs_ocr:
        parsed, ocr_stage = await _recover_with_ocr(data, parsed)
        result.stages.append(ocr_stage)
        result.warnings.extend(parsed.warnings[-4:])
    elif parsed.parser.startswith("ocr:"):
        # A raster image went through the OCR parser directly.
        report = parsed.metadata.get("ocr", {})
        result.stages.append(
            StageOutcome(
                "ocr",
                CapabilityState.AVAILABLE,
                items=int(report.get("words", 0)),
                detail=(
                    f"engine={report.get('engine')} "
                    f"mean confidence {float(report.get('mean_confidence', 0)) * 100:.0f}%"
                ),
            )
        )

    # --- classify ----------------------------------------------------------
    classification = classify(
        filename=blob.original_filename,
        extension=blob.extension,
        head_text=parsed.head_text,
        page_count=parsed.page_count,
        has_text_layer=parsed.has_text_layer,
        vector_segment_count=parsed.metadata.get("vector_objects"),
        drawing_signature=bool(parsed.metadata.get("drawing_signature")),
    )
    result.doc_type = classification.doc_type.value
    result.pages = parsed.page_count
    result.stages.append(
        StageOutcome(
            "classify",
            CapabilityState.AVAILABLE,
            detail=f"{classification.doc_type.value} "
            f"(confidence {classification.confidence:.2f}, via {classification.method})",
        )
    )

    # --- persist the document row -----------------------------------------
    title = _derive_title(parsed.metadata, blob.original_filename)
    revision = _derive_revision(parsed.head_text)
    issued_on = _derive_issue_date(parsed.head_text)
    # The identifier printed on the document, which is stable across revisions.
    # doc_id cannot serve this purpose: it is a content hash, so every revision
    # gets a different one by design.
    doc_number = revisions.derive_doc_number(
        title=title,
        filename=blob.original_filename,
        head_text=parsed.head_text,
        doc_type=classification.doc_type,
    )

    await db.execute(
        """
        INSERT INTO documents (
            doc_id, content_hash, title, doc_type, doc_type_confidence, doc_type_method,
            data_class, source_system, source_path, original_filename, mime_type,
            byte_size, page_count, has_text_layer, revision, issued_on, blob_path,
            ingest_job_id, metadata, doc_number, doc_number_method,
            parser, ocr_engine, ocr_mean_confidence, ocr_word_count,
            vector_objects, is_drawing, tables_found
        ) VALUES (
            %(doc_id)s, %(content_hash)s, %(title)s, %(doc_type)s, %(conf)s, %(method)s,
            %(data_class)s, %(source_system)s, %(source_path)s, %(filename)s, %(mime)s,
            %(byte_size)s, %(page_count)s, %(has_text)s, %(revision)s, %(issued_on)s,
            %(blob_path)s, %(job_id)s, %(metadata)s, %(doc_number)s, %(doc_number_method)s,
            %(parser)s, %(ocr_engine)s, %(ocr_conf)s, %(ocr_words)s,
            %(vector_objects)s, %(is_drawing)s, %(tables_found)s
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
            "doc_number": doc_number.value if doc_number else None,
            "doc_number_method": doc_number.method if doc_number else None,
            "blob_path": blob.path,
            "job_id": job_id,
            "parser": parsed.parser,
            "ocr_engine": (parsed.metadata.get("ocr") or {}).get("engine"),
            "ocr_conf": (parsed.metadata.get("ocr") or {}).get("mean_confidence"),
            "ocr_words": (parsed.metadata.get("ocr") or {}).get("words"),
            "vector_objects": parsed.metadata.get("vector_objects"),
            "is_drawing": bool(parsed.metadata.get("drawing_signature")),
            "tables_found": int(parsed.metadata.get("tables_found") or 0),
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

    # --- revision lineage --------------------------------------------------
    # Recomputed for the whole series, because a revision can arrive out of
    # order and an incremental update would leave the chain wrong. Runs before
    # the graph write so the Document node carries the settled currency rather
    # than an optimistic True that a later pass has to correct.
    revision_report: dict[str, Any] = {"status": "no_doc_number"}
    if doc_number:
        revision_report = await revisions.reconcile(doc_number.value)
        await graph_writer.link_revision_chain(revision_report)
    result.stages.append(
        StageOutcome(
            "revisions",
            CapabilityState.AVAILABLE,
            items=int(revision_report.get("documents") or 0),
            detail=_revision_detail(doc_number, revision_report),
        )
    )

    settled = await db.fetch_one(
        "SELECT is_current, superseded_by, valid_from, valid_to FROM documents WHERE doc_id = %s",
        (doc_id,),
    )
    await graph_writer.upsert_document(
        {
            "doc_id": doc_id,
            "title": title,
            "doc_type": classification.doc_type.value,
            "data_class": data_class.value,
            "source_system": source_system,
            "content_hash": blob.content_hash,
            "revision": revision,
            "doc_number": doc_number.value if doc_number else None,
            "issued_on": issued_on.isoformat() if issued_on else None,
            "page_count": parsed.page_count,
            "is_current": bool(settled["is_current"]) if settled else True,
            "valid_from": _iso(settled.get("valid_from")) if settled else None,
            "valid_to": _iso(settled.get("valid_to")) if settled else None,
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
    failure_terms: list[dict[str, Any]] = []
    llm_facts: list[dict[str, Any]] = []
    llm_calls = 0
    llm_rejected = 0
    llm_state = CapabilityState.NOT_CONFIGURED
    llm_detail: str | None = None
    llm_required_env: list[str] = []

    for chunk in chunks:
        cid = make_chunk_id(doc_id, chunk.ordinal)
        extraction = extract_all(chunk.text)
        degradation_cues += len(extraction.degradation_cues)

        # Deterministic failure vocabulary, recorded with its verbatim span so
        # the assertion is checkable against the source.
        for term in extraction.failure_terms:
            failure_terms.append({**term, "chunk_id": cid, "page": chunk.page_from})

        # Reasoning-dependent facts. Gated twice: on the provider being
        # configured at all, and on this chunk plausibly containing a causal
        # narrative, so a table of thickness readings costs nothing.
        llm_result = await llm_extract_facts(
            text=chunk.text,
            chunk_kind=chunk.kind,
            has_failure_vocabulary=bool(extraction.failure_terms),
        )
        llm_state = llm_result.state
        llm_detail = llm_result.detail
        llm_required_env = llm_result.required_env
        llm_calls += llm_result.calls
        llm_rejected += llm_result.rejected
        for fact in llm_result.facts:
            llm_facts.append({**fact.to_row(), "chunk_id": cid, "page": chunk.page_from})

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
    await _persist_extractions(
        doc_id=doc_id, failure_terms=failure_terms, llm_facts=llm_facts, data_class=data_class
    )

    result.stages.append(
        StageOutcome(
            "extract",
            CapabilityState.AVAILABLE,
            items=len(all_mentions),
            detail=(
                f"deterministic extractors (tag grammar + gazetteer); "
                f"{len(failure_terms)} failure-mode term(s), "
                f"{degradation_cues} degradation cue(s)"
            ),
        )
    )

    verified = sum(1 for f in llm_facts if f["quote_verified"])
    if llm_state is CapabilityState.AVAILABLE and llm_facts:
        result.stages.append(
            StageOutcome(
                "extract_llm",
                CapabilityState.AVAILABLE,
                items=verified,
                detail=(
                    f"{llm_calls} model call(s); {verified} fact(s) with a verified verbatim "
                    f"span, {llm_rejected} rejected for unverifiable evidence"
                ),
            )
        )
    else:
        state, detail, required_env = extraction_capability()
        result.stages.append(
            StageOutcome(
                "extract_llm",
                state if llm_state is not CapabilityState.ERROR else CapabilityState.ERROR,
                items=0,
                detail=llm_detail if llm_state is CapabilityState.ERROR else detail,
                required_env=llm_required_env or required_env,
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

    # --- structured records from prose --------------------------------------
    # Runs after the graph upsert because it links incidents to Equipment nodes
    # that stage creates. Turns an incident *document* into an incident *record*:
    # without it the corpus holds two investigated seal failures that RCA cannot
    # see, because nothing outside text search knows they exist.
    record_outcome = await _extract_structured_records(
        doc_id=doc_id,
        doc_type=classification.doc_type,
        doc_number=doc_number.value if doc_number else None,
        title=title,
        data_class=data_class,
    )
    if record_outcome is not None:
        result.stages.append(record_outcome)

    # --- P&ID digitisation --------------------------------------------------
    # Runs after the graph upsert because linking a detected tag to an Equipment
    # node needs that node to exist. Drawings only: everything else skips the
    # stage entirely rather than reporting an empty result that looks like a
    # detector finding nothing.
    if classification.doc_type is DocumentType.PID:
        pid_outcome = await _digitise_drawing(
            doc_id=doc_id, blob_path=blob.path, pages=parsed.page_count or 1, data_class=data_class
        )
        if pid_outcome is not None:
            result.stages.append(pid_outcome)

    # --- proactive matching -------------------------------------------------
    # The graph just changed. This is the trigger for the proactive path: match
    # what arrived against history, open actions, obligations and superseded
    # procedures, and raise notifications for what it finds. Real events only --
    # a document that produced no asset-linked record produces no notification.
    proactive_outcome = await _run_proactive_matching(doc_id=doc_id, title=title)
    if proactive_outcome is not None:
        result.stages.append(proactive_outcome)

    # Per-document counters the ingestion dashboard reports. Written here rather
    # than recomputed by the API so the numbers on screen are the numbers this
    # run actually produced, including how long it took.
    result.processing_ms = int((time.perf_counter() - started) * 1000)
    await db.execute(
        """
        UPDATE documents SET
            processing_ms       = %s,
            chunk_count         = %s,
            mention_count       = %s,
            graph_nodes_created = %s,
            graph_edges_created = %s
        WHERE doc_id = %s
        """,
        (
            result.processing_ms,
            len(chunks),
            len(all_mentions),
            len(new_assets),
            edges,
            doc_id,
        ),
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


async def _recover_with_ocr(
    data: bytes, parsed: ParsedDocument
) -> tuple[ParsedDocument, StageOutcome]:
    """Run OCR over a PDF that has no usable text layer.

    Returns the parsed document either enriched with recognised text, or
    unchanged with a stage outcome explaining why it could not be. It never
    invents text: with no provider configured the document keeps its metadata,
    keeps ``has_text_layer=False``, and is queued for review.
    """
    state, detail, required_env = ocr_capability()
    if state is not CapabilityState.AVAILABLE:
        return parsed, StageOutcome("ocr", state, detail=detail, required_env=required_env)

    result = await asyncio.to_thread(get_ocr_provider().ocr_pdf, data)
    if result.state is not CapabilityState.AVAILABLE:
        return parsed, StageOutcome(
            "ocr",
            result.state,
            detail=result.detail,
            required_env=result.required_env,
        )

    blocks = blocks_from_ocr(result.pages)
    if not blocks:
        return parsed, StageOutcome(
            "ocr",
            CapabilityState.AVAILABLE,
            items=0,
            detail=(
                f"engine={result.engine} recognised no text. The document is recorded "
                "with its metadata and queued for review."
            ),
        )

    report = result.quality_report()
    parsed.blocks = blocks
    parsed.parser = f"{parsed.parser}+ocr:{result.engine}"
    parsed.metadata["ocr"] = report
    # `has_text_layer` stays False on purpose: these characters were recognised,
    # not read. Anything downstream that treats OCR output as equivalent to an
    # embedded text layer is making an assumption it should not.
    parsed.warnings.extend(ocr_quality_warnings(result.pages))

    return parsed, StageOutcome(
        "ocr",
        CapabilityState.AVAILABLE,
        items=int(report["words"]),
        detail=(
            f"engine={result.engine} recovered {report['words']} words across "
            f"{report['pages']} page(s) at {float(report['mean_confidence']) * 100:.0f}% "
            f"mean confidence ({report['low_confidence_words']} low-confidence)"
        ),
    )


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
                    token_count, data_class, extraction_method, extraction_confidence
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    text = EXCLUDED.text,
                    context_header = EXCLUDED.context_header,
                    section_path = EXCLUDED.section_path,
                    token_count = EXCLUDED.token_count,
                    bbox = EXCLUDED.bbox,
                    extraction_method = EXCLUDED.extraction_method,
                    extraction_confidence = EXCLUDED.extraction_confidence
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
                    chunk.extraction_method,
                    chunk.extraction_confidence,
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


async def _persist_extractions(
    *,
    doc_id: str,
    failure_terms: list[dict[str, Any]],
    llm_facts: list[dict[str, Any]],
    data_class: DataClass,
) -> None:
    """Record every asserted fact with the span it was based on.

    Rejected extractions are stored too, with the reason. Discarding them would
    make the verbatim-validation rate unmeasurable -- and "0.0% of asserted facts
    lack a verified source span" is only a number if the denominator is kept.
    """
    if not failure_terms and not llm_facts:
        return

    rows: list[tuple[Any, ...]] = []
    for term in failure_terms:
        rows.append(
            (
                doc_id,
                term.get("chunk_id"),
                "failure_mode",
                term["extractor"],
                json.dumps(
                    {"failure_mode_code": term["failure_mode_code"], "phrase": term["phrase"]}
                ),
                term["quote"],
                True,  # located by literal search, so verified by construction
                None,
                term["confidence"],
                term["char_start"],
                term["char_end"],
                term.get("page"),
                data_class.value,
            )
        )
    for fact in llm_facts:
        rows.append(
            (
                doc_id,
                fact.get("chunk_id"),
                fact["kind"],
                fact["extractor"],
                json.dumps(fact["payload"]),
                fact["evidence_quote"],
                fact["quote_verified"],
                fact["reject_reason"],
                fact["confidence"],
                fact["char_start"],
                fact["char_end"],
                fact.get("page"),
                # A model inference is never an audit record, whatever the
                # document it came from was classified as.
                DataClass.MODEL_DERIVED.value,
            )
        )

    async with db.connection() as conn, conn.cursor() as cur:
        await cur.executemany(
            """
            INSERT INTO extractions (
                doc_id, chunk_id, kind, extractor, payload, evidence_quote,
                quote_verified, reject_reason, confidence, char_start, char_end,
                page, data_class
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            rows,
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


async def _extract_structured_records(
    *,
    doc_id: str,
    doc_type: DocumentType,
    doc_number: str | None,
    title: str,
    data_class: DataClass,
) -> StageOutcome | None:
    """Turn an incident or MOC document into graph nodes, deterministically.

    Chunks are read back from Postgres rather than passed in, so extraction sees
    exactly the text that was stored -- if chunking changed the section paths, the
    extractor finds out here rather than producing records that disagree with the
    passages a citation would open.

    Returns None for document types this does not apply to, so the ingestion
    dashboard shows the stage only where it means something.
    """
    if not records.is_extractable(doc_type):
        return None

    chunk_rows = await db.fetch_all(
        "SELECT chunk_id, text, section_path, page_from FROM document_chunks "
        "WHERE doc_id = %s ORDER BY ordinal",
        (doc_id,),
    )
    if not chunk_rows:
        return None

    try:
        if doc_type is DocumentType.INCIDENT_REPORT:
            incident = records.extract_incident(
                doc_id=doc_id, title=title, doc_number=doc_number, chunks=chunk_rows
            )
            if incident is None:
                return StageOutcome(
                    "structured_records",
                    CapabilityState.AVAILABLE,
                    items=0,
                    detail=(
                        "No incident record extracted: the document carries neither a cause "
                        "statement nor a corrective action under a recognised heading. "
                        "Reported rather than guessed."
                    ),
                )
            written = await record_writer.write_incident(incident, data_class=data_class)
            missing = (
                f"; fields not stated: {', '.join(incident.fields_missing)}"
                if incident.fields_missing
                else ""
            )
            return StageOutcome(
                "structured_records",
                CapabilityState.AVAILABLE,
                items=1,
                detail=(
                    f"{incident.incident_id}: {written['asset_links']} asset link(s), "
                    f"{written['corrective_actions']} corrective action(s), "
                    f"{written['related_incidents']} related incident(s){missing}"
                ),
            )

        change = records.extract_change(
            doc_id=doc_id, title=title, doc_number=doc_number, chunks=chunk_rows
        )
        if change is None:
            return StageOutcome(
                "structured_records",
                CapabilityState.AVAILABLE,
                items=0,
                detail="No change record extracted: no MOC number found.",
            )
        written = await record_writer.write_change(change, data_class=data_class)
        return StageOutcome(
            "structured_records",
            CapabilityState.AVAILABLE,
            items=1,
            detail=f"{change.moc_id}: {written['asset_links']} asset link(s)",
        )
    except Exception as exc:
        # A record that will not extract must not fail the ingestion of a
        # document that parsed, chunked and indexed correctly.
        log.error("records.extraction_failed", doc_id=doc_id, error=str(exc))
        return StageOutcome(
            "structured_records",
            CapabilityState.ERROR,
            items=0,
            detail=f"{type(exc).__name__}: {str(exc)[:200]}",
        )


async def _digitise_drawing(
    *, doc_id: str, blob_path: str, pages: int, data_class: DataClass
) -> StageOutcome | None:
    """Detect tags, instruments and lines on a drawing, and link them to assets.

    The capability this unlocks is not "we read the drawing" — it is that
    ``P-101B`` on the sheet becomes the *same node* as ``P-101B`` in the incident
    report, with coordinates. Everything the viewer highlights depends on it.

    Failures are contained: a drawing that will not digitise has still parsed,
    chunked, indexed and reached the graph, and losing the overlay is a smaller
    problem than losing the document.
    """
    try:
        path = storage.resolve_blob(blob_path)
    except Exception as exc:
        log.error("pid.blob_unresolvable", doc_id=doc_id, error=str(exc))
        return None

    # Split rather than one dict: mixing counters and a set gives the dict a
    # value type of `object`, and every arithmetic and set operation below then
    # has to be either cast or ignored.
    totals: dict[str, int] = {"detections": 0, "linked": 0, "connections": 0}
    unlinked: set[str] = set()
    errors: list[str] = []

    for page in range(1, max(1, pages) + 1):
        try:
            result = pid.digitise_page(str(path), page)
            written = await pid_writer.write(result, doc_id=doc_id, data_class=data_class)
            totals["detections"] += written["detections"]
            totals["linked"] += written["linked_assets"]
            totals["connections"] += written["connections"]
            unlinked.update(written["unlinked_tags"])
        except Exception as exc:
            errors.append(f"page {page}: {type(exc).__name__}")
            log.error("pid.page_failed", doc_id=doc_id, page=page, error=str(exc))

    detail = (
        f"{totals['detections']} detection(s) across {pages} page(s); "
        f"{totals['linked']} linked to canonical assets; "
        f"{totals['connections']} connection(s) recovered"
    )
    if unlinked:
        # Named, not buried. A tag drawn on the sheet that the corpus has never
        # ingested is the gap between what the plant has drawn and what it has
        # recorded, and it is one of the more useful things this finds.
        detail += f". Tags with no matching asset: {', '.join(sorted(unlinked))}"
    if errors:
        detail += f". Failed: {'; '.join(errors)}"

    return StageOutcome(
        "pid_digitisation",
        CapabilityState.ERROR if errors and not totals["detections"] else CapabilityState.AVAILABLE,
        items=totals["detections"],
        detail=detail + ".",
    )


async def _run_proactive_matching(*, doc_id: str, title: str) -> StageOutcome | None:
    """Fire the proactive matchers for whatever this document just asserted.

    Scoped to the assets this document actually touched, rather than the whole
    estate: ingesting one incident report should not re-evaluate every pump in
    the plant, and a notification storm on bulk ingest is how the feature gets
    turned off.

    Failures here never fail the ingest. The document parsed, chunked, indexed
    and reached the graph; not managing to raise a notification about it is a
    lesser problem, and the endpoint can be re-run.
    """
    rows = await db.fetch_all(
        """
        SELECT a.canonical_tag AS asset_tag, i.incident_id AS ref_id,
               coalesce(i.root_cause, i.immediate_cause, i.title) AS description,
               'incident.recorded' AS event_type
          FROM incidents i
          JOIN assets a ON a.asset_id = i.asset_id
         WHERE i.doc_id = %(doc_id)s
        UNION ALL
        SELECT a.canonical_tag, w.wo_id,
               coalesce(w.as_found, w.description, w.wo_id),
               'work_order.recorded'
          FROM work_orders w
          JOIN assets a ON a.asset_id = w.asset_id
         WHERE w.doc_id = %(doc_id)s
        """,
        {"doc_id": doc_id},
    )
    if not rows:
        return None

    # One evaluation per asset, using its most recent triggering record. A CMMS
    # export carrying fifteen work orders on one pump is one event about that
    # pump, not fifteen.
    by_asset: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_asset.setdefault(row["asset_tag"], row)

    raised = 0
    errors = 0
    for asset_tag, row in by_asset.items():
        try:
            candidates = await proactive.evaluate_event(
                event_type=row["event_type"],
                asset_tag=asset_tag,
                description=row["description"] or "",
                ref_id=row["ref_id"],
                doc_id=doc_id,
            )
            raised += len(await proactive.raise_notifications(candidates))
        except Exception as exc:
            errors += 1
            log.error("proactive.matching_failed", doc_id=doc_id, asset=asset_tag, error=str(exc))

    detail = (
        f"{len(by_asset)} asset(s) evaluated against history, open actions, obligations and "
        f"superseded procedures; {raised} notification(s) raised"
    )
    if errors:
        detail += f"; {errors} evaluation(s) failed"
    return StageOutcome(
        "proactive_matching",
        CapabilityState.ERROR if errors and not raised else CapabilityState.AVAILABLE,
        items=raised,
        detail=detail + ".",
    )


def _iso(value: Any) -> str | None:
    """Dates cross into Cypher as ISO strings; ``date()`` parses them there."""
    return value.isoformat() if value is not None else None


def _revision_detail(doc_number: Any, report: dict[str, Any]) -> str:
    """What the ingestion dashboard shows for the revision stage."""
    if not doc_number:
        return (
            "No document number found, so this document forms no revision series. "
            "Work-order and sensor exports legitimately have none."
        )
    status = report.get("status")
    if status == "conflict":
        return f"{doc_number.value}: {report.get('note')}"
    superseded = int(report.get("superseded") or 0)
    if superseded:
        return (
            f"{doc_number.value}: {report.get('documents')} revision(s), {superseded} "
            f"superseded, ordered by {report.get('basis')}"
        )
    return f"{doc_number.value}: the only revision held (via {doc_number.method})"


def _derive_title(metadata: dict[str, Any], filename: str) -> str:
    title = str(metadata.get("title") or "").strip()
    if title and len(title) > 3:
        return title[:300]
    return Path(filename).stem.replace("_", " ").replace("-", " ").strip()[:300] or filename


#: A document's own revision is stated as a *field* in its header block --
#: "Revision: 4", "Rev. 3", "| Issue | 2 |". A reference to some *other*
#: document's revision appears mid-sentence: "the startup procedure had been
#: revised to SOP-4412 revision 3", "SOP-4412 | Review discharge pressure limit |
#: DONE in revision 3".
#:
#: The earlier version searched the whole head text for the first occurrence of
#: the word and took the number after it, so every incident report that mentioned
#: the SOP it was about inherited that SOP's revision number. Two incident
#: reports and one MOC in the current corpus carried a revision they do not have,
#: and the revision machinery treats that field as ordering evidence -- so the
#: consequence was not cosmetic: documents can be declared superseded on the
#: strength of a number that belongs to a different document.
#:
#: Anchoring to the start of a line is what separates the two, and it is
#: deterministic: no model, no threshold. Markdown and table decoration may
#: precede it, nothing else may.
_REVISION_FIELD = re.compile(
    r"""
    (?:
        # (a) a labelled field: the colon is what makes it this document's own
        #     metadata rather than a mention of someone else's revision.
        (?i:rev(?:ision)?|issue) [ \t]* : [ \t]*
        (?P<labelled> \d{1,3}[A-Z]? | [A-Z] )
        (?![^\s|])                       # not the head of a longer token
      |
        # (b) the field occupies its own line, with nothing else on it.
        ^[ \t]* [|>#*-]{0,3} [ \t]*
        (?i:rev(?:ision)?|issue) \.? [ \t]* [:|#]? [ \t]*
        (?P<alone> \d{1,3}[A-Z]? | [A-Z] )
        [ \t]* \|? [ \t]* $
    )
""",
    re.MULTILINE | re.VERBOSE,
)

#: Only the header block is considered. A revision field that appears three pages
#: in is a reference to something else, whatever it looks like.
_REVISION_HEADER_CHARS = 1200


def _derive_revision(head_text: str) -> str | None:
    """The revision printed on *this* document, or None.

    None is the correct answer for most documents. An incident report does not
    have a revision, and inventing one for it is worse than leaving the column
    empty -- `order_revisions` uses this field as ordering evidence.
    """
    match = _REVISION_FIELD.search(head_text[:_REVISION_HEADER_CHARS])
    return (match.group("labelled") or match.group("alone")) if match else None


def _derive_issue_date(head_text: str) -> date | None:
    dates = extract_all(head_text[:2000]).dates
    if not dates:
        return None
    return date.fromisoformat(dates[0]["date"])
