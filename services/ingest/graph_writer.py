"""Idempotent graph upserts with provenance.

Every write here is a ``MERGE`` on a natural key, so re-ingesting a document
updates the graph instead of duplicating it. That property is what makes the
whole system safe to re-run, and it is the reason document ids are derived from
content hashes rather than generated.

Every asserted fact carries its evidence pointer -- source document, page,
extraction confidence, method, and the timestamp of assertion. Facts that a
revision can invalidate additionally carry ``valid_from`` / ``valid_to``. This
is not bookkeeping for its own sake: it is what makes a citation resolvable and
an audit possible, and it is what lets the compliance feature distinguish a
machine inference from a human attestation.
"""

from __future__ import annotations

from typing import Any

from services.common import graph
from services.common.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Cypher
# ---------------------------------------------------------------------------

_UPSERT_DOCUMENT = """
MERGE (d:Document {doc_id: $doc_id})
  ON CREATE SET d.created_at = datetime()
SET d.title          = $title,
    d.doc_type       = $doc_type,
    d.data_class     = $data_class,
    d.source_system  = $source_system,
    d.content_hash   = $content_hash,
    d.revision       = $revision,
    d.doc_number     = $doc_number,
    d.issued_on      = CASE WHEN $issued_on IS NULL THEN NULL ELSE date($issued_on) END,
    d.valid_from     = CASE WHEN $valid_from IS NULL THEN NULL ELSE date($valid_from) END,
    d.valid_to       = CASE WHEN $valid_to IS NULL THEN NULL ELSE date($valid_to) END,
    d.page_count     = $page_count,
    d.is_current     = $is_current,
    d.licence_note   = $licence_note,
    d.updated_at     = datetime()
RETURN d.doc_id AS doc_id
"""

#: Supersession, written as an explicit edge rather than inferred at read time.
#:
#: The chain is rebuilt from scratch on every reconciliation -- old SUPERSEDES
#: edges within the series are deleted first -- because a revision can arrive out
#: of order and leave a stale edge that no incremental update would notice. The
#: direction is newer-SUPERSEDES-older, so "what replaced this?" is a single hop
#: backwards and "is this the latest?" is the absence of an incoming edge.
_LINK_REVISIONS = """
UNWIND $pairs AS pair
MATCH (older:Document {doc_id: pair.older})
MATCH (newer:Document {doc_id: pair.newer})
MERGE (newer)-[r:SUPERSEDES]->(older)
SET r.basis      = $basis,
    r.data_class = 'model_derived',
    r.asserted_at = datetime()
RETURN count(r) AS links
"""

#: Removes supersession edges inside one series before rebuilding it, so a
#: correction (a mis-ordered revision, a document deleted) cannot leave a
#: contradictory edge behind.
_CLEAR_REVISIONS = """
MATCH (a:Document {doc_number: $doc_number})-[r:SUPERSEDES]->(b:Document {doc_number: $doc_number})
DELETE r
RETURN count(r) AS removed
"""

_UPSERT_CHUNK = """
UNWIND $chunks AS c
MERGE (chunk:Chunk {chunk_id: c.chunk_id})
  ON CREATE SET chunk.created_at = datetime()
SET chunk.doc_id       = $doc_id,
    chunk.ordinal      = c.ordinal,
    chunk.page         = c.page_from,
    chunk.section_path = c.section_path,
    chunk.kind         = c.kind,
    chunk.data_class   = $data_class
WITH chunk
MATCH (d:Document {doc_id: $doc_id})
MERGE (d)-[:HAS_CHUNK]->(chunk)
MERGE (chunk)-[:PART_OF]->(d)
RETURN count(chunk) AS chunks
"""

_UPSERT_ASSET = """
UNWIND $assets AS a
MERGE (e:Equipment {canonical_tag: a.canonical_tag})
  ON CREATE SET e.created_at = datetime(),
                e.first_seen_doc = a.first_seen_doc
SET e.asset_id    = a.asset_id,
    e.tag_kind    = a.tag_kind,
    e.class_code  = a.class_code,
    e.class_label = a.class_label,
    e.unit_prefix = a.unit_prefix,
    e.sequence_no = a.sequence_no,
    e.item_suffix = a.item_suffix,
    e.site        = a.site,
    e.data_class  = a.data_class,
    e.updated_at  = datetime()
WITH e, a
// Link the asset to its class in the taxonomy subgraph where one exists.
OPTIONAL MATCH (cls:EquipmentClass {code: a.class_code})
FOREACH (_ IN CASE WHEN cls IS NULL THEN [] ELSE [1] END |
    MERGE (e)-[:INSTANCE_OF]->(cls))
RETURN count(DISTINCT e) AS assets
"""

_UPSERT_INSTRUMENT = """
UNWIND $instruments AS a
MERGE (i:Instrument {tag: a.canonical_tag})
  ON CREATE SET i.created_at = datetime()
SET i.asset_id   = a.asset_id,
    i.variable   = a.class_code,
    i.function   = a.class_label,
    i.loop_no    = a.sequence_no,
    i.data_class = a.data_class,
    i.updated_at = datetime()
WITH i, a
MERGE (loop:Loop {loop_no: a.sequence_no})
  ON CREATE SET loop.created_at = datetime(), loop.data_class = a.data_class
MERGE (i)-[:BELONGS_TO_LOOP]->(loop)
RETURN count(DISTINCT i) AS instruments
"""

#: Mentions are stored as their own nodes, separate from the entities they
#: resolve to. The RESOLVES_TO edge carries the score, the method and the reason,
#: so a bad resolution can be inspected and reversed without re-ingesting.
_UPSERT_MENTIONS = """
UNWIND $mentions AS m
MERGE (mention:Mention {mention_id: m.mention_id})
  ON CREATE SET mention.created_at = datetime()
SET mention.surface_form = m.surface_form,
    mention.normalised   = m.normalised,
    mention.tag_kind     = m.tag_kind,
    mention.page         = m.page,
    mention.char_start   = m.char_start,
    mention.char_end     = m.char_end,
    mention.extractor    = m.extractor,
    mention.confidence   = m.extractor_confidence,
    mention.data_class   = m.data_class
WITH mention, m
MATCH (chunk:Chunk {chunk_id: m.chunk_id})
MERGE (mention)-[:APPEARS_IN]->(chunk)
WITH mention, m
OPTIONAL MATCH (eq:Equipment {canonical_tag: m.canonical_tag})
OPTIONAL MATCH (inst:Instrument {tag: m.canonical_tag})
WITH mention, m, coalesce(eq, inst) AS target
WHERE target IS NOT NULL
MERGE (mention)-[r:RESOLVES_TO]->(target)
SET r.confidence  = m.resolution_score,
    r.method      = m.resolution_method,
    r.reason      = m.resolution_reason,
    r.needs_review= m.needs_review,
    r.resolved_at = datetime()
RETURN count(r) AS resolved
"""

#: The DESCRIBES edge is the one that makes cross-document linkage visible: a
#: 2024 inspection report attaching itself to a pump previously known only from
#: a 2018 drawing. Its provenance properties are what the graph explorer renders
#: when an edge is clicked.
_LINK_DOCUMENT_ASSET = """
UNWIND $links AS l
MATCH (d:Document {doc_id: $doc_id})
OPTIONAL MATCH (eq:Equipment {canonical_tag: l.canonical_tag})
OPTIONAL MATCH (inst:Instrument {tag: l.canonical_tag})
WITH d, l, coalesce(eq, inst) AS target
WHERE target IS NOT NULL
MERGE (d)-[r:DESCRIBES]->(target)
  ON CREATE SET r.created_at = datetime()
SET r.page          = l.page,
    r.confidence    = l.confidence,
    r.method        = l.method,
    r.mention_count = l.mention_count,
    r.data_class    = $data_class,
    r.source_system = $source_system,
    r.evidence_chunks = l.evidence_chunks,
    r.valid_from    = CASE WHEN $issued_on IS NULL THEN NULL ELSE date($issued_on) END,
    r.asserted_at   = datetime()
RETURN count(r) AS links
"""

#: Siblings, never merges. The pair is symmetric, so both directions are written
#: and the query layer does not have to guess which way round it was recorded.
_LINK_SIBLINGS = """
UNWIND $pairs AS p
MATCH (a:Equipment {canonical_tag: p.a})
MATCH (b:Equipment {canonical_tag: p.b})
MERGE (a)-[r1:SIBLING_OF]->(b)
  ON CREATE SET r1.created_at = datetime()
SET r1.reason = p.reason, r1.confidence = p.confidence, r1.method = 'tag_grammar'
MERGE (b)-[r2:SIBLING_OF]->(a)
  ON CREATE SET r2.created_at = datetime()
SET r2.reason = p.reason, r2.confidence = p.confidence, r2.method = 'tag_grammar'
RETURN count(r1) AS pairs
"""

_UPSERT_WORK_ORDERS = """
UNWIND $work_orders AS w
MERGE (wo:WorkOrder {wo_id: w.wo_id})
  ON CREATE SET wo.created_at = datetime()
SET wo.wo_type        = w.wo_type,
    wo.status         = w.status,
    wo.priority       = w.priority,
    wo.description    = w.description,
    wo.as_found       = w.as_found,
    wo.as_left        = w.as_left,
    wo.coded_failure_mode = w.coded_failure_mode,
    wo.opened_on      = CASE WHEN w.opened_on IS NULL THEN NULL ELSE date(w.opened_on) END,
    wo.closed_on      = CASE WHEN w.closed_on IS NULL THEN NULL ELSE date(w.closed_on) END,
    wo.downtime_hrs   = w.downtime_hours,
    wo.cost           = w.cost,
    wo.source_system  = w.source_system,
    wo.data_class     = w.data_class,
    wo.updated_at     = datetime()
WITH wo, w
MATCH (d:Document {doc_id: $doc_id})
MERGE (wo)-[:RECORDED_IN]->(d)
WITH wo, w
OPTIONAL MATCH (e:Equipment {canonical_tag: w.canonical_tag})
FOREACH (_ IN CASE WHEN e IS NULL THEN [] ELSE [1] END |
    MERGE (wo)-[pr:PERFORMED_ON]->(e)
    SET pr.source_doc = $doc_id, pr.data_class = w.data_class, pr.asserted_at = datetime())
RETURN count(DISTINCT wo) AS work_orders
"""

_UPSERT_INCIDENTS = """
UNWIND $incidents AS i
MERGE (inc:Incident {incident_id: i.incident_id})
  ON CREATE SET inc.created_at = datetime()
SET inc.title            = i.title,
    inc.occurred_on      = CASE WHEN i.occurred_on IS NULL THEN NULL ELSE date(i.occurred_on) END,
    inc.severity         = i.severity,
    inc.event_type       = i.event_type,
    inc.immediate_cause  = i.immediate_cause,
    inc.root_cause       = i.root_cause,
    inc.investigation_status = i.investigation_status,
    inc.source_system    = i.source_system,
    inc.data_class       = i.data_class,
    inc.updated_at       = datetime()
WITH inc, i
OPTIONAL MATCH (e:Equipment {canonical_tag: i.canonical_tag})
FOREACH (_ IN CASE WHEN e IS NULL THEN [] ELSE [1] END |
    MERGE (inc)-[r:INVOLVED]->(e)
    SET r.source_doc = $doc_id, r.data_class = i.data_class, r.asserted_at = datetime())
RETURN count(DISTINCT inc) AS incidents
"""

_UPSERT_INSPECTIONS = """
UNWIND $inspections AS s
MERGE (ins:Inspection {inspection_id: s.inspection_id})
  ON CREATE SET ins.created_at = datetime()
SET ins.method          = s.method,
    ins.inspected_on    = CASE WHEN s.inspected_on IS NULL THEN NULL ELSE date(s.inspected_on) END,
    ins.thickness_mm    = s.thickness_mm,
    ins.min_required_mm = s.min_required_mm,
    ins.inspector       = s.inspector,
    ins.finding         = s.finding,
    ins.source_system   = s.source_system,
    ins.data_class      = s.data_class,
    ins.updated_at      = datetime()
WITH ins, s
FOREACH (_ IN CASE WHEN s.cml_id IS NULL THEN [] ELSE [1] END |
    MERGE (cml:CML {cml_id: s.cml_id})
      ON CREATE SET cml.created_at = datetime(), cml.data_class = s.data_class
    MERGE (ins)-[:MEASURED_AT]->(cml))
WITH ins, s
OPTIONAL MATCH (e:Equipment {canonical_tag: s.canonical_tag})
FOREACH (_ IN CASE WHEN e IS NULL THEN [] ELSE [1] END |
    MERGE (ins)-[r:PERFORMED_ON]->(e)
    SET r.source_doc = $doc_id, r.data_class = s.data_class, r.asserted_at = datetime())
RETURN count(DISTINCT ins) AS inspections
"""


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


#: The functional location is the permanent position in the process; the
#: equipment is the machine currently occupying it. Keeping them apart is what
#: lets the reliability question split in two:
#:   "does this position keep killing pumps?"  -> history on the FunctionalLocation
#:   "does this machine keep failing wherever   -> history on the Equipment serial
#:    we install it?"
#: Those have different answers and different fixes -- the first points at
#: process or installation, the second at manufacture or repair quality.
_UPSERT_FUNCTIONAL_LOCATIONS = """
UNWIND $locations AS fl
MERGE (loc:FunctionalLocation {fl_tag: fl.fl_tag})
  ON CREATE SET loc.created_at = datetime()
SET loc.description = fl.description,
    loc.data_class  = fl.data_class,
    loc.updated_at  = datetime()
WITH loc, fl
UNWIND fl.equipment AS tag
MATCH (e:Equipment {canonical_tag: tag})
MERGE (loc)-[r:OCCUPIED_BY]->(e)
  ON CREATE SET r.created_at = datetime()
SET r.source_doc = fl.source_doc,
    r.method     = 'record_field',
    r.data_class = fl.data_class,
    r.asserted_at = datetime()
RETURN count(DISTINCT loc) AS locations
"""


async def upsert_functional_locations(locations: list[dict[str, Any]]) -> int:
    if not locations:
        return 0
    rows = await graph.write(_UPSERT_FUNCTIONAL_LOCATIONS, locations=locations)
    return int(rows[0]["locations"]) if rows else 0


async def upsert_document(document: dict[str, Any]) -> None:
    await graph.write(
        _UPSERT_DOCUMENT,
        doc_id=document["doc_id"],
        title=document["title"],
        doc_type=document["doc_type"],
        data_class=document["data_class"],
        source_system=document["source_system"],
        content_hash=document["content_hash"],
        revision=document.get("revision"),
        doc_number=document.get("doc_number"),
        issued_on=document.get("issued_on"),
        valid_from=document.get("valid_from"),
        valid_to=document.get("valid_to"),
        page_count=document.get("page_count"),
        is_current=document.get("is_current", True),
        licence_note=document.get("licence_note"),
    )


async def link_revision_chain(report: dict[str, Any]) -> int:
    """Mirror a reconciled revision series into the graph.

    Takes the report from ``revisions.reconcile`` rather than re-querying, so the
    two stores cannot disagree about the ordering: whatever Postgres settled on
    is exactly what the graph is told.

    A conflicted series writes no edges at all. Asserting supersession the
    relational layer explicitly refused to assert would put a claim in the graph
    that nothing supports -- and graph facts are citable, so it would surface to
    an engineer as established.
    """
    doc_number = report.get("doc_number")
    if not doc_number:
        return 0

    await graph.write(_CLEAR_REVISIONS, doc_number=doc_number)
    if report.get("status") != "ok":
        return 0

    chain = report.get("chain") or []
    if len(chain) < 2:
        return 0

    # Every document at a level supersedes every document at the level below it.
    # SOP-4412 Rev 4 replaces both the Markdown and the scanned Rev 3, so both
    # edges are written -- a single edge to one representative would leave the
    # other looking current to any traversal that starts from it.
    pairs = [
        {"older": older["doc_id"], "newer": newer["doc_id"]}
        for lower, upper in zip(chain, chain[1:], strict=False)
        for older in lower["documents"]
        for newer in upper["documents"]
    ]
    rows = await graph.write(_LINK_REVISIONS, pairs=pairs, basis=report.get("basis", "unknown"))
    links = int(rows[0]["links"]) if rows else 0
    log.info("graph.revision_chain_linked", doc_number=doc_number, links=links)
    return links


async def upsert_chunks(doc_id: str, chunks: list[dict[str, Any]], data_class: str) -> int:
    if not chunks:
        return 0
    rows = await graph.write(_UPSERT_CHUNK, doc_id=doc_id, chunks=chunks, data_class=data_class)
    return int(rows[0]["chunks"]) if rows else 0


async def upsert_assets(assets: list[dict[str, Any]]) -> int:
    """Write equipment and instruments to their respective labels."""
    equipment = [a for a in assets if a.get("tag_kind") != "instrument"]
    instruments = [a for a in assets if a.get("tag_kind") == "instrument"]
    total = 0
    if equipment:
        rows = await graph.write(_UPSERT_ASSET, assets=equipment)
        total += int(rows[0]["assets"]) if rows else 0
    if instruments:
        rows = await graph.write(_UPSERT_INSTRUMENT, instruments=instruments)
        total += int(rows[0]["instruments"]) if rows else 0
    return total


async def upsert_mentions(mentions: list[dict[str, Any]]) -> int:
    if not mentions:
        return 0
    rows = await graph.write(_UPSERT_MENTIONS, mentions=mentions)
    return int(rows[0]["resolved"]) if rows else 0


async def link_document_to_assets(
    *,
    doc_id: str,
    links: list[dict[str, Any]],
    data_class: str,
    source_system: str,
    issued_on: str | None,
) -> int:
    if not links:
        return 0
    rows = await graph.write(
        _LINK_DOCUMENT_ASSET,
        doc_id=doc_id,
        links=links,
        data_class=data_class,
        source_system=source_system,
        issued_on=issued_on,
    )
    return int(rows[0]["links"]) if rows else 0


async def link_siblings(pairs: list[dict[str, Any]]) -> int:
    if not pairs:
        return 0
    rows = await graph.write(_LINK_SIBLINGS, pairs=pairs)
    return int(rows[0]["pairs"]) if rows else 0


async def upsert_work_orders(doc_id: str, work_orders: list[dict[str, Any]]) -> int:
    if not work_orders:
        return 0
    rows = await graph.write(_UPSERT_WORK_ORDERS, doc_id=doc_id, work_orders=work_orders)
    return int(rows[0]["work_orders"]) if rows else 0


async def upsert_incidents(doc_id: str, incidents: list[dict[str, Any]]) -> int:
    if not incidents:
        return 0
    rows = await graph.write(_UPSERT_INCIDENTS, doc_id=doc_id, incidents=incidents)
    return int(rows[0]["incidents"]) if rows else 0


async def upsert_inspections(doc_id: str, inspections: list[dict[str, Any]]) -> int:
    if not inspections:
        return 0
    rows = await graph.write(_UPSERT_INSPECTIONS, doc_id=doc_id, inspections=inspections)
    return int(rows[0]["inspections"]) if rows else 0
