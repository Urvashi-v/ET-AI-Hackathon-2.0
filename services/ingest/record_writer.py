"""Persist extracted records to Postgres and Neo4j.

Split from ``records.py`` on purpose: extraction is pure and testable without a
database, writing is I/O. The same separation the rest of the ingest path uses.

Idempotence works the same way it does for documents. ``incident_id`` and
``moc_id`` are the natural keys printed on the source, so re-ingesting a report
updates the record rather than duplicating it — and the two renditions of
INC-2019-07 (the Markdown source and the scanned PDF) converge on one node
rather than becoming two incidents that happen to share a date.

That convergence is why the writer merges rather than overwrites: whichever
rendition carries a field wins, so the scan's OCR gaps are filled by the
Markdown and vice versa. A field that is present in neither stays null, which is
a finding rather than a defect.
"""

from __future__ import annotations

from typing import Any

from services.common import db, graph
from services.common.logging import get_logger
from services.common.schemas import DataClass
from services.ingest.records import ChangeRecord, IncidentRecord

log = get_logger(__name__)


_UPSERT_INCIDENT_NODE = """
MERGE (i:Incident {incident_id: $incident_id})
  ON CREATE SET i.created_at = datetime()
SET i.title             = $title,
    i.occurred_on       = CASE WHEN $occurred_on IS NULL THEN NULL ELSE date($occurred_on) END,
    i.severity          = $severity,
    i.status            = $status,
    i.immediate_cause   = $immediate_cause,
    i.root_cause        = $root_cause,
    i.data_class        = $data_class,
    i.evidence_chunks   = $evidence_chunks,
    i.updated_at        = datetime()
WITH i
MATCH (d:Document {doc_id: $doc_id})
MERGE (d)-[:RECORDS]->(i)
MERGE (i)-[:DOCUMENTED_IN]->(d)
RETURN i.incident_id AS incident_id
"""

#: Links the incident to the equipment it happened to. ``INVOLVED`` rather than
#: ``PERFORMED_ON``: a work order is performed on an asset, an incident merely
#: involves it, and RCA traverses the two differently.
_LINK_INCIDENT_ASSET = """
UNWIND $tags AS tag
MATCH (i:Incident {incident_id: $incident_id})
MATCH (e:Equipment {canonical_tag: tag})
MERGE (i)-[r:INVOLVED]->(e)
SET r.data_class = $data_class,
    r.evidence_chunks = $evidence_chunks,
    r.confidence = 1.0,
    r.asserted_at = datetime()
RETURN count(r) AS links
"""

#: Corrective actions land as ``:CAPA`` with ``capa_id``, matching the ontology
#: Day 1 established and the property names the RCA endpoint already queries.
#: They are literally labelled CAPA-87 on the source document, so the existing
#: label is the right one -- adding a parallel ``:CorrectiveAction`` node for the
#: same rows would leave the RCA traversal finding nothing while the data sat
#: one label away.
#:
#: Both ``GENERATED`` and ``RESOLVED_BY`` are written. They say different things:
#: the incident *generated* this action, and the failure is *resolved by* it.
#: RCA traverses the first, lessons-learned the second.
_UPSERT_ACTIONS = """
UNWIND $actions AS a
MERGE (c:CAPA {capa_id: a.action_id})
  ON CREATE SET c.created_at = datetime()
SET c.action      = a.description,
    c.owner       = a.owner,
    c.due_date    = CASE WHEN a.due_on IS NULL THEN NULL ELSE date(a.due_on) END,
    c.status      = a.status,
    c.is_open     = a.is_open,
    c.data_class  = $data_class,
    c.updated_at  = datetime()
WITH c, a
MATCH (i:Incident {incident_id: $incident_id})
MERGE (i)-[g:GENERATED]->(c)
SET g.evidence_chunks = [a.chunk_id], g.data_class = $data_class
MERGE (i)-[r:RESOLVED_BY]->(c)
SET r.evidence_chunks = [a.chunk_id], r.data_class = $data_class
RETURN count(c) AS actions
"""

#: An incident that names another incident as the same mechanism. This is the
#: edge that makes "has this happened before?" a traversal instead of a search --
#: and in this corpus it is the edge nobody drew, which is precisely why the two
#: seal failures were investigated independently three years apart.
_LINK_RELATED = """
UNWIND $related AS other_id
MATCH (i:Incident {incident_id: $incident_id})
MATCH (o:Incident {incident_id: other_id})
MERGE (i)-[r:REFERENCES]->(o)
SET r.data_class = $data_class, r.asserted_at = datetime()
RETURN count(r) AS links
"""

_LINK_PROCEDURES = """
UNWIND $procedures AS doc_number
MATCH (i:Incident {incident_id: $incident_id})
MATCH (d:Document {doc_number: doc_number})
MERGE (i)-[r:CITES_PROCEDURE]->(d)
SET r.data_class = $data_class, r.asserted_at = datetime()
RETURN count(r) AS links
"""

#: ``:MOC`` with ``moc_id`` / ``change_desc`` / ``approved_on``, again matching
#: the ontology and the property names the RCA endpoint reads.
_UPSERT_CHANGE = """
MERGE (m:MOC {moc_id: $moc_id})
  ON CREATE SET m.created_at = datetime()
SET m.title       = $title,
    m.change_desc = $description,
    m.approved_on = CASE WHEN $raised_on IS NULL THEN NULL ELSE date($raised_on) END,
    m.change_type = $change_type,
    m.status      = $status,
    m.data_class  = $data_class,
    m.updated_at  = datetime()
WITH m
MATCH (d:Document {doc_id: $doc_id})
MERGE (d)-[:RECORDS]->(m)
RETURN m.moc_id AS moc_id
"""

_LINK_CHANGE_ASSET = """
UNWIND $tags AS tag
MATCH (m:MOC {moc_id: $moc_id})
MATCH (e:Equipment {canonical_tag: tag})
MERGE (m)-[r:CHANGED]->(e)
SET r.data_class = $data_class,
    r.date = m.approved_on,
    r.asserted_at = datetime()
RETURN count(r) AS links
"""


async def write_incident(record: IncidentRecord, *, data_class: DataClass) -> dict[str, Any]:
    """Write one incident to both stores. Safe to re-run."""
    evidence = _evidence_chunks(record)
    asset_id = await _resolve_asset(record.asset_tags)

    await db.execute(
        """
        INSERT INTO incidents (
            incident_id, doc_id, asset_id, raw_asset_tag, title, occurred_on, severity,
            event_type, narrative, immediate_cause, root_cause, investigation_status,
            source_system, data_class, metadata
        ) VALUES (
            %(incident_id)s, %(doc_id)s, %(asset_id)s, %(raw_tag)s, %(title)s, %(occurred_on)s,
            %(severity)s, %(event_type)s, %(narrative)s, %(immediate_cause)s, %(root_cause)s,
            %(status)s, %(source_system)s, %(data_class)s, %(metadata)s
        )
        ON CONFLICT (incident_id) DO UPDATE SET
            -- COALESCE with the *existing* value first for identity-ish columns,
            -- and with the incoming value first for content: a second rendition
            -- should fill gaps, never blank out what the first one found.
            asset_id             = COALESCE(EXCLUDED.asset_id, incidents.asset_id),
            occurred_on          = COALESCE(EXCLUDED.occurred_on, incidents.occurred_on),
            severity             = COALESCE(EXCLUDED.severity, incidents.severity),
            narrative            = COALESCE(EXCLUDED.narrative, incidents.narrative),
            immediate_cause      = COALESCE(EXCLUDED.immediate_cause, incidents.immediate_cause),
            root_cause           = COALESCE(EXCLUDED.root_cause, incidents.root_cause),
            investigation_status = COALESCE(
                EXCLUDED.investigation_status, incidents.investigation_status),
            metadata             = incidents.metadata || EXCLUDED.metadata
        """,
        {
            "incident_id": record.incident_id,
            "doc_id": record.doc_id,
            "asset_id": asset_id,
            "raw_tag": record.asset_tags[0] if record.asset_tags else None,
            "title": record.title,
            "occurred_on": record.occurred_on,
            "severity": record.severity,
            "event_type": "incident",
            "narrative": str(record.narrative) if record.narrative else None,
            "immediate_cause": str(record.immediate_cause) if record.immediate_cause else None,
            "root_cause": str(record.root_cause) if record.root_cause else None,
            "status": record.investigation_status,
            "source_system": "document_extraction",
            "data_class": data_class.value,
            "metadata": _metadata_json(record, evidence),
        },
    )

    await graph.write(
        _UPSERT_INCIDENT_NODE,
        incident_id=record.incident_id,
        doc_id=record.doc_id,
        title=record.title,
        occurred_on=record.occurred_on.isoformat() if record.occurred_on else None,
        severity=record.severity,
        status=record.investigation_status,
        immediate_cause=str(record.immediate_cause) if record.immediate_cause else None,
        root_cause=str(record.root_cause) if record.root_cause else None,
        data_class=data_class.value,
        evidence_chunks=evidence,
    )

    links = 0
    if record.asset_tags:
        rows = await graph.write(
            _LINK_INCIDENT_ASSET,
            incident_id=record.incident_id,
            tags=record.asset_tags,
            data_class=data_class.value,
            evidence_chunks=evidence,
        )
        links = int(rows[0]["links"]) if rows else 0

    actions = 0
    if record.corrective_actions:
        rows = await graph.write(
            _UPSERT_ACTIONS,
            incident_id=record.incident_id,
            data_class=data_class.value,
            actions=[
                {
                    "action_id": a.action_id,
                    "description": a.description,
                    "owner": a.owner,
                    "due_on": a.due_on.isoformat() if a.due_on else None,
                    "status": a.status,
                    "is_open": a.is_open,
                    "chunk_id": a.chunk_id,
                }
                for a in record.corrective_actions
            ],
        )
        actions = int(rows[0]["actions"]) if rows else 0

    related = 0
    if record.referenced_incidents:
        rows = await graph.write(
            _LINK_RELATED,
            incident_id=record.incident_id,
            related=record.referenced_incidents,
            data_class=data_class.value,
        )
        related = int(rows[0]["links"]) if rows else 0

    if record.referenced_procedures:
        await graph.write(
            _LINK_PROCEDURES,
            incident_id=record.incident_id,
            procedures=record.referenced_procedures,
            data_class=data_class.value,
        )

    log.info(
        "records.incident_written",
        incident_id=record.incident_id,
        asset_links=links,
        actions=actions,
        related=related,
        fields_missing=record.fields_missing,
    )
    return {
        "incident_id": record.incident_id,
        "asset_links": links,
        "corrective_actions": actions,
        "related_incidents": related,
        "fields_missing": record.fields_missing,
    }


async def write_change(record: ChangeRecord, *, data_class: DataClass) -> dict[str, Any]:
    await graph.write(
        _UPSERT_CHANGE,
        moc_id=record.moc_id,
        doc_id=record.doc_id,
        title=record.title,
        description=str(record.description) if record.description else None,
        raised_on=record.raised_on.isoformat() if record.raised_on else None,
        change_type=record.change_type,
        status=record.status,
        data_class=data_class.value,
    )
    links = 0
    if record.asset_tags:
        rows = await graph.write(
            _LINK_CHANGE_ASSET,
            moc_id=record.moc_id,
            tags=record.asset_tags,
            data_class=data_class.value,
        )
        links = int(rows[0]["links"]) if rows else 0
    log.info("records.change_written", moc_id=record.moc_id, asset_links=links)
    return {"moc_id": record.moc_id, "asset_links": links}


async def _resolve_asset(tags: list[str]) -> str | None:
    """Map the report's equipment reference to a canonical asset, or nothing.

    Returns None rather than creating an asset. An incident report naming an
    asset the corpus has never ingested is a gap worth seeing, and inventing the
    equipment node to satisfy a foreign key would hide it.
    """
    for tag in tags:
        row = await db.fetch_one(
            "SELECT asset_id FROM assets WHERE upper(canonical_tag) = upper(%s)", (tag,)
        )
        if row:
            return str(row["asset_id"])
    return None


def _evidence_chunks(record: IncidentRecord) -> list[str]:
    chunks = [
        f.chunk_id
        for f in (
            record.narrative,
            record.immediate_cause,
            record.root_cause,
            record.recurrence_note,
        )
        if f is not None
    ]
    chunks.extend(a.chunk_id for a in record.corrective_actions)
    return sorted(set(chunks))


def _metadata_json(record: IncidentRecord, evidence: list[str]) -> str:
    import json

    return json.dumps(
        {
            "functional_location": record.functional_location,
            "asset_tags": record.asset_tags,
            "referenced_incidents": record.referenced_incidents,
            "referenced_procedures": record.referenced_procedures,
            "corrective_actions": [
                {
                    "action_id": a.action_id,
                    "description": a.description,
                    "owner": a.owner,
                    "due_on": a.due_on.isoformat() if a.due_on else None,
                    "status": a.status,
                    "is_open": a.is_open,
                    "chunk_id": a.chunk_id,
                }
                for a in record.corrective_actions
            ],
            "fields_found": record.fields_found,
            "fields_missing": record.fields_missing,
            "evidence_chunks": evidence,
            "extraction_method": "deterministic_section_parse",
        }
    )
