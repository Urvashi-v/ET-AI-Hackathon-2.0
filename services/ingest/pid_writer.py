"""Persist P&ID detections and link them into the knowledge graph.

The point of digitising a drawing is not the drawing. It is that ``P-101B`` on
sheet 3 becomes the *same node* as ``P-101B`` in the incident report and
``P-101-B`` in the CMMS export — so asking "show me this pump" can return its
failure history, its procedure, and the rectangle it occupies on the P&ID.

Linking uses the same entity resolution as every other path. A tag read off a
drawing is matched against canonical assets exactly as a tag read out of prose
is, which means the drawing inherits the sibling protection, the separator
normalisation and the review queue rather than needing its own.

Unlinked detections are kept. A tag on a drawing that names equipment the corpus
has never ingested is one of the more useful things this pipeline finds — it is
the gap between what the plant has drawn and what it has recorded — and deleting
it to keep the table tidy would throw that away.
"""

from __future__ import annotations

import json
from typing import Any

from services.common import db, graph
from services.common.logging import get_logger
from services.common.schemas import DataClass
from services.ingest.pid import PidResult

log = get_logger(__name__)


#: Equipment appears at a place on a drawing. The coordinates live on the edge
#: rather than the node because one pump appears on several sheets, at different
#: positions, and the position belongs to the appearance.
_LINK_APPEARANCE = """
UNWIND $appearances AS a
MATCH (e:Equipment {canonical_tag: a.tag})
MATCH (d:Document {doc_id: $doc_id})
MERGE (e)-[r:APPEARS_ON]->(d)
SET r.page        = a.page,
    r.x0          = a.x0,
    r.y0          = a.y0,
    r.x1          = a.x1,
    r.y1          = a.y1,
    r.page_width  = a.page_width,
    r.page_height = a.page_height,
    r.method      = a.method,
    r.confidence  = a.confidence,
    r.data_class  = $data_class,
    r.asserted_at = datetime()
RETURN count(r) AS links
"""

#: Connectivity recovered from the sheet. Written as CONNECTS_ON rather than the
#: ontology's process-flow FEEDS, and the distinction is deliberate: this says
#: "a line on this drawing joins these two", not "product flows from one to the
#: other". Direction of flow is not recoverable from an undirected line segment,
#: and asserting FEEDS would put a claim in the graph that nothing supports.
_LINK_CONNECTION = """
UNWIND $connections AS c
MATCH (a:Equipment {canonical_tag: c.from_tag})
MATCH (b:Equipment {canonical_tag: c.to_tag})
MERGE (a)-[r:CONNECTS_ON {doc_id: $doc_id}]-(b)
SET r.page       = c.page,
    r.method     = c.method,
    r.confidence = c.confidence,
    r.data_class = $data_class,
    r.asserted_at = datetime()
RETURN count(r) AS links
"""


async def write(
    result: PidResult, *, doc_id: str, data_class: DataClass
) -> dict[str, Any]:
    """Store every detection, resolve tags to assets, and link the graph."""
    # Re-running the pipeline replaces this page's detections rather than
    # accumulating them. Detector parameters change; stale boxes from an older
    # tuning would sit in the overlay looking exactly like current ones.
    await db.execute(
        "DELETE FROM drawing_detections WHERE doc_id = %s AND page = %s",
        (doc_id, result.page),
    )

    ids: list[int] = []
    linked = 0
    unlinked_tags: list[str] = []

    for detection in result.detections:
        asset_id = None
        if detection.normalised:
            asset_id = await _resolve(detection.normalised)
            if asset_id:
                linked += 1
            elif detection.kind == "tag":
                unlinked_tags.append(detection.normalised)

        row = await db.fetch_one(
            """
            INSERT INTO drawing_detections (
                doc_id, page, kind, text, normalised, x0, y0, x1, y1,
                page_width, page_height, method, confidence, linked_asset_id,
                properties, data_class
            ) VALUES (
                %(doc_id)s, %(page)s, %(kind)s, %(text)s, %(normalised)s,
                %(x0)s, %(y0)s, %(x1)s, %(y1)s, %(pw)s, %(ph)s,
                %(method)s, %(confidence)s, %(asset_id)s, %(properties)s, %(data_class)s
            )
            ON CONFLICT (doc_id, page, kind, x0, y0, x1, y1) DO UPDATE SET
                text = EXCLUDED.text,
                normalised = EXCLUDED.normalised,
                confidence = EXCLUDED.confidence,
                linked_asset_id = EXCLUDED.linked_asset_id,
                properties = EXCLUDED.properties
            RETURNING detection_id
            """,
            {
                "doc_id": doc_id,
                "page": result.page,
                "kind": detection.kind,
                "text": detection.text,
                "normalised": detection.normalised,
                "x0": detection.x0,
                "y0": detection.y0,
                "x1": detection.x1,
                "y1": detection.y1,
                "pw": result.page_width,
                "ph": result.page_height,
                "method": detection.method,
                "confidence": detection.confidence,
                "asset_id": asset_id,
                "properties": json.dumps(detection.properties, default=str),
                "data_class": data_class.value,
            },
        )
        ids.append(int(row["detection_id"]) if row else 0)

    connections_written = await _write_connections(
        result, doc_id=doc_id, detection_ids=ids, data_class=data_class
    )
    graph_links = await _link_graph(result, doc_id=doc_id, data_class=data_class)

    log.info(
        "pid.written",
        doc_id=doc_id,
        page=result.page,
        detections=len(ids),
        linked_assets=linked,
        unlinked_tags=len(unlinked_tags),
        connections=connections_written,
        graph_links=graph_links,
    )
    return {
        "detections": len(ids),
        "linked_assets": linked,
        "unlinked_tags": sorted(set(unlinked_tags)),
        "connections": connections_written,
        "graph_appearances": graph_links,
        "by_kind": {
            kind: len(result.of_kind(kind))
            for kind in ("tag", "instrument_bubble", "line_segment")
        },
    }


async def _write_connections(
    result: PidResult, *, doc_id: str, detection_ids: list[int], data_class: DataClass
) -> int:
    written = 0
    for connection in result.connections:
        if connection.from_index >= len(detection_ids) or connection.to_index >= len(
            detection_ids
        ):
            continue
        from_id = detection_ids[connection.from_index]
        to_id = detection_ids[connection.to_index]
        if not from_id or not to_id or from_id == to_id:
            continue
        await db.execute(
            """
            INSERT INTO drawing_connections (
                doc_id, page, from_detection, to_detection, via_detections,
                method, confidence, properties, data_class
            ) VALUES (
                %(doc_id)s, %(page)s, %(from_id)s, %(to_id)s, %(via)s,
                %(method)s, %(confidence)s, %(properties)s, %(data_class)s
            )
            ON CONFLICT (from_detection, to_detection, method) DO UPDATE SET
                confidence = EXCLUDED.confidence,
                properties = EXCLUDED.properties
            """,
            {
                "doc_id": doc_id,
                "page": result.page,
                "from_id": from_id,
                "to_id": to_id,
                "via": [detection_ids[i] for i in connection.via if i < len(detection_ids)],
                "method": connection.method,
                "confidence": connection.confidence,
                "properties": json.dumps(connection.properties, default=str),
                "data_class": data_class.value,
            },
        )
        written += 1
    return written


async def _link_graph(result: PidResult, *, doc_id: str, data_class: DataClass) -> int:
    """Write APPEARS_ON edges for every tag that resolved to an asset."""
    appearances = []
    for detection in result.detections:
        if detection.kind not in ("tag", "instrument_bubble") or not detection.normalised:
            continue
        appearances.append(
            {
                "tag": detection.normalised,
                "page": result.page,
                "x0": detection.x0,
                "y0": detection.y0,
                "x1": detection.x1,
                "y1": detection.y1,
                "page_width": result.page_width,
                "page_height": result.page_height,
                "method": detection.method,
                "confidence": detection.confidence,
            }
        )
    if not appearances:
        return 0

    rows = await graph.write(
        _LINK_APPEARANCE,
        doc_id=doc_id,
        appearances=appearances,
        data_class=data_class.value,
    )
    links = int(rows[0]["links"]) if rows else 0

    connections = []
    for connection in result.connections:
        source = result.detections[connection.from_index]
        target = result.detections[connection.to_index]
        if not source.normalised or not target.normalised:
            continue
        if source.normalised == target.normalised:
            continue
        connections.append(
            {
                "from_tag": source.normalised,
                "to_tag": target.normalised,
                "page": result.page,
                "method": connection.method,
                "confidence": connection.confidence,
            }
        )
    if connections:
        await graph.write(
            _LINK_CONNECTION,
            doc_id=doc_id,
            connections=connections,
            data_class=data_class.value,
        )
    return links


async def _resolve(normalised: str) -> str | None:
    """Match a drawing tag to a canonical asset, or return nothing.

    Checks the alias table as well as the canonical column, so a drawing that
    writes ``P-101-B`` links to the same asset as the CMMS export writing
    ``P-101B`` -- the resolution work from Day 2, reused rather than reimplemented.
    """
    row = await db.fetch_one(
        """
        SELECT a.asset_id
          FROM assets a
         WHERE upper(a.canonical_tag) = upper(%(tag)s)
         UNION
        SELECT al.asset_id
          FROM asset_aliases al
         WHERE upper(al.normalised) = upper(%(tag)s)
            OR upper(al.surface_form) = upper(%(tag)s)
         LIMIT 1
        """,
        {"tag": normalised},
    )
    return str(row["asset_id"]) if row else None
