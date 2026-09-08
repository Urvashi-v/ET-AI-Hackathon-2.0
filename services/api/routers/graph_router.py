"""Knowledge graph endpoints.

``GET /api/v1/graph/{asset_id}`` returns the neighbourhood around one asset for
the graph explorer: nodes, typed edges, edge provenance and the chunk ids that
evidence each edge. Clicking an edge in the UI therefore leads to the passage
that asserted it, which is what makes "we connected these documents" checkable
rather than a claim.

Filters mirror what the retrieval layer uses: hop bound and edge-type selection,
so the explorer and the copilot see the same graph through the same lens.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from fastapi import APIRouter, Query

from services.common import db, graph
from services.common.errors import NotFoundError
from services.common.schemas import GraphEdge, GraphNode, GraphResponse
from services.retrieval.graph_retrieval import _ALL_EDGE_TYPES, neighbourhood

router = APIRouter(prefix="/graph", tags=["knowledge graph"])


@router.get("/schema", summary="Ontology as installed in the running database")
async def schema() -> dict[str, Any]:
    """Report the graph's real shape, not the intended one.

    Labels, relationship types and counts are read back from Neo4j, so this
    endpoint shows what actually exists after ingestion.
    """
    labels = await graph.read("CALL db.labels() YIELD label RETURN label ORDER BY label")
    rel_types = await graph.read(
        "CALL db.relationshipTypes() YIELD relationshipType "
        "RETURN relationshipType ORDER BY relationshipType"
    )
    counts = await graph.read(
        "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS n ORDER BY n DESC"
    )
    rel_counts = await graph.read(
        "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS n ORDER BY n DESC"
    )
    ontology = await graph.read(
        "MATCH (o:OntologyVersion) RETURN o.version AS version, "
        "o.node_labels AS declared_labels, o.relationship_types AS declared_relationships "
        "ORDER BY o.version DESC LIMIT 1"
    )
    return {
        "ontology_version": ontology[0] if ontology else None,
        "labels_present": [r["label"] for r in labels],
        "relationship_types_present": [r["relationshipType"] for r in rel_types],
        "node_counts": [dict(r) for r in counts],
        "relationship_counts": [dict(r) for r in rel_counts],
    }


@router.get("/{asset_id}", response_model=GraphResponse, summary="Asset neighbourhood")
async def asset_graph(
    asset_id: str,
    hops: int = Query(2, ge=1, le=3),
    edges: str | None = Query(
        None, description="Comma-separated relationship types, e.g. FEEDS,SIBLING_OF"
    ),
    as_of: date | None = Query(
        None, description="Filter temporal edges to those valid on this date"
    ),
    limit: int = Query(300, ge=10, le=1000),
) -> GraphResponse:
    row = await db.fetch_one(
        "SELECT canonical_tag FROM assets "
        "WHERE asset_id = %(key)s OR upper(canonical_tag) = upper(%(key)s)",
        {"key": asset_id},
    )
    if not row:
        # Truthful empty result rather than a 404 with no information: the caller
        # learns the anchor does not exist and what to do about it.
        return GraphResponse(
            anchor=asset_id,
            anchor_found=False,
            hops=hops,
            edge_types=_selected_edges(edges),
            as_of=as_of,
            nodes=[],
            edges=[],
            detail=(
                f"No asset matching '{asset_id}' exists in the corpus. Assets appear only "
                "once a document mentioning them has been ingested."
            ),
        )

    anchor_tag = row["canonical_tag"]
    edge_types = _selected_edges(edges)
    result = await neighbourhood(
        anchor_tag=anchor_tag, hops=hops, edge_types=edge_types, limit=limit
    )

    nodes = [GraphNode(**n) for n in result["nodes"]]
    graph_edges = [GraphEdge(**e) for e in result["edges"]]

    if as_of is not None:
        graph_edges = [e for e in graph_edges if _valid_on(e, as_of)]
        keep = {e.source for e in graph_edges} | {e.target for e in graph_edges}
        nodes = [n for n in nodes if n.id in keep]

    return GraphResponse(
        anchor=anchor_tag,
        anchor_found=True,
        hops=hops,
        edge_types=edge_types,
        as_of=as_of,
        nodes=nodes,
        edges=graph_edges,
        truncated=result["truncated"],
        detail=(
            f"{len(nodes)} nodes and {len(graph_edges)} edges within {hops} hop(s) of {anchor_tag}."
            + (" Result truncated at the node limit." if result["truncated"] else "")
        ),
    )


@router.get("/{asset_id}/evidence/{edge_id}", summary="Evidence behind one graph edge")
async def edge_evidence(asset_id: str, edge_id: str) -> dict[str, Any]:
    """Resolve an edge's evidence chunk ids to real passages.

    This is what makes a graph edge auditable: an assertion the system displays
    must lead back to the text that produced it.
    """
    rows = await graph.read(
        "MATCH ()-[r]->() WHERE elementId(r) = $edge_id "
        "RETURN type(r) AS type, properties(r) AS props",
        edge_id=edge_id,
    )
    if not rows:
        raise NotFoundError(f"No graph edge with id {edge_id}.")

    props = rows[0]["props"] or {}
    chunk_ids = list(props.get("evidence_chunks") or [])
    passages: list[dict[str, Any]] = []
    if chunk_ids:
        passages = [
            dict(r)
            for r in await db.fetch_all(
                """
                SELECT c.chunk_id, c.text, c.page_from, c.section_path,
                       c.data_class::text AS data_class,
                       d.doc_id, d.title, d.doc_type::text AS doc_type, d.source_system
                  FROM document_chunks c
                  JOIN documents d ON d.doc_id = c.doc_id
                 WHERE c.chunk_id = ANY(%s)
                """,
                (chunk_ids,),
            )
        ]

    return {
        "edge_id": edge_id,
        "type": rows[0]["type"],
        "properties": {
            k: (v.iso_format() if hasattr(v, "iso_format") else v)
            for k, v in props.items()
            if k != "evidence_chunks"
        },
        "evidence_chunk_ids": chunk_ids,
        "evidence": passages,
        "detail": (
            "This edge carries no chunk-level evidence pointer."
            if not chunk_ids
            else f"{len(passages)} of {len(chunk_ids)} evidence chunks resolved."
        ),
    }


def _selected_edges(edges: str | None) -> list[str]:
    if not edges:
        return list(_ALL_EDGE_TYPES)
    requested = [e.strip().upper() for e in edges.split(",") if e.strip()]
    return [e for e in requested if e in _ALL_EDGE_TYPES] or list(_ALL_EDGE_TYPES)


def _valid_on(edge: GraphEdge, when: date) -> bool:
    """Temporal filter. Edges with no validity window are treated as always
    valid -- absence of a date is not evidence of expiry."""
    props = edge.properties
    valid_from = props.get("valid_from")
    valid_to = props.get("valid_to")
    if valid_from and str(valid_from)[:10] > when.isoformat():
        return False
    return not (valid_to and str(valid_to)[:10] < when.isoformat())
