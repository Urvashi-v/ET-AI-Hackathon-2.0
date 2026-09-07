"""GraphRAG: retrieval that traverses.

Rather than retrieving k chunks by similarity, this anchors on the entities named
in the question, traverses the graph to gather a *structurally coherent*
subgraph, and then returns both the facts and the chunks that evidence them. The
context handed to a generator is then a connected account of the asset rather
than a bag of loosely similar passages.

Two constraints keep it from exploding:

* **hop bound** -- an unbounded traversal from a well-connected node returns the
  whole plant;
* **edge sets by intent** -- traversing every edge type drowns the answer in
  irrelevant context. A diagnostic question wants failure history and siblings;
  an impact question wants process topology and isolation points.

Everything returned carries its provenance, so a graph fact can be cited the same
way a passage can.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from services.common import graph
from services.common.logging import get_logger
from services.common.schemas import DataClass, GraphEntityRef, GraphFact, QueryIntent

log = get_logger(__name__)

#: Which relationships are worth walking, per intent.
EDGE_SETS: dict[QueryIntent, list[str]] = {
    QueryIntent.DIAGNOSTIC: [
        "PERFORMED_ON",
        "RECORDED",
        "OF_MODE",
        "CAUSED_BY",
        "SIBLING_OF",
        "HAS_COMPONENT",
        "INVOLVED",
        "GENERATED",
        "CHANGED",
        "DESCRIBES",
    ],
    QueryIntent.MULTI_HOP: [
        "FEEDS",
        "ISOLATES",
        "SIBLING_OF",
        "CONNECTS",
        "OCCUPIED_BY",
        "HAS_POSITION",
        "CONTAINS",
        "MEASURES",
        "DESCRIBES",
    ],
    QueryIntent.AGGREGATE: ["PERFORMED_ON", "RECORDED", "OF_MODE", "INSTANCE_OF", "INVOLVED"],
    QueryIntent.PROCEDURAL: ["DESCRIBES", "HAS_STEP", "IMPLEMENTS", "ISOLATES"],
    QueryIntent.COMPARATIVE: ["SIBLING_OF", "INSTANCE_OF", "PERFORMED_ON", "DESCRIBES"],
    QueryIntent.LOOKUP: ["DESCRIBES", "INSTANCE_OF", "OCCUPIED_BY"],
    QueryIntent.UNANSWERABLE: ["DESCRIBES"],
}

_MAX_PATHS = 400


@dataclass(slots=True)
class GraphRetrievalResult:
    facts: list[GraphFact] = field(default_factory=list)
    evidence_chunk_ids: list[str] = field(default_factory=list)
    anchors_found: list[str] = field(default_factory=list)
    anchors_missing: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0
    truncated: bool = False
    #: Nodes touched during traversal, keyed by display id. Collected as the
    #: facts are built rather than re-queried, so "which entities were used"
    #: is answered by what actually happened rather than by a second guess.
    entities: dict[str, dict[str, Any]] = field(default_factory=dict)

    def as_ranked_chunks(self) -> list[dict[str, Any]]:
        """Evidence chunks in graph-proximity order, for fusion."""
        return [
            {"chunk_id": cid, "retriever": "graph", "rank": rank}
            for rank, cid in enumerate(self.evidence_chunk_ids, start=1)
        ]

    def entity_refs(self, limit: int = 40) -> list[GraphEntityRef]:
        """The graph nodes that contributed, anchors first.

        This is what the copilot shows under "graph entities used". It is derived
        from the traversal, so it cannot list a node the query did not touch.
        """
        refs = [
            GraphEntityRef(
                node_id=node_id,
                label=meta["label"],
                display=meta["display"],
                role="anchor" if node_id in self.anchors_found else "neighbour",
                hops=0 if node_id in self.anchors_found else 1,
                data_class=meta.get("data_class"),
            )
            for node_id, meta in self.entities.items()
        ]
        refs.sort(key=lambda r: (r.role != "anchor", r.label, r.display))
        return refs[:limit]

    def note_entity(
        self,
        node_id: str,
        labels: list[str],
        data_class: DataClass,
        title: str | None = None,
    ) -> None:
        """Record a node the traversal touched.

        ``node_id`` is the identity used for fact deduplication; ``title`` is what
        a human should see. They differ for documents, whose identity is a content
        hash -- "doc_6de8cb37..." tells an engineer nothing, "incident 2019 seal
        failure" tells them everything.
        """
        if not node_id or node_id in self.entities:
            return
        # Neo4j returns every label on the node; the most specific one is the
        # useful one, and the ontology puts it last.
        self.entities[node_id] = {
            "label": labels[-1] if labels else "Node",
            "data_class": data_class,
            "display": title or node_id,
        }


_TRAVERSE = """
MATCH (anchor)
WHERE (anchor:Equipment AND anchor.canonical_tag IN $tags)
   OR (anchor:Instrument AND anchor.tag IN $tags)
CALL {
    WITH anchor
    MATCH path = (anchor)-[:%(rels)s*1..%(hops)d]-(n)
    RETURN path LIMIT %(limit)d
}
UNWIND relationships(path) AS r
WITH DISTINCT startNode(r) AS s, r, endNode(r) AS e
RETURN
    coalesce(s.canonical_tag, s.tag, s.doc_id, s.wo_id, s.incident_id,
             s.inspection_id, s.code, s.chunk_id, s.title, elementId(s)) AS subject,
    labels(s) AS subject_labels,
    s.title AS subject_title,
    type(r) AS predicate,
    coalesce(e.canonical_tag, e.tag, e.doc_id, e.wo_id, e.incident_id,
             e.inspection_id, e.code, e.chunk_id, e.title, elementId(e)) AS object,
    labels(e) AS object_labels,
    e.title AS object_title,
    properties(r) AS rel_props,
    coalesce(r.evidence_chunks, []) AS evidence_chunks,
    r.confidence AS confidence,
    coalesce(r.data_class, e.data_class, s.data_class) AS data_class
LIMIT %(limit)d
"""

_ANCHOR_CHECK = """
UNWIND $tags AS tag
OPTIONAL MATCH (e:Equipment {canonical_tag: tag})
OPTIONAL MATCH (i:Instrument {tag: tag})
RETURN tag, (e IS NOT NULL OR i IS NOT NULL) AS found
"""

#: Document types worth reaching for, per intent. The edge sets above decide
#: which *facts* to gather; this decides which *evidence* is relevant. Without
#: it the graph leg ranks purely by how often a chunk mentions the asset, which
#: floats incident reports and work orders to the top of every question --
#: including "how do I isolate this pump?", where the procedure is what was
#: asked for and the incident history is noise.
_INTENT_DOC_TYPES: dict[QueryIntent, list[str]] = {
    QueryIntent.PROCEDURAL: ["sop", "manual", "permit", "pid"],
    QueryIntent.DIAGNOSTIC: ["incident_report", "work_order", "moc", "inspection_report"],
    QueryIntent.AGGREGATE: ["work_order", "inspection_report", "incident_report"],
    QueryIntent.COMPARATIVE: ["work_order", "incident_report", "datasheet"],
    QueryIntent.MULTI_HOP: ["pid", "sop", "datasheet", "isometric"],
    QueryIntent.LOOKUP: ["datasheet", "sop", "manual", "pid"],
    QueryIntent.UNANSWERABLE: [],
}

#: Chunks reached *through the graph* rather than by resembling the question's
#: wording. Two routes, and both are needed:
#:
#: 1. **Direct mention** -- the chunk names the asset. Strong but narrow.
#: 2. **Document association** -- the chunk belongs to a document that DESCRIBES
#:    the asset. This is the route that matters for procedures: an SOP names the
#:    pump once, in its scope line, and then never again. Every individual step
#:    is about that pump without saying so, and a mention-only query cannot see
#:    it. Following Equipment <-[:DESCRIBES]- Document -[:HAS_CHUNK]-> Chunk is
#:    exactly the traversal a flat index cannot do.
#:
#: Document-type preference by intent orders the result, so a procedural question
#: gets the procedure and a diagnostic one gets the failure history.
_ANCHOR_CHUNKS = """
CALL {
    // Route 1: chunks that mention the asset directly.
    MATCH (m:Mention)-[:RESOLVES_TO]->(target)
    WHERE (target:Equipment AND target.canonical_tag IN $tags)
       OR (target:Instrument AND target.tag IN $tags)
    MATCH (m)-[:APPEARS_IN]->(c:Chunk)
    RETURN c AS chunk, count(m) AS mention_count, 0 AS route
UNION
    // Route 2: chunks of documents that describe the asset.
    MATCH (d:Document)-[:DESCRIBES]->(target)
    WHERE (target:Equipment AND target.canonical_tag IN $tags)
       OR (target:Instrument AND target.tag IN $tags)
    MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
    RETURN c AS chunk, 0 AS mention_count, 1 AS route
}
WITH chunk, max(mention_count) AS mention_count, min(route) AS route
MATCH (chunk)-[:PART_OF]->(d:Document)
WITH chunk, d.doc_type AS doc_type, mention_count, route,
     CASE WHEN size($preferred_types) = 0 THEN 1
          WHEN d.doc_type IN $preferred_types THEN 0
          ELSE 1 END AS type_rank
RETURN chunk.chunk_id AS chunk_id, doc_type, mention_count, route, type_rank
ORDER BY type_rank ASC, route ASC, mention_count DESC, chunk.ordinal ASC
LIMIT $limit
"""


async def anchors_present(tags: list[str]) -> tuple[list[str], list[str]]:
    """Split requested tags into those the graph knows and those it does not.

    Reported honestly: a question about an asset that is not in the corpus must
    lead to an abstention naming the asset, never to an answer about a
    similarly-named one.
    """
    if not tags:
        return [], []
    rows = await graph.read(_ANCHOR_CHECK, tags=tags)
    found = [r["tag"] for r in rows if r["found"]]
    missing = [r["tag"] for r in rows if not r["found"]]
    return found, missing


async def retrieve(
    *,
    tags: list[str],
    intent: QueryIntent,
    hops: int = 2,
    limit: int = 50,
) -> GraphRetrievalResult:
    started = time.perf_counter()
    result = GraphRetrievalResult()

    if not tags:
        result.elapsed_ms = (time.perf_counter() - started) * 1000
        return result

    found, missing = await anchors_present(tags)
    result.anchors_found = found
    result.anchors_missing = missing
    if not found:
        result.elapsed_ms = (time.perf_counter() - started) * 1000
        return result

    edge_types = EDGE_SETS.get(intent, EDGE_SETS[QueryIntent.LOOKUP])
    cypher = _TRAVERSE % {
        "rels": "|".join(edge_types),
        "hops": max(1, min(hops, 3)),
        "limit": _MAX_PATHS,
    }

    rows = await graph.read(cypher, tags=found)
    seen: set[tuple[str, str, str]] = set()
    evidence: list[str] = []

    for row in rows:
        key = (str(row["subject"]), row["predicate"], str(row["object"]))
        if key in seen:
            continue
        seen.add(key)
        props = {k: v for k, v in (row.get("rel_props") or {}).items() if k != "evidence_chunks"}
        fact_class = _data_class(row.get("data_class"))
        result.note_entity(
            str(row["subject"]), row["subject_labels"], fact_class, row.get("subject_title")
        )
        result.note_entity(
            str(row["object"]), row["object_labels"], fact_class, row.get("object_title")
        )
        result.facts.append(
            GraphFact(
                subject=f"{str(row['subject'])} ({'/'.join(row['subject_labels'])})",
                predicate=row["predicate"],
                object=f"{str(row['object'])} ({'/'.join(row['object_labels'])})",
                properties=_stringify(props),
                evidence_chunk_ids=list(row.get("evidence_chunks") or []),
                confidence=row.get("confidence"),
                data_class=_data_class(row.get("data_class")),
            )
        )
        for chunk_id in row.get("evidence_chunks") or []:
            if chunk_id not in evidence:
                evidence.append(chunk_id)

    anchor_chunks = await graph.read(
        _ANCHOR_CHUNKS,
        tags=found,
        limit=limit,
        preferred_types=_INTENT_DOC_TYPES.get(intent, []),
    )
    for row in anchor_chunks:
        if row["chunk_id"] not in evidence:
            evidence.append(row["chunk_id"])

    result.evidence_chunk_ids = evidence[:limit]
    result.truncated = len(rows) >= _MAX_PATHS
    result.elapsed_ms = (time.perf_counter() - started) * 1000

    log.debug(
        "graph_retrieval.done",
        anchors=found,
        facts=len(result.facts),
        evidence=len(result.evidence_chunk_ids),
        elapsed_ms=round(result.elapsed_ms, 1),
    )
    return result


async def neighbourhood(
    *,
    anchor_tag: str,
    hops: int = 2,
    edge_types: list[str] | None = None,
    limit: int = 300,
) -> dict[str, Any]:
    """Nodes and edges around one asset, for the graph explorer screen."""
    # Relationship types and hop counts cannot be query parameters in Cypher, so
    # this template is interpolated. Percent-formatting rather than str.format
    # because Cypher map literals use braces, which .format would try to expand.
    # The interpolated values are never user-supplied: `rels` comes from the
    # _ALL_EDGE_TYPES allow-list and the numerics are clamped ints.
    rels = "|".join(edge_types or _ALL_EDGE_TYPES)
    cypher = """
    MATCH (anchor)
    WHERE (anchor:Equipment AND anchor.canonical_tag = $tag)
       OR (anchor:Instrument AND anchor.tag = $tag)
    CALL {
        WITH anchor
        MATCH path = (anchor)-[:%(rels)s*1..%(hops)d]-(n)
        RETURN path LIMIT %(limit)d
    }
    UNWIND relationships(path) AS r
    WITH DISTINCT r
    RETURN elementId(startNode(r)) AS source_id,
           labels(startNode(r))    AS source_labels,
           properties(startNode(r)) AS source_props,
           elementId(endNode(r))   AS target_id,
           labels(endNode(r))      AS target_labels,
           properties(endNode(r))  AS target_props,
           elementId(r)            AS edge_id,
           type(r)                 AS edge_type,
           properties(r)           AS edge_props
    LIMIT %(limit)d
    """ % {"rels": rels, "hops": max(1, min(hops, 3)), "limit": limit}  # noqa: UP031

    rows = await graph.read(cypher, tag=anchor_tag)
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    for row in rows:
        for prefix in ("source", "target"):
            node_id = row[f"{prefix}_id"]
            if node_id not in nodes:
                props = _stringify(row[f"{prefix}_props"])
                nodes[node_id] = {
                    "id": node_id,
                    "labels": row[f"{prefix}_labels"],
                    "properties": _display_props(props),
                    "data_class": props.get("data_class"),
                }
        edge_props = _stringify(row["edge_props"])
        edges.append(
            {
                "id": row["edge_id"],
                "type": row["edge_type"],
                "source": row["source_id"],
                "target": row["target_id"],
                "properties": {k: v for k, v in edge_props.items() if k != "evidence_chunks"},
                "evidence_chunk_ids": list(row["edge_props"].get("evidence_chunks") or []),
                "confidence": row["edge_props"].get("confidence"),
            }
        )

    return {"nodes": list(nodes.values()), "edges": edges, "truncated": len(rows) >= limit}


_ALL_EDGE_TYPES = [
    "DESCRIBES",
    "SIBLING_OF",
    "INSTANCE_OF",
    "PERFORMED_ON",
    "RECORDED",
    "OF_MODE",
    "INVOLVED",
    "GENERATED",
    "CHANGED",
    "FEEDS",
    "ISOLATES",
    "MEASURES",
    "CONNECTS",
    "OCCUPIED_BY",
    "HAS_POSITION",
    "CONTAINS",
    "HAS_COMPONENT",
    "BELONGS_TO_LOOP",
    "MEASURED_AT",
    "RECORDED_IN",
    "APPLIES_TO",
    "SATISFIES",
    "IMPLEMENTS",
    "PROVES",
]

#: Properties that clutter a graph view without informing it.
_HIDDEN_PROPS = {"created_at", "updated_at", "asserted_at", "resolved_at", "evidence_chunks"}


def _display_props(props: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in props.items() if k not in _HIDDEN_PROPS}


def _stringify(props: dict[str, Any]) -> dict[str, Any]:
    """Neo4j temporal types are not JSON-serialisable; render them as ISO text."""
    out: dict[str, Any] = {}
    for key, value in (props or {}).items():
        if hasattr(value, "iso_format"):
            out[key] = value.iso_format()
        elif isinstance(value, list | tuple):
            out[key] = [v.iso_format() if hasattr(v, "iso_format") else v for v in value]
        else:
            out[key] = value
    return out


def _data_class(value: Any) -> DataClass:
    try:
        return DataClass(value)
    except (ValueError, TypeError):
        return DataClass.MODEL_DERIVED
