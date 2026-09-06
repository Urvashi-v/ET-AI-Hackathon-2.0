# Ontology

The schema the whole system rests on. A good one makes the five capabilities
nearly trivial; a bad one makes them impossible.

Defined in `database/cypher/001_constraints.cypher` (constraints and indexes) and
`002_ontology_seed.cypher` (taxonomy as data). `GET /api/v1/graph/schema` reads
it back from the running database.

## Design principles

1. **Every node carries provenance.** Every asserted fact traces to a document
   and page. This is what makes citations real and audits possible.
2. **Edges are temporal.** `valid_from` / `valid_to` on anything a revision can
   invalidate. A pump's impeller diameter in 2019 is not its diameter today, and
   an RCA over stale facts is worse than no RCA.
3. **Confidence is a first-class property.** Extraction is probabilistic. Store
   the score, filter on it, show it. Low-confidence links are surfaced for review
   rather than silently asserted.
4. **The observation is separate from the interpretation.** A `Mention` (this
   string appeared here) is not an `Equipment` (this asset exists). Keeping both
   is what lets a bad resolution be undone without re-ingesting.
5. **The ontology is data, not code.** Equipment classes and failure modes are
   nodes in a taxonomy subgraph, extensible without a deploy.

## Node labels

### Asset hierarchy (ISA-95 / ISO 14224 shape)

| Label | Key | Meaning |
|---|---|---|
| `Site` `Plant` `Area` `System` | `code` | the organisational tree above equipment |
| `FunctionalLocation` | `fl_tag` | the **position** in the process — permanent |
| `Equipment` | `canonical_tag` | the **machine** currently in that position — swappable |
| `Component` `Part` | `component_id` / `part_no` | maintainable sub-assemblies and spares |
| `EquipmentClass` | `code` | ISO 14224-style taxonomy; seeded |

### Process topology (from P&IDs)

| Label | Key | Meaning |
|---|---|---|
| `Instrument` | `tag` | ISA 5.1 instrument; `PIC-101` decodes to Pressure/Indicate/Control |
| `Loop` | `loop_no` | the control loop an instrument belongs to |
| `Line` | `line_no` | piping run |
| `Valve` | `tag` | isolation and control elements |

### Evidence

| Label | Key | Meaning |
|---|---|---|
| `Document` | `doc_id` | content-hash derived; carries `doc_type`, `data_class`, revision |
| `Chunk` | `chunk_id` | retrievable passage with page and section provenance |
| `Mention` | `mention_id` | an extracted string occurrence, **before** resolution |

### History

| Label | Key | Meaning |
|---|---|---|
| `WorkOrder` | `wo_id` | the CMMS record; `as_found` is the evidence, `as_left` is not |
| `FailureEvent` | `event_id` | the interpreted failure, distinct from the order that recorded it |
| `FailureMode` | `code` | ISO 14224-style vocabulary; seeded |
| `Incident` | `incident_id` | HSE event with narrative and stated causes |
| `Inspection` `CML` | `inspection_id` / `cml_id` | readings against fixed monitoring locations |
| `MOC` | `moc_id` | approved change — the most frequently missed causal factor |
| `CAPA` `CorrectiveAction` | `capa_id` / `action_id` | remedial commitments and their status |

### Procedures and compliance

| Label | Key | Meaning |
|---|---|---|
| `SOP` `Procedure` `Step` | `proc_id` / `step_id` | procedures decomposed into ordered steps |
| `Requirement` | `req_id` | one atomised, independently testable obligation |
| `Control` | `control_id` | what the plant does about a requirement |
| `Evidence` | `evidence_id` | the proof that the control operates |

### People

| Label | Key | Meaning |
|---|---|---|
| `Role` | `name` | seeded role slots, used for expert routing on abstention |
| `Person` | `person_id` | pseudonymised; **never seeded** — only from real records |

## Relationship types, grouped by the question each answers

```
STRUCTURE    (:Site)-[:CONTAINS]->(:Plant)-[:CONTAINS]->(:System)
             (:System)-[:HAS_POSITION]->(:FunctionalLocation)
             (:FunctionalLocation)-[:OCCUPIED_BY {from,to}]->(:Equipment)
             (:Equipment)-[:HAS_COMPONENT]->(:Component)-[:USES_PART]->(:Part)
             (:Equipment)-[:SIBLING_OF]->(:Equipment)      ← duty/standby pairs
             (:Equipment)-[:INSTANCE_OF]->(:EquipmentClass)

TOPOLOGY     (:Equipment)-[:FEEDS {line_no}]->(:Equipment)
             (:Line)-[:CONNECTS]->(:Equipment)
             (:Valve)-[:ISOLATES]->(:Equipment)     ← "how do I isolate this?"
             (:Instrument)-[:MEASURES]->(:Equipment|:Line)
             (:Instrument)-[:BELONGS_TO_LOOP]->(:Loop)

EVIDENCE     (:Document)-[:HAS_CHUNK]->(:Chunk)-[:PART_OF]->(:Document)
             (:Mention)-[:APPEARS_IN]->(:Chunk)
             (:Mention)-[:RESOLVES_TO {confidence,method,reason}]->(:Equipment)
             (:Document)-[:DESCRIBES {page,confidence,evidence_chunks}]->(:Equipment)
             (:Document)-[:SUPERSEDES]->(:Document)

HISTORY      (:WorkOrder)-[:PERFORMED_ON]->(:Equipment)
             (:WorkOrder)-[:RECORDED_IN]->(:Document)
             (:FailureEvent)-[:OF_MODE]->(:FailureMode)
             (:FailureEvent)-[:CAUSED_BY {evidence,confidence}]->(:Cause)
             (:Incident)-[:INVOLVED]->(:Equipment)
             (:Incident)-[:GENERATED]->(:CAPA)
             (:MOC)-[:CHANGED {date}]->(:Equipment)
             (:Inspection)-[:MEASURED_AT]->(:CML)

COMPLIANCE   (:Requirement)-[:APPLIES_TO]->(:EquipmentClass|:Plant)
             (:Control)-[:SATISFIES {coverage,last_checked}]->(:Requirement)
             (:Procedure)-[:IMPLEMENTS]->(:Control)
             (:Evidence)-[:PROVES]->(:Control)
             (:Requirement)-[:AMENDED_BY {date}]->(:Requirement)
```

## Which of these are populated today

Populated by the Day 1 write path, from real ingested documents:

`Document` · `Chunk` · `Mention` · `Equipment` · `Instrument` · `Loop` ·
`FunctionalLocation` · `WorkOrder` · `Inspection` · `CML` · `EquipmentClass` ·
`FailureMode` · `Role` · `Requirement`, and the edges `HAS_CHUNK`, `PART_OF`,
`APPEARS_IN`, `RESOLVES_TO`, `DESCRIBES`, `PERFORMED_ON`, `RECORDED_IN`,
`MEASURED_AT`, `SIBLING_OF`, `INSTANCE_OF`, `OCCUPIED_BY`, `BELONGS_TO_LOOP`,
`APPLIES_TO`.

Defined with constraints but **not yet populated**, because doing so requires
extraction from narrative prose (capability-gated on a generation provider) or a
P&ID computer-vision pipeline that is not built:

`FailureEvent` · `Incident` (from prose) · `MOC` · `CAPA` · `Control` ·
`Evidence` · `Procedure`/`Step` · `Line` · `Valve` · `Site`/`Plant`/`Area` ·
`Component`/`Part` · `Person`, and the edges `FEEDS`, `ISOLATES`, `CONNECTS`,
`MEASURES`, `CAUSED_BY`, `OF_MODE`, `INVOLVED`, `GENERATED`, `CHANGED`,
`SATISFIES`, `IMPLEMENTS`, `PROVES`, `AMENDED_BY`, `SUPERSEDES`.

This distinction is deliberate and checkable: the constraints exist so the shape
is fixed and the queries that depend on them are already written, but no node is
created without a document to evidence it. `GET /api/v1/graph/schema` reports the
live counts, so the gap between "defined" and "populated" is always visible
rather than asserted here.

## The queries that prove the graph earns its keep

Each answers something a vector-only system cannot. `B.5` is the linkage-
completeness metric the evaluation harness reports.

```cypher
// Q1 · MULTI-HOP: what else is affected if P-101B is taken out of service?
MATCH (e:Equipment {canonical_tag: 'P-101B'})
MATCH path = (e)-[:FEEDS*1..3]->(downstream)
OPTIONAL MATCH (e)-[:SIBLING_OF]->(spare:Equipment)
OPTIONAL MATCH (v:Valve)-[:ISOLATES]->(e)
RETURN downstream.canonical_tag AS impacted, length(path) AS hops,
       collect(DISTINCT spare.canonical_tag) AS available_spare,
       collect(DISTINCT v.tag) AS isolation_points
ORDER BY hops;

// Q3 · THE COMPLIANCE GAP: the gap IS the absence of an edge.
MATCH (r:Requirement)-[:APPLIES_TO]->(:EquipmentClass)<-[:INSTANCE_OF]-(e:Equipment)
OPTIONAL MATCH (ctrl:Control)-[:SATISFIES]->(r)
OPTIONAL MATCH (ev:Evidence)-[:PROVES]->(ctrl)
WITH r, e, ctrl, max(ev.date) AS latest_evidence
WHERE ctrl IS NULL OR latest_evidence IS NULL
   OR latest_evidence < date() - duration({months: coalesce(r.frequency_months, 12)})
RETURN r.source_standard + ' ' + r.clause AS requirement, e.canonical_tag AS asset,
       CASE WHEN ctrl IS NULL THEN 'NO CONTROL' ELSE 'EVIDENCE STALE' END AS gap_type;

// B.2 · SILENT DRIFT: documents stale since an MOC changed the asset.
MATCH (m:MOC)-[c:CHANGED]->(e:Equipment)<-[:DESCRIBES]-(d:Document)
WHERE d.revised_on < c.date
RETURN e.canonical_tag, m.moc_id, m.change_desc, c.date AS changed_on,
       d.doc_id, d.type, d.revised_on AS last_revised
ORDER BY changed_on DESC;

// B.5 · LINKAGE COMPLETENESS: the evaluation metric, as a query.
MATCH (e:Equipment)<-[:RESOLVES_TO]-(m:Mention)-[:APPEARS_IN]->(:Chunk)-[:PART_OF]->(d:Document)
WITH e, count(DISTINCT d) AS docs, count(DISTINCT d.source_system) AS systems
RETURN count(e) AS total_entities,
       sum(CASE WHEN docs > 1 THEN 1 ELSE 0 END) AS multi_doc_entities,
       sum(CASE WHEN systems > 1 THEN 1 ELSE 0 END) AS cross_system_entities;
```
