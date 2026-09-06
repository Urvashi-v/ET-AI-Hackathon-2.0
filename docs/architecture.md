# Architecture

One shared industrial knowledge substrate. The five capabilities in the brief are
not five products — they are query patterns over one ingestion pipeline, one
entity layer, one knowledge graph, one retrieval service and one evidence store.

## The seven layers

```
┌────────────────────────────────────────────────────────────────────────────────┐
│ L7  EXPERIENCE          web/  — vanilla HTML/CSS/JS, served by the API at /ui   │
│     field.html (mobile) │ copilot.html │ graph.html │ reliability.html │        │
│     compliance.html     │ ingestion.html │ index.html                           │
└──────────────────────────────────┬─────────────────────────────────────────────┘
                                   │  REST + SSE (same origin, no CORS shim)
┌──────────────────────────────────┴─────────────────────────────────────────────┐
│ L6  AGENTS              services/api/routers/{rca,compliance,notifications}.py  │
│     RCA · Compliance · Proactive.  Deterministic parts run; reasoning parts are │
│     capability-gated and say so.                                                │
└──────────────────────────────────┬─────────────────────────────────────────────┘
┌──────────────────────────────────┴─────────────────────────────────────────────┐
│ L5  RETRIEVAL           services/retrieval/                                     │
│     intent → [lexical ‖ dense ‖ graph] → fusion(RRF) → rerank → assemble →      │
│     generate → citation binding → verify → confidence → answer | abstain        │
└──────────────────────────────────┬─────────────────────────────────────────────┘
┌──────────────────────────────────┴─────────────────────────────────────────────┐
│ L4  STORES                                                                      │
│   ┌───────────────────────────────────────────┐  ┌───────────────┐  ┌────────┐  │
│   │ PostgreSQL 16 + pgvector                  │  │ Neo4j 5       │  │ Redis 7│  │
│   │ records · chunks · BM25 postings · vectors│  │ knowledge     │  │ queue  │  │
│   │ query log · citations · review queue      │  │ graph         │  │ events │  │
│   └───────────────────────────────────────────┘  └───────────────┘  └────────┘  │
└──────────────────────────────────┬─────────────────────────────────────────────┘
┌──────────────────────────────────┴─────────────────────────────────────────────┐
│ L3  KNOWLEDGE CONSTRUCTION   services/ingest/{extract,resolve,graph_writer}.py  │
│     extract → normalise → parse → block → score → decide → upsert w/ provenance │
└──────────────────────────────────┬─────────────────────────────────────────────┘
┌──────────────────────────────────┴─────────────────────────────────────────────┐
│ L2  DOCUMENT UNDERSTANDING   services/ingest/{classify,parsers/,chunk}.py       │
│     classify → parse (pdf│text│docx│tabular) → structure-aware chunk → index     │
└──────────────────────────────────┬─────────────────────────────────────────────┘
┌──────────────────────────────────┴─────────────────────────────────────────────┐
│ L1  INGESTION           services/ingest/{storage,worker}.py                     │
│     validate → content-address → dedup by hash → durable job → reliable queue    │
└────────────────────────────────────────────────────────────────────────────────┘
        ▲                                                                    │
        └───────── FEEDBACK: thumbs, corrections, review queue ◄─────────────┘
```

## The three paths

### Write path — documents become knowledge (asynchronous)

```
file → validate → store (content-addressed) → dedup → queue
     → classify → parse → chunk → BM25 index → embed*
     → extract (regex + gazetteer) → resolve (normalise/parse/block/score)
     → persist (Postgres) → upsert (Neo4j, MERGE + provenance)
     → emit "graph.changed"
```

`*` capability-gated. Every stage records its own outcome in
`ingestion_jobs.stage_report`, including the stages that could not run and the
environment variables that would enable them.

**Idempotence.** Document ids are SHA-256 content hashes; chunk and mention ids
are derived deterministically from them. Every write is an upsert on a natural
key, so re-ingesting a corpus converges instead of double-counting. This is
asserted by an integration test, because the alternative — silently doubled work
orders — corrupts every statistic without any visible symptom.

### Read path — questions become grounded answers (synchronous)

```
question → intent classification (deterministic rules)
         → entity linking (tag grammar + user context for "the B pump")
         → parallel: BM25 ‖ pgvector* ‖ graph traversal
         → reciprocal rank fusion, weighted by intent
         → cross-encoder rerank*
         → context assembly (graph triples first, strongest passage last)
         → grounded generation*  → citation binding → claim verification
         → confidence scoring → ANSWER | CAVEAT | ABSTAIN
```

### Proactive path — knowledge finds the user (event-driven)

The event bus runs today: ingestion publishes `graph.changed` and
`document.ingested`, and `/api/v1/events/stream` carries them to the dashboard's
live panel. The **pattern-matching engine** that turns an event into a pushed
warning is not implemented; `/api/v1/notifications` reports that in its `engine`
field and returns an empty list rather than sample alerts.

## Entity resolution — the load-bearing piece

The same pump appears as six different strings. If they are not unified, the
graph is six disconnected islands and every headline claim is false.

```
normalise   NFKC · uppercase · collapse every separator variant (incl. U+2011)
            · expand PUMP→P · strip leading zeros
parse       tag grammars → {unit, class, sequence, suffix}
block       key = kind|class|sequence — never compare across blocks (O(n²))
score       parse agreement → string similarity → alias evidence
decide      merge | link_sibling | needs_review | separate
```

The rule that protects everything downstream:

| Pair | Score | Relation | Action |
|---|---|---|---|
| `P-101B` / `P101B` | 0.97 | same | merge |
| `P-101B` / `10-P-101-B` | 0.85 | same | canonical form already unifies these |
| **`P-101A` / `P-101B`** | **0.30** | **sibling** | **`SIBLING_OF` edge, never merge** |
| `10-P-101-B` / `20-P-101-B` | 0.15 | different | separate |

A naive edit-distance matcher merges `P-101A` and `P-101B` — one character in
six — and silently halves every failure statistic. The decision is four-way, not
two-way: the ambiguous middle band creates the link and flags it for review
rather than guessing.

## Functional location versus equipment

The functional location is the permanent position in the process; the equipment
is the machine currently occupying it. Keeping them apart lets the reliability
question split in two, with different answers and different fixes:

* *"Does this position keep killing pumps?"* → history on the `FunctionalLocation`
  → points at process or installation: cavitation, misalignment, piping strain.
* *"Does this machine keep failing wherever we install it?"* → history on the
  `Equipment` serial → points at manufacture or repair quality.

`CDU1-PUMP-101` parses perfectly well as pump `P-101`, so without care the
extractor creates a phantom third pump beside the real `P-101A` and `P-101B`.
Tags introduced as functional locations are masked out of tag extraction and
recovered separately from the record's own field.

## Provenance

Two orthogonal labels, on everything.

**`data_class`** — where a displayed value came from. Rendered as a badge on
every dashboard value; never inferred client-side.

| Value | Meaning |
|---|---|
| `real_source_document` | extracted from a real ingested document |
| `synthetic_test_data` | from `data/synthetic/generate.py`; labelled everywhere |
| `model_derived` | a model inference — never an audit record |
| `calculated_metric` | deterministic computation over stored rows |
| `human_attested` | confirmed by a named person; the only audit-grade class |
| `reference_taxonomy` | curated vocabulary the system was configured with |

**`CapabilityStatus`** — whether a stage ran. `available`, `disabled`,
`provider_not_configured` (with the required environment variables named),
`not_implemented`, `error`.

Every asserted graph fact additionally carries its source document, page,
extraction method, confidence and assertion timestamp, plus `valid_from` /
`valid_to` where a revision can invalidate it.

## Ontology

See `docs/ontology.md`. Constraints in `database/cypher/001_constraints.cypher`;
taxonomy seed (equipment classes, failure modes, roles) in `002_ontology_seed.cypher`.
`GET /api/v1/graph/schema` reads the ontology back from the running database, so
what it reports is what actually exists rather than what was intended.
