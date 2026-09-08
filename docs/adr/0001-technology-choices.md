# ADR-0001 — Technology choices

**Status:** accepted · **Date:** 2026-09-06 (Day 1)

Judges rarely ask *what* you used. They ask *why*. This records the reason for
every box in the architecture, and — as importantly — what was rejected.

## Context

Every additional service costs roughly two hours of integration and adds one
class of demo-day failure. The build target is the smallest stack that can
honestly demonstrate the architecture, not the most impressive-sounding one.

## Decisions

### PostgreSQL 16 + pgvector — relational store, chunk store, lexical index, dense index

One engine carries four jobs. The alternative — OpenSearch for lexical, Qdrant
for vectors — adds two services, two client libraries and two failure modes to
buy capabilities this corpus size does not need.

**Rejected:** Qdrant. It wins on payload filtering at scale and on quantisation.
The honest trigger to adopt it is a measured one: when payload-filtered ANN over
the chunk table becomes the bottleneck. Saying "we moved to Qdrant when filtering
on 200k chunks became the bottleneck" is a good answer; "we used Qdrant because
it is popular" is not.

**Rejected:** OpenSearch. See the BM25 decision below — the lexical index we
need is small and domain-specific, and building it in Postgres let us control
the tokenizer, which mattered more than the engine did.

### A hand-built Okapi BM25 rather than Postgres full-text search

`to_tsvector('english', 'P-101B')` produces the lexemes `{p, 101b}`. The single
most important exact-match object in an industrial corpus stops being searchable
as itself. Since embeddings *also* blur `P-101B` against `P-101A` and `P-102B`,
losing it lexically means losing it entirely.

So `services/retrieval/lexical.py` owns the tokenizer — it recognises industrial
tags through the shared grammar and emits them whole, in canonical form, so
`P 101 B`, `10-P-101-B` and `P‑101‑B` all index as one token. Migration
`002_lexical_index.sql` stores the postings and computes the standard BM25
formula in SQL.

The cost is roughly 150 lines. The benefit is that the lexical leg is genuinely
good at the thing this domain needs it for, rather than nominally present.

### Neo4j 5 Community — the knowledge graph

Cypher traversal answers what no vector index can: multi-hop impact ("what is
downstream of P-101B?"), aggregation across time, and the compliance gap, which
is literally the *absence* of an edge. Cypher is also readable enough to show on
a slide.

**Constraint discovered on Day 1:** `NODE KEY` constraints are Enterprise-only.
Composite keys are expressed as a single unique property instead, with the
existence half enforced by the loader (see `database/cypher/001_constraints.cypher`).

**Rejected:** modelling the graph as a Postgres adjacency table. It would remove
a service, but variable-length traversal and the "gap is a missing edge" query
become painful, and those are the two things the graph exists for.

### Redis 7 — reliable queue, event bus, cache

A single 400-page scanned drawing must never block a query, so ingestion is
asynchronous. The queue uses `BLMOVE` into a per-worker processing list with an
explicit ack, so a worker that dies mid-job leaves its work recoverable —
a property plain `BRPOP` does not have.

The same instance carries the event bus that drives the live ingestion panel and
will drive the proactive engine.

**Rejected:** RabbitMQ/Celery. Redis was already required for the event bus, and
the queue semantics needed here fit in about 120 readable lines.

### FastAPI + Pydantic — API and typed contracts

Async streaming for SSE, and typed contracts that become the OpenAPI document
for free. Pydantic models are also where the truthfulness contract lives:
`CapabilityStatus` and `DataClass` are types, so a stage cannot quietly omit
them.

### Vanilla HTML/CSS/JS — the dashboard

Required by the project brief. It also turned out to cost little: `web/js/api.js`
and `web/js/ui.js` provide what a component library would, in ~400 lines, and the
force-directed graph in `web/js/graphview.js` is ~250 lines with no dependency.
No build step means the frontend is served directly by the API from the same
origin, so there is no CORS shim and no path by which the UI could be pointed at
fixtures.

### Deterministic, rule-based query intent classification

Not a placeholder. The intents are separated by stable lexical markers ("how
many", "why does", "what are the steps"), so rules are accurate here, cost
nothing, and give the same answer every time — which the evaluation harness
depends on. The reported `method` names the rule that fired, so a misroute is
diagnosable rather than mysterious.

Measured on the Day 1 golden set: 0.905 routing accuracy. The two failures are
both diagnostic questions phrased without a causal marker, and they are reported
rather than patched by adding their exact wording to the rules — tuning a
classifier to its own benchmark is how a benchmark stops meaning anything.

### No fake providers anywhere

Embeddings, generation, reranking and OCR are all capability-gated. With none
configured the system still performs real query understanding, real BM25, real
graph traversal, real fusion and real citation binding — and, since the
extractive composer was added, still answers: from verbatim spans of the cited
passages, which is why groundedness is a structural property here rather than a
score to be defended. `ABSTAIN_NO_ANSWER` is the answer when the evidence is
insufficient, not when a provider is missing.

A random-vector fallback would produce plausible-looking neighbours and silently
poison every retrieval metric. That failure is invisible, which is exactly why
the rule is absolute rather than pragmatic.

## Consequences

* `docker compose up` starts five containers and needs no credentials.
* Adding a provider is an `.env` change plus a migration re-run; no code changes.
* The corpus ceiling of this stack is roughly a single plant. The sizing table in
  the README says what changes beyond that, and why.
