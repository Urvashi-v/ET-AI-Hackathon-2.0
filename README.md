# Unified Asset & Operations Brain

> *"The plant already knows the answer. It just can't remember where it wrote it down."*

One shared industrial knowledge substrate. Document ingestion, entity resolution,
a provenance-carrying knowledge graph, and hybrid retrieval — with the five
capabilities from the brief built as query patterns over it rather than as five
separate demos.

**Day 1 of the build.** This README separates what runs from what does not. See
[Feature status](#feature-status).

---

## Measured results

Produced by `python eval/run_eval.py` against a live stack, over 21 golden
questions and the 6-document corpus described below. Re-runnable; every run is
saved to `eval/results/` with the configuration it ran under.

| Metric | Value | Notes |
|---|---|---|
| Context recall (answerable) | **0.941** | did retrieval reach the documents the question needs |
| Context precision | 0.402 | 8 passages returned per question, unreranked |
| Entity recall | 0.821 | asset tags correctly linked from the question |
| Intent routing accuracy | 0.905 | deterministic rule classifier |
| **Abstention recall on unanswerable** | **1.000** | 4 deliberately unanswerable questions, all refused |
| Abstentions naming a referral | 1.000 | never a bare refusal — names what's missing and who owns it |
| Mention resolution rate | 100% | 61 mentions, all resolved to a canonical asset |
| Multi-document assets | 57.1% | assets evidenced by more than one document |
| p50 / p95 latency | 0.071 s / 0.102 s | end-to-end, retrieval-only configuration |
| Answer correctness | **not measurable** | no generation provider configured — reported as such, never as zero |

Corpus behind those numbers: 6 documents → 75 chunks → 2,538 BM25 postings →
61 mentions → 7 canonical assets, 15 work orders, 24 inspection readings,
20 atomised requirements.

**Read the last row carefully.** With no LLM configured the system produces no
prose answers, so answer correctness cannot be measured. The harness reports
`not_measurable` rather than `0.0`, because a zero would imply the system tried
and failed.

---

## Quickstart

Requires Docker and Python 3.11+. No credentials needed.

```bash
cp .env.example .env    # then set POSTGRES_PASSWORD and NEO4J_PASSWORD
docker compose up -d --build
python data/synthetic/generate.py
python scripts/ingest_dir.py data/synthetic/generated --data-class synthetic_test_data --wait
```

Then open **http://localhost:8000** (API docs at `/docs`, Neo4j browser at
`:7474`).

```bash
docker compose exec api python scripts/load_requirements.py   # compliance corpus
python eval/run_eval.py                                       # the numbers above
python -m pytest -q -m "not integration"                      # 219 unit tests
python -m pytest -q -m integration                            # 55 API tests
```

`make help` lists every target. On Windows without GNU make, run the commands
directly — they are all one-liners.

---

## Feature status

Honest categories. Nothing below is described as working when it is mocked.

### Implemented and verified

| Capability | Evidence |
|---|---|
| **Universal document ingestion** | PDF, Markdown/text, DOCX, CSV, JSON. Classifier routes by title-block keyword, document number, vector density and filename, and reports which signal decided. Structure-aware chunking per document type. |
| **Idempotent ingestion** | Content-hash document ids, deterministic chunk/mention ids, upserts throughout. Re-submitting a corpus accepts 0 files and changes no counts — asserted by test. |
| **Industrial tag normalisation** | Six spellings of one pump unify: `P-101B`, `P101B`, `P 101 B`, `10-P-101-B`, `P-101-B`, `P‑101‑B` (U+2011). Plus `CDU1-PUMP-101B`, `Pump 101 B`, `P-0101B`. ISA 5.1 instruments, line numbers, KKS designations. |
| **Sibling protection** | `P-101A` and `P-101B` are **never merged** — scored 0.30 → `SIBLING_OF` edge. 82 tests on the tag grammar alone. |
| **Functional location vs equipment** | Modelled separately, so "does this *position* kill pumps?" and "does this *machine* keep failing?" stay distinct questions. |
| **Entity resolution with review queue** | Four-way decision: merge / link sibling / needs review / separate. Ambiguous cases are surfaced, not guessed. |
| **Knowledge graph** | Neo4j, 45 constraints and indexes. Every asserted fact carries source document, page, method, confidence and assertion time. |
| **Lexical retrieval** | Real Okapi BM25 over a tag-preserving tokenizer, in Postgres. Needs no credentials. |
| **Graph retrieval (GraphRAG)** | Intent-scoped edge traversal plus two evidence routes: direct mention, and documents that `DESCRIBES` the asset — the route that finds a procedure whose steps never repeat the tag. |
| **Reciprocal rank fusion** | Rank-based, intent-weighted. Degrades cleanly when a leg is unavailable. |
| **Citations** | Every passage resolves to chunk, document, page and section. Graph edges resolve to the passages that asserted them. |
| **Calibrated abstention** | Five independent signals. A question naming an asset not in the corpus is capped and refused, naming the asset. |
| **Evaluation harness** | 21 golden questions across 6 categories, 19% deliberately unanswerable. Runs and reports even when accuracy is zero. |
| **Dashboard** | 7 pages, vanilla HTML/CSS/JS, no framework, no build step. Force-directed graph explorer written from scratch. |
| **Provenance labelling** | Six `data_class` values on every displayed value, rendered as a badge. |
| **Background jobs** | Reliable Redis queue with per-worker in-flight lists, ack/nack, stale reclaim on restart. |
| **SSE streaming** | Live ingestion events and streamed query pipeline stages — real stage completions, not timers. |

### Requires credentials — interface built, provider absent

Each reports `provider_not_configured` and names the variables. Nothing is faked.

| Capability | Set in `.env` |
|---|---|
| **Dense retrieval** (pgvector) | `EMBEDDING_PROVIDER=openai` + `OPENAI_API_KEY`, or `EMBEDDING_PROVIDER=local` + `sentence-transformers` |
| **Grounded answer generation** | `LLM_PROVIDER=openai\|anthropic` + `LLM_MODEL` + `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` |
| **LLM extraction** of failure modes, causes, obligations from prose | as above |
| **Cross-encoder reranking** | `RERANKER_PROVIDER=local` |
| **OCR** for scanned documents and P&ID rasters | `OCR_PROVIDER=paddle\|tesseract` (adapter not written) |

### Partially implemented

| Capability | What runs | What does not |
|---|---|---|
| **Maintenance / RCA intelligence** | Evidence gathering across the asset, its siblings, inspections, MOCs and open CAPAs. Reliability metrics (MTBF, downtime, corrosion rate) computed from stored rows. Similar events ranked by structural signals, each with the reason it matched. | The causal tree down to a systemic cause. Capability-gated; returns `causal_tree: null`, never a template. |
| **Compliance intelligence** | 20 atomised requirements loaded with per-requirement provenance. Coverage-gap and evidence-staleness detection as real graph queries. | Content-gap detection (does the procedure text *entail* the obligation?) needs NLI. Evidence-package generation not implemented. |
| **Incident / MOC extraction** | Incident and MOC documents ingest, chunk by causal section, and link to assets. | `Incident`, `MOC` and `CAPA` *nodes* are not created from narrative prose — that needs the LLM extractor. Tabular incidents would populate them today. |

### Not implemented — declared, not pretended

| Capability | Status |
|---|---|
| **Proactive push engine** | The event bus runs and carries `graph.changed`. The pattern-matching engine that turns an event into a pushed warning does not exist. `/api/v1/notifications` reports `not_implemented` and returns an **empty list** — no sample alerts. |
| **P&ID topology reconstruction** | Classifier detects drawings and routes them; symbol detection, line tracing and graph reconstruction are not built. |
| **Lessons-learned clustering** | Composite incident similarity and community detection not built. |
| **Evidence-package export** | Reports `not_implemented`. The schema to support it exists (content hashes, verified quotes, `data_class`). |
| **Authentication / multi-tenancy / PII redaction** | None. See [docs/security.md](docs/security.md) before deploying anything. |

### External integrations — interface only

`CMMS_CONNECTOR`, `S3_CONNECTOR_ENABLED`, `SHAREPOINT_CONNECTOR_ENABLED` exist as
configuration with a defined shape. **None is connected**, and each reports
`not_configured`. No connector emits synthetic records.

---

## What you need to provide

Nothing is required to run the stack. To lift specific ceilings:

1. **An LLM API key** (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`) plus `LLM_MODEL`
   and `LLM_PROVIDER`. This is the single highest-value addition: it enables
   grounded answer generation, prose extraction of failure modes and causes, the
   RCA causal tree, and compliance content-gap detection — and it makes answer
   correctness measurable.
2. **An embedding provider.** `EMBEDDING_PROVIDER=openai` reuses the same key; or
   `local` for a fully air-gapped path (needs `sentence-transformers`, not in
   `requirements.txt` because it pulls in torch). Enables the dense retrieval leg.
3. **Real industrial documents.** The largest quality lever. `data/corpus/` ships
   empty by design with [SOURCES.md](data/corpus/SOURCES.md) naming legally
   usable sources per document class, and a manifest schema that requires
   provenance and licence per file. A real P&ID and ten real work orders would
   change more than any model upgrade.
4. **Verbatim regulatory text.** All 20 shipped requirements are
   `paraphrase_for_demo` and are labelled as such everywhere, including on the
   compliance dashboard. See [data/requirements/README.md](data/requirements/README.md).

---

## Data honesty

Four categories, distinguished in the database, the API and the UI:

| Badge | Meaning |
|---|---|
| `REAL SOURCE` | extracted from a real ingested document |
| `SYNTHETIC` | from `data/synthetic/generate.py` — not a real plant, not evidence |
| `MODEL-DERIVED` | a model inference; never an audit record |
| `CALCULATED` | deterministic computation over stored rows |
| `ATTESTED` | confirmed by a named person — the only audit-grade class |
| `TAXONOMY` | curated reference vocabulary |

The demo corpus is **entirely synthetic** and says so in every file's own
content, in its manifest, in every database row and on every dashboard badge.
It is deterministic: same seed, byte-identical output, verified by test. The
plant model and failure history are scripted, not sampled — only the surface
forms vary, which is where the deliberate messiness lives (six tag spellings,
dropdown-default failure codes, missing close-out dates, mixed date formats,
technician shorthand). See [data/synthetic/SCHEMA.md](data/synthetic/SCHEMA.md).

---

## The demo spine

Everything converges on one deeply documented asset: **P-101B**, a standby crude
charge pump.

```
P-101B ──SIBLING_OF──> P-101A          duty/standby pair, linked never merged
   │                        │
   │                        └── 2022 seal failure (INC-2022-19) — same mechanism
   ├──OCCUPIED_BY── CDU1-PUMP-101      the position, distinct from the machine
   ├──DESCRIBES──── 2019 incident report      dry running during startup
   ├──DESCRIBES──── CMMS work order export    15 orders, 2019–2025
   ├──DESCRIBES──── MOC-2023-07               impeller trimmed 330→310 mm
   └──DESCRIBES──── SOP-4412 rev 3            limit reduced 12→10 barg
```

Ask *"why does P-101B keep failing on the seal?"* and retrieval reaches the 2019
incident, the 2022 incident on the **sibling**, and the work orders — three
documents that share no identifier and were never linked in any source system.

---

## Repository layout

```
services/
  common/     config · logging · errors · db · graph · bus · tags · schemas · migrate
  api/        FastAPI app + routers (health, ingest, query, assets, graph, rca,
              compliance, notifications, feedback, events)
  ingest/     storage · classify · parsers/ · chunk · extract · resolve ·
              embeddings · graph_writer · pipeline · worker
  retrieval/  intent · lexical (BM25) · dense · graph_retrieval · fusion ·
              generate · confidence · pipeline
database/
  migrations/ 4 SQL migrations — core schema, BM25 index, pgvector, enum extension
  cypher/     constraints and indexes · ontology seed
web/          7 HTML pages · css/base.css · js/{api,ui,graphview}.js
data/
  corpus/     real documents (empty by design) + SOURCES.md + manifest schema
  synthetic/  deterministic generator + SCHEMA.md
  requirements/ atomised requirements with per-entry provenance
eval/         golden.jsonl (21 questions) · run_eval.py · results/
tests/        274 tests — 219 unit, 55 integration
docs/         architecture · ontology · security · adr/
```

Languages, each with a real purpose: Python (services, eval, generators),
JavaScript (dashboard), HTML, CSS, SQL (migrations, BM25 scoring function),
Cypher (constraints, ontology, traversals), YAML (compose, CI), Shell (scripts),
JSON (requirements, manifests, eval fixtures), Markdown (docs, synthetic corpus).

---

## Verifying

```bash
docker compose ps                            # 5 containers, health status
curl -s localhost:8000/health | python -m json.tool
python -m pytest -q -m "not integration"     # 219 passed, no Docker needed
python -m pytest -q -m integration           # 55 passed, needs the stack
python -m ruff check services eval tests data scripts
python -m mypy services                      # clean over 51 source files
python eval/run_eval.py
```

---

## Known limitations

Stated plainly, because a limitation you name is worth more than one a reviewer
finds.

1. **No answer generation without a credential.** Every query returns
   `ABSTAIN_NO_GENERATOR`. Retrieval, fusion, citation binding and confidence all
   run and the evidence is real, but there is no prose answer to grade.
2. **Cross-system linkage is 0%.** Everything ingested so far carries one
   `source_system` label, so the platform's headline claim is not yet
   demonstrated numerically. Ingesting a second source with a different label is
   what makes that metric mean something. The dashboard says this in place of
   showing a flattering number.
3. **Comparative questions score 0.0 context recall** (1 question). "Which of the
   two pumps has more downtime?" names no parseable tag, so the graph leg has no
   anchor. Needs an aggregation router.
4. **Diagnostic intent accuracy is 0.33** (3 questions). Two are phrased without a
   causal marker. Deliberately *not* fixed by adding their exact wording to the
   rules — tuning a classifier to its own benchmark makes the benchmark
   meaningless.
5. **`Incident`, `MOC` and `CAPA` nodes are not created from prose.** The
   documents ingest and link, but the structured nodes need the LLM extractor. So
   RCA reports `incidents: 0` for P-101B even though two incident reports about
   it are ingested and retrievable.
6. **No bounding boxes**, so citations resolve to page and section but not to a
   highlighted span. `pypdf` gives reading order, not per-span geometry; the
   schema and citation contract already carry the field.
7. **Reranking is not configured**, so the fused RRF order is used unchanged.
   This is reported on every query rather than silently skipped.
8. **One unresolved review item** in the demo corpus: `P-101` appears without an
   item suffix alongside `P-101A`/`P-101B`. It is flagged for review rather than
   silently asserted as a third pump — the intended behaviour, visible on the
   ingestion page.
9. **Neo4j Community** has no `NODE KEY` constraints, so composite keys are
   single-property unique constraints with existence enforced by the loader.
10. **Not deployable.** No auth, no multi-tenancy, no PII redaction, no TLS, no
    rate limiting. See [docs/security.md](docs/security.md).

---

## Documentation

* [docs/architecture.md](docs/architecture.md) — seven layers, three paths, entity resolution
* [docs/ontology.md](docs/ontology.md) — labels, edges, and which are populated today
* [docs/security.md](docs/security.md) — what is enforced and what is not
* [docs/adr/0001-technology-choices.md](docs/adr/0001-technology-choices.md) — why each component, and what was rejected
* `/docs` on the running API — OpenAPI, generated from the Pydantic contracts
