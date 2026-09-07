# Unified Asset & Operations Brain

> *"The plant already knows the answer. It just can't remember where it wrote it down."*

One shared industrial knowledge substrate. Document ingestion, entity resolution,
a provenance-carrying knowledge graph, and hybrid retrieval — with the five
capabilities from the brief built as query patterns over it rather than as five
separate demos.

**Day 4 of the build.** This README separates what runs from what does not. See
[Feature status](#feature-status).

---

## Measured results

Produced by `python eval/run_eval.py` against a live stack, over 25 golden
questions and the 11-document corpus described below. Re-runnable; every run is
saved to `eval/results/` with the configuration it ran under.

| Metric | Value | Notes |
|---|---|---|
| Context recall (answerable) | **0.952** | did retrieval reach the documents the question needs |
| Context precision | 0.410 | 8 passages returned per question, after cross-encoder reranking |
| Entity recall | 0.844 | asset tags correctly linked from the question |
| Intent routing accuracy | 0.840 | deterministic rule classifier |
| **Abstention recall on unanswerable** | **0.750** | 4 deliberately unanswerable questions; 3 refused |
| **False abstention rate** | **0.095** | 2 of 21 answerable questions withheld — the cost side of abstaining |
| Abstentions naming a referral | 1.000 | never a bare refusal — names what's missing and who owns it |
| Mention resolution rate | 100% | 101 mentions, all resolved to a canonical asset |
| Cross-system assets | 46.7% | assets evidenced by more than one *source system* |
| Revision series reconciled | 5 | 1 with a real supersession chain, 0 needing human resolution |
| p50 / p95 latency | 1.90 s / 2.39 s | end-to-end; the cross-encoder is most of it |
| Extractions with a verified verbatim span | **100%** | every asserted fact traces to text that literally occurs in the source |
| OCR mean confidence | 0.95 | 203 words recovered from the scanned document |
| Answer correctness | **not measurable** | no LLM configured; extraction answers, but correctness needs a grader |

Per category:

| Category | n | ctx recall | intent | abstained |
|---|---|---|---|---|
| lookup | 5 | 1.000 | 1.000 | 0.000 |
| procedural | 4 | 1.000 | 1.000 | 0.000 |
| aggregate | 3 | 1.000 | 1.000 | 0.000 |
| diagnostic | 3 | 1.000 | 0.333 | 0.000 |
| multi_hop | 3 | 1.000 | 1.000 | 0.333 |
| comparative | 3 | 0.667 | 0.333 | 0.333 |
| unanswerable | 4 | n/a | 1.000 | 0.750 |

Corpus behind those numbers: 11 documents (4 real PDFs, 2 CSV exports, 5
Markdown) → 98 chunks → 101 mentions → 15 canonical assets → 2 incidents,
5 corrective actions, 1 management-of-change record, 20 requirements.

### Agent capabilities, measured on P-101B

Produced by `python scripts/demo_spine.py` against the live stack.

| Capability | Operational | What it produced |
|---|---|---|
| **RCA** | yes, no credential | 6 candidate causes ranked from 14 recorded statements; leading cause *dry running* with 5 records across 4 documents; MTBF 908.5 days; 2 open CAPAs surfaced from the sibling |
| **Lessons learned** | yes, real embeddings | 2 incidents compared, both matched — 0.73 and 0.74 cosine, one on the same pump, one on the sibling; the sibling's 2 open actions listed |
| **Compliance** | yes, evidence-backed | 20 requirements evaluated; on V-102: 4 satisfied, 5 gaps, 44.4% of 9 decidable. On P-101B: 5 gaps, 2 needing verification, 13 not evaluable |
| **Proactive** | yes, event-driven | 4 notifications from one work-order event, each with its evidence and audience |

**Two rows deserve reading together.** Abstention recall and false abstention are
a pair: either can be driven to a perfect score by a system that always abstains
or never does, and only both together say anything. 0.750 / 0.095 means the
system refuses three of four unanswerable questions while withholding one in ten
answerable ones.

**Answer correctness stays `not_measurable`.** The extractive answerer produces
real answers without a credential, but grading them against reference text needs
a judge the project does not have. `not_measurable` rather than `0.0`, because a
zero would imply the system tried and failed.

---

## Quickstart

Requires Docker and Python 3.11+. No credentials needed.

```bash
cp .env.example .env    # then set POSTGRES_PASSWORD and NEO4J_PASSWORD
docker compose up -d --build

python data/synthetic/generate.py        # CSV exports + Markdown reports
python data/synthetic/generate_pdfs.py   # real PDFs, incl. an image-only scan

python scripts/ingest_dir.py data/synthetic/generated \
  --data-class synthetic_test_data --source-system synthetic_cmms --wait
python scripts/ingest_dir.py data/synthetic/generated_pdf \
  --data-class synthetic_test_data --source-system pdf_corpus --wait
```

Then open **http://localhost:8000** (API docs at `/docs`, Neo4j browser at
`:7474`).

```bash
docker compose exec api python scripts/load_requirements.py   # compliance corpus
python eval/run_eval.py                                       # the numbers above
python -m pytest -q -m "not integration"                      # 357 unit tests
python -m pytest -q -m integration                            # 64 API tests
```

The OCR tests skip unless `tesseract` is on your PATH. It is installed in the
image, so to run them where it lives:

```bash
docker compose exec api sh -c "cd /app && python -m pytest tests/test_pdf_and_ocr.py -q"
```

`make help` lists every target. On Windows without GNU make, run the commands
directly — they are all one-liners.

---

## Feature status

Honest categories. Nothing below is described as working when it is mocked.

### Implemented and verified

| Capability | Evidence |
|---|---|
| **Universal document ingestion** | PDF, Markdown/text, DOCX, CSV, JSON, images. Classifier routes by title-block keyword, document number, **vector density** and filename, and reports which signal decided. Structure-aware chunking per document type. See [docs/ingestion.md](docs/ingestion.md). |
| **Real PDF parsing** | pdfplumber over pdfminer.six. Word geometry → **bounding boxes on every block**; ruled tables extracted as structured rows with the header repeated, and their region excluded from the prose pass so nothing is indexed twice. |
| **Real OCR** | tesseract 5.3, installed in the image. Offline, no credentials, no network call — the air-gap story stays intact. Per-word bbox and confidence, reading order grouped by `(block, paragraph, line)`. Verified: 203 words at 0.95 mean confidence from an image-only PDF. |
| **Drawing detection** | Text-chars-per-vector-object ratio, decided per page *before* table extraction. Schematic 2.2, ruled table 35.5, prose 183.9. Also suppresses the phantom tables a schematic's grid lines would otherwise produce. |
| **Extraction provenance** | Every chunk carries `extraction_method` and `extraction_confidence`. A character *read* from a text layer (1.0) and one *recognised* by OCR (the weakest word's confidence) are different kinds of fact, and the record says which. |
| **Verbatim-span validation** | Every asserted fact must be supported by a span that literally occurs in the source. Rejections are **stored with their reason**, so the rate is measurable rather than merely claimed. Currently 100% verified. |
| **Failure-vocabulary extraction** | ISO 14224-style codes recovered from free text, then compared with the CMMS-coded field: `agree` / `recoded` (dropdown default) / `disagree` (both kept, neither overwritten). |
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
| **Calibrated abstention** | Six independent signals plus three hard gates that a weighted sum cannot outvote: an asset not in the corpus, a proper noun the corpus has never recorded (a different site), and an answer that does not cover what was asked. Each abstention names the specific thing that was missing. |
| **Evaluation harness** | 25 golden questions across 6 categories plus 4 deliberately unanswerable. Measures abstention recall *and* false-abstention rate together, because either alone is gameable. Runs and reports even when accuracy is zero. |
| **Dashboard** | 7 pages, vanilla HTML/CSS/JS, no framework, no build step. Force-directed graph explorer written from scratch. Ingestion page shows real jobs: pages, chunks, entities, graph nodes, per-document duration, how each document was read (text layer vs OCR, with confidence), drawing/table flags, and errors. |
| **Provenance labelling** | Six `data_class` values on every displayed value, rendered as a badge. |
| **Background jobs** | Reliable Redis queue with per-worker in-flight lists, ack/nack, stale reclaim on restart. |
| **Dense retrieval** | Real `BAAI/bge-small-en-v1.5` through ONNX Runtime, 384-dim vectors in pgvector with an HNSW index. No API key, no torch, no network call after first download. Query and passage embedded asymmetrically, as the model was trained. |
| **Cross-encoder reranking** | Real `Xenova/ms-marco-MiniLM-L-6-v2`. Reads query and passage together and reorders the fused shortlist. No hand-rolled similarity anywhere — a fabricated score would reorder plausibly and make every retrieval metric describe something else. |
| **Extractive grounded answering** | Answers are built **only** from verbatim sentences in retrieved passages, so the copilot cannot answer a plant question from general knowledge — a structural guarantee, not a prompt instruction. Every sentence carries the citation of the chunk it came from. Needs no credential. |
| **Query decomposition** | Compound and comparative questions split into independently retrieved sub-questions, folded back into fusion at a discount. Conservative by design: only clearly separable clauses. |
| **Document revision lineage** | Revisions grouped by document number (`doc_id` is a content hash, so it cannot group them), ordered into revision *levels*, and linked with `SUPERSEDES` in both stores. Same-revision documents in different formats are recognised as renditions, not a sequence. Unorderable series are flagged for a human, never guessed. |
| **Source viewer** | Clicking a citation opens the exact extracted span with the cited sentence highlighted, the rendered PDF page beside it, the entities found in it, and — first, before content — whether the document has been superseded. |
| **Model warm-up** | ONNX sessions reach steady speed only after several inferences (measured: 15.6 s → 4.7 s → 1.8 s). Paid at startup on synthetic strings so the first real question is not the slowest. |
| **Structured record extraction** | Incident and MOC records parsed deterministically from section headings and field labels — no LLM. Every field carries the chunk that asserted it. A document without the expected structure yields **no record** rather than a guessed one, and two renditions of one report converge on a single node. |
| **Root cause analysis** | Candidate causes aggregated from cause statements the plant recorded, matched against ISO 14224 modes and a condition vocabulary, ranked by independent occurrences, subject weight, recency and cross-type corroboration. **Abstains below two records.** Open corrective actions queried across the sibling pair. |
| **Lessons learned** | Incident similarity from four independent signals — semantic (the real embedding model, over already-stored vectors), failure mechanism, graph proximity, documentary cross-reference — each returned with the evidence that produced it. Empty result when there is no precedent. |
| **Compliance evaluation** | Four verdict states, evaluated per testability mode. Only evidence-document and graph-state obligations can be decided from records; procedure-text obligations return a candidate control for human verification. `coverage_pct_of_decidable` names its own denominator. |
| **Proactive event path** | Ingestion → graph changed → precedent, compliance, open-action and superseded-procedure matching → notification. Rows in Postgres, pushed over SSE, de-duplicated on pattern among unacknowledged findings. No timers anywhere. |
| **SSE streaming** | Live ingestion events and streamed query pipeline stages — real stage completions, not timers. |

### Requires credentials — interface built, provider absent

Each reports `provider_not_configured` and names the variables. Nothing is faked.

| Capability | Set in `.env` |
|---|---|
| **Dense retrieval** (pgvector) | `EMBEDDING_PROVIDER=openai` + `OPENAI_API_KEY`, or `EMBEDDING_PROVIDER=local` + `sentence-transformers` |
| **Grounded answer generation** | `LLM_PROVIDER=openai\|anthropic` + `LLM_MODEL` + `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` |
| **LLM extraction** of causal chains, actions and obligations from prose | as above. The adapter, schema constraint and verbatim validation are built and unit-tested; only the provider is absent. |
| **Cross-encoder reranking** | `RERANKER_PROVIDER=local` |

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
8. **Verbatim regulatory text.** All 20 shipped requirements are
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
  api/        FastAPI app + routers (health, ingest, documents, query, assets,
              graph, rca, compliance, notifications, feedback, events)
  agents/     rca (cause ranking) · compliance (four verdict states) ·
              lessons (incident similarity) · proactive (the event path)
  ingest/     storage · classify · parsers/ (pdf·text·docx·tabular·image) · ocr ·
              chunk · extract · llm_extract · resolve · revisions · records ·
              record_writer · embeddings · graph_writer · pipeline · worker
  retrieval/  intent (+decomposition) · lexical (BM25) · dense (pgvector) ·
              graph_retrieval · fusion (RRF) · rerank (cross-encoder) ·
              compose (extractive) · generate (LLM) · confidence · warmup ·
              pipeline
database/
  migrations/ 6 SQL migrations — core schema, BM25 index, pgvector, enum
              extension, extraction provenance + timing, revision lineage
  cypher/     constraints and indexes · ontology seed
web/          7 HTML pages · css/base.css · js/{api,ui,graphview,sourceviewer}.js
data/
  corpus/     real documents (empty by design) + SOURCES.md + manifest schema
  synthetic/  deterministic generators (CSV/Markdown + real PDFs) + SCHEMA.md
  requirements/ atomised requirements with per-entry provenance
eval/         golden.jsonl (25 questions) · run_eval.py · results/
tests/        421 tests — 357 unit, 64 integration
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

1. **RCA candidate causes are aggregated, not reasoned.** The agent ranks
   mechanisms that recorded evidence names. It does not build a causal *tree*
   down to a systemic cause, and it cannot infer a mechanism nobody wrote down.
   That is the deliberate trade for being unable to produce a confident RCA about
   a pump it has no evidence for.
2. **Compliance decides 5 of 20 requirements for a pump.** The rest need a
   permit system (not connected), records the corpus does not hold, or human
   judgement on procedure text. Reported as `not_evaluable`, never as passing.
3. **All 20 requirements are `paraphrase_for_demo`.** No verbatim regulatory text
   is loaded, because none was supplied. The schema, provenance field and
   evaluation logic are real; the clause wording is not quotable and the API
   says so on every response.
4. **Lessons learned has two incidents to compare against.** The four signals
   and the thresholds are real, but the discrimination they provide is barely
   exercised at this corpus size. More incident reports is the single highest-
   value data addition.
5. **No abstractive generation without a credential, and correctness is
   therefore ungraded.** The extractive answerer produces real cited answers with
   no credential, so the copilot is not merely a search box. But grading answers
   against reference text needs a judge, so answer correctness is reported as
   `not_measurable` rather than as a number.
6. **Extraction cannot combine two half-answers into one sentence.** It selects
   sentences; it does not synthesise. A question whose answer is spread across
   two documents gets both sentences, not the synthesis a reader might want.
   That is the deliberate trade for being structurally unable to hallucinate.
7. **False abstention rate is 0.095** — 2 of 21 answerable questions withheld.
   Both are vocabulary mismatches: the answer is correct but reuses none of the
   question's distinctive words ("which documents *describe* this location"),
   and the relevance measure is lexical. An entailment model would fix it; a
   lower threshold would only trade these for wrong answers.
8. **One unanswerable question is answered with a caveat.** "What is the NPSH
   required for P-101B?" scores 0.36 relevance against a 0.34 floor, because the
   corpus genuinely *discusses* NPSH — an MOC notes the datasheet values no
   longer describe the machine — without stating the value. Nudging the floor to
   0.38 would score 4/4 and mean nothing; the threshold is set from measured
   separation, not from this question.
9. **Comparative questions score 0.667 context recall** (3 questions). "Which of
   the two pumps has more downtime?" names no parseable tag, so the graph leg has
   no anchor. Needs an aggregation router.
10. **Diagnostic intent accuracy is 0.33** (3 questions). Two are phrased without a
   causal marker. Deliberately *not* fixed by adding their exact wording to the
   rules — tuning a classifier to its own benchmark makes the benchmark
   meaningless.
11. **Reranking is ~2.0 s of a ~3.1 s query.** A 6-layer cross-encoder over 25
   candidates on a container CPU. `RERANK_CANDIDATES` and `ONNX_THREADS` are the
   dials; a GPU or a smaller shortlist both help. Retrieval itself (BM25 + dense
   + graph) totals ~155 ms.
12. **`Incident`, `MOC` and `CAPA` nodes are not created from prose.** The
   documents ingest and link, but the structured nodes need the LLM extractor. So
   RCA reports `incidents: 0` for P-101B even though two incident reports about
   it are ingested and retrievable.
13. **Bounding boxes are per block, not per sentence.** The source viewer does
   highlight the cited sentence in the *extracted text*, but it locates it by
   string match, and the stored rectangle still surrounds the whole passage. So
   the rendered PDF page is shown without a box drawn on the cited line.
   Per-sentence geometry needs word offsets carried through chunk splitting,
   which is not done.
14. **OCR reading order is good, not perfect.** Tesseract's `--psm 3` layout
   analysis handles the corpus correctly, but a form with columns aligned across
   a page can still interleave. The per-word geometry needed to detect and fix
   that is stored; the correction is not written.
15. **The P&ID is classified, not understood.** Vector density routes it to the
   drawing pipeline and its text layer is indexed, so tags on the sheet are
   searchable. Symbol detection, line tracing and topology reconstruction are
   not built, so `FEEDS` / `ISOLATES` edges do not exist.

16. **One unresolved review item** in the demo corpus: `P-101` appears without an
   item suffix alongside `P-101A`/`P-101B`. It is flagged for review rather than
   silently asserted as a third pump — the intended behaviour, visible on the
   ingestion page.
17. **Neo4j Community** has no `NODE KEY` constraints, so composite keys are
    single-property unique constraints with existence enforced by the loader.
18. **The corpus is synthetic content in real containers.** The PDFs are genuine
    PDF files — real text layers, real ruled tables, a real image-only scan, real
    vector geometry — but the plant they describe is invented and every page says
    so. Real industrial documents remain the single largest quality lever; see
    [data/corpus/SOURCES.md](data/corpus/SOURCES.md).
19. **Not deployable.** No auth, no multi-tenancy, no PII redaction, no TLS, no
    rate limiting. See [docs/security.md](docs/security.md).

---

## Documentation

* [docs/architecture.md](docs/architecture.md) — seven layers, three paths, entity resolution
* [docs/ingestion.md](docs/ingestion.md) — the write path: parsers, OCR, provenance, extraction, failure handling
* [docs/retrieval.md](docs/retrieval.md) — the read path: three retrievers, fusion, reranking, extractive answering, abstention
* [docs/agents.md](docs/agents.md) — RCA, compliance, lessons learned, and the proactive path
* [docs/ontology.md](docs/ontology.md) — labels, edges, and which are populated today
* [docs/security.md](docs/security.md) — what is enforced and what is not
* [docs/adr/0001-technology-choices.md](docs/adr/0001-technology-choices.md) — why each component, and what was rejected
* `/docs` on the running API — OpenAPI, generated from the Pydantic contracts
