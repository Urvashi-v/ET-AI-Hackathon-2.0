# Final build report

**Unified Asset & Operations Brain** — ET AI Hackathon 2026, Problem Statement 08.
Seven days. Written on the last one, against the system as it actually stands.

**This is a production-oriented prototype, not a production system.** It does a
real job end to end, from a destroyed database, with no credentials. It also has
no authentication, one tenant, one machine, and a corpus of eleven documents.
Both halves of that sentence matter, and the second half is why the phrase
"production-ready" does not appear anywhere in this repository.

---

## 1. Architecture

One shared knowledge substrate. The five capabilities in the brief are query
patterns over it, not five separate products — which is the whole architectural
bet: a work order, an incident report and a P&ID become the *same* asset's
history, so a question can cross documents that share no identifier.

| Layer | What it does | Where |
|---|---|---|
| **L7 Experience** | 8 vanilla HTML/CSS/JS surfaces, served same-origin by the API | `web/` |
| **L6 Agents** | RCA · compliance · lessons learned · proactive | `services/agents/` |
| **L5 Retrieval & reasoning** | intent → lexical ‖ dense ‖ graph → RRF → rerank → assemble → compose → verify → confidence | `services/retrieval/` |
| **L4 Knowledge stores** | PostgreSQL 16 + pgvector · Neo4j 5 · Redis 7 | `database/`, compose |
| **L3 Knowledge construction** | extract → normalise → parse → block → score → decide → upsert with provenance | `services/ingest/` |
| **L2 Document understanding** | classify → parse (pdf│text│docx│tabular│image) → OCR → chunk → index | `services/ingest/parsers/` |
| **L1 Ingestion** | validate → content-address → dedup → durable job → reliable queue | `services/ingest/` |

Three paths: the **write path** (asynchronous, per-stage reporting), the **read
path** (synchronous, ten stages), and the **proactive path** (the event bus runs;
the pattern-matching engine that would push a warning is not implemented and the
API says so).

Runtime: 5 containers, one command, no credentials. Full detail in
[`docs/architecture.md`](docs/architecture.md).

**Design decisions that carry the most weight**

* **Entity resolution is four-way, not two-way.** `P-101B` and `P101B` merge;
  `P-101A` and `P-101B` are linked as siblings and *never* merged; the ambiguous
  middle band goes to a review queue. A fuzzy matcher that merges one character
  of difference halves every failure statistic in the plant, silently.
* **Answers are extractive by default.** With no LLM configured, answers are
  assembled from verbatim spans of cited passages, so answering from general
  knowledge is structurally impossible rather than discouraged by a prompt.
* **Provenance is a column, not a convention.** Every asserted graph fact carries
  its source document, page, extraction method, confidence, and validity window.
  Every displayable value carries a `data_class` the frontend renders as a badge
  and never infers.
* **A stage that cannot run says so.** `CapabilityStatus` names the missing
  capability and the exact environment variables that would enable it. Nothing
  silently degrades into fabrication.

---

## 2. What is genuinely implemented

Everything in this section runs today, from a clean checkout, with no
credentials.

### Ingestion and document understanding
* Upload or path-based ingestion, content-addressed by SHA-256, deduplicated,
  queued to a durable worker with a visibility timeout.
* **Idempotent**: re-ingesting a corpus converges instead of double-counting.
  Asserted by an integration test, because the alternative — silently doubled
  work orders — corrupts every statistic with no visible symptom.
* Parsers for PDF, plain text, Markdown, DOCX, tabular (CSV) and image.
* **OCR on pages with no text layer**, via Tesseract, offline, no credential.
  The corpus includes an image-only scanned incident report specifically to
  exercise it.
* Document classification, structure-aware chunking, per-stage outcome reporting
  including the stages that could not run and why.

### Knowledge construction
* Tag extraction with a grammar that handles `P-101B`, `P101B`, `P 101 B`,
  `10-P-101-B`, `P-101-B`, and does **not** conflate `P-101A` with `P-101B`.
* Functional location kept distinct from equipment, so "does this position kill
  pumps?" and "does this machine fail wherever it goes?" are different questions
  with different answers.
* Revision lineage: documents grouped by document number, ordered into revision
  levels by label then date, with renditions (the same revision in two formats)
  held at one level. **Where evidence cannot order two documents, both stay
  current and a conflict is flagged** rather than guessed.
* Deterministic record extraction — work orders, incidents, inspections, CAPAs,
  MOCs — from section headings and field labels. No LLM.
* Neo4j upserts, every edge carrying provenance.

### Retrieval
* Intent classification (deterministic rules), sub-question decomposition.
* **Three retrievers in parallel**: Okapi BM25 implemented in SQL over a
  tag-preserving tokenizer, pgvector HNSW cosine over real ONNX embeddings
  (`BAAI/bge-small-en-v1.5`), and intent-scoped graph traversal with two evidence
  routes.
* **Reciprocal rank fusion**, rank-based and intent-weighted, sub-question legs
  discounted.
* **Real cross-encoder reranking** (`Xenova/ms-marco-MiniLM-L-6-v2` through ONNX
  Runtime), with a content-keyed LRU cache.
* **Extractive answer composition** with IDF-weighted relevance.
* **Citation binding and verification**: every claim re-checked for verbatim
  containment in the passage it cites.
* **Confidence**: six weighted signals plus four hard gates (unknown asset,
  off-topic answer, unknown term, live-state request).
* **Abstention with a referral**: what is missing, which document class should
  hold it, which role owns it.

### Agents
* **RCA**: candidate causes aggregated from actual records and ranked by
  occurrence count, subject weight, recency and cross-type corroboration.
  Abstains below two pieces of evidence. Reports MTBF as `insufficient_data`
  rather than computing one from two events.
* **Compliance**: four testability modes → four verdict states, with coverage
  reported over the *decidable* subset and requirement provenance counted by
  `text_status`.
* **Lessons learned**: four weighted signals (semantic over stored vectors,
  mechanism, structural, documentary), each match explaining why it matched.
* **Proactive**: matchers over real graph state, delivered on a real event bus.

### P&ID digitisation
* Tag detection from pdfplumber word geometry; ISA instrument bubbles via
  `cv2.HoughCircles`; pipe runs via `cv2.HoughLinesP` merged into runs;
  connection recovery between them.
* 465 detections on one sheet, 12 distinct tags, **12 of 12 linked** to canonical
  assets. Clickable, with the geometry each was found at.

### Experience
* Eight surfaces, vanilla HTML/CSS/ES modules, no framework and no build step.
* Loading, empty, error, and not-configured rendered as four *different* states.
* Citations open the source document at the page, with the span highlighted; a
  superseded document warns before its content.
* Mobile field view with offline caching of previously retrieved content and
  browser-native speech recognition **only where the browser genuinely has it**.
* Live graph explorer with a deterministic layout, evidence on every edge.
* ROI model where every field is labelled `USER INPUT`, `ASSUMPTION`, `MEASURED`
  or `CALCULATED`, and a sensitivity panel that shows the width of the offer.

### Engineering
* One error envelope across 41 operations; `x-request-id` on every response and
  every log line; structured JSON logging that never logs a secret.
* 7 forward-only SQL migrations, applied on startup and reported by `/health`.
* 486 tests (484 pass, 2 skip). Ruff clean, ruff-format clean, **mypy clean over
  69 source files**.
* `scripts/clean_start_test.sh` — 20 checks from destroyed volumes.
* `eval/run_eval.py` — 55-question benchmark, ten metric families.
* `scripts/gen_api_docs.py --check` — fails if the API reference has drifted.
* `scripts/capture_screenshots.py` — README screenshots regenerated from the
  running system.
* CI: lint, format, types, unit tests, corpus determinism, requirement
  provenance, then the full stack and the evaluation harness.

---

## 3. Test results

All figures below are from runs on the final code.

### Clean-room startup — **4 runs, 20/20 each**

`./scripts/clean_start_test.sh` destroys the Docker volumes first, so it cannot
pass on state left behind by an earlier run. It rebuilds the images, starts the
stack, applies migrations, generates the corpus, ingests Markdown/CSV and PDFs
(including the scanned page and the P&ID), loads requirements, then asserts the
pipeline produced something and the read path works.

| Run | Result | Notes |
|---|---|---|
| 1 | 20/20 | baseline |
| 2 | 20/20 | repeat, unchanged code |
| 3 | 20/20 | after the Day 7 API, StrEnum and accessibility changes |
| 4 | 20/20 | after the revision-extraction fix, with the image rebuilt |

An earlier run found a real gap and failed on it: `reportlab` is a dev-only
dependency absent from the runtime image, so a genuinely clean environment could
not generate the test PDFs. The script and the README were fixed, not the test.

### Test suite — **484 passed, 2 skipped** (486 collected)

402 unit, 84 integration. The 2 skips are OCR tests that need `tesseract` on the
path; they run inside the image, where it is installed.

### Static analysis

| Check | Result |
|---|---|
| `ruff check services eval tests data scripts` | clean |
| `ruff format --check` | 103 files already formatted |
| `mypy services` | **clean, 69 source files** |

mypy was *not* clean at the start of Day 7 — seven errors, one of which was a
genuine latent bug: `ComposedClaim.marker_index` was monkey-patched onto the
class after definition, so every use site was an `attr-defined` error that looks
identical to a real `AttributeError`. It is a normal property now.

### Surface consistency — **24/24**

`scripts/verify_surfaces.py` asserts every dashboard surface reads from the same
backend, with no second data path.

---

## 4. Evaluation results

Run `20260908T195645Z-day7-final` — 55 questions, 0 errored, against a stack
rebuilt from destroyed volumes minutes earlier.

| Metric | Result | How |
|---|---|---|
| Entity precision / recall / F1 | **0.929 / 0.907 / 0.918** | 11 hand-labelled documents; 39 TP, 3 FP, 4 FN |
| Citation validity | **1.000** | 440/440 cited chunks re-fetched, snippets verified against stored text |
| Groundedness | **1.000** | 170/170 claims verbatim in the passage they cite |
| Compliance gap detection | **1.000** | 13/13 hand-determined verdicts |
| Mention resolution | **1.000** | 104 of 104 extracted tags reached a canonical asset |
| Drawing tag linkage | **1.000** | 12 of 12 P&ID tags resolved to an asset |
| Context recall | 0.956 | required documents present in retrieved context |
| Context precision | 0.426 | fixed 8 passages returned per question |
| Intent routing | 0.836 | deterministic rule classifier |
| Abstention recall (unanswerable) | **0.667** | 6 of 9 refused |
| False abstention rate | **0.109** | 5 of 46 answerable questions withheld |
| Abstentions naming a referral | 1.000 | never a bare refusal |
| Cross-system assets | 0.438 | 7 of 16 evidenced by more than one source system |
| Latency p50 / p95 | **2.79 s / 4.04 s** | 55-question benchmark |
| **Answer correctness** | **Not measured** | no judge configured |
| **Time-to-answer improvement** | **Not measured** | protocol written, study NOT YET RUN |

**How to read the pairs.** Abstention recall and false abstention only mean
something together: either can be driven to a perfect score by a system that
always or never refuses. Same for entity precision and recall — recall alone
hides invented equipment.

**Groundedness is not correctness.** 1.000 says every claim appears verbatim in
the passage it cites. It does not say the claim answers the question.

**p95 moves between runs on this host.** The Day 6 run of the identical benchmark
measured 6.52 s; this one measured 4.04 s. Nothing in the retrieval path changed.
Docker Desktop on Windows does not give the container its own cores. Quote the
range, not the better half.

### What the benchmark caught

Each of these was found by measurement, not by reading code.

1. **A tag-extraction bug that both invented and destroyed assets.** `T-101
   P-101A` on the P&ID matched as one tag `T-101 P`. Precision 0.848 → 0.929.
2. **A latency cliff.** p95 reached 17 s under load; the reranker was re-scoring
   identical candidate sets. A content-keyed cache brought repeats to ~130 ms.
3. **A stale answer that looked perfect.** "What is the current vibration reading
   on P-101A?" returned a real, well-cited reading from 2023. A live-state gate
   now refuses present-tense measurement questions and says why.
4. **A document inheriting another document's revision number.** An incident
   report saying *"revised to SOP-4412 revision 3"* was recorded as revision 3
   itself. Since revision labels are ordering evidence, a fabricated one can
   declare a real document superseded. Now read only from a labelled field in the
   document's own header — which also *recovered* two revisions that had been
   missed entirely.
5. **A citation that knew it was superseded and did not say so.** `is_current`
   reached the confidence score and the assembled context but not the citation
   the browser renders. A superseded procedure looked identical to the current
   one until you opened it.

Three of the harness's own metrics were also wrong and were fixed. In one case —
a pressure-vessel clause scoped to a heat exchanger — **the system was right and
the reference set was wrong**, which is why a reference set has to be reviewable.

---

## 5. Required credentials

**None to run the stack, the demo, or the evaluation.** Embeddings, reranking and
OCR all run locally on CPU with no key and no network call after the first model
download.

| Variable | Required | Without it |
|---|---|---|
| `POSTGRES_PASSWORD` | **yes** | the stack will not start |
| `NEO4J_PASSWORD` | **yes** | the stack will not start |
| `LLM_PROVIDER` + `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | no | answers are extractive rather than synthesised; correctness stays unmeasurable; RCA returns a ranked cause list rather than a narrative causal tree; compliance content-gap detection does not run |
| `EMBEDDING_PROVIDER=openai` | no | the local ONNX embedder is used |
| `RERANKER_PROVIDER` | no | the fused RRF order is used unchanged, and the API reports that |
| `OCR_PROVIDER` | no | scanned pages ingest with no text and the stage says so |
| `CMMS_CONNECTOR`, `S3_*`, `SHAREPOINT_*` | no | interfaces exist, nothing is connected, and each reports `not_configured` |

**No secret ever reaches the browser.** The frontend is served same-origin by the
API and holds no credential. `.env` is gitignored; `.env.example` carries empty
values and names what each one unlocks.

---

## 6. Known limitations

Stated plainly. A limitation you name is worth more than one a reviewer finds.

1. **No authentication, and this is the largest gap.** Single-tenant, bound to
   localhost by the compose file. The `role` field personalises retrieval; it is
   not an authorisation boundary and withholds nothing from a caller who claims a
   different role. Deploying this anywhere shared needs an identity layer,
   per-document access control, and an audit log of who read what.
2. **The corpus is 11 synthetic documents.** Every number here is real and
   reproducible on that corpus, and none of it establishes behaviour at plant
   scale. Untested: BM25 index build time, HNSW recall at scale, graph traversal
   fan-out on a dense plant model, and whether the resolver's blocking key holds
   when one block contains thousands of tags.
3. **Answer correctness is unmeasured.** No judge is configured. Groundedness
   (1.000) is reported and correctness is not.
4. **Latency p95 is 4.04 s, not the sub-2-second target**, and it swings between
   runs on this host. The cross-encoder is ~83% of it.
5. **Abstention recall is 0.667.** The three misses ask for a field that a real
   document of that type would contain but this one does not — a seal part
   number, a shift roster — which lexical relevance cannot distinguish from a
   document that answers the question.
6. **No time-to-answer study has been run.** The protocol is written in
   `docs/time-to-answer.md`. The ROI model labels its time inputs as assumptions
   in the field, in the result, and in a banner.
7. **All 20 requirements are paraphrases**, not verbatim standard text, and are
   labelled `paraphrase_for_demo` end to end. Nothing here is fit for an audit.
8. **Equipment-symbol detection is not implemented.** Tags, instrument bubbles
   and pipe runs are detected without training data; recognising a pump by its
   silhouette is not. `data/pid_training/` specifies the dataset that would be
   needed. **No accuracy figure is quoted for it, because none has been measured.**
9. **The proactive engine is matchers, not a pattern engine.** The event bus is
   real and carries real events; what does not exist is a general rule engine
   that turns an arbitrary event into a pushed warning.
10. **31 of 41 API operations return `dict[str, Any]`** rather than a declared
    response model, so a generated client gets typed requests and untyped
    responses for most of the surface.
11. **Single-node everything.** One Postgres, one Neo4j, one Redis, one worker.
    No replication, no backup strategy, no horizontal scale path exercised.
12. **External connectors are interfaces only.** CMMS, S3 and SharePoint have
    configuration and status reporting; none has been connected to a real system.

### Claims that must NOT be made about this system

* Not "production-ready". **Production-oriented prototype.**
* Not "99% accurate" or any accuracy figure — **answer correctness is not
  measured**.
* Not "reduces search from 30 minutes to 30 seconds" — **no user study has been
  run**.
* Not "audit-ready compliance" — the requirements are paraphrases and 10 of 20
  are not machine-decidable.
* Not "reads any P&ID" — it reads text, bubbles and lines; it does not recognise
  equipment symbols.
* Not "works on real plant data" — it has never been run on a real plant corpus.
* Not "secure" — there is no authentication.

---

## 7. Demo flow

A 12-step, 5–7 minute walkthrough around **P-101B**, a standby crude charge pump,
is written in [`docs/demo_script.md`](docs/demo_script.md) with the expected
figure at each step. Every step is a live request; there is no demo mode.

The short version:

1. The problem — one pump, six documents, four formats, two systems, no shared id
2. Ingestion, and the per-stage report that names what could not run
3. Six tag spellings resolved to one asset — with `P-101A` linked, never merged
4. "Why did the mechanical seal on P-101B fail after startup?" → 0.93, 8 citations
5. Click a citation → the source page, the span, the superseded-revision badge
6. "…for P-999Z?" → abstains with a referral; "current vibration reading" → refused as stale
7. RCA: 5 candidate causes ranked from 15 recorded statements, confidence 0.27 and it says so
8. The cross-document join: MOC-2023-07 trimmed the impeller and the datasheet was never updated
9. Precedents: two prior incidents, each explaining why it matched
10. Compliance on V-102: 4 satisfied, 5 gaps, coverage 44.4% **of 9 decidable**
11. The P&ID: 465 detections, 12/12 tags linked — and symbol detection is *not* implemented
12. The phone view at the machine: same backend, same citations, same abstention

Before presenting: `docker compose exec api python scripts/demo_spine.py`. It
walks the same path in the terminal in ~40 seconds. If it completes, the demo
works. **Do not** run `clean_start_test.sh` in the ten minutes before presenting
— it destroys the volumes first.

---

## 8. Does it work from a clean clone?

Yes, and this is the claim that has been tested hardest.

```bash
git clone <repo> && cd <repo>
cp .env.example .env          # set POSTGRES_PASSWORD and NEO4J_PASSWORD
./scripts/clean_start_test.sh # ~12 minutes: destroys volumes, rebuilds, ingests, tests
```

Four runs, 20/20 each, from destroyed volumes. The one earlier failure was real
(`reportlab` missing from the runtime image) and was fixed in the script rather
than papered over.

The demo has been run repeatedly against those fresh stacks — `demo_spine.py`
walks all seven capability areas and returns real figures each time, and the
screenshots in `docs/screenshots/` were captured from a stack rebuilt from
nothing.

**The clone was tested as a clone, not assumed.** A fresh `git clone` was checked
for every file the startup script touches, confirmed to contain no `.env`, and
used to generate the synthetic corpus on Windows. All eight files hashed
byte-identically to the same corpus generated inside the Linux container:

```
406464a6…  inspection_ut_readings.csv          bfca8c02…  MANIFEST.json
70c72de3…  sop_4412_crude_charge_pump_startup.md   c6946121…  incident_2019_seal_failure.md
b782afb0…  work_orders_cmms_export.csv         cc7c1dbe…  moc_2023_07_impeller_trim.md
bfafb5ea…  incident_2022_seal_failure.md       dcdd7a43…  sop_4412_rev4_…startup.md
```

That equality is the point, not a curiosity: document ids are SHA-256 over file
bytes, so a newline translated between platforms changes a document's identity
and re-ingestion silently creates duplicates instead of converging. It happened
once. `.gitattributes` and an explicit `newline="\n"` are why it no longer does.

---

## 9. Future improvements

In the order that would add the most.

1. **Real documents.** `data/corpus/` ships empty with
   [SOURCES.md](data/corpus/SOURCES.md) naming legally usable sources per class.
   One real P&ID and ten real work orders would change more than any model
   upgrade — the abbreviations alone would rewrite the parser.
2. **An identity and authorisation layer.** The single largest gap between this
   and something deployable.
3. **The time-to-answer study.** The protocol exists; running it converts the
   ROI model's central assumption into a measurement.
4. **A judge for answer correctness.** An LLM provider plus reference answers,
   or an afternoon of hand-grading 55 answers.
5. **Response models for the remaining 31 operations**, so a generated client is
   typed on both sides.
6. **Equipment-symbol detection.** `data/pid_training/` specifies the dataset;
   a few hundred annotated sheets would make it real.
7. **A general proactive rule engine**, so an arbitrary event — a reading
   crossing a limit, a permit expiring — can raise a notification without code.
8. **Latency**: a GPU, or a distilled reranker, or a cheaper first-pass filter
   before the cross-encoder.
9. **Scale testing** at 10k and 100k documents, to find where the blocking key
   and the traversal fan-out break down.

---

## Appendix — how to verify every claim in this report

```bash
docker compose up -d
docker compose exec api python -m pytest tests -q     # 484 passed, 2 skipped
docker compose exec api python -m mypy services       # clean, 69 files
python -m ruff check services eval tests data scripts # clean
python scripts/gen_api_docs.py --check                # API reference is current
python eval/run_eval.py                               # the numbers in section 4
python eval/compare_runs.py day7-final <your-run>     # diff against this report
./scripts/clean_start_test.sh                         # section 8, from nothing
docker compose exec api python scripts/demo_spine.py  # section 7, in the terminal
```

Every figure in this document came out of one of those commands.
