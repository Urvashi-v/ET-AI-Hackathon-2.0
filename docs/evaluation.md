# Evaluation methodology

Every number this project reports comes from `python eval/run_eval.py` executed
against a live stack. There is no benchmark file with numbers typed into it, and
nothing below was estimated. Where a thing has not been measured, this document
says so and says what measuring it would take.

Runs are written to `eval/results/<timestamp>-<tag>.json` with the system
configuration they ran under — providers, index sizes, entity counts, app
version — so a number can always be traced back to the system that produced it.
`python eval/compare_runs.py` diffs any two runs and flags movements larger than
a 0.02 noise floor.

The run quoted throughout: **`20260908T195645Z-day7-final`**, 55 questions,
0 errored, against a stack rebuilt from destroyed volumes minutes earlier.

---

## 1. What is measured, and how

### The golden set

`eval/golden.jsonl` — 55 questions written against the corpus, each carrying the
category it belongs to, the intent it should route to, and the document ids that
must appear in the retrieved context.

| Category | n | What it tests |
|---|---|---|
| lookup | 10 | a single fact stated in one document |
| unanswerable | 9 | the corpus does not contain the answer — the system must refuse |
| procedural | 8 | a sequence of steps from an SOP |
| multi_hop | 6 | a fact reachable only by joining two documents through the graph |
| aggregate | 6 | counting or summarising across records |
| diagnostic | 5 | why something failed |
| comparative | 5 | two assets or two revisions set against each other |
| compliance | 3 | an obligation and the evidence for it |
| drawing | 3 | something detected on the P&ID |

**16.4% of the set is deliberately unanswerable.** That fraction is the point.
A retrieval benchmark made only of answerable questions rewards a system that
always answers, which is the failure mode that matters most in a plant: a
confident, well-cited, wrong answer costs more than a refusal.

The unanswerable questions are not nonsense. They ask for a seal part number
that no ingested document records, a shift roster the corpus does not contain,
the vibration setpoint of an asset that does not exist, and the *current* reading
of a real instrument — things a technician would plausibly ask and the corpus
genuinely cannot answer.

### Retrieval

| Metric | Result | Method |
|---|---|---|
| Context recall | 0.956 | fraction of answerable questions whose required documents appeared in the retrieved context |
| Context precision | 0.426 | fraction of returned passages that were required; 8 passages returned per question |
| Entity recall | 0.577 | fraction of expected asset tags present in the retrieved context |
| Mean citations | 8.0 | per answered question |
| Mean graph facts | 14.4 | triples assembled into context per question |

Context precision of 0.426 is low, and deliberately so: the pipeline returns a
fixed 8 passages regardless of how many are needed, so a question answerable
from one passage scores 0.125 by construction. Recall is the metric being
optimised, because a missing passage cannot be recovered downstream while an
extra one is only cost.

### Entity extraction

Scored against `eval/entities.jsonl` — 11 documents hand-labelled with every
asset tag that genuinely appears in them.

| | Result |
|---|---|
| Precision (micro) | **0.929** |
| Recall (micro) | **0.907** |
| F1 (micro) | **0.918** |
| Precision (macro) | 0.955 |
| Recall (macro) | 0.934 |
| Counts | 39 TP · 3 FP · 4 FN |

Precision and recall are reported together on purpose. Recall alone hides
invented equipment; precision alone hides equipment that was silently dropped. A
single defect on Day 6 did both at once — `T-101 P-101A` on the P&ID matched as
one tag `T-101 P`, creating a machine that does not exist and losing the duty
pump — and only the pair made it visible. Precision went 0.848 → 0.929 when it
was fixed.

### Linkage

| Metric | Result | Method |
|---|---|---|
| Mention resolution | **1.000** | 104 of 104 extracted tags reached a canonical asset |
| Drawing tag linkage | **1.000** | 12/12 tags detected on the P&ID resolved to an asset |
| Cross-system assets | 0.438 | 7 of 16 assets evidenced by more than one source system |
| Review queue open | 1 | one ambiguous resolution awaiting a human |

### Citation validity

**1.000 — 440 of 440.** Every citation returned across the whole run is
re-fetched from the store by its chunk id, and the snippet the answer showed is
verified to be present in the stored text. A citation that pointed at a chunk
that does not exist, or quoted text the chunk does not contain, would fail here.

This is the metric that makes the others meaningful. A retrieval score describes
nothing if the evidence trail behind it does not resolve.

### Groundedness

**1.000 — 170 of 170 claims.** Each claim in each answer is checked for verbatim
containment in the passage it cites.

That result is close to structural rather than lucky: with no LLM configured,
answers are assembled by `services/retrieval/compose.py` from verbatim spans of
the cited chunks. The composer *cannot* produce a sentence that is not in a cited
passage, so it cannot answer from general knowledge. Groundedness measures that
the property holds end to end; it is not evidence that a generative model would
behave the same way.

**Groundedness is not correctness.** It says the answer came from the corpus. It
does not say the answer is right, or that it is the right passage.

### Abstention

| Metric | Result | Reading |
|---|---|---|
| Recall on unanswerable | **0.667** | 6 of 9 unanswerable questions refused |
| False abstention rate | **0.109** | 5 of 46 answerable questions withheld |
| Abstentions naming a referral | 1.000 | never a bare refusal — always "ask X" or "check Y" |

These two rates are a pair and neither means anything alone. A system that always
abstains scores 1.000 on the first; one that never abstains scores 0.000 on the
second. The operating point chosen here refuses roughly two thirds of what it
cannot answer at the cost of withholding roughly one answerable question in ten.

The three misses are informative. Two ask for a field that a real document of
that type would contain but this one does not — a seal part number, a shift
roster — and lexical relevance cannot distinguish "this document is about the
right thing but lacks the field" from "this document answers the question". The
original four unanswerable questions from Day 3 still score 0.750 unchanged: the
rate fell because the benchmark got harder, not because the system got worse.

### Compliance gap detection

**1.000 — 13 of 13.** Scored against `eval/compliance_expectations.jsonl`, which
records a hand-determined verdict for each (requirement, asset) pair whose answer
is decidable from stored evidence.

Requirements whose testability mode is `procedure_text` or `permit_record_field`
are **excluded rather than graded**, because their correct answer is "a human
must decide" and scoring a machine against that would be measuring the wrong
thing. The reported figure is accuracy over the decidable subset, and the
coverage of that subset is reported alongside it as `coverage_pct_of_decidable`.

One entry in this reference set was wrong and the system was right: a
pressure-vessel clause had been scoped to a heat exchanger. The expectation file
was corrected, not the code.

### Latency

| | Result |
|---|---|
| p50 | **2.79 s** |
| p95 | **4.04 s** |
| max | 5.83 s |

Per-leg, over the same 55 questions:

| Leg | p50 | p95 | max |
|---|---|---|---|
| Rerank | 2610 ms | 3372 ms | 3998 ms |
| Graph | 0 ms | 315 ms | 2720 ms |
| Dense | 100 ms | 153 ms | 271 ms |
| Lexical | 5 ms | 13 ms | 24 ms |

**The cross-encoder is 83% of p95.** It is a 6-layer transformer scoring 25
candidates on a container CPU with no GPU. The dials are `RERANK_CANDIDATES` and
`ONNX_THREADS`. A repeated question is served from the reranker cache in ~130 ms.

**This number moves between runs, and the honest thing is to say so.** The Day 6
run of the identical benchmark measured p95 6.52 s and max 13.38 s; this one
measured 4.04 s and 5.83 s. Nothing in the retrieval path changed between them.
The host is Docker Desktop on Windows, where the container does not own its
cores, and CPU inference latency swings with whatever else the machine is doing.
Quote the range, not the better half.

Either way p95 misses a sub-2-second target, and the measured number is printed
rather than the target.

---

## 2. What is *not* measured

### Answer correctness — `not_measured`

Answers are produced and their groundedness is measured. Correctness is not.
Grading an answer against a reference requires a judge — a human, or a model
capable enough to be trusted with the comparison — and neither is configured.

A string-overlap score presented as accuracy would be worse than no number,
because it would be quoted as accuracy. The field reads `not_measured`, never
`0.0`, and never a proxy dressed up as the real thing.

This run produced **41 answers** from the extractive composer with no provider
configured, and all 170 of their claims were verbatim in the passage they cite.
What is missing is the judge, not the answers.

**To measure it:** set `LLM_PROVIDER` and the corresponding key, add reference
answers to `eval/golden.jsonl`, and score with a judge prompt. Or grade 55
answers by hand, which for this corpus is an afternoon.

### Time-to-answer improvement — `NOT YET RUN`

The headline claim this class of system is sold on — "reduces search from 30
minutes to 30 seconds" — requires a controlled study with real technicians. The
protocol is written in `docs/time-to-answer.md`: participants, task set,
counterbalancing, what is timed, what is recorded.

**It has not been run.** No participant has used this system under measurement.
The ROI model in `/ui/impact.html` therefore labels its time inputs
`ASSUMPTION`, in the field, in the result, and in a banner.

### Behaviour at plant scale — not established

The corpus is 11 documents, 98 chunks, 104 mentions, 16 assets. Everything above is real and
reproducible on that corpus, and none of it says how the system behaves on
100,000 documents. Specifically untested: BM25 index build time, HNSW recall at
scale, graph traversal fan-out on a dense plant model, and whether entity
resolution's blocking key holds up when one block contains thousands of tags.

---

## 3. Reproducing a run

```bash
docker compose up -d
docker compose exec api python data/synthetic/generate.py
docker compose exec api python scripts/ingest_dir.py data/synthetic/generated \
  --data-class synthetic_test_data --source-system synthetic_cmms --wait
docker compose exec api python scripts/load_requirements.py

python eval/run_eval.py --tag my-run
python eval/compare_runs.py day7-final my-run     # runs resolve by tag
```

`run_eval.py` writes both the summary and the per-question result, so a metric
can be traced to the questions that produced it. `compare_runs.py` tracks 16
metric paths and reports any movement above 0.02 as significant.

The whole thing from a destroyed database, including the eval:

```bash
./scripts/clean_start_test.sh
```

---

## 4. What the benchmark has caught

Kept as a record of what the harness is for. Each of these was found by the
measurement rather than by reading the code.

1. **A tag-extraction bug that both invented and destroyed assets.** `T-101
   P-101A` matched as one tag. Entity precision 0.848 → 0.929.
2. **A latency cliff.** p95 reached 17 s under sustained load; the reranker was
   re-scoring identical candidate sets. A content-keyed cache brought repeats to
   ~130 ms and p95 to 6.52 s.
3. **A stale answer that looked perfect.** "What is the current vibration reading
   on P-101A?" returned a real, well-cited reading from 2023 — correct in every
   respect except that it was two years old. A live-state gate now refuses
   present-tense measurement questions and says why. Abstention recall
   0.556 → 0.667.
4. **An eval field that contradicted the file it was in.** `answer_quality`
   reported "no answers were produced" while `groundedness` in the same document
   reported 170 verified claims across 44 answers. The extractive composer had
   made the text stale. Corrected to `not_measured` with an accurate reason.

5. **A document inheriting another document's revision number.** Revision was
   read by taking the number after the first occurrence of the word "revision"
   anywhere in the header, so an incident report saying "the procedure had been
   revised to SOP-4412 revision 3" was recorded as revision 3 itself. Because
   `order_revisions` treats that field as ordering evidence, a fabricated label
   can declare a real document superseded. Now read only from a labelled field in
   the document's own header block — which also recovered two revisions that had
   been missed entirely.
6. **A citation that knew it was superseded and did not say so.** `is_current`
   reached the confidence score and the assembled context but not the citation
   the browser renders, so a superseded procedure looked identical to the current
   one until you opened it.

Three of the harness's own metrics were also wrong and were fixed: intent
expectations missing for two categories added later, a drawing linkage rate
computed above 1.0, and the compliance expectation described above. In that last
case the system was right and the reference set was wrong — which is the reason
a reference set has to be reviewable rather than assumed.

---

## Related

* `eval/golden.jsonl` · `eval/entities.jsonl` · `eval/compliance_expectations.jsonl`
* `eval/metrics.py` — every metric function, one per family
* `docs/time-to-answer.md` — the study protocol, NOT YET RUN
* `docs/retrieval.md` — the pipeline these metrics describe
