# Demo script — 12 steps, 5–7 minutes

One asset, followed through every capability. **P-101B**, the standby crude
charge pump on CDU-1.

Everything below is live system state. There is no demo mode, no fixture server
and no scripted response — each step is a real request against the running API,
and if a stage has broken the demo shows a smaller number rather than the same
reassuring output. The figures quoted are what the system produced on the run
this document was written against; yours will match if the corpus is the same.

**Before you start**

```bash
docker compose up -d
docker compose exec api python scripts/demo_spine.py --asset P-101B
```

`demo_spine.py` walks the same path in the terminal and takes about 40 seconds.
Run it once before presenting: if it completes, every step below will work. If
you have time for nothing else, run *that* and read from it.

Open `http://localhost:8000/ui/index.html`.

---

## The story in one line

*A pump keeps failing. The answer was already written down — in four different
documents, by four different people, in four different systems, none of whom
talked to each other.*

---

## Step 1 · The problem, stated once (0:00 – 0:30)

**Say:** "P-101B is a crude charge pump. It has failed twice on the mechanical
seal. The reason is recorded — across an incident report from 2019 that exists
only as a scan, a second incident in 2022, a management-of-change record that
trimmed the impeller, and a work order history in the CMMS. Nobody has ever read
all four together."

**Show:** the Overview page. Point at the corpus counts — they are read live from
`/api/v1/documents` and `/api/v1/assets/stats`, not typed into the HTML.

> 11 documents · 98 chunks · 104 mentions · 16 canonical assets · 2 source systems

---

## Step 2 · Ingestion is real, and says what it could not do (0:30 – 1:15)

**Go to:** Ingestion.

**Say:** "Five document classes, five formats. One of these PDFs has no text
layer at all — it is a photograph of a 2019 incident report."

**Show:** the job list, then open one job. The **stage report** is the point:
every stage reports its own outcome, including the ones that could not run and
the environment variables that would enable them.

**The line to land:** "Nothing here silently skips. If OCR had not been
available, that stage would say so and name the variable — it would not quietly
return an empty page and let you think the document was blank."

---

## Step 3 · Six spellings, one pump (1:15 – 2:00)

**Go to:** Knowledge graph, search `P-101B`.

**Say:** "The same pump is written six ways across these documents: `P-101B`,
`P101B`, `P 101 B`, `10-P-101-B`, `P-101-B`. Entity resolution unifies them."

**Show:** the neighbourhood — 50 nodes, 84 edges across 2 hops. Click any edge
to see the evidence behind it: source document, page, extraction method,
confidence, and when it was asserted.

**The line to land:** point at `P-101A` sitting beside `P-101B` with a
`SIBLING_OF` edge between them, not merged.

> "One character apart. A fuzzy matcher merges these and halves every failure
> statistic in the plant. The resolver scores them 0.30, links them as siblings,
> and never merges them. That decision is four-way — merge, link, flag for
> review, separate — and the ambiguous middle goes to a human."

---

## Step 4 · The question a technician actually asks (2:00 – 2:45)

**Go to:** Copilot. Ask:

> **Why did the mechanical seal on P-101B fail after startup?**

**Show, while it streams:** each retrieval stage reporting as it completes —
intent classification, then lexical, dense and graph running in parallel, then
fusion, rerank, assembly.

**Expect:** intent `diagnostic` (0.88), confidence **0.93 ANSWER**, 8 citations,
retrieval via dense + graph + lexical, ~3–5 s.

**Say:** "No LLM is configured. That answer was assembled from verbatim spans of
the cited passages — which means it *structurally cannot* answer from general
knowledge. Groundedness on the benchmark is 1.000 across 170 claims, and that is
close to a property of the design rather than a lucky score."

---

## Step 5 · Every claim opens the document it came from (2:45 – 3:30)

**Show:** click a citation marker in the answer. The source viewer opens the
actual document, at the actual page, with the cited span highlighted.

**Say:** "Citation validity on the benchmark is 440 out of 440. Every cited chunk
is re-fetched and the snippet re-verified against stored text. A system that
cites is easy; a system whose citations resolve is the part that takes the work."

**Then — the one to slow down for:** point at the citation list. Two of the
passages are from **superseded** revisions of SOP-4412 and are badged as such.

> "Rev 3 and rev 4 of the same procedure are both in the corpus, and the answer
> drew on both. The badge is not decoration: the confidence score has a currency
> signal that reads it, and opening a superseded document puts a red banner
> across the top — *do not work to this document* — naming what replaced it."

---

## Step 6 · It refuses (3:30 – 4:00)

**Ask:**

> **What is the vibration alarm setpoint for P-999Z?**

**Expect:** `ABSTAIN_AND_ROUTE`, confidence 0.35, and a referral naming what is
missing, which document class should contain it, and which role owns it.

**Then ask the harder one:**

> **What is the current vibration reading on P-101A?**

**Say:** "This one is nastier. There *is* a real, well-cited vibration reading
for P-101A in the corpus — from 2023. Everything about answering it would look
right except that it would be two years stale. A live-state gate refuses
present-tense measurement questions and says why."

**The line to land:** "On the benchmark it refuses two thirds of what it cannot
answer, and withholds about one answerable question in ten. Both numbers are
published. Either one alone can be made perfect by a system that always or never
refuses."

---

## Step 7 · Root cause, ranked from records (4:00 – 4:45)

**Go to:** Reliability, asset `P-101B`.

**Expect:**

> 9 work orders · 1 incident · 1 sibling with 4 events · 2 open CAPAs · 1 MOC
> MTBF 908.5 days · overall confidence 0.27
>
> 1. Failure mode ELP — mechanical seal failed — 4 records (3 self, 1 sibling)
> 2. Dry running / loss of seal flush — 4 records (3 self, 1 sibling)
> 3. Failure mode OHE — running hot — 1 record

**Say:** "Each candidate cause is a statement aggregated from actual records,
ranked by how many records support it, how strong the subject relationship is,
how recent it is, and whether more than one document type corroborates it. Below
two pieces of evidence it declines to rank at all."

**The line to land:** "Confidence is 0.27 and it says so. This is a
five-work-order history — that is genuinely weak evidence, and the system is not
going to pretend otherwise to look better in a demo."

---

## Step 8 · The cross-document join (4:45 – 5:15)

**Show:** in the RCA evidence, the MOC record — `MOC-2023-07`, impeller trim —
sitting alongside the 2019 and 2022 incidents.

**Say:** "The impeller was trimmed under a management-of-change in 2023. The
equipment datasheet was never updated — the MOC's own action table says `NOT
DONE`. The 2022 incident happened on the sibling pump, and its corrective action
covers both."

**The line to land:** "That connection lives in three documents in two systems.
It is not in anyone's head, and no search box would have found it — the join is
through the graph, not through a keyword."

---

## Step 9 · Has this happened before? (5:15 – 5:45)

**Show:** the precedent panel.

> INC-2019-07 · 0.49 · [P-101B] 2019-03-22
>  — cause statements are semantically close (0.72 cosine); same equipment
>
> INC-2022-19 · 0.46 · [P-101A] 2022-08-04 · **2 open actions**
>  — cause statements semantically close (0.74); identical equipment in the same
>    service (P-101A is a sibling)

**Say:** "Four signals, weighted: semantic similarity of the cause statements
against stored embeddings, shared failure mechanism, structural relationship
between the assets, and documentary overlap. Each match shows *why* it matched.
Below 0.35 it returns nothing rather than the least-bad row."

---

## Step 10 · Compliance, with the limits stated (5:45 – 6:20)

**Go to:** Compliance. Evaluate `V-102` (a pressure vessel — more of the
requirement corpus applies to it than to a pump).

**Expect:**

> 20 requirements loaded · 4 satisfied · 5 gaps · 1 needs verification ·
> 10 not evaluable · **coverage 44.4% of 9 decidable**

**Say:** "Four testability modes. Two can be decided from stored evidence — is
there a document of the required type, in date? does the graph hold the required
relationship? The other two cannot: a procedure-text obligation returns a
candidate control for a human to verify, and a permit-record obligation needs a
permit system that is not connected."

**The two lines to land:**

> "The denominator is the decidable subset, and it says so. Reporting 4 out of 20
> would be a made-up number."
>
> "And every requirement here is a **paraphrase**, not verbatim standard text.
> The API returns that breakdown and the dashboard renders it. Nothing in this
> repository reproduces the text of a paywalled standard, and no finding here is
> fit for an actual audit."

---

## Step 11 · The drawing (6:20 – 6:45)

**Go to:** the P&ID viewer, or the P&ID panel in the field view.

**Expect:** 465 detections on one sheet, 12 distinct tags, **12 of 12 linked** to
canonical assets. Click `P-101B` on the drawing → its dossier.

**Say:** "Classical computer vision, no training data: text geometry from the PDF
for tags, Hough circles for ISA instrument bubbles, Hough lines merged into pipe
runs."

**The line to land:** "Equipment *symbol* detection — recognising a pump by its
shape — is **not implemented**. `data/pid_training/` describes the dataset that
would be required. There is no fabricated accuracy figure anywhere in this
project."

---

## Step 12 · At the machine (6:45 – 7:00)

**Go to:** `field.html`, ideally on a phone or a narrow window.

**Show:** the asset hero, collapsed dossier panels with counts on the summaries,
the big ask box, and the sync bar stating what is cached and how old it is.

**Say:** "Same backend, same citations, same abstention. The panels are collapsed
because a technician standing at the pump does not want four screens of scrolling
before the question box."

**Close on:**

> "One substrate. Ingestion, entity resolution, a knowledge graph that keeps its
> provenance, and hybrid retrieval — and the five capabilities in the brief are
> query patterns over it, not five separate demos. Every number you have seen came
> out of the running system, and the ones that have not been measured are labelled
> *not measured* rather than filled in."

---

## If something goes wrong

| Symptom | Cause | Do this |
|---|---|---|
| First query takes ~20 s | ONNX models loading on first inference | Run `demo_spine.py` once beforehand — it warms them |
| Page looks stale after an edit | browser cache | Static assets are served `no-cache`; hard-refresh once |
| A panel shows "API unreachable" | stack not up | `docker compose ps`, then `docker compose up -d` |
| Graph canvas is blank | no asset selected | Search a tag, or arrive via `graph.html?asset=P-101B` |
| Compliance shows 0 requirements | corpus not loaded | `docker compose exec api python scripts/load_requirements.py` |
| Everything is empty | volumes destroyed, nothing ingested | `./scripts/clean_start_test.sh` (~10 min, rebuilds everything) |

**Do not** re-run `clean_start_test.sh` in the ten minutes before presenting. It
destroys the volumes first.

---

## Questions you will be asked

**"Is this using GPT-4?"**
No LLM is configured at all. Answers are extractive — assembled from verbatim
spans of cited passages. `LLM_PROVIDER` and a key would enable generation; the
interface is built and the capability reports itself as not configured. The
demo you just saw needs no credential of any kind.

**"Is the data real?"**
No, and it says so on every value. The corpus is synthetic, generated
deterministically from a written plant model, and badged `Synthetic` throughout.
`data/corpus/` is where real documents go; it ships empty because inventing one
would be the exact failure this project is built to avoid.

**"What's the accuracy?"**
Entity F1 0.918, citation validity 1.000, groundedness 1.000, compliance
detection 1.000 — all measured, all reproducible. **Answer correctness is not
measured**: grading against a reference needs a judge and none is configured.
Reported as `not_measured`, never as a number.

**"How much time does it save?"**
Unknown. No user study has been run. The protocol is written in
`docs/time-to-answer.md` and marked NOT YET RUN, and the ROI model labels its
time inputs as assumptions in the field, in the result and in a banner.

**"Is it production-ready?"**
No — it is a production-oriented prototype. There is no authentication, it is
single-tenant, it runs on one machine, and it has been proven on 11 documents.
`README.md` lists what would have to be true first.
