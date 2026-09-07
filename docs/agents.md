# The intelligence agents

Three capabilities over one substrate: root cause analysis, compliance
evaluation, and lessons learned — plus the event path that runs all three
unprompted. None of them is a separate product; each is a query pattern over the
same documents, entities, graph and retrieval stack.

Each has a cheap, plausible, wrong version of itself. Naming them is the fastest
way to explain what these actually do:

| The wrong version | Why it is tempting | What is built instead |
|---|---|---|
| An LLM writes a fluent RCA for any pump you name | It reads like expertise and always works | Causes aggregated from cause statements the plant wrote down, ranked by independent evidence, abstaining below two records |
| "94% compliant" | One satisfying number | Four verdict states; the percentage is taken only over requirements the system can actually decide, with the denominator shown |
| Any two pump failures match | Something always appears in the panel | Four independent signals, each returned with its evidence, and an empty result when there is no precedent |
| Notifications on a timer | The demo always has something to show | Matchers over stored state, triggered by ingestion, de-duplicated by pattern |

---

## Structured records: the prerequisite

Until Day 4 the corpus held incident *documents* and no incident *records*. RCA
reported `incidents: 0` for a pump with two investigated seal failures on file,
because nothing had turned the prose into nodes. The facts were retrievable by
text search and invisible to everything else.

`services/ingest/records.py` fixes that deterministically. An incident report is
not free prose — it has a document number, a date, an equipment reference, an
immediate cause, a root cause and a table of corrective actions, all under
headings, because a form template produced it. Field labels and section headings
are matched case-insensitively against the chunks the structure-aware chunker
already produced, and **every extracted field carries the chunk that asserted
it**, so an `Incident` node opens back to the sentence behind it.

Two properties worth stating:

* **A document that does not present the expected structure yields no record.**
  A missing incident is a visible gap; a guessed one becomes a node, gets cited,
  and is believed.
* **Renditions converge.** The Markdown source and the scanned PDF of
  INC-2019-07 produce one incident, not two, because the key is the identifier
  printed on the document. Whichever rendition carries a field fills it, so the
  scan's OCR gaps are covered by the Markdown.

The OCR path needed its own work: tesseract flattens a printed CAPA table into
running text, and a pipe-delimited parser silently produced zero corrective
actions for a report listing two. Rows are now recovered positionally as well.

---

## 1. Root cause analysis

`services/agents/rca.py`. Candidate causes are **aggregated, not generated**.

Cause statements are collected from incident root/immediate causes, work-order
as-found notes, and the same fields on identical equipment. Each is matched
against two vocabularies: the ISO 14224 failure *modes* the ingest path already
extracts, and a set of *conditions* — dry running, throttled suction, cavitation,
procedure gap — because "the seal failed" is the mode and "it ran dry because the
suction was throttled" is the cause an engineer wants ranked.

Ranking combines four measurable signals:

* **independent occurrences** across distinct documents — two investigations
  reaching the same conclusion separately is the strongest evidence a plant
  produces;
* **subject weight** — this asset above a sibling, but siblings count;
* **recency**, with a five-year half-life;
* **corroboration across evidence types** — an investigation and a work-order
  note agreeing beats two of either.

`as_left` is deliberately not evidence: what was done about a failure is a
remedy, not a mechanism.

**It abstains below two records.** One recorded failure tells you what happened
once; presenting it as "leading candidate, confidence 0.9" is the LLM failure
mode reproduced with arithmetic. It also abstains when no statement names a
recognised mechanism, rather than filing them under "other" — an "other" bucket
accumulates unrelated statements and then ranks first on volume.

Open corrective actions are queried across the **sibling pair**, which is where
the payoff is: two CAPAs raised in 2022 against the duty pump, still open, and
directly about the mechanism that later took out the standby pump.

## 2. Compliance

`services/agents/compliance.py`. Requirements carry a `testable_by` mode, and
each mode is evaluated by what it can actually support:

| Mode | Can it be decided? | How |
|---|---|---|
| `evidence_document` | Yes | A record must exist, cover this asset, and be recent enough for the stated frequency. The inspection either happened within twelve months or it did not. |
| `graph_state` | Yes | A structural condition — "documents shall be revised when an approved change alters the asset" is the absence of an edge after an MOC. |
| `procedure_text` | **No** | Retrieval finds the candidate procedure and the cross-encoder scores it, but neither can confirm a clause is *satisfied*. Returns the control and asks a human. |
| `permit_record_field` | **No** | Needs the permit-to-work system, which is not connected. |

Hence four states — `satisfied`, `gap`, `needs_verification`, `not_evaluable` —
and `coverage_pct_of_decidable`, whose name states its own denominator. Merging
"we checked and it fails" with "we could not check" is the specific dishonesty a
compliance dashboard is most tempted into, because it produces one number.

An obligation is only tested against a record type the system holds. A pump
without a thickness survey has not breached a machine-guarding rule, and a report
that cries breach on category errors is one nobody reads.

## 3. Lessons learned

`services/agents/lessons.py`. Similarity from four independent signals, each
returned with the evidence that produced it:

* **semantic** — cosine between cause statements, using the same real embedding
  model the copilot retrieves with, computed against the *already stored* chunk
  vectors so there is no second index to drift;
* **mechanism** — overlap of the failure vocabulary the RCA agent ranks with;
* **structural** — same equipment > sibling > same functional location > same
  class;
* **documentary** — one report citing the other, or both citing one procedure.
  Rare and decisive: it is a human having already made the link.

An incident scores as its *best* matching chunk rather than its mean; averaging
the root-cause paragraph with the corrective-action table dilutes the signal that
matters. Below the similarity floor nothing is returned, and "this failure has no
precedent on file" is a real answer — it means the corrective actions have to be
worked out rather than looked up.

## 4. The proactive path

```
new work order / incident
    -> graph changed
    -> historical pattern matching   (lessons)
    -> compliance matching
    -> open-action matching
    -> superseded-procedure matching
    -> notification candidate
    -> notification
```

Triggered from the ingest pipeline once records are written and the graph is
updated, scoped to the assets the document actually touched — ingesting one
report should not re-evaluate the estate, and a notification storm on bulk
ingest is how the feature gets turned off.

Three rules govern raising one, because a notification interrupts a person:

* **it carries its evidence** — the incident it matched, the requirement it
  touches, the action already open;
* **it is new** — de-duplicated on `pattern_id` among *unacknowledged*
  notifications, so the same finding does not fire twice while it is on
  someone's list, but can fire again after they acknowledge it and the condition
  persists;
* **it clears a stricter threshold than browsing** — precedent worth showing
  someone who went looking is not automatically worth interrupting them for.

Notifications are rows in Postgres, pushed on `/api/v1/events/stream`. The store
is the source of truth and the stream is the push half of the same fact, so the
field page re-reads on an event rather than trusting the payload — which means a
missed event self-heals on the next one.

The most valuable output is not a prediction. It is *"this exact failure happened
on the sister pump in 2022, the investigation raised two corrective actions, and
both are still open"* — a statement of fact that was always available and that
nobody had assembled.

---

## Seeing it work

```bash
docker compose exec api python scripts/demo_spine.py
```

One asset through ingestion, graph, copilot, RCA, compliance, lessons and the
proactive path, fetched live from the running API at the moment it prints.
