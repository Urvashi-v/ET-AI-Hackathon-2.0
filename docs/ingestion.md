# The write path

Bytes in, resolved knowledge out. Every stage records what it did or why it could
not, and no stage substitutes invented output for a real one.

```
file
 → validate (extension allow-list, size, magic bytes)
 → store content-addressed (SHA-256)          ── deduplicate
 → durable job record → reliable Redis queue
 → parse (pdfplumber | text | docx | tabular | image)
 → OCR if there is no text layer              ── tesseract
 → classify (title block, doc number, vector density, filename)
 → chunk (strategy per document type)
 → index (Okapi BM25) · embed (capability-gated)
 → extract (tag grammar · gazetteer · failure vocabulary · LLM where reasoning is needed)
 → verbatim-span validation                   ── reject what cannot be evidenced
 → resolve (normalise → parse → block → score → decide)
 → persist (PostgreSQL) → upsert (Neo4j, MERGE with provenance)
 → emit graph.changed
```

## Deduplication

The document id **is** the content hash: `doc_<first 24 hex of SHA-256>`. Chunk
and mention ids derive deterministically from it. Every write is an upsert on a
natural key.

The consequence is that re-ingesting a corpus converges instead of duplicating.
That property is load-bearing rather than merely tidy: the alternative is
double-counted work orders, which corrupts every failure statistic computed over
the graph while looking completely normal. An integration test asserts that a
second submission of the same bytes accepts zero files and changes no counts.

Identity is content, not filename. The same drawing arriving from a CAD vault and
from an email attachment under two different names is one document.

## Parsers

| Format | Parser | What it preserves |
|---|---|---|
| PDF | pdfplumber (pdfminer.six) | word geometry → block bounding boxes; ruled tables as structured rows; vector-object counts |
| PDF, no text layer | pypdfium2 render → tesseract | per-word bbox and confidence, reading order by (block, paragraph, line) |
| Images | tesseract | as above |
| Markdown / text | in-house | heading hierarchy → `section_path`; numbered-step detection |
| DOCX | python-docx | Word style names → heading hierarchy; tables as rows |
| CSV / JSON | in-house | column mapping per source system; a composed NL summary per record |

### Why the table region is excluded from the prose pass

A ruled table's cells are extracted as `table_row` blocks with the header
repeated per row, and that region is then subtracted from the word set the prose
pass sees. Without the subtraction the same numbers are indexed twice — once with
their column labels and once as a meaningless run of digits — and the second copy
competes with the first at retrieval time.

### Drawing detection

The discriminator is the **ratio** of text characters to vector objects, not
either number alone. A dense numeric table has plenty of both; a schematic has
comparable geometry and almost no text, because its content is topology.

Measured on the corpus:

| Document | Vector objects | Text chars | chars/object | Verdict |
|---|---|---|---|---|
| P&ID schematic | 176 | 392 | 2.2 | **drawing** |
| Ruled thickness table | 16 | 568 | 35.5 | not a drawing |
| Prose procedure | 11 | 2 023 | 183.9 | not a drawing |

The verdict is decided **per page, before table extraction**, because a
schematic's frame and grid lines look exactly like table rules. Running table
extraction over a drawing produces dozens of rows of fragments (`": LS | : H"`)
that are worse than no output at all.

## OCR

`tesseract`, installed in the image. Real, offline, no credentials, and no
network call at run time — which is what keeps the air-gapped deployment story
intact.

`image_to_data` is used rather than `image_to_string`, because the former returns
per-word geometry and confidence and the latter throws both away. What comes back
is words with bounding boxes and confidence, which is exactly what the provenance
contract requires: an anchorable citation and a quality number that can be
trended per document rather than assumed.

**Reading order.** Words are grouped by `(block, paragraph, line)`. Tesseract
restarts `line_num` inside every paragraph, so grouping on `(block, line)` merges
a heading with the first line of the body beneath it — and since the merged words
are then ordered by x position, the output interleaves:

```
IMMEDIATE CAUSE
The outboard mechanical seal failed.
        ↓  grouped on (block, line)
"The IMMEDIATE outboard mechanical CAUSE seal failed."
```

That is not a cosmetic defect. It destroys every phrase the extractors look for:
before the fix the scanned incident report yielded zero failure-mode terms;
after it, `mechanical seal failed → ELP` with a verified verbatim span.

**Quality is reported, never assumed.** Mean confidence per document, count of
words below the 60% threshold, and a per-page warning when a page was read badly.
Low-confidence text is retained but marked — silently passing garbage into the
graph is worse than passing nothing, because nothing is visibly missing while
garbage is invisibly wrong.

## Provenance on every extracted item

| Field | Where it comes from |
|---|---|
| `doc_id` | SHA-256 of the content |
| `page` | parser, per block |
| `bbox` | word geometry (text layer) or OCR word boxes; `NULL` when a chunk spans a page break, because a rectangle across two pages would be a lie |
| `extraction_method` | `pdfplumber.text_layer` · `pdfplumber.table` · `ocr` · `python-docx` · `tabular` · `mixed` |
| `extraction_confidence` | 1.0 for a read text layer; the **weakest** word confidence for OCR |

The confidence rule is deliberate: a chunk is only as trustworthy as its worst
line, so the minimum is carried rather than the mean. And a character *read* from
an embedded text layer is not the same kind of fact as a character *recognised*
by OCR — anything downstream that treats them alike is making an assumption it
should not, so the method travels with the text all the way to the citation.

## Extraction

**Deterministic first.** Tag grammars, dates, quantities with units, document and
clause references, and an ISO 14224-style failure vocabulary. High precision,
zero cost, reproducible — and it covers the large majority of what is in
industrial text.

The failure vocabulary earns its place by making one specific comparison
possible:

| Coded field | Narrative says | Verdict |
|---|---|---|
| `OTHER` | "seal faces scored" | **recoded** — the dropdown default carries no diagnostic content |
| `ELP` | "seal faces scored" | agree |
| `BRD` | "seal faces scored" | **disagree** — both recorded, neither overwritten |

The structured field lies and the free text tells the truth: failure codes are
picked from a dropdown by a tired technician at the end of a shift. Disagreement
is surfaced, not resolved — the coded field is what the plant's own records
assert, and replacing it silently would be the same failure this system exists to
prevent.

**A model only where reasoning is genuinely required**: the causal chain in a
narrative, the action actually taken, the obligation inside a regulatory
paragraph. Gated twice — on the provider being configured, and on the chunk
plausibly containing a narrative at all (long enough, right kind, already showing
failure vocabulary). A table of thickness readings costs nothing.

### The verbatim-span guard

Every asserted fact must be supported by a span that **literally occurs** in the
source chunk. One string containment test, whitespace-insensitive so a reflowed
quote still counts, no second model, fully deterministic.

* A tag the model asserts that is not in the text → the whole extraction is
  rejected. A phantom asset corrupts every count computed over the graph,
  invisibly.
* A paraphrased quote → that field is rejected with the reason recorded.
* A field with no quote → rejected.
* A failure mode outside the closed vocabulary → rejected.
* `null` → not a rejection. Null is a correct answer and is preferred to a guess.

**Rejections are stored, not discarded.** Keeping them is what makes the rate
measurable: "0.0% of asserted facts lack a verified source span" is only a number
if the denominator survives. The `extractions` table holds both.

## Failure handling

| Situation | Behaviour |
|---|---|
| One page malformed | Skipped with a warning; the rest of the document still ingests |
| One document fails | Recorded as failed on the job with the error; the batch continues |
| No text layer | Not an error — a routing decision. Reported as "scan, routed to OCR" |
| No OCR provider | Document recorded with metadata, queued for review, **no text invented** |
| Encrypted PDF | Reported; no blocks |
| Garbage bytes, wrong magic | Rejected at the storage boundary before anything is written |
| Empty file | Rejected at the storage boundary |
| Unknown extension | Recorded as unreadable with the reason |
| Worker dies mid-job | The job stays in its in-flight list and is reclaimed on restart |
| Job fails twice | Marked `failed` with the reason; not retried forever |

## Current corpus

10 documents → 91 chunks → 92 mentions → 15 canonical assets, 56 extractions all
with verified verbatim spans, 1 document read by OCR (203 words at 95% mean
confidence), 1 classified as a drawing.
