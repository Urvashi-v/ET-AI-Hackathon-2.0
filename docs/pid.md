# P&ID digitisation

A P&ID is the densest engineering document a plant holds and the one least
served by text search. It states, graphically, that this pump discharges into
that exchanger through that line, protected by that relief valve — and none of
that survives being turned into a bag of words.

The capability that matters is not "we read the drawing". It is that **`P-101B`
on sheet 3 becomes the same node** as `P-101B` in the incident report and
`P-101-B` in the CMMS export, with coordinates — so "show me this pump" returns
its failure history, its procedure, and the rectangle it occupies on the sheet.

## Four detectors, in descending order of trust

| Detector | Method | Training data | Status |
|---|---|---|---|
| Tag localisation | PDF word geometry + the project's tag grammar | none | **available** |
| Instrument bubbles | `cv2.HoughCircles` | none | **available** |
| Process lines | `cv2.HoughLinesP`, merged into runs | none | **available (partial)** |
| Equipment symbols | needs a trained detector | **required** | **not implemented** |

### 1. Tag localisation — exact, not inferred

Vector PDFs carry per-word geometry, so a tag on the sheet already *has* a
rectangle; there is nothing to infer. Words are grouped into runs on the same
baseline and every prefix of each run is offered to the tag grammar — the same
code that reads tags out of incident reports.

The grouping is necessary because PDF extraction splits on kerning, not meaning:
`P-101B` can arrive as `P`, `-`, `101B`. A detector that only parses single
words misses most tags on a drawing.

Confidence reflects how many words had to be joined. A tag read from one word is
certain — the geometry is exact and the grammar is deterministic. One assembled
from three rests on a guess about spacing, and the score says so.

### 2. Instrument bubbles — classical, untrained

ISA 5.1 draws instruments as circles of a consistent size. A Hough circle
transform finds circles by voting for centres in an accumulator built from image
gradients. It does not need to be taught what an instrument looks like, and its
failure modes are geometric and inspectable — it misses broken outlines, it
invents circles in dense hatching — which is more than can be said for an
unmeasured model.

Each circle is then named by the tag drawn inside it. An unnamed circle is nearly
useless: "there is an instrument here" is worth much less than "PIC-101 is here".

### 3. Process lines — partial, and honest about it

A probabilistic Hough transform recovers straight segments. Two things fragment
a single drawn pipe: Canny reports *both* gradient edges of a stroke, and
`HoughLinesP` breaks a run wherever the accumulator dips. Raw output for one A3
sheet was **857 segments**, which is not 857 pipes. Segments are grouped by
orientation and position across the run, then merged along it — giving **439
runs**, which is a number an engineer would recognise.

Topology is the conservative subset: two symbols are connected when one detected
segment demonstrably touches both. Collinear joining across symbol gaps, elbow
following and process-versus-signal-line discrimination are **not** implemented,
so the result is a *lower bound* on connectivity. Nine connections were recovered
from the demo sheet. The absence of a connection means nothing.

### 4. Equipment symbols — declared, not faked

Telling a centrifugal pump from a vessel from an exchanger means recognising a
silhouette, and there is no classical transform for that. No pretrained detector
ships with ISA symbology. The stage reports `not_implemented` and emits nothing.

[`data/pid_training/README.md`](../data/pid_training/README.md) specifies exactly
what dataset would be needed: ~200 annotated sheets minimum, 1,000+ for
production, 11 classes matching the graph ontology, split **by drawing office**
rather than randomly — because symbol conventions drift between contractors and
a random split reports an accuracy that will not survive new drawings.

## On confidence, and on accuracy

The stored numbers are **detector-specific quality measures, not probabilities**.
A Hough circle's score is radius agreement with the sheet's dominant bubble size;
a tag's is word-assembly certainty. They are comparable within a method and
meaningless across methods, and the database column carries a comment saying so.

**No accuracy figure is reported anywhere in this project**, because none has
been measured against a labelled ground truth. Inventing one would be the single
most dishonest thing here — a plausible mAP number is far more dangerous than an
admitted gap.

## What is stored

`drawing_detections` holds one row per detected thing: kind, text, the rectangle
in PDF points, the page dimensions it was measured against, the detector that
found it, its quality score, and the canonical asset it resolved to.

`linked_asset_id` is nullable and **NULL is a real answer**. A tag on a drawing
naming equipment the corpus has never ingested is the gap between what the plant
has drawn and what it has recorded — one of the more useful things this pipeline
finds. On the demo sheet: 11 of 12 tags linked; `HV-1502` and `T-101` did not,
and the viewer says so.

`drawing_connections` holds recovered topology separately, because an edge is not
a thing on the page — it is a relationship between two things on the page, and
conflating them makes "how many symbols are on this sheet?" unanswerable.

In the graph, `Equipment -[:APPEARS_ON]-> Document` carries the coordinates on
the *edge*, because one pump appears on several sheets at different positions and
the position belongs to the appearance. Recovered connectivity is written as
`CONNECTS_ON`, deliberately **not** the ontology's `FEEDS`: a line joins two
symbols, but direction of flow is not recoverable from an undirected segment, and
asserting `FEEDS` would put a claim in the graph that nothing supports.

## The viewer

`web/js/drawingviewer.js` renders the page image with an SVG overlay. The overlay
is separate from the raster so the same render serves every highlight, changing
the selection costs no round trip, and a box stays crisp when the reader zooms.

Line segments are **hidden by default**: there are hundreds, they are the least
reliable detector, and drawing them all over the drawing they were detected from
obscures the tags that matter. One toggle away, and the toggle says how many.

Clicking a detection shows the detector and its parameters, not just a number —
"Hough circle, radius 21 px against a sheet median of 20" is a claim a reader can
evaluate.

## Measured on the demo sheet

```
tag_localisation       available          12 tags (11 linked to assets)
instrument_bubbles     available          14 circles
line_segments          available          439 runs (857 raw segments merged)
topology               available          9 connections
equipment_symbols      not_implemented    0 — needs a trained detector
```

Run it yourself:

```bash
docker compose exec api python scripts/verify_surfaces.py
```
