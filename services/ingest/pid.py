"""P&ID digitisation.

A P&ID is the densest engineering document a plant holds and the one least
served by text search. It states, graphically, that this pump discharges into
that exchanger through that line, protected by that relief valve — and none of
that survives being turned into a bag of words.

What this module does, and what it honestly cannot
--------------------------------------------------
Four detectors, in descending order of how much they can be trusted:

**1. Tag localisation** — reliable, and the one that matters most.
   Vector PDFs carry per-word geometry, so a tag on the sheet has a rectangle
   already; there is nothing to infer. Words are grouped into candidate tags and
   run through the project's own tag grammar, which is the same code that reads
   tags out of prose. A tag found here is as trustworthy as a tag found in an
   incident report, and it comes with coordinates.

**2. Instrument bubbles** — reliable, classical, untrained.
   ISA 5.1 draws instruments as circles of a consistent size. A Hough circle
   transform finds circles; it does not need to be taught what an instrument
   looks like, and its failures are geometric and inspectable rather than
   mysterious. Circles are then associated with the tag text inside them.

**3. Process lines** — partial.
   A probabilistic Hough transform recovers straight segments, which on a P&ID
   are pipe runs. Recovering *topology* from them means joining collinear
   segments across symbol gaps, distinguishing process lines from instrument
   signal lines and from the title block's ruling, and following elbows. What is
   implemented is the honest subset: segments are detected, filtered to plausible
   pipe runs, and used to connect symbols that a segment demonstrably touches at
   both ends. Longer chains are not claimed.

**4. Equipment symbols** — not implemented, and not faked.
   Distinguishing a centrifugal pump from a vessel from a heat exchanger by
   silhouette is a learned-model problem. There is no classical transform for
   it, and no pretrained model ships with knowledge of ISA symbology. The
   training-data structure and exactly what dataset would be required are
   documented in ``data/pid_training/README.md``; this module reports the
   capability as ``not_implemented`` rather than emitting boxes with invented
   labels.

On confidence
-------------
The numbers stored are **detector-specific quality measures, not probabilities**.
A Hough circle's confidence here is derived from how well its radius matches the
sheet's dominant instrument-bubble radius; a tag's reflects how many PDF words had
to be joined to form it, since the grammar itself is deterministic and the
geometry exact.
They are comparable within a method and meaningless across methods, and the
schema comment says so. No accuracy figure is reported anywhere, because none has
been measured against a labelled ground truth — and inventing one would be the
single most dishonest thing this project could do.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from services.common.logging import get_logger
from services.common.schemas import CapabilityState
from services.common.tags import parse as parse_tag

log = get_logger(__name__)

#: Rendering scale for the raster passed to OpenCV. 3x on a typical A3 sheet puts
#: an ISA instrument bubble at roughly 40 px diameter, which is comfortably above
#: the Hough transform's useful floor without making the image so large that the
#: accumulator fills with noise.
RENDER_SCALE = 3.0

#: Instrument bubbles on one sheet are drawn at one size. These bounds are in
#: rendered pixels at RENDER_SCALE and bracket that size generously -- the
#: dominant radius is then found from the detections themselves, so the bounds
#: only have to exclude rivets and title-block circles.
MIN_BUBBLE_RADIUS = 10
MAX_BUBBLE_RADIUS = 60

#: A line shorter than this is lettering, hatching or a leader, not a pipe run.
MIN_LINE_LENGTH_PX = 40

#: Pipe runs on a P&ID are drawn orthogonally. Segments more than this far from
#: horizontal or vertical are almost always leaders, dimension lines or the
#: diagonal strokes inside a symbol.
ORTHOGONAL_TOLERANCE_DEG = 4.0

#: How close a line end must come to a symbol's box to count as touching it.
#: In PDF points, so it scales with the sheet rather than the render.
TOUCH_TOLERANCE_PT = 6.0

#: Words this far apart on the same text line are joined before tag parsing.
#: A PDF may emit "P", "-", "101B" as three words, and the tag grammar cannot
#: read what the extractor never put together.
WORD_JOIN_GAP_PT = 3.0


@dataclass(slots=True)
class Detection:
    kind: str
    x0: float
    y0: float
    x1: float
    y1: float
    method: str
    confidence: float
    text: str | None = None
    normalised: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "text": self.text,
            "normalised": self.normalised,
            "x0": round(self.x0, 2),
            "y0": round(self.y0, 2),
            "x1": round(self.x1, 2),
            "y1": round(self.y1, 2),
            "method": self.method,
            "confidence": round(self.confidence, 4),
            "properties": self.properties,
        }


@dataclass(slots=True)
class Connection:
    from_index: int
    to_index: int
    via: list[int]
    method: str
    confidence: float
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PidResult:
    page: int
    page_width: float
    page_height: float
    detections: list[Detection] = field(default_factory=list)
    connections: list[Connection] = field(default_factory=list)
    #: Per-detector state, so the dashboard can say which stages ran rather than
    #: showing an empty overlay that looks like a drawing with nothing on it.
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)

    def of_kind(self, kind: str) -> list[Detection]:
        return [d for d in self.detections if d.kind == kind]


def digitise_page(pdf_path: str, page_number: int = 1) -> PidResult:
    """Run every available detector over one drawing page.

    Ordered so that each stage can use the previous one's output: tags locate
    text, circles locate instruments, tags inside circles name them, and lines
    connect what the first two found.
    """
    import pdfplumber

    with pdfplumber.open(pdf_path) as pdf:
        if page_number < 1 or page_number > len(pdf.pages):
            raise ValueError(f"page {page_number} out of range (1..{len(pdf.pages)})")
        page = pdf.pages[page_number - 1]
        result = PidResult(
            page=page_number,
            page_width=float(page.width),
            page_height=float(page.height),
        )
        words = page.extract_words(keep_blank_chars=False, use_text_flow=False)

    # --- 1. tags, from the PDF's own word geometry --------------------------
    tags = _detect_tags(words)
    result.detections.extend(tags)
    result.stages["tag_localisation"] = {
        "state": CapabilityState.AVAILABLE.value,
        "detections": len(tags),
        "method": "pdf_word_geometry + tag grammar",
        "detail": (
            f"{len(words)} word(s) on the page; {len(tags)} parsed as equipment, instrument "
            "or line tags. Geometry comes from the PDF, so these coordinates are exact "
            "rather than inferred."
        ),
    }

    # --- 2 & 3. raster detectors --------------------------------------------
    raster = _render(pdf_path, page_number, result.page_width, result.page_height)
    if raster is None:
        for stage in ("instrument_bubbles", "line_segments"):
            result.stages[stage] = {
                "state": CapabilityState.ERROR.value,
                "detections": 0,
                "detail": "Page could not be rendered to raster; geometric detection skipped.",
            }
        return result

    image, scale_x, scale_y = raster

    circles = _detect_bubbles(image, scale_x, scale_y)
    _name_bubbles(circles, tags)
    result.detections.extend(circles)
    result.stages["instrument_bubbles"] = {
        "state": CapabilityState.AVAILABLE.value,
        "detections": len(circles),
        "method": "cv2.HoughCircles (HOUGH_GRADIENT)",
        "detail": (
            f"{len(circles)} circle(s) matching ISA 5.1 instrument-bubble geometry. "
            "Classical transform, no training data. Confidence measures radius agreement "
            "with the sheet's dominant bubble size and is not a probability."
        ),
    }

    lines = _detect_lines(image, scale_x, scale_y)
    result.detections.extend(lines)
    result.stages["line_segments"] = {
        "state": CapabilityState.AVAILABLE.value,
        "detections": len(lines),
        "method": "cv2.HoughLinesP, filtered to orthogonal runs",
        "detail": (
            f"{len(lines)} straight segment(s) plausibly pipe runs. Collinear joining "
            "across symbol gaps and elbow following are not implemented, so long "
            "process routes are not reconstructed."
        ),
    }

    # --- 4. connectivity ----------------------------------------------------
    symbols = tags + circles
    offset = 0  # symbols occupy the first len(symbols) slots of result.detections
    line_offset = len(symbols)
    result.connections = _connect(symbols, lines, symbol_offset=offset, line_offset=line_offset)
    result.stages["topology"] = {
        "state": CapabilityState.AVAILABLE.value,
        "detections": len(result.connections),
        "method": "line endpoints touching symbol boxes",
        "detail": (
            f"{len(result.connections)} connection(s) where one detected segment demonstrably "
            "touches two symbols. Multi-segment routes are not chained, so this is a lower "
            "bound on the sheet's real connectivity, not a complete topology."
        ),
    }

    # --- 5. equipment symbols: declared, not faked --------------------------
    result.stages["equipment_symbols"] = {
        "state": CapabilityState.NOT_IMPLEMENTED.value,
        "detections": 0,
        "method": "requires a trained detector",
        "detail": (
            "Classifying a pump against a vessel against an exchanger by silhouette is a "
            "learned-model problem with no classical equivalent, and no pretrained model "
            "ships with ISA symbology. See data/pid_training/README.md for the dataset "
            "this needs. No boxes are emitted rather than emitting guessed labels."
        ),
    }

    log.info(
        "pid.digitised",
        page=page_number,
        tags=len(tags),
        bubbles=len(circles),
        lines=len(lines),
        connections=len(result.connections),
    )
    return result


# ---------------------------------------------------------------------------
# 1. Tag localisation
# ---------------------------------------------------------------------------


def _detect_tags(words: list[dict[str, Any]]) -> list[Detection]:
    """Find equipment, instrument and line tags, with their rectangles.

    Words are joined before parsing because PDF text extraction splits on
    kerning, not on meaning: "P-101B" can arrive as three words. Joining runs of
    adjacent words on the same baseline and parsing every prefix of the run
    recovers the tag without hard-coding what tags look like -- the tag grammar
    already knows that.
    """
    detections: list[Detection] = []
    seen: set[tuple[str, int, int]] = set()

    for run in _word_runs(words):
        for length in range(min(4, len(run)), 0, -1):
            for start in range(len(run) - length + 1):
                group = run[start : start + length]
                text = "".join(w["text"] for w in group)
                if len(text) < 3:
                    continue
                parsed = parse_tag(text)
                if not parsed.parsed:
                    continue
                x0 = min(float(w["x0"]) for w in group)
                y0 = min(float(w["top"]) for w in group)
                x1 = max(float(w["x1"]) for w in group)
                y1 = max(float(w["bottom"]) for w in group)
                key = (parsed.canonical, int(x0), int(y0))
                if key in seen:
                    continue
                seen.add(key)
                detections.append(
                    Detection(
                        kind="tag",
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                        method="pdf_word_geometry",
                        # The geometry is exact and the grammar is deterministic:
                        # a tag read from a single PDF word is certain. What is
                        # uncertain is the *assembly* -- joining "P", "-", "101B"
                        # into one tag is an inference about kerning, and each
                        # extra word joined is another chance the join was wrong.
                        confidence=round(1.0 - 0.1 * (len(group) - 1), 4),
                        text=text,
                        normalised=parsed.canonical,
                        properties={
                            "tag_kind": parsed.kind.value if parsed.kind else None,
                            "words": len(group),
                        },
                    )
                )
    return detections


def _word_runs(words: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group words into runs on the same baseline, split at wide gaps."""
    if not words:
        return []
    ordered = sorted(words, key=lambda w: (round(float(w["top"]), 1), float(w["x0"])))
    runs: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = [ordered[0]]

    for word in ordered[1:]:
        previous = current[-1]
        same_line = abs(float(word["top"]) - float(previous["top"])) <= 2.0
        gap = float(word["x0"]) - float(previous["x1"])
        if same_line and gap <= WORD_JOIN_GAP_PT:
            current.append(word)
        else:
            runs.append(current)
            current = [word]
    runs.append(current)
    return runs


# ---------------------------------------------------------------------------
# 2 & 3. Raster detectors
# ---------------------------------------------------------------------------


def _render(
    pdf_path: str, page_number: int, page_width: float, page_height: float
) -> tuple[Any, float, float] | None:
    """Render one page to a greyscale array, with PDF-point scale factors."""
    try:
        import numpy as np
        import pypdfium2
    except ImportError as exc:  # pragma: no cover - dependency guard
        log.error("pid.render_unavailable", error=str(exc))
        return None

    try:
        pdf = pypdfium2.PdfDocument(pdf_path)
        try:
            bitmap = pdf[page_number - 1].render(scale=RENDER_SCALE, grayscale=True)
            image = np.asarray(bitmap.to_pil())
        finally:
            pdf.close()
    except Exception as exc:
        log.error("pid.render_failed", page=page_number, error=str(exc))
        return None

    height, width = image.shape[:2]
    # Pixels back to PDF points. Computed from the actual raster rather than
    # assumed from RENDER_SCALE, because pdfium rounds to whole pixels and a
    # half-pixel error at 3x is a visible offset on the overlay.
    return image, page_width / width, page_height / height


def _detect_bubbles(image: Any, scale_x: float, scale_y: float) -> list[Detection]:
    """ISA instrument bubbles, via a Hough circle transform.

    No training data and no learned weights: the transform votes for circle
    centres in an accumulator built from image gradients. That is a real
    detector, its failure modes are geometric (it misses broken outlines, it
    invents circles in dense hatching), and both are inspectable by looking at
    the sheet — which is more than can be said for an unmeasured model.
    """
    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        log.error("pid.cv_unavailable", error=str(exc))
        return []

    blurred = cv2.medianBlur(image, 5)
    found = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        # Two instrument bubbles are never drawn closer than their own diameter.
        minDist=MIN_BUBBLE_RADIUS * 2.0,
        # Canny high threshold. A P&ID is line art on white, so edges are strong
        # and a high value keeps the accumulator from filling with texture.
        param1=120,
        # Accumulator threshold: how many votes a centre needs. Low enough to
        # find a bubble whose outline is broken by the tag text crossing it.
        param2=30,
        minRadius=MIN_BUBBLE_RADIUS,
        maxRadius=MAX_BUBBLE_RADIUS,
    )
    if found is None:
        return []

    circles = np.round(found[0, :]).astype(int)
    radii = [int(r) for _, _, r in circles]
    # The sheet's dominant bubble size, used to score each detection. Drawings
    # are consistent; a circle far from the mode is more likely a symbol detail
    # or a piece of the title block than an instrument.
    dominant = float(np.median(radii)) if radii else 0.0

    detections: list[Detection] = []
    for cx, cy, r in circles:
        agreement = 1.0 - min(1.0, abs(r - dominant) / max(dominant, 1.0))
        detections.append(
            Detection(
                kind="instrument_bubble",
                x0=float((cx - r) * scale_x),
                y0=float((cy - r) * scale_y),
                x1=float((cx + r) * scale_x),
                y1=float((cy + r) * scale_y),
                method="hough_circle",
                # Radius agreement with the sheet's dominant bubble size.
                # Explicitly not a probability: see the module docstring.
                confidence=round(float(agreement), 4),
                properties={
                    "radius_px": int(r),
                    "dominant_radius_px": round(dominant, 1),
                    "accumulator_threshold": 30,
                },
            )
        )
    return detections


def _name_bubbles(bubbles: list[Detection], tags: list[Detection]) -> None:
    """Attach the tag text drawn inside each bubble.

    An unnamed circle is nearly useless -- "there is an instrument here" is worth
    much less than "PIC-101 is here" -- and on a P&ID the tag is always drawn
    inside the bubble, so containment is the whole association rule.
    """
    for bubble in bubbles:
        inside = [t for t in tags if bubble.contains(*t.centre)]
        if not inside:
            continue
        # Instrument tags first: a bubble containing both "PIC" and a line
        # number is labelled by the instrument.
        inside.sort(key=lambda t: (t.properties.get("tag_kind") != "instrument", -t.confidence))
        best = inside[0]
        bubble.text = best.text
        bubble.normalised = best.normalised
        bubble.properties["named_by"] = best.normalised
        bubble.properties["tag_kind"] = best.properties.get("tag_kind")


def _detect_lines(image: Any, scale_x: float, scale_y: float) -> list[Detection]:
    """Straight runs, via a probabilistic Hough transform.

    Filtered to near-orthogonal segments above a minimum length, because that is
    what a pipe run looks like on a P&ID and it is not what lettering, hatching,
    leaders or the diagonal strokes inside a symbol look like.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:  # pragma: no cover
        return []

    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    segments = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        # Votes needed for a line. Tuned to the render scale: at 3x a real pipe
        # run contributes far more than this, and lettering does not.
        threshold=60,
        minLineLength=MIN_LINE_LENGTH_PX,
        # A pipe run broken by a symbol or a line-number label should still come
        # back as one segment.
        maxLineGap=8,
    )
    if segments is None:
        return []

    detections: list[Detection] = []
    for x1, y1, x2, y2 in segments[:, 0, :]:
        angle = abs(math.degrees(math.atan2(float(y2 - y1), float(x2 - x1)))) % 180.0
        orthogonal = min(angle, abs(angle - 90.0), abs(angle - 180.0))
        if orthogonal > ORTHOGONAL_TOLERANCE_DEG:
            continue
        length_px = math.hypot(float(x2 - x1), float(y2 - y1))
        detections.append(
            Detection(
                kind="line_segment",
                x0=float(min(x1, x2) * scale_x),
                y0=float(min(y1, y2) * scale_y),
                x1=float(max(x1, x2) * scale_x),
                y1=float(max(y1, y2) * scale_y),
                method="hough_lines_p",
                # Longer segments are more certainly pipe runs. Saturating at
                # 400 px so a very long segment does not dominate the scale.
                confidence=round(min(1.0, length_px / 400.0), 4),
                properties={
                    "length_px": round(length_px, 1),
                    "orientation": "horizontal" if angle < 45 or angle > 135 else "vertical",
                },
            )
        )
    return _merge_duplicates(detections)


def _merge_duplicates(lines: list[Detection]) -> list[Detection]:
    """Reassemble the fragments Hough returns into whole pipe runs.

    Two things fragment a single drawn pipe. Canny reports *both* gradient edges
    of a stroke, so every line arrives twice; and ``HoughLinesP`` breaks a run
    wherever the accumulator dips — at a crossing, a label, a symbol. The raw
    output for one A3 sheet was 857 segments, which is not 857 pipes.

    Segments are grouped by orientation and by position across the run, then
    merged along it where they overlap or nearly touch. The result is a count of
    *runs*, which is the number an engineer would recognise.

    ``_MERGE_JOIN_PT`` is the one number that matters here: it is how far apart
    two collinear fragments can be and still be called one pipe. Set too high it
    bridges genuinely separate pipes and invents connectivity; set too low the
    sheet stays fragmented. It is deliberately smaller than the smallest symbol
    on a P&ID, so a run broken *by* a symbol is not silently rejoined through it.
    """
    groups: dict[tuple[str, int], list[Detection]] = {}
    for line in lines:
        horizontal = line.properties.get("orientation") == "horizontal"
        across = round((line.y0 if horizontal else line.x0) / _MERGE_BIN_PT)
        groups.setdefault((line.properties.get("orientation", "?"), across), []).append(line)

    merged: list[Detection] = []
    for (orientation, _), group in groups.items():
        horizontal = orientation == "horizontal"
        # Sorted along the run so a single forward pass can extend or close.
        group.sort(key=lambda d: d.x0 if horizontal else d.y0)

        current = group[0]
        fragments = 1
        for candidate in group[1:]:
            current_end = current.x1 if horizontal else current.y1
            candidate_start = candidate.x0 if horizontal else candidate.y0
            if candidate_start - current_end <= _MERGE_JOIN_PT:
                current = _extend(current, candidate, horizontal)
                fragments += 1
                continue
            merged.append(_finalise(current, fragments))
            current = candidate
            fragments = 1
        merged.append(_finalise(current, fragments))
    return merged


def _extend(base: Detection, extra: Detection, horizontal: bool) -> Detection:
    """Grow a run to cover a fragment that continues it."""
    base.x0 = min(base.x0, extra.x0)
    base.y0 = min(base.y0, extra.y0)
    base.x1 = max(base.x1, extra.x1)
    base.y1 = max(base.y1, extra.y1)
    return base


def _finalise(line: Detection, fragments: int) -> Detection:
    """Recompute length and confidence for a merged run.

    Confidence is recomputed from the merged extent rather than inherited from
    whichever fragment happened to be first: a run assembled from nine fragments
    is a longer, better-supported line than any one of them, and reporting the
    first fragment's score would understate it.
    """
    length = (
        line.x1 - line.x0
        if line.properties.get("orientation") == "horizontal"
        else line.y1 - line.y0
    )
    line.properties["length_pt"] = round(length, 1)
    line.properties["fragments_merged"] = fragments
    line.properties.pop("length_px", None)
    line.confidence = round(min(1.0, length / 120.0), 4)
    return line


#: Bin width across a run, in PDF points. Wide enough to catch both gradient
#: edges of one drawn stroke, narrow enough to keep two parallel pipes apart --
#: P&IDs do not route pipes closer than this.
_MERGE_BIN_PT = 4.0

#: How far two collinear fragments can be apart and still be one pipe. Smaller
#: than the smallest symbol on the sheet, so a run genuinely interrupted by a
#: valve or a bubble is not rejoined straight through it.
_MERGE_JOIN_PT = 6.0


# ---------------------------------------------------------------------------
# 4. Topology
# ---------------------------------------------------------------------------


def _connect(
    symbols: list[Detection],
    lines: list[Detection],
    *,
    symbol_offset: int,
    line_offset: int,
) -> list[Connection]:
    """Join symbols that one detected segment touches at both ends.

    Deliberately the conservative version. Real P&ID topology needs collinear
    segments joined across the gaps symbols punch in them, elbows followed round
    corners, and process lines told apart from instrument signal lines. None of
    that is implemented, so what is returned is a *lower bound* on connectivity
    -- every connection claimed is one a segment demonstrably makes, and the
    absence of a connection means nothing.
    """
    connections: list[Connection] = []
    seen: set[tuple[int, int]] = set()

    for line_index, line in enumerate(lines):
        ends = ((line.x0, line.y0), (line.x1, line.y1))
        touched: list[int] = []
        for symbol_index, symbol in enumerate(symbols):
            if any(_near(symbol, x, y) for x, y in ends):
                touched.append(symbol_index)
        if len(touched) < 2:
            continue
        for i in range(len(touched)):
            for j in range(i + 1, len(touched)):
                a, b = sorted((touched[i], touched[j]))
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                connections.append(
                    Connection(
                        from_index=symbol_offset + a,
                        to_index=symbol_offset + b,
                        via=[line_offset + line_index],
                        method="line_endpoint_adjacency",
                        confidence=line.confidence,
                        properties={"orientation": line.properties.get("orientation")},
                    )
                )
    return connections


def _near(symbol: Detection, x: float, y: float) -> bool:
    return (
        symbol.x0 - TOUCH_TOLERANCE_PT <= x <= symbol.x1 + TOUCH_TOLERANCE_PT
        and symbol.y0 - TOUCH_TOLERANCE_PT <= y <= symbol.y1 + TOUCH_TOLERANCE_PT
    )


def capability() -> dict[str, Any]:
    """What this module can and cannot do, for the health endpoint."""
    try:
        import cv2  # noqa: F401

        cv_state = CapabilityState.AVAILABLE.value
        cv_detail = "OpenCV available; Hough circle and line detection enabled."
    except ImportError:
        cv_state = CapabilityState.NOT_CONFIGURED.value
        cv_detail = "opencv-python-headless is not installed; only tag localisation runs."

    return {
        "tag_localisation": {
            "state": CapabilityState.AVAILABLE.value,
            "detail": "Exact coordinates from PDF word geometry, parsed by the project tag grammar.",
        },
        "instrument_bubbles": {"state": cv_state, "detail": cv_detail},
        "line_segments": {"state": cv_state, "detail": cv_detail},
        "equipment_symbols": {
            "state": CapabilityState.NOT_IMPLEMENTED.value,
            "detail": (
                "Needs a trained detector. Dataset requirements are documented in "
                "data/pid_training/README.md. No accuracy figure is reported because none "
                "has been measured."
            ),
        },
    }
