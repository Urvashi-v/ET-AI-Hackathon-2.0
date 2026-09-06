"""Document classification and routing.

There is no single ingestion pipeline: a work-order CSV and a scanned P&ID share
nothing but a file extension. This module decides which specialist pipeline a
file goes to, and -- as important -- reports *which signal decided*, with a
confidence, so a wrong routing decision is diagnosable rather than mysterious.

Signals, in the order they are cheap:

1. **Extension / MIME** -- coarse routing: tabular vs document vs image.
2. **Text-layer presence** -- a PDF with no extractable text is a scan and needs
   OCR. This is a cheap check with a huge branch behind it.
3. **Vector-content density** -- very high line-segment count with low text
   density is the signature of a drawing, even when it is "a PDF".
4. **Header keywords** -- the phrases that appear in the first page of each
   document class.
5. **Filename convention** -- weak, but real corpora are named consistently.

A VLM fallback for genuinely ambiguous first pages is defined as a capability
(``OCR_PROVIDER`` / vision model) and is *not* silently emulated: when rules are
ambiguous and no provider is configured, the document is classified ``unknown``
with the ambiguity recorded, and it enters the review queue.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from services.common.schemas import DocumentType

#: Phrases that identify a document class from its first page. Ordered by
#: specificity; the first match with the highest weight wins.
_KEYWORD_RULES: tuple[tuple[DocumentType, float, tuple[str, ...]], ...] = (
    (
        DocumentType.PID,
        0.95,
        (
            "piping and instrumentation diagram",
            "piping & instrumentation diagram",
            "process flow diagram",
        ),
    ),
    # A bare "P&ID" is weak on its own: documents constantly *refer* to drawings
    # without being one. An MOC listing "P&ID CDU-1 sheet 3" under documents to
    # update is not a P&ID. Position weighting (below) does most of the work, and
    # the lower base weight keeps it from beating a title-block match.
    (DocumentType.PID, 0.6, ("p&id", "p & id")),
    (
        DocumentType.WORK_ORDER,
        0.92,
        ("work order", "maintenance notification", "job card", "wo number", "work order no"),
    ),
    (
        DocumentType.SOP,
        0.92,
        (
            "standard operating procedure",
            "operating instruction",
            "start-up procedure",
            "startup procedure",
            "shutdown procedure",
        ),
    ),
    (
        DocumentType.INSPECTION_REPORT,
        0.92,
        (
            "ultrasonic thickness",
            "thickness survey",
            "inspection report",
            "radiographic test report",
            "condition monitoring location",
            "magnetic particle",
        ),
    ),
    (
        DocumentType.INCIDENT_REPORT,
        0.92,
        (
            "incident report",
            "near miss report",
            "near-miss report",
            "incident investigation",
            "root cause analysis",
        ),
    ),
    (
        DocumentType.MOC,
        0.9,
        ("management of change", "moc no", "change request", "modification proposal"),
    ),
    (DocumentType.HAZOP, 0.9, ("hazop", "hazard and operability")),
    (DocumentType.PERMIT, 0.9, ("permit to work", "hot work permit", "job safety analysis")),
    (
        DocumentType.NCR_CAPA,
        0.88,
        ("non-conformance", "nonconformance", "corrective action report"),
    ),
    (
        DocumentType.DATASHEET,
        0.85,
        ("equipment datasheet", "pump data sheet", "technical data sheet", "api 610"),
    ),
    (
        DocumentType.REGULATION,
        0.85,
        ("oisd-std", "factories act", "gazette of india", "peso", "central pollution control"),
    ),
    (
        DocumentType.MANUAL,
        0.8,
        ("installation, operation and maintenance", "o&m manual", "iom manual"),
    ),
)

#: Filename hints. Real corpora are named consistently, but the separator is
#: usually ``_``, which ``\b`` treats as a word character -- so ``\bsop\b`` does
#: not match ``sop_4412_startup.md``. These patterns use an explicit separator
#: class instead.
_SEP = r"(?:^|[_\-.\s])"
_SEP_END = r"(?:[_\-.\s0-9]|$)"
_FILENAME_RULES: tuple[tuple[DocumentType, re.Pattern[str]], ...] = (
    (DocumentType.PID, re.compile(rf"{_SEP}(pid|p&id|pnid){_SEP_END}", re.I)),
    (DocumentType.WORK_ORDER, re.compile(rf"{_SEP}(wo|work[_-]?orders?|notif){_SEP_END}", re.I)),
    (DocumentType.SOP, re.compile(rf"{_SEP}(sop|procedure|proc){_SEP_END}", re.I)),
    (
        DocumentType.INSPECTION_REPORT,
        re.compile(rf"{_SEP}(insp\w*|ut|thickness|ndt|rt){_SEP_END}", re.I),
    ),
    (
        DocumentType.INCIDENT_REPORT,
        re.compile(rf"{_SEP}(inc|incidents?|nearmiss|near[_-]miss){_SEP_END}", re.I),
    ),
    (DocumentType.MOC, re.compile(rf"{_SEP}moc{_SEP_END}", re.I)),
    (
        DocumentType.REGULATION,
        re.compile(rf"{_SEP}(oisd|factories?[_-]act|peso|cpcb|cea){_SEP_END}", re.I),
    ),
)

#: A document number in the opening lines is a title-block signal and is the
#: strongest text evidence available short of an explicit title phrase.
_DOC_NUMBER_RULES: tuple[tuple[DocumentType, re.Pattern[str]], ...] = (
    (DocumentType.SOP, re.compile(r"\bSOP[-\s]?\d{3,6}\b")),
    (DocumentType.MOC, re.compile(r"\bMOC[-\s]?\d{2,4}[-/]\d{2,4}\b")),
    (DocumentType.INCIDENT_REPORT, re.compile(r"\bINC[-\s]?\d{2,4}[-/]\d{2,4}\b")),
    (DocumentType.NCR_CAPA, re.compile(r"\bNCR[-\s]?\d{2,6}\b")),
)

#: Characters from the start of the text within which a keyword is treated as a
#: title-block signal rather than a passing reference.
_TITLE_WINDOW = 600
#: Multiplier applied to a keyword found beyond that window.
_REFERENCE_PENALTY = 0.55

_TABULAR_EXTENSIONS = {".csv"}
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@dataclass(frozen=True, slots=True)
class Classification:
    doc_type: DocumentType
    confidence: float
    method: str
    pipeline: str  # text | tabular | scanned | drawing | structured_json
    needs_ocr: bool = False
    ambiguous_between: tuple[DocumentType, ...] = ()
    notes: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "doc_type": self.doc_type.value,
            "confidence": round(self.confidence, 3),
            "method": self.method,
            "pipeline": self.pipeline,
            "needs_ocr": self.needs_ocr,
            "ambiguous_between": [d.value for d in self.ambiguous_between],
            "notes": self.notes,
        }


def classify(
    *,
    filename: str,
    extension: str,
    head_text: str,
    page_count: int | None = None,
    has_text_layer: bool | None = None,
    vector_segment_count: int | None = None,
) -> Classification:
    """Route a document.

    ``head_text`` is the extracted text of roughly the first page. It may be
    empty -- that is itself the signal that identifies a scan.
    """
    ext = extension.lower()
    lowered = head_text.lower()

    # 1. Structured/tabular sources are records, not prose, and must not be
    #    chunked as prose.
    if ext in _TABULAR_EXTENSIONS:
        return Classification(
            doc_type=_tabular_subtype(filename, lowered),
            confidence=0.9,
            method="extension:tabular",
            pipeline="tabular",
            notes="Row-oriented source; each record becomes one chunk with a serialised header.",
        )
    if ext == ".json":
        return Classification(
            doc_type=_tabular_subtype(filename, lowered),
            confidence=0.9,
            method="extension:structured_json",
            pipeline="structured_json",
        )

    # 2. Raster images can only be read with OCR.
    if ext in _IMAGE_EXTENSIONS:
        return Classification(
            doc_type=_filename_type(filename) or DocumentType.UNKNOWN,
            confidence=0.5 if _filename_type(filename) else 0.2,
            method="extension:image",
            pipeline="scanned",
            needs_ocr=True,
            notes="Raster image: text recovery requires an OCR provider.",
        )

    # 3. A PDF with no text layer is a scan.
    scanned = has_text_layer is False or (ext == ".pdf" and len(lowered.strip()) < 40)

    # 4. Drawing signature: dense vector geometry, sparse text.
    if vector_segment_count is not None and vector_segment_count > 400 and len(lowered) < 4000:
        return Classification(
            doc_type=DocumentType.PID,
            confidence=0.8,
            method="vector_density",
            pipeline="drawing",
            needs_ocr=scanned,
            notes=(
                f"{vector_segment_count} vector segments with sparse text: drawing signature. "
                "Topology reconstruction requires the drawing CV pipeline."
            ),
        )

    # 5. Header keywords, weighted by where they appear.
    #
    # Position matters as much as presence. "Work order" in a precondition
    # ("confirm no open work order restricts operation") is a reference; the same
    # phrase in a title block identifies the document. Without this, an SOP whose
    # section 3 mentions work orders is filed as a work order.
    matches: list[tuple[DocumentType, float, str, int]] = []
    for doc_type, weight, keywords in _KEYWORD_RULES:
        for keyword in keywords:
            position = lowered.find(keyword)
            if position == -1:
                continue
            effective = weight * (1.0 if position < _TITLE_WINDOW else _REFERENCE_PENALTY)
            matches.append((doc_type, effective, keyword, position))

    # A document number near the top is a title-block signal.
    head = head_text[:_TITLE_WINDOW]
    for doc_type, pattern in _DOC_NUMBER_RULES:
        found = pattern.search(head)
        if found:
            matches.append((doc_type, 0.93, f"doc_number:{found.group(0)}", found.start()))

    # The filename agrees or it does not; when it agrees it breaks ties.
    guessed = _filename_type(filename)
    if guessed:
        matches.append((guessed, 0.5, "filename", _TITLE_WINDOW))

    if matches:
        matches.sort(key=lambda m: (-m[1], m[3]))
        best_type, best_weight, best_kw, _ = matches[0]
        # Agreement between independent signals raises confidence; disagreement
        # from a different document type lowers it and is reported.
        agreeing = {m[2] for m in matches if m[0] is best_type}
        rivals = tuple({m[0] for m in matches if m[0] is not best_type})
        confidence = min(0.97, best_weight + 0.03 * (len(agreeing) - 1))
        if rivals:
            confidence = max(0.4, confidence - 0.08 * len(rivals))
        pipeline = (
            "drawing" if best_type is DocumentType.PID else ("scanned" if scanned else "text")
        )
        return Classification(
            doc_type=best_type,
            confidence=confidence,
            method=f"keyword:{best_kw}",
            pipeline=pipeline,
            needs_ocr=scanned,
            ambiguous_between=rivals,
            notes=(
                f"{len(agreeing)} agreeing signal(s)"
                + (
                    f"; rival types considered: {', '.join(r.value for r in rivals)}"
                    if rivals
                    else ""
                )
            ),
        )

    # 7. Genuinely unknown. Say so; do not guess a type to make the UI tidier.
    return Classification(
        doc_type=DocumentType.UNKNOWN,
        confidence=0.0,
        method="none",
        pipeline="scanned" if scanned else "text",
        needs_ocr=scanned,
        notes=(
            "No classification signal matched. A vision-model fallback would run here "
            "when a provider is configured; the document is queued for review instead."
        ),
    )


def _tabular_subtype(filename: str, lowered_head: str) -> DocumentType:
    guessed = _filename_type(filename)
    if guessed:
        return guessed
    if any(k in lowered_head for k in ("downtime", "failure", "wo_id", "work_order")):
        return DocumentType.WORK_ORDER
    if any(k in lowered_head for k in ("thickness", "cml", "corrosion")):
        return DocumentType.INSPECTION_REPORT
    return DocumentType.UNKNOWN


def _filename_type(filename: str) -> DocumentType | None:
    stem = Path(filename).stem
    for doc_type, pattern in _FILENAME_RULES:
        if pattern.search(stem):
            return doc_type
    return None
