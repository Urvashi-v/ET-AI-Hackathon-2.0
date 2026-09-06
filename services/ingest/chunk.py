"""Structure-aware chunking.

Fixed-size windows are where most RAG systems quietly lose their accuracy: a
512-token cut splits step 7 of a procedure from the warning that governs it, and
splits a thickness table from its header row. So the strategy is chosen by
document type:

======================  ==========================================================
SOP / procedure         one chunk per step, with the governing precondition
                        carried along -- a step without its precondition is a
                        safety hazard, not a chunk
Work order / record     one chunk per record, header serialised into the text
Inspection report       one chunk per CML row, plus a report-level summary
Incident report         semantic sections (narrative / immediate cause / root
                        cause / actions) as separate chunks
Manual / prose          split on heading hierarchy, never inside a table
P&ID                    not chunked as text at all -- topology is not prose
======================  ==========================================================

Every chunk also gets a **contextual header** ("From SOP-4412 rev 3 (Crude Charge
Pump Startup), section 5 -- Priming") stored separately from the body. The header
is prepended for embedding, which resolves the orphan references that otherwise
make a chunk unretrievable, while the body stays byte-identical to the source so
that citation quotes can be verified verbatim against it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any

from services.common.schemas import DocumentType
from services.ingest.parsers.base import ParsedDocument, TextBlock

#: Target chunk size in characters. Prose only; records and steps are atomic.
TARGET_CHARS = 1400
MAX_CHARS = 2400
MIN_CHARS = 120
OVERLAP_CHARS = 160

#: A numbered procedure step. Two accepted shapes, because real procedures use
#: both: a dotted number needs no trailing punctuation ("4.1 Open the valve"),
#: while a bare number does ("4. Open the valve" / "4) Open the valve"). Demanding
#: punctuation in every case silently drops every step in a dotted procedure;
#: accepting a bare number without it would turn "10 barg limit applies" into a
#: step.
_STEP_START = re.compile(
    r"^\s*(?:step\s+)?(?P<no>\d+(?:\.\d+)+|\d+(?=[.)]))(?P<punct>[.)])?\s+(?P<body>\S.*)$",
    re.I,
)
_PRECONDITION = re.compile(
    r"\b(precondition|prerequisite|before starting|warning|caution|danger|note)\b", re.I
)
#: A heading whose entire section states conditions governing the rest of the
#: procedure rather than steps to perform. Everything under such a heading
#: becomes a document-level governing condition.
_GOVERNING_SECTION = re.compile(
    r"\b(safety|precondition|prerequisite|ppe|protective equipment|general requirement)", re.I
)
_INCIDENT_SECTIONS = (
    (
        "narrative",
        re.compile(r"\b(what happened|narrative|description of (the )?event|sequence)\b", re.I),
    ),
    ("immediate_cause", re.compile(r"\b(immediate cause|direct cause)\b", re.I)),
    ("root_cause", re.compile(r"\b(root cause|basic cause|underlying cause)\b", re.I)),
    (
        "corrective_action",
        re.compile(r"\b(corrective action|capa|recommendation|action taken)\b", re.I),
    ),
)


@dataclass(slots=True)
class Chunk:
    ordinal: int
    text: str
    context_header: str | None = None
    section_path: str | None = None
    page_from: int | None = None
    page_to: int | None = None
    char_start: int = 0
    char_end: int = 0
    bbox: list[float] | None = None
    kind: str = "prose"
    #: Weakest extraction confidence among the blocks that formed this chunk.
    #: The minimum, not the mean: a chunk is only as trustworthy as its worst
    #: sentence, and an OCR line read at 40% drags the whole passage down.
    extraction_confidence: float = 1.0
    #: How the text was obtained: pdfplumber.text_layer, pdfplumber.table, ocr,
    #: python-docx, tabular. Carried to the citation so a reader can tell a read
    #: character from a recognised one.
    extraction_method: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def embedding_text(self) -> str:
        """What is embedded: header + body. The body alone is what gets cited."""
        return f"{self.context_header}\n{self.text}" if self.context_header else self.text

    @property
    def token_estimate(self) -> int:
        # Deterministic character-based estimate. Exact token counts are
        # tokenizer-specific and only matter once a provider is configured.
        return max(1, len(self.embedding_text) // 4)


def build_context_header(
    *,
    doc_title: str,
    doc_type: DocumentType,
    revision: str | None,
    section_path: str | None,
) -> str:
    parts = [f"From {doc_title}"]
    if revision:
        parts.append(f"rev {revision}")
    parts.append(f"({doc_type.value.replace('_', ' ')})")
    header = " ".join(parts)
    if section_path:
        header += f", section: {section_path}"
    return header


def chunk_document(
    parsed: ParsedDocument,
    *,
    doc_type: DocumentType,
    doc_title: str,
    revision: str | None = None,
) -> list[Chunk]:
    """Dispatch to the strategy for this document type."""
    if doc_type is DocumentType.PID:
        chunks = _chunk_drawing(parsed)
    elif doc_type in (DocumentType.WORK_ORDER, DocumentType.INSPECTION_REPORT) and parsed.records:
        chunks = _chunk_records(parsed)
    elif doc_type is DocumentType.SOP:
        chunks = _chunk_procedure(parsed)
    elif doc_type is DocumentType.INCIDENT_REPORT:
        chunks = _chunk_incident(parsed)
    elif parsed.records:
        chunks = _chunk_records(parsed)
    else:
        chunks = _chunk_prose(parsed)

    for chunk in chunks:
        chunk.context_header = build_context_header(
            doc_title=doc_title,
            doc_type=doc_type,
            revision=revision,
            section_path=chunk.section_path,
        )
    return chunks


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


def _chunk_records(parsed: ParsedDocument) -> list[Chunk]:
    """One chunk per record. Records are naturally atomic and self-describing."""
    chunks: list[Chunk] = []
    for ordinal, block in enumerate(b for b in parsed.blocks if b.kind == "record"):
        chunks.append(
            Chunk(
                ordinal=ordinal,
                text=block.text,
                section_path=block.section_path,
                page_from=block.page,
                page_to=block.page,
                char_start=block.char_start,
                char_end=block.char_end,
                bbox=block.bbox,
                kind="record",
                extraction_confidence=block.extraction_confidence,
                extraction_method=block.metadata.get("extraction_method"),
                metadata=dict(block.metadata),
            )
        )
    if not chunks:
        return _chunk_prose(parsed)
    return chunks


def _chunk_procedure(parsed: ParsedDocument) -> list[Chunk]:
    """One chunk per numbered step, carrying the conditions that govern it.

    Two scopes, because procedures use both and conflating them loses safety
    information:

    * a **document-level** condition comes from a dedicated section -- Safety,
      Preconditions, Prerequisites, PPE -- and governs every step that follows
      it, right to the end of the document;
    * a **section-level** condition is a warning or caution written inline above
      a run of steps, and stops applying at the next heading.

    Losing either association is a safety defect, not a retrieval inconvenience:
    a step retrieved without "confirm suction pressure first" reads as complete
    and is not.
    """
    chunks: list[Chunk] = []
    ordinal = 0
    document_governing: list[str] = []
    section_governing: list[str] = []
    current_section: str | None = None
    in_governing_section = False

    # The step currently being accumulated. Procedure text is routinely
    # hard-wrapped, so a step's body continues across following lines until the
    # next step or heading. Emitting only the first line silently truncates the
    # instruction -- losing "the limit is 10 barg" from a pressure step is
    # exactly the failure this module exists to prevent.
    pending: dict[str, Any] | None = None

    def flush() -> None:
        nonlocal pending, ordinal
        if pending is None:
            return
        governing = pending["governing"]
        # The step's own words are the chunk body, and nothing else. Splicing the
        # governing conditions into the body looks safer but is not: the same
        # 500-character safety preamble then appears in every step of the
        # procedure, which inflates each chunk's length, makes all of them look
        # alike, and measurably degrades retrieval of the specific step someone
        # asked for. The association is preserved in metadata instead, where the
        # UI renders it alongside the step and nothing is lost.
        body = " ".join(pending["lines"]).strip()
        confidence, method = merge_provenance(pending["blocks"])
        chunks.append(
            Chunk(
                ordinal=ordinal,
                text=body,
                section_path=pending["section"],
                page_from=pending["page"],
                page_to=pending["page"],
                char_start=pending["char_start"],
                char_end=pending["char_start"] + len(body),
                bbox=merge_bbox(pending["blocks"]),
                extraction_confidence=confidence,
                extraction_method=method,
                kind="step",
                metadata={
                    "step_no": pending["step_no"],
                    "has_governing_condition": bool(governing),
                    "governing_conditions": governing,
                },
            )
        )
        ordinal += 1
        pending = None

    def emit_governing_section(lines: list[str], section: str | None, block: TextBlock) -> None:
        """Emit a Safety / Preconditions section as a chunk in its own right.

        Without this, the content of those sections is only ever seen as a
        preamble attached to other chunks, so "what PPE is required?" has nothing
        precise to match. The section is a real part of the procedure and has to
        be directly retrievable.
        """
        nonlocal ordinal
        text = " ".join(lines).strip()
        if not text:
            return
        chunks.append(
            Chunk(
                ordinal=ordinal,
                text=text,
                section_path=section,
                page_from=block.page,
                page_to=block.page,
                char_start=block.char_start,
                char_end=block.char_start + len(text),
                bbox=block.bbox,
                extraction_confidence=block.extraction_confidence,
                extraction_method=block.metadata.get("extraction_method"),
                kind="precondition",
                metadata={"governs_following_steps": True},
            )
        )
        ordinal += 1

    governing_buffer: list[str] = []
    governing_block: TextBlock | None = None
    governing_section_name: str | None = None

    # Content that is neither a step nor a governing condition: the document's
    # front matter (number, revision, effective date, "applies to"), scope
    # paragraphs, records sections. Dropping it loses the very lines that tie a
    # procedure to the assets it governs -- and with them the DESCRIBES edge that
    # makes the procedure reachable from the pump at all.
    prose_buffer: list[str] = []
    prose_block: TextBlock | None = None

    def close_governing_section() -> None:
        nonlocal governing_buffer, governing_block
        if governing_buffer and governing_block is not None:
            emit_governing_section(governing_buffer, governing_section_name, governing_block)
        governing_buffer = []
        governing_block = None

    def close_prose() -> None:
        nonlocal prose_buffer, prose_block, ordinal
        text = " ".join(prose_buffer).strip()
        if text and prose_block is not None:
            chunks.append(
                Chunk(
                    ordinal=ordinal,
                    text=text,
                    section_path=current_section or prose_block.section_path,
                    page_from=prose_block.page,
                    page_to=prose_block.page,
                    char_start=prose_block.char_start,
                    char_end=prose_block.char_start + len(text),
                    bbox=prose_block.bbox,
                    extraction_confidence=prose_block.extraction_confidence,
                    extraction_method=prose_block.metadata.get("extraction_method"),
                    kind="prose",
                )
            )
            ordinal += 1
        prose_buffer = []
        prose_block = None

    for block in parsed.blocks:
        if block.kind == "heading":
            flush()
            close_governing_section()
            close_prose()
            current_section = block.section_path or block.text
            section_governing = []
            in_governing_section = bool(_GOVERNING_SECTION.search(block.text))
            governing_section_name = current_section if in_governing_section else None
            continue

        for line in _split_lines(block):
            stripped = line.strip()
            if not stripped:
                continue

            match = _STEP_START.match(stripped)
            if match:
                flush()
                close_governing_section()
                close_prose()
                pending = {
                    "step_no": match.group("no"),
                    "lines": [stripped],
                    "governing": document_governing + section_governing,
                    "section": current_section or block.section_path,
                    "page": block.page,
                    "char_start": block.char_start,
                    "blocks": [block],
                }
                continue

            if pending is not None:
                pending["lines"].append(stripped)
                if block not in pending["blocks"]:
                    pending["blocks"].append(block)
                continue

            if in_governing_section:
                document_governing.append(stripped)
                governing_buffer.append(stripped)
                if governing_block is None:
                    governing_block = block
                continue

            if _PRECONDITION.search(stripped):
                section_governing.append(stripped)
            prose_buffer.append(stripped)
            if prose_block is None:
                prose_block = block

    flush()
    close_governing_section()
    close_prose()

    if not chunks:
        return _chunk_prose(parsed)
    return chunks


def _chunk_incident(parsed: ParsedDocument) -> list[Chunk]:
    """Split on causal sections so the lessons engine can retrieve causes
    without dragging the whole narrative along."""
    chunks: list[Chunk] = []
    ordinal = 0
    current_label = "narrative"
    buffer: list[TextBlock] = []

    def flush() -> None:
        nonlocal ordinal, buffer
        if not buffer:
            return
        text = "\n".join(b.text for b in buffer).strip()
        if len(text) >= MIN_CHARS or current_label != "narrative":
            chunks.append(
                Chunk(
                    ordinal=ordinal,
                    text=text,
                    section_path=buffer[0].section_path or current_label,
                    page_from=buffer[0].page,
                    page_to=buffer[-1].page,
                    char_start=buffer[0].char_start,
                    char_end=buffer[-1].char_end,
                    bbox=merge_bbox(buffer),
                    extraction_confidence=merge_provenance(buffer)[0],
                    extraction_method=merge_provenance(buffer)[1],
                    kind="incident_section",
                    metadata={"section": current_label},
                )
            )
            ordinal += 1
        buffer = []

    for block in parsed.blocks:
        label = _incident_section_of(block.text)
        if label and label != current_label:
            flush()
            current_label = label
        buffer.append(block)
    flush()

    if not chunks:
        return _chunk_prose(parsed)
    return chunks


def _chunk_drawing(parsed: ParsedDocument) -> list[Chunk]:
    """A drawing is not prose.

    Its value is topology, which becomes graph nodes and edges. The only text
    chunk produced is one descriptive summary carrying the tags found on the
    sheet, so the drawing is still reachable by search.
    """
    text = parsed.full_text.strip()
    if not text:
        return []
    return [
        Chunk(
            ordinal=0,
            text=text[:MAX_CHARS],
            section_path="drawing text layer",
            page_from=1,
            page_to=parsed.page_count,
            char_start=0,
            char_end=min(len(text), MAX_CHARS),
            bbox=merge_bbox(parsed.blocks),
            extraction_confidence=merge_provenance(parsed.blocks)[0],
            extraction_method=merge_provenance(parsed.blocks)[1],
            kind="summary",
            metadata={
                "note": "Drawing text layer only. Topology reconstruction is a separate "
                "pipeline and does not produce prose chunks.",
                "truncated": len(text) > MAX_CHARS,
            },
        )
    ]


def _chunk_prose(parsed: ParsedDocument) -> list[Chunk]:
    """Heading-aware prose chunking with sentence-boundary splits."""
    chunks: list[Chunk] = []
    ordinal = 0
    buffer: list[TextBlock] = []
    buffer_len = 0

    def flush() -> None:
        nonlocal ordinal, buffer, buffer_len
        if not buffer:
            return
        text = "\n".join(b.text for b in buffer).strip()
        if not text:
            buffer, buffer_len = [], 0
            return
        for piece_start, piece in _split_long(text):
            chunks.append(
                Chunk(
                    ordinal=ordinal,
                    text=piece,
                    section_path=buffer[0].section_path,
                    page_from=buffer[0].page,
                    page_to=buffer[-1].page,
                    char_start=buffer[0].char_start + piece_start,
                    char_end=buffer[0].char_start + piece_start + len(piece),
                    bbox=merge_bbox(buffer),
                    extraction_confidence=merge_provenance(buffer)[0],
                    extraction_method=merge_provenance(buffer)[1],
                    kind="table_row" if buffer[0].kind == "table_row" else "prose",
                )
            )
            ordinal += 1
        buffer, buffer_len = [], 0

    last_section: str | None = None
    for block in parsed.blocks:
        if block.kind == "heading":
            flush()
            last_section = block.section_path or block.text
            continue
        # A table row is atomic and never merged with prose.
        if block.kind == "table_row":
            flush()
            chunks.append(
                Chunk(
                    ordinal=ordinal,
                    text=block.text,
                    section_path=block.section_path or last_section,
                    page_from=block.page,
                    page_to=block.page,
                    char_start=block.char_start,
                    char_end=block.char_end,
                    bbox=block.bbox,
                    kind="table_row",
                    extraction_confidence=block.extraction_confidence,
                    extraction_method=block.metadata.get("extraction_method"),
                    metadata=dict(block.metadata),
                )
            )
            ordinal += 1
            continue
        if buffer and (
            buffer_len + len(block.text) > TARGET_CHARS
            or (block.section_path or last_section) != buffer[0].section_path
        ):
            flush()
        if not buffer and not block.section_path and last_section:
            # `replace`, not `TextBlock(**block.__dict__)`: TextBlock is a
            # slots dataclass and has no __dict__.
            block = replace(block, section_path=last_section)
        buffer.append(block)
        buffer_len += len(block.text)
    flush()
    return chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def merge_bbox(blocks: list[TextBlock]) -> list[float] | None:
    """Union of the blocks' bounding boxes, when they share a page.

    A chunk spanning a page break has no single rectangle, so it gets none
    rather than a misleading one that spans both pages.
    """
    boxes = [b.bbox for b in blocks if b.bbox]
    if not boxes:
        return None
    pages = {b.page for b in blocks if b.bbox}
    if len(pages) > 1:
        return None
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def merge_provenance(blocks: list[TextBlock]) -> tuple[float, str | None]:
    """Weakest confidence, and the extraction method if the blocks agree."""
    if not blocks:
        return 1.0, None
    confidence = min(b.extraction_confidence for b in blocks)
    methods = {
        b.metadata.get("extraction_method") for b in blocks if b.metadata.get("extraction_method")
    }
    method = methods.pop() if len(methods) == 1 else ("mixed" if methods else None)
    return round(confidence, 4), method


def _split_lines(block: TextBlock) -> list[str]:
    return block.text.splitlines() or [block.text]


def _incident_section_of(text: str) -> str | None:
    head = text[:160]
    for label, pattern in _INCIDENT_SECTIONS:
        if pattern.search(head):
            return label
    return None


def _split_long(text: str) -> list[tuple[int, str]]:
    """Split oversized text at sentence boundaries, with a small overlap.

    The overlap keeps a claim that straddles a boundary retrievable from either
    side; without it, exactly the sentences that span a cut become invisible.
    """
    if len(text) <= MAX_CHARS:
        return [(0, text)]

    pieces: list[tuple[int, str]] = []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    current: list[str] = []
    current_len = 0
    cursor = 0
    start = 0

    for sentence in sentences:
        if current and current_len + len(sentence) > TARGET_CHARS:
            piece = " ".join(current)
            pieces.append((start, piece))
            tail = piece[-OVERLAP_CHARS:] if len(piece) > OVERLAP_CHARS else piece
            start = max(0, cursor - len(tail))
            current = [tail]
            current_len = len(tail)
        current.append(sentence)
        current_len += len(sentence) + 1
        cursor += len(sentence) + 1

    if current:
        pieces.append((start, " ".join(current)))
    return pieces
