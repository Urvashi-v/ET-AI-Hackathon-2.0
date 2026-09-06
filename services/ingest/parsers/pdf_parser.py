"""PDF parser.

Built on pdfplumber (pdfminer.six underneath) rather than plain text extraction,
because three things have to survive parsing and cannot be recovered afterwards:

**Bounding boxes.** Recorded per block, derived from word geometry. This is what
makes "open the citation and highlight the sentence on the source page" possible
at all. Merging parsing into chunking is the classic mistake that throws it away.

**Table structure.** A thickness table flattened to prose loses the row-column
association that makes the numbers mean anything, and a model reading it will
confidently report the wrong cell. Ruled tables are extracted as structured rows
with their header repeated per row, and the region they occupy is excluded from
the surrounding prose pass so the same text is not indexed twice.

**Drawing characteristics.** A P&ID is often "a PDF" by extension and a drawing
by content. Counting vector objects -- lines, rectangles, curves -- against text
density is a real, cheap signal that separates the two, and it is only available
here, at parse time.

Pages with no usable text layer are reported as such rather than yielding an
empty block; the pipeline then routes them to OCR, or records honestly that it
could not read them.
"""

from __future__ import annotations

import io
from typing import Any

from services.common.logging import get_logger
from services.ingest.parsers.base import ParsedDocument, TextBlock

log = get_logger(__name__)

#: A page with fewer than this many characters of extractable text is treated as
#: having no usable text layer, and is a candidate for the OCR path.
MIN_CHARS_PER_PAGE = 20

#: Minimum vector-object count before a page can be considered a drawing at all.
#: A ruled table or a page border runs to tens of objects; a schematic runs to
#: hundreds or thousands.
DRAWING_VECTOR_THRESHOLD = 120

#: Characters of text per vector object, below which the page is a drawing.
#:
#: The ratio is the discriminator, not either number alone. A dense numeric table
#: has many vector objects *and* a lot of text; a schematic has comparable
#: geometry and almost none, because its content is topology rather than prose.
#: Measured on the corpus: schematic 0.7 chars/object, ruled table 35,
#: prose page 184.
DRAWING_TEXT_PER_VECTOR = 4.0

#: Vertical gap (in points) that separates one paragraph from the next. Larger
#: than normal line spacing, smaller than a section break.
PARAGRAPH_GAP = 4.0


class PdfParser:
    name = "pdfplumber"

    def can_parse(self, extension: str) -> bool:
        return extension.lower() == ".pdf"

    def parse(self, data: bytes, *, filename: str) -> ParsedDocument:
        encrypted, metadata, warnings = self._probe(data)
        if encrypted:
            return ParsedDocument(
                blocks=[],
                page_count=metadata.get("pages"),
                has_text_layer=None,
                parser=self.name,
                warnings=["PDF is encrypted and no password was supplied."],
                metadata=metadata,
            )

        try:
            import pdfplumber
        except ImportError:  # pragma: no cover - dependency is pinned
            return ParsedDocument(
                blocks=[],
                parser=self.name,
                warnings=["pdfplumber is not installed; PDFs cannot be read."],
            )

        blocks: list[TextBlock] = []
        offset = 0
        pages_with_text = 0
        page_count = 0
        vector_objects = 0
        text_chars = 0
        pages_without_text: list[int] = []
        table_count = 0
        drawing_pages = 0

        try:
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                page_count = len(pdf.pages)
                for page_no, page in enumerate(pdf.pages, start=1):
                    try:
                        page_blocks, page_stats = self._parse_page(page, page_no, offset)
                    except Exception as exc:
                        # One malformed page must not lose the whole document.
                        warnings.append(f"Page {page_no} could not be parsed: {type(exc).__name__}")
                        continue

                    vector_objects += page_stats["vector_objects"]
                    text_chars += page_stats["chars"]
                    table_count += page_stats["tables"]
                    drawing_pages += page_stats["drawing_pages"]

                    if page_stats["chars"] < MIN_CHARS_PER_PAGE:
                        pages_without_text.append(page_no)
                        continue

                    pages_with_text += 1
                    blocks.extend(page_blocks)
                    offset = page_blocks[-1].char_end + 2 if page_blocks else offset
        except Exception as exc:
            return ParsedDocument(
                blocks=[],
                page_count=page_count or metadata.get("pages"),
                has_text_layer=None,
                parser=self.name,
                warnings=[f"PDF could not be opened: {type(exc).__name__}: {str(exc)[:160]}"],
                metadata=metadata,
            )

        if pages_without_text:
            preview = ", ".join(str(p) for p in pages_without_text[:10])
            extra = len(pages_without_text) - 10
            suffix = "" if extra <= 0 else f" (+{extra} more)"
            warnings.append(
                f"{len(pages_without_text)} of {page_count} page(s) have no usable text "
                f"layer and need OCR: {preview}{suffix}."
            )

        # Sparse text over dense vector geometry is the signature of a drawing.
        # A single drawing page is enough: a P&ID sheet bound into a document
        # pack is still a drawing.
        is_drawing = drawing_pages > 0 or _is_drawing(vector_objects, text_chars)

        metadata.update(
            {
                "pages_total": page_count,
                "pages_with_text": pages_with_text,
                "pages_without_text": pages_without_text,
                "vector_objects": vector_objects,
                "text_chars": text_chars,
                "tables_found": table_count,
                "drawing_pages": drawing_pages,
                "drawing_signature": is_drawing,
            }
        )

        return ParsedDocument(
            blocks=blocks,
            page_count=page_count,
            has_text_layer=pages_with_text > 0,
            parser=self.name,
            warnings=warnings,
            metadata=metadata,
        )

    # ------------------------------------------------------------------ page

    def _parse_page(
        self, page: Any, page_no: int, offset: int
    ) -> tuple[list[TextBlock], dict[str, int]]:
        blocks: list[TextBlock] = []
        cursor = offset

        vector_objects = len(page.lines) + len(page.rects) + len(page.curves)
        words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
        page_chars = sum(len(w["text"]) for w in words)

        # Whether this page is a drawing has to be decided before tables are
        # extracted. A schematic's grid and frame lines look exactly like table
        # rules, so running table extraction over a drawing produces dozens of
        # rows of fragments (": LS | : H") that are worse than no output at all.
        page_is_drawing = _is_drawing(vector_objects, page_chars)

        # Tables first, so their region can be excluded from the prose pass and
        # the same text is not indexed twice.
        table_regions: list[tuple[float, float, float, float]] = []
        tables = [] if page_is_drawing else page.find_tables()
        for table_index, table in enumerate(tables):
            rows = table.extract()
            if not rows:
                continue
            header = [(cell or "").strip() for cell in rows[0]]
            # A real table has a header. Ruled geometry that yields mostly empty
            # header cells is a frame, a border or a form outline.
            if sum(1 for cell in header if cell) < 2:
                continue
            table_regions.append(table.bbox)
            for row_index, row in enumerate(rows[1:], start=1):
                cells = [(cell or "").strip() for cell in row]
                pairs = [
                    f"{head}: {value}" for head, value in zip(header, cells, strict=False) if value
                ]
                if not pairs:
                    continue
                text = " | ".join(pairs)
                blocks.append(
                    TextBlock(
                        text=text,
                        page=page_no,
                        char_start=cursor,
                        char_end=cursor + len(text),
                        bbox=[float(v) for v in table.bbox],
                        kind="table_row",
                        extraction_confidence=1.0,
                        metadata={
                            "extraction_method": "pdfplumber.table",
                            "table_index": table_index,
                            "row_index": row_index,
                            "columns": header,
                        },
                    )
                )
                cursor += len(text) + 1

        prose_words = [w for w in words if not _inside_any(w, table_regions)]
        chars = sum(len(w["text"]) for w in prose_words)

        for paragraph in _group_paragraphs(prose_words):
            text = " ".join(w["text"] for w in paragraph).strip()
            if not text:
                continue
            blocks.append(
                TextBlock(
                    text=text,
                    page=page_no,
                    char_start=cursor,
                    char_end=cursor + len(text),
                    bbox=_bbox_of(paragraph),
                    kind="prose",
                    # An embedded text layer is read, not inferred: the
                    # characters are exactly what the producer wrote.
                    extraction_confidence=1.0,
                    metadata={"extraction_method": "pdfplumber.text_layer"},
                )
            )
            cursor += len(text) + 2

        return blocks, {
            "vector_objects": vector_objects,
            "chars": chars,
            "tables": len(table_regions),
            "drawing_pages": 1 if page_is_drawing else 0,
        }

    # ----------------------------------------------------------------- probe

    def _probe(self, data: bytes) -> tuple[bool, dict[str, Any], list[str]]:
        """Cheap pypdf pass for encryption and document metadata."""
        metadata: dict[str, Any] = {}
        warnings: list[str] = []
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(data))
            metadata["pages"] = len(reader.pages)
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception:
                    return True, metadata, warnings
            info = dict(reader.metadata or {})
            for key, target in (
                ("/Title", "title"),
                ("/Author", "author"),
                ("/Subject", "subject"),
                ("/Producer", "producer"),
            ):
                value = info.get(key)
                if value:
                    metadata[target] = str(value)[:300]
        except Exception as exc:
            warnings.append(f"PDF metadata probe failed: {type(exc).__name__}")
        return False, metadata, warnings


# --------------------------------------------------------------------- helpers


def _is_drawing(vector_objects: int, text_chars: int) -> bool:
    """Dense vector geometry with almost no text is a schematic.

    The ratio does the work. A numeric table has comparable geometry but plenty
    of text; a drawing's content is topology, so it carries almost none.
    """
    if vector_objects < DRAWING_VECTOR_THRESHOLD:
        return False
    return (text_chars / vector_objects) < DRAWING_TEXT_PER_VECTOR


def _inside_any(word: dict[str, Any], regions: list[tuple[float, float, float, float]]) -> bool:
    for x0, top, x1, bottom in regions:
        if (
            word["x0"] >= x0 - 1
            and word["x1"] <= x1 + 1
            and word["top"] >= top - 1
            and word["bottom"] <= bottom + 1
        ):
            return True
    return False


def _group_paragraphs(words: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group words into lines, then lines into paragraphs by vertical gap.

    Reading order matters: two-column text interleaved by naive extraction
    becomes nonsense, and a procedure that reads as nonsense is worse than one
    that is missing.
    """
    if not words:
        return []

    lines: dict[float, list[dict[str, Any]]] = {}
    for word in words:
        # Snap to the nearest half point so words on one visual line, whose tops
        # differ by sub-pixel amounts, group together.
        key = round(float(word["top"]) * 2) / 2
        lines.setdefault(key, []).append(word)

    paragraphs: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    previous_bottom: float | None = None

    for key in sorted(lines):
        line = sorted(lines[key], key=lambda w: float(w["x0"]))
        line_top = min(float(w["top"]) for w in line)
        if previous_bottom is not None and (line_top - previous_bottom) > PARAGRAPH_GAP:
            if current:
                paragraphs.append(current)
            current = []
        current.extend(line)
        previous_bottom = max(float(w["bottom"]) for w in line)

    if current:
        paragraphs.append(current)
    return paragraphs


def _bbox_of(words: list[dict[str, Any]]) -> list[float] | None:
    if not words:
        return None
    return [
        min(float(w["x0"]) for w in words),
        min(float(w["top"]) for w in words),
        max(float(w["x1"]) for w in words),
        max(float(w["bottom"]) for w in words),
    ]
