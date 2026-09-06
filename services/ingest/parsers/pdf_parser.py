"""PDF parser.

Extracts the embedded text layer page by page, keeping the page number and the
character offsets that citations resolve against.

Two honest limits, both reported rather than hidden:

* **Scanned PDFs.** A page with no text layer yields no text. Rather than
  emitting an empty block and pretending the document was read, the parser marks
  the page and sets ``has_text_layer=False`` so the pipeline records
  ``ocr: provider_not_configured`` and queues the document for review.
* **Bounding boxes.** ``pypdf`` gives reading-order text, not per-span geometry,
  so ``bbox`` is left ``None`` here. Span-level highlight needs a layout-aware
  parser; the schema and citation contract already carry the field so that
  adding one is a parser change, not a migration.
"""

from __future__ import annotations

import io
from typing import Any

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from services.common.logging import get_logger
from services.ingest.parsers.base import ParsedDocument, TextBlock

log = get_logger(__name__)

#: A page with fewer than this many characters of extractable text is treated as
#: having no usable text layer.
_MIN_CHARS_PER_PAGE = 20


class PdfParser:
    name = "pypdf"

    def can_parse(self, extension: str) -> bool:
        return extension.lower() == ".pdf"

    def parse(self, data: bytes, *, filename: str) -> ParsedDocument:
        warnings: list[str] = []
        blocks: list[TextBlock] = []

        try:
            reader = PdfReader(io.BytesIO(data))
        except PdfReadError as exc:
            return ParsedDocument(
                blocks=[],
                page_count=None,
                has_text_layer=None,
                parser=self.name,
                warnings=[f"PDF could not be opened: {exc}"],
            )

        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                return ParsedDocument(
                    blocks=[],
                    page_count=len(reader.pages),
                    has_text_layer=None,
                    parser=self.name,
                    warnings=["PDF is encrypted and no password was supplied."],
                )

        offset = 0
        pages_with_text = 0
        for page_no, page in enumerate(reader.pages, start=1):
            try:
                text = (page.extract_text() or "").strip()
            except Exception as exc:  # a single malformed page must not fail the file
                warnings.append(f"Page {page_no} could not be extracted: {type(exc).__name__}")
                continue

            if len(text) < _MIN_CHARS_PER_PAGE:
                warnings.append(
                    f"Page {page_no} has no usable text layer "
                    f"({len(text)} chars): OCR required to read it."
                )
                continue

            pages_with_text += 1
            blocks.append(
                TextBlock(
                    text=text,
                    page=page_no,
                    char_start=offset,
                    char_end=offset + len(text),
                    kind="prose",
                    metadata={"page_chars": len(text)},
                )
            )
            offset += len(text) + 2

        page_count = len(reader.pages)
        has_text_layer = pages_with_text > 0
        if page_count and pages_with_text < page_count:
            warnings.append(
                f"{page_count - pages_with_text} of {page_count} pages had no text layer."
            )

        metadata: dict[str, object] = {
            "pages_with_text": pages_with_text,
            "pages_total": page_count,
        }
        try:
            info: dict[Any, Any] = dict(reader.metadata or {})
            for key, target in (
                ("/Title", "title"),
                ("/Author", "author"),
                ("/Subject", "subject"),
            ):
                value = info.get(key)
                if value:
                    metadata[target] = str(value)[:300]
        except Exception:
            pass

        return ParsedDocument(
            blocks=blocks,
            page_count=page_count,
            has_text_layer=has_text_layer,
            parser=self.name,
            warnings=warnings,
            metadata=metadata,
        )
