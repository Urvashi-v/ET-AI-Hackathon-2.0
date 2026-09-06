"""Parser registry.

The router in ``services/ingest/classify.py`` decides *what* a document is; this
registry decides *which parser reads it*. Formats with no parser return a
``ParsedDocument`` carrying the reason -- an unreadable file is recorded as
unreadable, never as an empty document.
"""

from __future__ import annotations

from services.ingest.parsers.base import ParsedDocument, Parser, TextBlock
from services.ingest.parsers.docx_parser import DocxParser
from services.ingest.parsers.pdf_parser import PdfParser
from services.ingest.parsers.tabular_parser import TabularParser
from services.ingest.parsers.text_parser import TextParser

_REGISTRY: tuple[Parser, ...] = (
    TextParser(),
    PdfParser(),
    TabularParser(),
    DocxParser(),
)

#: Extensions that can only be read with OCR. Handled by the pipeline, which
#: reports ``ocr: provider_not_configured`` rather than emitting empty text.
OCR_ONLY_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def get_parser(extension: str) -> Parser | None:
    ext = extension.lower()
    for parser in _REGISTRY:
        if parser.can_parse(ext):
            return parser
    return None


def parse_document(data: bytes, *, filename: str, extension: str) -> ParsedDocument:
    parser = get_parser(extension)
    if parser is None:
        reason = (
            "Raster image: an OCR provider must be configured to read it (OCR_PROVIDER)."
            if extension.lower() in OCR_ONLY_EXTENSIONS
            else f"No parser is registered for '{extension}'."
        )
        return ParsedDocument(
            blocks=[],
            page_count=None,
            has_text_layer=False,
            parser="none",
            warnings=[reason],
        )
    return parser.parse(data, filename=filename)


__all__ = [
    "OCR_ONLY_EXTENSIONS",
    "ParsedDocument",
    "Parser",
    "TextBlock",
    "get_parser",
    "parse_document",
]
