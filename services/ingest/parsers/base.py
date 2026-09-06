"""Parser contract.

A parser turns bytes into :class:`ParsedDocument` -- text blocks that each carry
their own provenance (page, character offsets, and a bounding box where the
format provides one). Provenance is captured *here*, at parse time, because it
cannot be recovered later: merging parsing into chunking is the classic mistake
that makes span-level citation impossible.

A parser never invents content. If a page yields nothing, it emits a block
recording that fact, with the reason, so the pipeline can queue the document for
review rather than silently indexing an empty document.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class TextBlock:
    """One extracted unit of text with its provenance."""

    text: str
    page: int | None = None
    char_start: int = 0
    char_end: int = 0
    bbox: list[float] | None = None
    section_path: str | None = None
    kind: str = "prose"  # prose | heading | table_row | record | step
    extraction_confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ParsedDocument:
    blocks: list[TextBlock]
    page_count: int | None = None
    has_text_layer: bool | None = None
    parser: str = "unknown"
    warnings: list[str] = field(default_factory=list)
    #: Structured records recovered from tabular/record-shaped sources. These go
    #: to relational tables and graph properties, not through prose chunking.
    records: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks if b.text)

    @property
    def head_text(self) -> str:
        """Roughly the first page, used by the classifier."""
        first_page = [b.text for b in self.blocks if b.page in (None, 1)]
        return "\n".join(first_page)[:6000] or self.full_text[:6000]


class Parser(Protocol):
    name: str

    def can_parse(self, extension: str) -> bool: ...

    def parse(self, data: bytes, *, filename: str) -> ParsedDocument: ...
