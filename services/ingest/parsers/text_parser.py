"""Plain-text and Markdown parser.

Recovers heading hierarchy into ``section_path``, which is the single most
useful piece of chunk metadata available: it is what lets a chunk say "SOP-4412
rev 3 > 5 Priming" instead of floating free, and it is what the contextual chunk
header is built from.
"""

from __future__ import annotations

import re

from services.ingest.parsers.base import ParsedDocument, TextBlock

_ATX_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_SETEXT_UNDERLINE = re.compile(r"^(=|-){3,}\s*$")
#: Numbered section headings, e.g. "5.2 Priming" -- ubiquitous in SOPs and
#: standards, and absent from Markdown's own grammar.
#:
#: The trailing `[^.]` matters more than it looks. A numbered *step*
#: ("4.1 Open the suction valve fully.") has the same shape as a numbered
#: *heading* ("4 Startup sequence"), and the only reliable difference in real
#: procedures is that a step is a sentence and ends in a full stop. Without this
#: guard every step in an SOP is consumed as a heading, the procedure chunker
#: finds no steps at all, and the document silently loses its structure.
#: The title must also be short and contain no sentence break. Procedure text is
#: routinely hard-wrapped, and the first line of a wrapped step ("4.6 Confirm
#: discharge pressure develops within 30 seconds. The discharge pressure") ends
#: mid-sentence on a non-period character -- indistinguishable from a heading
#: without these two extra constraints.
_NUMBERED_HEADING = re.compile(r"^\s*(\d+(?:\.\d+){0,3})[.)]?\s+([A-Z][^\n]{2,58}[^.\s])\s*$")
_SENTENCE_BREAK = re.compile(r"[.!?]\s")


class TextParser:
    name = "text"

    def can_parse(self, extension: str) -> bool:
        return extension.lower() in {".txt", ".md", ".markdown"}

    def parse(self, data: bytes, *, filename: str) -> ParsedDocument:
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError:
            content = data.decode("latin-1")

        blocks: list[TextBlock] = []
        warnings: list[str] = []
        heading_stack: list[tuple[int, str]] = []

        offset = 0
        buffer: list[str] = []
        buffer_start = 0

        def flush() -> None:
            nonlocal buffer, buffer_start
            text = "\n".join(buffer).strip()
            if text:
                blocks.append(
                    TextBlock(
                        text=text,
                        page=1,
                        char_start=buffer_start,
                        char_end=buffer_start + len(text),
                        section_path=" > ".join(h[1] for h in heading_stack) or None,
                        kind="prose",
                    )
                )
            buffer = []

        lines = content.splitlines()
        for idx, line in enumerate(lines):
            line_start = offset
            offset += len(line) + 1

            level, title = _heading_of(line, lines[idx + 1] if idx + 1 < len(lines) else "")
            if title is not None:
                flush()
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, title))
                blocks.append(
                    TextBlock(
                        text=title,
                        page=1,
                        char_start=line_start,
                        char_end=line_start + len(line),
                        section_path=" > ".join(h[1] for h in heading_stack),
                        kind="heading",
                    )
                )
                buffer_start = offset
                continue

            if not line.strip():
                flush()
                buffer_start = offset
                continue

            if not buffer:
                buffer_start = line_start
            buffer.append(line)

        flush()

        if not blocks:
            warnings.append("File contained no extractable text.")

        return ParsedDocument(
            blocks=blocks,
            page_count=1,
            has_text_layer=True,
            parser=self.name,
            warnings=warnings,
            metadata={"line_count": len(lines), "char_count": len(content)},
        )


def _heading_of(line: str, next_line: str) -> tuple[int, str | None]:
    m = _ATX_HEADING.match(line)
    if m:
        return len(m.group(1)), m.group(2).strip()
    if line.strip() and _SETEXT_UNDERLINE.match(next_line or ""):
        return (1 if next_line.startswith("=") else 2), line.strip()
    m = _NUMBERED_HEADING.match(line)
    if m and not _SENTENCE_BREAK.search(m.group(2)):
        return m.group(1).count(".") + 1, f"{m.group(1)} {m.group(2).strip()}"
    return 0, None
