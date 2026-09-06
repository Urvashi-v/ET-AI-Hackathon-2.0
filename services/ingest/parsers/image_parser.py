"""Raster image parser.

A photographed permit, a scanned drawing, a phone picture of a work-order form.
There is no text layer to read, so the only honest options are OCR or nothing —
and "nothing" must be recorded as nothing rather than as an empty document.

Word geometry from OCR is carried through to the block bounding boxes, so a
citation into a scanned page can still be anchored, and the per-word confidence
becomes the block's ``extraction_confidence``. That number is the difference
between "the system read this" and "the system guessed at this", and the
dashboard shows which.
"""

from __future__ import annotations

from services.common.logging import get_logger
from services.common.schemas import CapabilityState
from services.ingest.ocr import LOW_CONFIDENCE_THRESHOLD, OCRPage, get_ocr_provider
from services.ingest.parsers.base import ParsedDocument, TextBlock

log = get_logger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


class ImageParser:
    name = "ocr"

    def can_parse(self, extension: str) -> bool:
        return extension.lower() in IMAGE_EXTENSIONS

    def parse(self, data: bytes, *, filename: str) -> ParsedDocument:
        result = get_ocr_provider().ocr_image(data)

        if result.state is not CapabilityState.AVAILABLE:
            return ParsedDocument(
                blocks=[],
                page_count=1,
                has_text_layer=False,
                parser=self.name,
                warnings=[result.detail or "OCR is unavailable."],
                metadata={
                    "ocr_state": result.state.value,
                    "ocr_required_env": result.required_env,
                },
            )

        blocks = blocks_from_ocr(result.pages)
        return ParsedDocument(
            blocks=blocks,
            page_count=len(result.pages),
            # There was no text layer; the text came from recognition. Saying
            # otherwise would hide that these characters are inferred.
            has_text_layer=False,
            parser=f"{self.name}:{result.engine}",
            warnings=ocr_quality_warnings(result.pages),
            metadata={"ocr": result.quality_report()},
        )


def blocks_from_ocr(pages: list[OCRPage]) -> list[TextBlock]:
    """Turn OCR word geometry into text blocks, one per recognised line group.

    Lines are grouped by tesseract's own block index, which respects columns —
    joining everything on whitespace would run a two-column page together, the
    same reading-order failure that makes naive PDF extraction useless.
    """
    blocks: list[TextBlock] = []
    offset = 0

    for page in pages:
        groups: dict[int, list] = {}
        for word in page.words:
            groups.setdefault(word.block, []).append(word)

        for block_index in sorted(groups):
            words = sorted(groups[block_index], key=lambda w: (w.paragraph, w.line, w.bbox[0]))
            if not words:
                continue

            # Keyed on (paragraph, line). Tesseract restarts line numbering
            # inside every paragraph, so keying on line alone merges a heading
            # with the first line of the body beneath it -- and since the merged
            # words are then ordered by x position, "IMMEDIATE CAUSE" and "The
            # outboard mechanical seal failed" come out interleaved as
            # "The IMMEDIATE outboard mechanical CAUSE seal failed". That breaks
            # every phrase the extractors look for.
            lines: dict[tuple[int, int], list] = {}
            for word in words:
                lines.setdefault((word.paragraph, word.line), []).append(word)
            text = "\n".join(
                " ".join(w.text for w in sorted(lines[key], key=lambda w: w.bbox[0]))
                for key in sorted(lines)
            ).strip()
            if not text:
                continue

            confidence = sum(w.confidence for w in words) / len(words)
            blocks.append(
                TextBlock(
                    text=text,
                    page=page.page,
                    char_start=offset,
                    char_end=offset + len(text),
                    bbox=[
                        min(w.bbox[0] for w in words),
                        min(w.bbox[1] for w in words),
                        max(w.bbox[2] for w in words),
                        max(w.bbox[3] for w in words),
                    ],
                    kind="prose",
                    extraction_confidence=round(confidence, 4),
                    metadata={
                        "extraction_method": "ocr",
                        "word_count": len(words),
                        "low_confidence_words": sum(
                            1 for w in words if w.confidence * 100 < LOW_CONFIDENCE_THRESHOLD
                        ),
                        "page_width": page.width,
                        "page_height": page.height,
                    },
                )
            )
            offset += len(text) + 2

    return blocks


def ocr_quality_warnings(pages: list[OCRPage]) -> list[str]:
    """Report OCR quality honestly, per page.

    A page recognised at 55% mean confidence has been read badly, and saying so
    is what lets a reviewer decide whether to trust anything extracted from it.
    """
    warnings: list[str] = []
    for page in pages:
        if not page.words:
            warnings.append(f"OCR recognised no text on page {page.page}.")
            continue
        mean = page.mean_confidence * 100
        if mean < LOW_CONFIDENCE_THRESHOLD:
            warnings.append(
                f"Page {page.page}: mean OCR confidence {mean:.0f}% is below the "
                f"{LOW_CONFIDENCE_THRESHOLD:.0f}% threshold. Text was retained but should "
                "be reviewed before anything extracted from it is relied on."
            )
        elif page.low_confidence_words:
            warnings.append(
                f"Page {page.page}: {page.low_confidence_words} of {len(page.words)} words "
                f"recognised below {LOW_CONFIDENCE_THRESHOLD:.0f}% confidence."
            )
    return warnings
