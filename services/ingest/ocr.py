"""OCR providers.

A scanned document is bytes with no text layer. Without OCR the pipeline can
record that the document exists and nothing else — which is the honest outcome,
and what Day 1 reported.

``tesseract`` is a real engine: offline, no credentials, no network call at run
time. That last property matters beyond convenience — it keeps the air-gapped
deployment story intact, which is the single biggest objection from the plants
this is aimed at.

What OCR returns here is not a blob of text. It is **words with bounding boxes
and per-word confidence**, which is what the provenance contract requires: a
citation that can be anchored on the page, and a quality number that can be
trended per document rather than assumed.

Low-confidence words are retained but marked. Silently passing garbage into the
graph is worse than passing nothing, because nothing is visibly missing while
garbage is invisibly wrong.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from services.common.config import get_settings
from services.common.logging import get_logger
from services.common.schemas import CapabilityState

log = get_logger(__name__)

#: Per-word confidence (0-100 from tesseract) below which a word is kept but
#: flagged. Tesseract reports -1 for non-text regions, which is filtered out
#: separately.
LOW_CONFIDENCE_THRESHOLD = 60.0

#: Render resolution for PDF pages. 300 DPI is the usual floor for reliable OCR
#: of engineering text; below roughly 200 the small rotated text on a drawing
#: becomes unreadable.
RENDER_DPI = 300


@dataclass(slots=True)
class OCRWord:
    """One recognised word with its geometry and confidence."""

    text: str
    bbox: list[float]  # [x0, y0, x1, y1] in pixels of the rendered page
    confidence: float  # 0-1
    line: int = 0
    #: Tesseract restarts `line_num` inside every paragraph, so a line is only
    #: identified by the triple (block, paragraph, line). Grouping on
    #: (block, line) alone merges the first line of two different paragraphs --
    #: which silently interleaves a heading into the body text beneath it and
    #: breaks phrases that extraction depends on.
    paragraph: int = 0
    block: int = 0


@dataclass(slots=True)
class OCRPage:
    page: int
    text: str
    words: list[OCRWord] = field(default_factory=list)
    width: float = 0.0
    height: float = 0.0

    @property
    def mean_confidence(self) -> float:
        if not self.words:
            return 0.0
        return sum(w.confidence for w in self.words) / len(self.words)

    @property
    def low_confidence_words(self) -> int:
        return sum(1 for w in self.words if w.confidence * 100 < LOW_CONFIDENCE_THRESHOLD)


@dataclass(slots=True)
class OCRResult:
    state: CapabilityState
    pages: list[OCRPage] = field(default_factory=list)
    engine: str = "none"
    detail: str | None = None
    required_env: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def mean_confidence(self) -> float:
        words = [w for page in self.pages for w in page.words]
        return sum(w.confidence for w in words) / len(words) if words else 0.0

    @property
    def word_count(self) -> int:
        return sum(len(page.words) for page in self.pages)

    def quality_report(self) -> dict[str, Any]:
        """Per-document OCR quality. A credible number for the eval slide, and a
        real signal for the review queue."""
        return {
            "engine": self.engine,
            "pages": len(self.pages),
            "words": self.word_count,
            "mean_confidence": round(self.mean_confidence, 4),
            "low_confidence_words": sum(p.low_confidence_words for p in self.pages),
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


class OCRProvider(Protocol):
    name: str

    def ocr_pdf(self, data: bytes, *, max_pages: int | None = None) -> OCRResult: ...

    def ocr_image(self, data: bytes) -> OCRResult: ...


class DisabledOCRProvider:
    """The honest no-op. Says why it did nothing and never invents text."""

    name = "none"

    def _unavailable(self) -> OCRResult:
        return OCRResult(
            state=CapabilityState.NOT_CONFIGURED,
            engine="none",
            detail=(
                "This document has no usable text layer and no OCR provider is configured, "
                "so its text was not recovered. The document is recorded with its metadata "
                "and queued for review. No text is invented."
            ),
            required_env=["OCR_PROVIDER"],
        )

    def ocr_pdf(self, data: bytes, *, max_pages: int | None = None) -> OCRResult:
        return self._unavailable()

    def ocr_image(self, data: bytes) -> OCRResult:
        return self._unavailable()


class TesseractOCRProvider:
    """Tesseract via pytesseract, with pypdfium2 for page rendering.

    Uses ``image_to_data`` rather than ``image_to_string``: the former returns
    the per-word geometry and confidence that make a citation anchorable and OCR
    quality measurable. The latter returns a string and throws that away.
    """

    name = "tesseract"

    def __init__(self, dpi: int = RENDER_DPI) -> None:
        self.dpi = dpi

    def _probe(self) -> tuple[bool, str]:
        try:
            import pytesseract

            version = pytesseract.get_tesseract_version()
            return True, f"tesseract {version}"
        except ImportError:
            return False, "pytesseract is not installed"
        except Exception as exc:  # binary missing or not on PATH
            return False, f"tesseract binary unavailable: {type(exc).__name__}"

    def ocr_image(self, data: bytes) -> OCRResult:
        started = time.perf_counter()
        ok, detail = self._probe()
        if not ok:
            return OCRResult(
                state=CapabilityState.ERROR,
                engine=self.name,
                detail=f"OCR_PROVIDER=tesseract is selected but {detail}.",
                required_env=["OCR_PROVIDER"],
            )
        try:
            from PIL import Image

            image = Image.open(io.BytesIO(data))
            page = self._recognise(image, page_number=1)
        except Exception as exc:
            log.error("ocr.image_failed", error=str(exc))
            return OCRResult(
                state=CapabilityState.ERROR,
                engine=self.name,
                detail=f"OCR failed: {type(exc).__name__}: {str(exc)[:200]}",
            )
        return OCRResult(
            state=CapabilityState.AVAILABLE,
            pages=[page],
            engine=self.name,
            detail=detail,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    def ocr_pdf(self, data: bytes, *, max_pages: int | None = None) -> OCRResult:
        started = time.perf_counter()
        ok, detail = self._probe()
        if not ok:
            return OCRResult(
                state=CapabilityState.ERROR,
                engine=self.name,
                detail=f"OCR_PROVIDER=tesseract is selected but {detail}.",
                required_env=["OCR_PROVIDER"],
            )

        pages: list[OCRPage] = []
        try:
            import pypdfium2 as pdfium

            document = pdfium.PdfDocument(data)
            total = len(document)
            limit = min(total, max_pages) if max_pages else total
            scale = self.dpi / 72.0
            for index in range(limit):
                image = document[index].render(scale=scale).to_pil()
                pages.append(self._recognise(image, page_number=index + 1))
            document.close()
        except Exception as exc:
            log.error("ocr.pdf_failed", error=str(exc))
            return OCRResult(
                state=CapabilityState.ERROR,
                engine=self.name,
                detail=f"OCR failed: {type(exc).__name__}: {str(exc)[:200]}",
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )

        elapsed = (time.perf_counter() - started) * 1000
        log.info(
            "ocr.completed",
            engine=self.name,
            pages=len(pages),
            words=sum(len(p.words) for p in pages),
            elapsed_ms=round(elapsed, 1),
        )
        return OCRResult(
            state=CapabilityState.AVAILABLE,
            pages=pages,
            engine=self.name,
            detail=detail,
            elapsed_ms=elapsed,
        )

    def _recognise(self, image: Any, *, page_number: int) -> OCRPage:
        import pytesseract

        data = pytesseract.image_to_data(
            image, output_type=pytesseract.Output.DICT, config="--psm 3"
        )
        words: list[OCRWord] = []
        for index, raw_text in enumerate(data["text"]):
            text = (raw_text or "").strip()
            confidence = float(data["conf"][index])
            # Tesseract reports -1 for layout regions that contain no word.
            if not text or confidence < 0:
                continue
            left, top = float(data["left"][index]), float(data["top"][index])
            width, height = float(data["width"][index]), float(data["height"][index])
            words.append(
                OCRWord(
                    text=text,
                    bbox=[left, top, left + width, top + height],
                    confidence=confidence / 100.0,
                    line=int(data["line_num"][index]),
                    paragraph=int(data["par_num"][index]),
                    block=int(data["block_num"][index]),
                )
            )

        return OCRPage(
            page=page_number,
            text=_reflow(words),
            words=words,
            width=float(image.width),
            height=float(image.height),
        )


def _reflow(words: list[OCRWord]) -> str:
    """Reassemble words into lines using tesseract's own block/line grouping.

    Joining on whitespace alone would run every line of a two-column page
    together, which is the same reading-order failure that makes naive PDF text
    extraction useless for procedures.
    """
    lines: dict[tuple[int, int, int], list[OCRWord]] = {}
    for word in words:
        lines.setdefault((word.block, word.paragraph, word.line), []).append(word)

    rendered: list[str] = []
    for key in sorted(lines):
        ordered = sorted(lines[key], key=lambda w: w.bbox[0])
        rendered.append(" ".join(w.text for w in ordered))
    return "\n".join(rendered).strip()


def get_ocr_provider() -> OCRProvider:
    settings = get_settings()
    if settings.ocr_provider == "tesseract":
        return TesseractOCRProvider()
    if settings.ocr_provider == "paddle":
        # Selected but not adapted. Reported as such rather than silently
        # falling back to a different engine, which would make the quality
        # numbers describe something other than what was configured.
        log.warning("ocr.provider_not_implemented", provider="paddle")
        return DisabledOCRProvider()
    return DisabledOCRProvider()


def ocr_capability() -> tuple[CapabilityState, str, list[str]]:
    """What the pipeline reports for its OCR stage."""
    settings = get_settings()
    if settings.ocr_provider == "none":
        return (
            CapabilityState.NOT_CONFIGURED,
            "No OCR provider is configured. Documents without a text layer are recorded "
            "with their metadata and queued for review; their text is not recovered.",
            ["OCR_PROVIDER"],
        )
    if settings.ocr_provider == "paddle":
        return (
            CapabilityState.NOT_IMPLEMENTED,
            "OCR_PROVIDER=paddle is selected but the PaddleOCR adapter is not implemented. "
            "Use OCR_PROVIDER=tesseract.",
            ["OCR_PROVIDER"],
        )
    ok, detail = TesseractOCRProvider()._probe()
    if not ok:
        return (CapabilityState.ERROR, f"OCR_PROVIDER=tesseract but {detail}.", ["OCR_PROVIDER"])
    return (CapabilityState.AVAILABLE, detail, [])
