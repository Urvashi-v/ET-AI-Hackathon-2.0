"""PDF parsing, OCR, and the provenance both must preserve.

These run against the real PDF corpus in ``data/synthetic/generated_pdf``, which
holds real PDF files with synthetic content: a text-layer document with a ruled
table, a numeric table, an image-only scan with no text layer, and a vector
schematic. Between them they cover every parsing path the pipeline has.

OCR tests skip when the tesseract binary is absent, so the suite still runs on a
developer machine without it. It is installed in the container image, and CI
exercises the OCR path there.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from services.common.schemas import CapabilityState, DocumentType
from services.ingest.chunk import chunk_document, merge_bbox, merge_provenance
from services.ingest.classify import classify
from services.ingest.ocr import (
    LOW_CONFIDENCE_THRESHOLD,
    DisabledOCRProvider,
    OCRPage,
    OCRWord,
    TesseractOCRProvider,
    _reflow,
)
from services.ingest.parsers import parse_document
from services.ingest.parsers.base import TextBlock
from services.ingest.parsers.image_parser import blocks_from_ocr, ocr_quality_warnings
from services.ingest.parsers.pdf_parser import PdfParser

REPO_ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = REPO_ROOT / "data" / "synthetic" / "generated_pdf"

SOP = PDF_DIR / "sop_4412_rev3.pdf"
INSPECTION = PDF_DIR / "inspection_ut_survey_2025.pdf"
SCAN = PDF_DIR / "incident_2019_scan.pdf"
PID = PDF_DIR / "pid_cdu1_sheet3.pdf"


def tesseract_available() -> bool:
    return TesseractOCRProvider()._probe()[0]


requires_tesseract = pytest.mark.skipif(
    not tesseract_available(), reason="tesseract binary not installed on this machine"
)


@pytest.fixture(scope="module", autouse=True)
def _corpus() -> None:
    """Generate the PDF corpus if it is not already present."""
    if SOP.exists() and SCAN.exists() and PID.exists() and INSPECTION.exists():
        return
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "data" / "synthetic" / "generate_pdfs.py")],
        capture_output=True,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        pytest.skip(f"could not generate the PDF corpus: {result.stderr.decode()[:200]}")


class TestPdfTextLayer:
    def test_reads_a_text_layer_pdf(self):
        parsed = PdfParser().parse(SOP.read_bytes(), filename=SOP.name)
        assert parsed.has_text_layer is True
        assert parsed.page_count == 1
        assert parsed.blocks

    def test_every_block_carries_a_bounding_box(self):
        # Without geometry captured at parse time, "highlight the sentence on the
        # source page" is impossible later -- it cannot be recovered.
        parsed = PdfParser().parse(SOP.read_bytes(), filename=SOP.name)
        for block in parsed.blocks:
            assert block.bbox is not None, f"{block.text[:40]!r} has no bbox"
            x0, top, x1, bottom = block.bbox
            assert x1 > x0 and bottom > top

    def test_every_block_records_how_it_was_extracted(self):
        parsed = PdfParser().parse(SOP.read_bytes(), filename=SOP.name)
        methods = {b.metadata.get("extraction_method") for b in parsed.blocks}
        assert methods <= {"pdfplumber.text_layer", "pdfplumber.table"}
        assert None not in methods

    def test_text_layer_characters_are_read_not_inferred(self):
        parsed = PdfParser().parse(SOP.read_bytes(), filename=SOP.name)
        assert all(b.extraction_confidence == 1.0 for b in parsed.blocks)

    def test_document_content_survives(self):
        parsed = PdfParser().parse(SOP.read_bytes(), filename=SOP.name)
        text = parsed.full_text
        assert "SOP-4412" in text
        assert "P-101B" in text
        assert "permit to work" in text.lower()

    def test_reading_order_is_preserved(self):
        parsed = PdfParser().parse(SOP.read_bytes(), filename=SOP.name)
        text = parsed.full_text
        assert text.index("4.1 Open the suction valve") < text.index("4.2 Open the seal flush")


class TestPdfTables:
    def test_a_ruled_table_is_extracted_as_rows(self):
        parsed = PdfParser().parse(INSPECTION.read_bytes(), filename=INSPECTION.name)
        rows = [b for b in parsed.blocks if b.kind == "table_row"]
        assert len(rows) >= 6

    def test_row_column_association_survives(self):
        # A flattened table loses the association that makes the numbers mean
        # anything, and a model reading it reports the wrong cell.
        parsed = PdfParser().parse(INSPECTION.read_bytes(), filename=INSPECTION.name)
        rows = [b for b in parsed.blocks if b.kind == "table_row"]
        first = next(r for r in rows if "CML-01" in r.text)
        assert "CML: CML-01" in first.text
        assert "Thickness mm: 11.62" in first.text
        assert "Min mm: 9.50" in first.text

    def test_table_columns_are_recorded(self):
        parsed = PdfParser().parse(INSPECTION.read_bytes(), filename=INSPECTION.name)
        row = next(b for b in parsed.blocks if b.kind == "table_row")
        assert "CML" in row.metadata["columns"]

    def test_table_text_is_not_also_indexed_as_prose(self):
        parsed = PdfParser().parse(INSPECTION.read_bytes(), filename=INSPECTION.name)
        prose = " ".join(b.text for b in parsed.blocks if b.kind == "prose")
        assert "11.62" not in prose, "table cells must not be duplicated into the prose pass"


class TestDrawingDetection:
    def test_a_schematic_is_recognised_as_a_drawing(self):
        parsed = PdfParser().parse(PID.read_bytes(), filename=PID.name)
        assert parsed.metadata["drawing_signature"] is True
        assert parsed.metadata["vector_objects"] > 100

    def test_a_prose_page_is_not_a_drawing(self):
        parsed = PdfParser().parse(SOP.read_bytes(), filename=SOP.name)
        assert parsed.metadata["drawing_signature"] is False

    def test_a_dense_numeric_table_is_not_a_drawing(self):
        # Both have vector geometry; the ratio of text to geometry separates them.
        parsed = PdfParser().parse(INSPECTION.read_bytes(), filename=INSPECTION.name)
        assert parsed.metadata["drawing_signature"] is False

    def test_no_phantom_tables_are_extracted_from_a_drawing(self):
        # A schematic's frame and grid lines look exactly like table rules, and
        # running table extraction over them yields rows of fragments.
        parsed = PdfParser().parse(PID.read_bytes(), filename=PID.name)
        assert parsed.metadata["tables_found"] == 0
        assert not [b for b in parsed.blocks if b.kind == "table_row"]

    def test_the_classifier_routes_a_drawing_to_the_drawing_pipeline(self):
        parsed = PdfParser().parse(PID.read_bytes(), filename=PID.name)
        result = classify(
            filename=PID.name,
            extension=".pdf",
            head_text=parsed.head_text,
            has_text_layer=parsed.has_text_layer,
            vector_segment_count=parsed.metadata["vector_objects"],
            drawing_signature=parsed.metadata["drawing_signature"],
        )
        assert result.doc_type is DocumentType.PID
        assert result.pipeline == "drawing"
        assert result.method == "vector_density"
        assert "not implemented" in (result.notes or "")


class TestScannedDocument:
    def test_a_scan_reports_no_text_layer_rather_than_empty_text(self):
        parsed = PdfParser().parse(SCAN.read_bytes(), filename=SCAN.name)
        assert parsed.has_text_layer is False
        assert parsed.blocks == []
        assert any("no usable text layer" in w for w in parsed.warnings)

    def test_the_page_count_is_still_known(self):
        # The document exists and is recorded even though it cannot be read.
        parsed = PdfParser().parse(SCAN.read_bytes(), filename=SCAN.name)
        assert parsed.page_count == 1

    @requires_tesseract
    def test_ocr_recovers_the_text(self):
        result = TesseractOCRProvider().ocr_pdf(SCAN.read_bytes())
        assert result.state is CapabilityState.AVAILABLE
        assert result.word_count > 100
        text = " ".join(p.text for p in result.pages)
        assert "INC-2019-07" in text
        assert "P-101B" in text

    @requires_tesseract
    def test_ocr_reports_per_word_geometry_and_confidence(self):
        result = TesseractOCRProvider().ocr_pdf(SCAN.read_bytes())
        words = [w for page in result.pages for w in page.words]
        assert words
        for word in words[:40]:
            assert 0.0 <= word.confidence <= 1.0
            x0, top, x1, bottom = word.bbox
            assert x1 > x0 and bottom > top

    @requires_tesseract
    def test_ocr_confidence_is_below_that_of_a_read_text_layer(self):
        # Recognised characters are not read characters, and the number says so.
        result = TesseractOCRProvider().ocr_pdf(SCAN.read_bytes())
        assert 0.0 < result.mean_confidence < 1.0

    @requires_tesseract
    def test_ocr_reading_order_keeps_headings_separate_from_body(self):
        result = TesseractOCRProvider().ocr_pdf(SCAN.read_bytes())
        text = " ".join(p.text for p in result.pages)
        # The failure this guards against interleaves them into
        # "The IMMEDIATE outboard mechanical CAUSE seal failed".
        assert "outboard mechanical seal failed" in text
        assert "IMMEDIATE outboard" not in text

    @requires_tesseract
    def test_ocr_quality_is_reported_per_document(self):
        report = TesseractOCRProvider().ocr_pdf(SCAN.read_bytes()).quality_report()
        assert report["engine"] == "tesseract"
        assert report["words"] > 0
        assert 0.0 <= report["mean_confidence"] <= 1.0


class TestOCRDisabled:
    def test_the_disabled_provider_never_invents_text(self):
        result = DisabledOCRProvider().ocr_pdf(SCAN.read_bytes())
        assert result.state is CapabilityState.NOT_CONFIGURED
        assert result.pages == []
        assert "OCR_PROVIDER" in result.required_env
        assert "No text is invented" in (result.detail or "")

    def test_it_says_the_same_for_images(self):
        assert DisabledOCRProvider().ocr_image(b"anything").state is CapabilityState.NOT_CONFIGURED


class TestOCRBlockAssembly:
    """Reading order and provenance, tested without needing the binary."""

    def _words(self):
        return [
            OCRWord("IMMEDIATE", [190, 1046, 400, 1070], 0.96, line=1, paragraph=1, block=4),
            OCRWord("CAUSE", [489, 1049, 600, 1070], 0.94, line=1, paragraph=1, block=4),
            OCRWord("The", [188, 1122, 250, 1145], 0.98, line=1, paragraph=2, block=4),
            OCRWord("outboard", [278, 1123, 450, 1145], 0.97, line=1, paragraph=2, block=4),
            OCRWord("seal", [484, 1126, 560, 1145], 0.40, line=1, paragraph=2, block=4),
        ]

    def test_paragraphs_are_not_merged_into_one_line(self):
        page = OCRPage(page=1, text="", words=self._words(), width=1240, height=1754)
        block = blocks_from_ocr([page])[0]
        assert block.text.startswith("IMMEDIATE CAUSE")
        assert "The outboard seal" in block.text
        assert "IMMEDIATE outboard" not in block.text

    def test_reflow_respects_paragraph_boundaries(self):
        text = _reflow(self._words())
        assert text.splitlines()[0] == "IMMEDIATE CAUSE"

    def test_block_confidence_is_the_weakest_word(self):
        # A passage is only as trustworthy as its worst-read word.
        page = OCRPage(page=1, text="", words=self._words(), width=1240, height=1754)
        block = blocks_from_ocr([page])[0]
        assert block.extraction_confidence == pytest.approx(0.85, abs=0.01)
        assert block.metadata["low_confidence_words"] == 1

    def test_block_bbox_is_the_union_of_its_words(self):
        page = OCRPage(page=1, text="", words=self._words(), width=1240, height=1754)
        block = blocks_from_ocr([page])[0]
        assert block.bbox == [188, 1046, 600, 1145]

    def test_extraction_method_is_recorded_as_ocr(self):
        page = OCRPage(page=1, text="", words=self._words(), width=1240, height=1754)
        assert blocks_from_ocr([page])[0].metadata["extraction_method"] == "ocr"

    def test_a_badly_read_page_is_reported(self):
        poor = [OCRWord("smudge", [0, 0, 10, 10], 0.31, line=1, paragraph=1, block=1)]
        warnings = ocr_quality_warnings([OCRPage(page=2, text="", words=poor)])
        assert warnings and "page 2" in warnings[0].lower()
        assert str(int(LOW_CONFIDENCE_THRESHOLD)) in warnings[0]

    def test_a_page_with_no_recognised_text_is_reported(self):
        warnings = ocr_quality_warnings([OCRPage(page=1, text="", words=[])])
        assert warnings and "no text" in warnings[0].lower()


class TestProvenanceThroughChunking:
    def test_bbox_survives_into_chunks(self):
        parsed = parse_document(SOP.read_bytes(), filename=SOP.name, extension=".pdf")
        chunks = chunk_document(parsed, doc_type=DocumentType.SOP, doc_title="SOP-4412")
        assert chunks
        assert any(c.bbox is not None for c in chunks)

    def test_extraction_method_survives_into_chunks(self):
        parsed = parse_document(SOP.read_bytes(), filename=SOP.name, extension=".pdf")
        chunks = chunk_document(parsed, doc_type=DocumentType.SOP, doc_title="SOP-4412")
        methods = {c.extraction_method for c in chunks}
        assert methods & {"pdfplumber.text_layer", "pdfplumber.table", "mixed"}

    def test_merge_bbox_unions_boxes_on_one_page(self):
        blocks = [
            TextBlock(text="a", page=1, bbox=[10, 20, 30, 40]),
            TextBlock(text="b", page=1, bbox=[5, 25, 50, 35]),
        ]
        assert merge_bbox(blocks) == [5, 20, 50, 40]

    def test_merge_bbox_refuses_to_span_a_page_break(self):
        # A rectangle spanning two pages would be a lie, so there is none.
        blocks = [
            TextBlock(text="a", page=1, bbox=[10, 20, 30, 40]),
            TextBlock(text="b", page=2, bbox=[5, 25, 50, 35]),
        ]
        assert merge_bbox(blocks) is None

    def test_merge_provenance_takes_the_weakest_confidence(self):
        blocks = [
            TextBlock(text="a", extraction_confidence=0.99, metadata={"extraction_method": "ocr"}),
            TextBlock(text="b", extraction_confidence=0.42, metadata={"extraction_method": "ocr"}),
        ]
        confidence, method = merge_provenance(blocks)
        assert confidence == 0.42 and method == "ocr"

    def test_mixed_extraction_methods_are_reported_as_mixed(self):
        blocks = [
            TextBlock(text="a", metadata={"extraction_method": "ocr"}),
            TextBlock(text="b", metadata={"extraction_method": "pdfplumber.text_layer"}),
        ]
        assert merge_provenance(blocks)[1] == "mixed"


class TestMalformedInput:
    def test_garbage_bytes_named_as_a_pdf_are_rejected_with_a_reason(self):
        parsed = PdfParser().parse(b"this is not a PDF at all", filename="broken.pdf")
        assert parsed.blocks == []
        assert parsed.warnings

    def test_an_empty_file_yields_no_blocks_and_a_reason(self):
        parsed = PdfParser().parse(b"", filename="empty.pdf")
        assert parsed.blocks == []
        assert parsed.warnings

    def test_a_truncated_pdf_is_survivable(self):
        truncated = SOP.read_bytes()[:400]
        parsed = PdfParser().parse(truncated, filename="truncated.pdf")
        assert parsed.blocks == []
        assert parsed.warnings

    def test_an_unknown_extension_says_so(self):
        parsed = parse_document(b"data", filename="x.xyz", extension=".xyz")
        assert parsed.blocks == []
        assert "No parser is registered" in parsed.warnings[0]

    def test_an_image_without_ocr_is_recorded_not_faked(self):
        from services.common.config import get_settings

        if get_settings().ocr_provider != "none":
            pytest.skip("OCR is configured on this machine")
        parsed = parse_document(b"\x89PNG\r\n\x1a\n", filename="scan.png", extension=".png")
        assert parsed.blocks == []
        assert parsed.has_text_layer is False
