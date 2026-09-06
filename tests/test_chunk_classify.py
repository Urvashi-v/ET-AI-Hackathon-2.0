"""Document classification and structure-aware chunking.

Chunking is where most RAG systems quietly lose their accuracy, and the losses
are invisible: a step separated from its precondition still retrieves, it is
just unsafe. These tests pin the structural guarantees.
"""

from __future__ import annotations

import pytest

from services.common.schemas import DocumentType
from services.ingest.chunk import chunk_document
from services.ingest.classify import classify
from services.ingest.parsers.base import ParsedDocument, TextBlock
from services.ingest.parsers.tabular_parser import TabularParser, compose_record_summary
from services.ingest.parsers.text_parser import TextParser

SOP_TEXT = """# SOP-4412 Crude Charge Pump Startup

Document: SOP-4412
Revision: 3

## 3 Preconditions

Precondition: Confirm the suction drum V-102 level is above the low-low trip setpoint.

## 4 Startup sequence

4.1 Open the suction valve fully.

4.2 Open the seal flush supply and confirm flow at the seal pot.

4.3 Confirm discharge pressure develops within 30 seconds. The limit is 10 barg.
"""

INCIDENT_TEXT = """# Incident Report INC-2019-07

## What happened

The standby pump was started and seal leakage was observed within four minutes.

## Immediate cause

The outboard mechanical seal failed with scored faces.

## Root cause

Startup was carried out with the suction valve throttled, so the faces ran dry.

## Corrective action

CAPA-88 recommends a suction pressure interlock.
"""


def parse_markdown(text: str) -> ParsedDocument:
    return TextParser().parse(text.encode("utf-8"), filename="doc.md")


class TestClassification:
    @pytest.mark.parametrize(
        "filename,head,expected",
        [
            ("sop_4412_crude_charge_pump_startup.md", SOP_TEXT, DocumentType.SOP),
            ("incident_2019_seal_failure.md", INCIDENT_TEXT, DocumentType.INCIDENT_REPORT),
            (
                "moc_2023_07_impeller_trim.md",
                "# Management of Change MOC-2023-07",
                DocumentType.MOC,
            ),
            ("drawing.pdf", "PIPING AND INSTRUMENTATION DIAGRAM CDU-1 SHEET 3", DocumentType.PID),
            ("ut_survey.pdf", "ULTRASONIC THICKNESS SURVEY REPORT", DocumentType.INSPECTION_REPORT),
        ],
    )
    def test_routes_by_title_block_signals(self, filename, head, expected):
        result = classify(
            filename=filename,
            extension=f".{filename.rsplit('.', 1)[1]}",
            head_text=head,
            has_text_layer=True,
        )
        assert result.doc_type is expected

    def test_a_passing_reference_to_a_pid_does_not_make_a_document_a_pid(self):
        # An MOC listing "P&ID CDU-1 sheet 3" under documents-to-update is not a
        # drawing. Position weighting is what separates a title from a reference.
        head = (
            "# Management of Change MOC-2023-07\n"
            + ("filler. " * 120)
            + "\nUpdate P&ID CDU-1 sheet 3."
        )
        result = classify(
            filename="moc_2023_07.md", extension=".md", head_text=head, has_text_layer=True
        )
        assert result.doc_type is DocumentType.MOC

    def test_a_procedure_mentioning_work_orders_is_not_a_work_order(self):
        result = classify(
            filename="sop_4412_startup.md",
            extension=".md",
            head_text=SOP_TEXT + "\nConfirm no open work order restricts operation.",
            has_text_layer=True,
        )
        assert result.doc_type is DocumentType.SOP

    def test_underscore_separated_filenames_are_matched(self):
        # `\bsop\b` does not match "sop_4412..." because `_` is a word character.
        result = classify(
            filename="sop_4412_startup.md",
            extension=".md",
            head_text="no signal here",
            has_text_layer=True,
        )
        assert result.doc_type is DocumentType.SOP

    def test_a_scan_is_routed_to_ocr_and_says_so(self):
        result = classify(filename="scan.pdf", extension=".pdf", head_text="", has_text_layer=False)
        assert result.needs_ocr and result.pipeline == "scanned"

    def test_no_signal_yields_unknown_rather_than_a_guess(self):
        result = classify(
            filename="a1b2c3.pdf",
            extension=".pdf",
            head_text="lorem ipsum dolor sit amet",
            has_text_layer=True,
        )
        assert result.doc_type is DocumentType.UNKNOWN
        assert result.confidence == 0.0
        assert "review" in (result.notes or "")

    def test_rival_types_are_reported_not_hidden(self):
        result = classify(
            filename="mixed.md",
            extension=".md",
            head_text="INCIDENT REPORT\nSTANDARD OPERATING PROCEDURE",
            has_text_layer=True,
        )
        assert result.ambiguous_between


class TestProcedureChunking:
    def test_one_chunk_per_step(self):
        chunks = chunk_document(
            parse_markdown(SOP_TEXT), doc_type=DocumentType.SOP, doc_title="SOP-4412", revision="3"
        )
        steps = [c for c in chunks if c.kind == "step"]
        assert {c.metadata["step_no"] for c in steps} == {"4.1", "4.2", "4.3"}

    def test_governing_precondition_travels_with_its_steps(self):
        # A step without its precondition is a safety hazard, not a chunk. The
        # association is carried in metadata rather than spliced into the body:
        # repeating the whole safety preamble inside every step dilutes each
        # chunk and measurably degrades retrieval of the specific step asked for.
        chunks = chunk_document(
            parse_markdown(SOP_TEXT), doc_type=DocumentType.SOP, doc_title="SOP-4412", revision="3"
        )
        steps = [c for c in chunks if c.kind == "step"]
        assert steps and all(c.metadata["has_governing_condition"] for c in steps)
        conditions = " ".join(steps[0].metadata["governing_conditions"])
        assert "low-low trip setpoint" in conditions

    def test_the_governing_section_is_retrievable_in_its_own_right(self):
        # "What PPE is required?" must have something precise to match.
        chunks = chunk_document(
            parse_markdown(SOP_TEXT), doc_type=DocumentType.SOP, doc_title="SOP-4412", revision="3"
        )
        preconditions = [c for c in chunks if c.kind == "precondition"]
        assert preconditions
        assert "low-low trip setpoint" in " ".join(c.text for c in preconditions)

    def test_a_step_body_is_not_diluted_by_the_safety_preamble(self):
        chunks = chunk_document(
            parse_markdown(SOP_TEXT), doc_type=DocumentType.SOP, doc_title="SOP-4412", revision="3"
        )
        step = next(c for c in chunks if c.metadata.get("step_no") == "4.1")
        assert step.text.startswith("4.1 Open the suction valve")
        assert "Governing condition" not in step.text

    def test_context_header_names_the_document_and_section(self):
        chunks = chunk_document(
            parse_markdown(SOP_TEXT), doc_type=DocumentType.SOP, doc_title="SOP-4412", revision="3"
        )
        header = chunks[0].context_header
        assert "SOP-4412" in header and "rev 3" in header

    def test_the_body_stays_byte_identical_for_verbatim_citation(self):
        chunks = chunk_document(
            parse_markdown(SOP_TEXT), doc_type=DocumentType.SOP, doc_title="SOP-4412", revision="3"
        )
        for chunk in chunks:
            assert chunk.context_header not in chunk.text
            assert chunk.text in chunk.embedding_text


class TestIncidentChunking:
    def test_splits_on_causal_sections(self):
        chunks = chunk_document(
            parse_markdown(INCIDENT_TEXT),
            doc_type=DocumentType.INCIDENT_REPORT,
            doc_title="INC-2019-07",
        )
        sections = {c.metadata.get("section") for c in chunks}
        assert {"root_cause", "corrective_action"} <= sections

    def test_root_cause_is_retrievable_without_the_narrative(self):
        chunks = chunk_document(
            parse_markdown(INCIDENT_TEXT),
            doc_type=DocumentType.INCIDENT_REPORT,
            doc_title="INC-2019-07",
        )
        root = next(c for c in chunks if c.metadata.get("section") == "root_cause")
        assert "throttled" in root.text
        assert "four minutes" not in root.text


class TestRecordChunking:
    CSV = (
        "# SYNTHETIC TEST DATA banner line\n"
        "wo_id,equipment_tag,order_type,long_text,downtime_hrs,opened_on\n"
        "WO-4471,P-101B,CORRECTIVE,Replaced outboard seal after dry running,14.0,2024-03-12\n"
        "WO-2140,P101B,CORRECTIVE,Seal faces scored,16.0,22/03/2019\n"
    )

    def test_leading_comment_lines_are_not_read_as_the_header(self):
        # Without this the banner becomes the column names and every row maps
        # into one garbage field -- a silent failure, because parsing "succeeds".
        parsed = TabularParser().parse(self.CSV.encode("utf-8"), filename="wo.csv")
        assert len(parsed.records) == 2
        assert parsed.records[0]["wo_id"] == "WO-4471"

    def test_one_chunk_per_record(self):
        parsed = TabularParser().parse(self.CSV.encode("utf-8"), filename="wo.csv")
        chunks = chunk_document(parsed, doc_type=DocumentType.WORK_ORDER, doc_title="CMMS export")
        assert len(chunks) == 2
        assert all(c.kind == "record" for c in chunks)

    def test_mixed_date_formats_both_normalise(self):
        parsed = TabularParser().parse(self.CSV.encode("utf-8"), filename="wo.csv")
        assert parsed.records[0]["opened_on"] == "2024-03-12"
        assert parsed.records[1]["opened_on"] == "2019-03-22"

    def test_unmapped_columns_are_kept_and_reported(self):
        csv = "wo_id,mystery_column\nWO-1,important value\n"
        parsed = TabularParser().parse(csv.encode("utf-8"), filename="wo.csv")
        assert parsed.records[0]["extra"]["mystery_column"] == "important value"
        assert any("Unmapped columns" in w for w in parsed.warnings)

    def test_record_summary_states_only_what_the_row_contains(self):
        summary = compose_record_summary(
            {
                "wo_id": "WO-4471",
                "asset_tag": "P-101B",
                "downtime_hours": 14.0,
                "as_found": "Seal faces scored",
            },
            {},
        )
        assert "WO-4471" in summary and "P-101B" in summary
        assert "14.0 h" in summary and "Seal faces scored" in summary
        # Nothing is inferred: fields that were absent must not appear.
        assert "cost" not in summary.lower() and "closed" not in summary.lower()

    def test_record_summary_is_deterministic(self):
        record = {"wo_id": "WO-1", "asset_tag": "P-101B", "status": "CLOSED"}
        assert len({compose_record_summary(record, {}) for _ in range(20)}) == 1


class TestDrawingChunking:
    def test_a_drawing_is_not_chunked_as_prose(self):
        parsed = ParsedDocument(
            blocks=[TextBlock(text="P-101B E-104 V-102 PIC-101", page=1)],
            page_count=1,
            has_text_layer=True,
            parser="pypdf",
        )
        chunks = chunk_document(parsed, doc_type=DocumentType.PID, doc_title="CDU-1 sheet 3")
        assert len(chunks) == 1
        assert chunks[0].kind == "summary"
        assert "Topology" in chunks[0].metadata["note"]


class TestParserHonesty:
    def test_an_empty_document_reports_why_rather_than_returning_nothing(self):
        parsed = TextParser().parse(b"", filename="empty.md")
        assert parsed.blocks == []
        assert parsed.warnings and "no extractable text" in parsed.warnings[0]

    def test_unparseable_json_is_reported(self):
        parsed = TabularParser().parse(b"{not json", filename="broken.json")
        assert parsed.records == []
        assert any("could not be parsed" in w for w in parsed.warnings)
