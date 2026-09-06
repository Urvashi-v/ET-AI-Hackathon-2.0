"""Deterministic extraction.

The extractor's precision matters more than its recall: a false positive creates
a phantom asset that then accumulates evidence and distorts every count computed
over the graph, invisibly. Most of these tests are about things that must *not*
be extracted.
"""

from __future__ import annotations

import pytest

from services.ingest.extract import (
    DEGRADATION_CUES,
    extract_all,
    extract_tags,
    find_functional_locations,
    validate_extraction,
)

WORK_ORDER_TEXT = (
    "Work order WO-4471 on P-101B (functional location CDU1-PUMP-101) closed 12/03/2024: "
    "replaced outboard mechanical seal after vibration high on the 10-P-101-B pump. "
    "Suction pressure 2.5 barg, discharge 12 barg. Refer OISD-STD-105 clause 4.2 and "
    'SOP-4412. PIC-101 indicated drift. Line 6"-P-1501-A1A was isolated. Slight leak '
    "noted earlier; temporary repair applied. CAPA-88 still open. Spare pump P-101A was "
    "started. Thickness 8.2 mm against 6.0 mm minimum."
)


class TestTagExtraction:
    def test_finds_equipment_instruments_and_lines(self):
        canonical = {m.canonical for m in extract_all(WORK_ORDER_TEXT).tags}
        assert {"P-101B", "P-101A", "PIC-101", '6"-P-1501'} <= canonical

    def test_tag_variants_in_one_passage_resolve_to_one_canonical_form(self):
        tags = extract_all("P-101B and 10-P-101-B are the same pump").tags
        assert {t.canonical for t in tags} == {"P-101B"}
        assert len(tags) == 2, "both surface forms must be retained as separate mentions"

    def test_offsets_point_at_the_actual_substring(self):
        for mention in extract_all(WORK_ORDER_TEXT).tags:
            assert WORK_ORDER_TEXT[mention.char_start : mention.char_end] == mention.surface_form

    def test_sibling_pumps_are_extracted_as_distinct_entities(self):
        canonical = {m.canonical for m in extract_all(WORK_ORDER_TEXT).tags}
        assert "P-101A" in canonical and "P-101B" in canonical


class TestFalsePositiveSuppression:
    """Document references and functional locations are not equipment."""

    @pytest.mark.parametrize(
        "reference,phantom",
        [
            ("SOP-4412", "P-4412"),
            ("CAPA-88", "A-88"),
            ("OISD-STD-105", "D-105"),
            ("MOC-2023", "C-2023"),
            ("INC-2019", "C-2019"),
        ],
    )
    def test_document_references_do_not_become_assets(self, reference, phantom):
        canonical = {m.canonical for m in extract_all(f"Refer {reference} for detail.").tags}
        assert phantom not in canonical
        assert not canonical, f"{reference} must produce no asset mentions at all"

    def test_functional_location_does_not_become_a_phantom_equipment_node(self):
        # CDU1-PUMP-101 parses perfectly well as pump P-101. Left alone it
        # creates a third pump alongside the real P-101A and P-101B.
        result = extract_all("Work order on P-101B, functional location CDU1-PUMP-101.")
        assert {m.canonical for m in result.tags} == {"P-101B"}
        assert len(result.functional_locations) == 1

    def test_functional_locations_are_still_captured_separately(self):
        locations = find_functional_locations("Functional location: CDU1-PUMP-101")
        assert locations and locations[0]["surface_form"] == "CDU1-PUMP-101"

    def test_a_psv_is_an_asset_not_a_document_reference(self):
        # PSV-204 is a pressure safety valve with an ISA-style tag. It must reach
        # the tag grammar, unlike SOP/CAPA/OISD prefixes.
        canonical = {m.canonical for m in extract_all("PSV-204 is due for testing.").tags}
        assert "PSV-204" in canonical

    def test_ordinary_prose_produces_no_tags(self):
        text = "The technician arrived at 2 am and found the area is 100 percent clear."
        assert not extract_all(text).tags


class TestSupportingExtractors:
    def test_document_references(self):
        refs = {(r["kind"], r["number"]) for r in extract_all(WORK_ORDER_TEXT).document_refs}
        assert {("WO", "4471"), ("OISD", "105"), ("SOP", "4412"), ("CAPA", "88")} <= refs

    def test_clause_references(self):
        assert "4.2" in {c["clause"] for c in extract_all(WORK_ORDER_TEXT).clause_refs}

    def test_dates_in_mixed_formats(self):
        assert "2024-03-12" in {d["date"] for d in extract_all(WORK_ORDER_TEXT).dates}
        assert "2024-03-12" in {d["date"] for d in extract_all("closed 2024-03-12").dates}

    def test_quantities_always_carry_a_unit(self):
        quantities = extract_all(WORK_ORDER_TEXT).quantities
        assert ("barg", 2.5) in {(q["unit"], q["value"]) for q in quantities}
        assert all(q["unit"] for q in quantities), "a bare number is never a fact here"

    def test_bare_numbers_are_not_quantities(self):
        assert not extract_all("There were 14 of them.").quantities

    def test_degradation_language_is_detected(self):
        cues = {c["cue"] for c in extract_all(WORK_ORDER_TEXT).degradation_cues}
        assert {"slight leak", "vibration high", "temporary repair"} <= cues

    def test_every_declared_cue_is_detectable(self):
        for cue in DEGRADATION_CUES:
            found = {c["cue"] for c in extract_all(f"Note: {cue} observed.").degradation_cues}
            assert cue in found


class TestVerbatimEvidenceGuard:
    """The cheapest hallucination defence in existence: one containment check."""

    def test_accepts_a_tag_and_quotes_that_occur_in_the_source(self):
        ok, reason = validate_extraction(
            source_text=WORK_ORDER_TEXT,
            asserted_tag="P-101B",
            evidence_quotes=["replaced outboard mechanical seal"],
        )
        assert ok and reason is None

    def test_rejects_a_hallucinated_tag(self):
        ok, reason = validate_extraction(
            source_text=WORK_ORDER_TEXT, asserted_tag="P-999Z", evidence_quotes=[]
        )
        assert not ok and "does not occur" in reason

    def test_rejects_a_paraphrased_quote(self):
        ok, reason = validate_extraction(
            source_text=WORK_ORDER_TEXT,
            asserted_tag="P-101B",
            evidence_quotes=["the seal was swapped out"],
        )
        assert not ok and "verbatim" in reason

    def test_is_deterministic(self):
        outcomes = {
            validate_extraction(
                source_text=WORK_ORDER_TEXT, asserted_tag="P-101B", evidence_quotes=["seal"]
            )[0]
            for _ in range(50)
        }
        assert outcomes == {True}


class TestGazetteer:
    def test_learned_alias_is_matched(self):
        tags = extract_tags("the B pump was started", gazetteer={"the b pump": "P-101B"})
        assert [t.canonical for t in tags] == ["P-101B"]
        assert tags[0].extractor == "gazetteer"

    def test_gazetteer_does_not_override_a_parsed_tag(self):
        tags = extract_tags("P-101B tripped", gazetteer={"p-101b": "P-999Z"})
        assert [t.extractor for t in tags] == ["regex:tag"]
