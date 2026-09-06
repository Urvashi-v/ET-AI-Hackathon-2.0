"""Failure-term extraction, the verbatim-evidence guard, and deduplication.

The verbatim guard is the one that matters most. It is the cheapest hallucination
defence in existence — one string containment test, no second model, fully
deterministic — and it runs regardless of which extractor produced the claim.
"""

from __future__ import annotations

import pytest

from services.common.ids import document_id, sha256_bytes
from services.common.schemas import CapabilityState
from services.ingest.extract import (
    FAILURE_TERMS,
    UNINFORMATIVE_FAILURE_CODES,
    compare_coded_and_extracted,
    extract_all,
    extract_failure_terms,
    validate_extraction,
)
from services.ingest.llm_extract import (
    ALLOWED_FAILURE_MODES,
    _validate_response,
    extraction_capability,
    is_worth_extracting,
    locate_span,
    verify_span,
)

WORK_ORDER_TEXT = (
    "Attended for high vibration on P-101B. Found outboard seal faces scored, evidence of "
    "dry running during startup. Operator confirms suction valve was throttled. Replaced "
    "outboard mechanical seal."
)


class TestFailureTerminology:
    def test_recovers_the_failure_mode_from_the_narrative(self):
        codes = {t["failure_mode_code"] for t in extract_failure_terms(WORK_ORDER_TEXT)}
        assert "ELP" in codes  # seal faces scored
        assert "VIB" in codes  # high vibration

    def test_every_term_carries_a_verbatim_span(self):
        for term in extract_failure_terms(WORK_ORDER_TEXT):
            assert WORK_ORDER_TEXT[term["char_start"] : term["char_end"]] == term["quote"]

    def test_longest_phrase_wins_over_a_weaker_overlapping_one(self):
        terms = extract_failure_terms("Found seal faces scored on strip-down.")
        phrases = {t["phrase"] for t in terms}
        assert "seal faces scored" in phrases
        assert "seal leak" not in phrases

    def test_no_overlapping_double_counting(self):
        terms = extract_failure_terms(WORK_ORDER_TEXT)
        spans = sorted((t["char_start"], t["char_end"]) for t in terms)
        for (_, end), (start, _) in zip(spans, spans[1:], strict=False):
            assert start >= end, "extracted spans must not overlap"

    def test_ordinary_prose_yields_nothing(self):
        assert extract_failure_terms("The technician arrived at 2 am and signed the permit.") == []

    def test_every_declared_phrase_is_detectable(self):
        for phrase in FAILURE_TERMS:
            found = {t["phrase"] for t in extract_failure_terms(f"Report notes: {phrase}.")}
            assert phrase in found

    def test_every_code_is_in_the_allowed_vocabulary(self):
        # The gazetteer and the LLM must agree on the vocabulary, or aggregation
        # across the two becomes meaningless.
        for code, _ in FAILURE_TERMS.values():
            assert code in ALLOWED_FAILURE_MODES

    def test_it_is_deterministic(self):
        runs = {
            tuple(t["failure_mode_code"] for t in extract_failure_terms(WORK_ORDER_TEXT))
            for _ in range(30)
        }
        assert len(runs) == 1

    def test_it_is_wired_into_the_main_extractor(self):
        assert extract_all(WORK_ORDER_TEXT).counts()["failure_terms"] > 0


class TestCodedVersusNarrative:
    """The structured field lies; the free text tells the truth."""

    def test_a_dropdown_default_is_recoded_from_the_narrative(self):
        terms = extract_failure_terms(WORK_ORDER_TEXT)
        verdict = compare_coded_and_extracted("OTHER", terms)
        assert verdict["verdict"] == "recoded"
        assert verdict["extracted_mode"] == "ELP"
        assert verdict["evidence_quote"] in WORK_ORDER_TEXT

    @pytest.mark.parametrize("code", sorted(UNINFORMATIVE_FAILURE_CODES - {""}))
    def test_every_uninformative_code_triggers_recoding(self, code):
        assert (
            compare_coded_and_extracted(code, extract_failure_terms(WORK_ORDER_TEXT))["verdict"]
            == "recoded"
        )

    def test_agreement_is_reported_as_agreement(self):
        assert (
            compare_coded_and_extracted("ELP", extract_failure_terms(WORK_ORDER_TEXT))["verdict"]
            == "agree"
        )

    def test_disagreement_is_surfaced_not_silently_overridden(self):
        verdict = compare_coded_and_extracted("BRD", extract_failure_terms(WORK_ORDER_TEXT))
        assert verdict["verdict"] == "disagree"
        assert verdict["coded_mode"] == "BRD"
        assert verdict["extracted_mode"] == "ELP"
        assert "neither is overwritten" in verdict["detail"]

    def test_no_narrative_evidence_is_stated_as_such(self):
        assert compare_coded_and_extracted("ELP", [])["verdict"] == "no_text_evidence"


class TestVerbatimGuard:
    def test_accepts_a_quote_that_occurs_in_the_source(self):
        assert verify_span("seal faces scored", WORK_ORDER_TEXT)

    def test_tolerates_reflowed_whitespace(self):
        # A model that wrapped the quote across a line has still quoted it.
        assert verify_span("seal   faces\nscored", WORK_ORDER_TEXT)

    def test_rejects_a_paraphrase(self):
        assert not verify_span("the seal was worn out", WORK_ORDER_TEXT)

    def test_rejects_an_empty_quote(self):
        assert not verify_span("", WORK_ORDER_TEXT)
        assert not verify_span("   ", WORK_ORDER_TEXT)

    def test_locates_the_span_for_citation_anchoring(self):
        span = locate_span("seal faces scored", WORK_ORDER_TEXT)
        assert span is not None
        assert WORK_ORDER_TEXT[span[0] : span[1]] == "seal faces scored"

    def test_an_absent_quote_has_no_span(self):
        assert locate_span("never written anywhere", WORK_ORDER_TEXT) is None

    def test_the_legacy_guard_still_rejects_a_hallucinated_tag(self):
        ok, reason = validate_extraction(
            source_text=WORK_ORDER_TEXT, asserted_tag="P-999Z", evidence_quotes=[]
        )
        assert not ok and "does not occur" in reason


class TestLLMResponseValidation:
    """Validation runs on the response regardless of which model produced it."""

    def _response(self, **overrides) -> str:
        import json

        payload = {
            "equipment_tag_raw": "P-101B",
            "failure_mode": "ELP",
            "as_found_condition": "outboard seal faces scored",
            "suspected_cause": "dry running during startup",
            "confidence": 0.88,
            "evidence": [
                {"field": "failure_mode", "quote": "seal faces scored"},
                {"field": "as_found_condition", "quote": "outboard seal faces scored"},
                {"field": "suspected_cause", "quote": "dry running during startup"},
            ],
        }
        payload.update(overrides)
        return json.dumps(payload)

    def test_a_well_formed_response_is_accepted(self):
        facts, rejected = _validate_response(
            self._response(), source_text=WORK_ORDER_TEXT, model="test"
        )
        assert rejected == 0
        assert all(f.quote_verified for f in facts)
        assert {f.kind for f in facts} == {"failure_mode", "as_found_condition", "suspected_cause"}

    def test_a_hallucinated_tag_rejects_the_whole_extraction(self):
        # A phantom asset corrupts every count computed over the graph, so this
        # is rejected outright rather than downgraded.
        facts, rejected = _validate_response(
            self._response(equipment_tag_raw="P-999Z"), source_text=WORK_ORDER_TEXT, model="test"
        )
        assert rejected == 1
        assert not facts[0].quote_verified
        assert "does not occur" in facts[0].reject_reason

    def test_a_paraphrased_quote_is_rejected(self):
        import json

        payload = json.loads(self._response())
        payload["evidence"] = [{"field": "failure_mode", "quote": "the seal had worn away"}]
        facts, rejected = _validate_response(
            json.dumps(payload), source_text=WORK_ORDER_TEXT, model="test"
        )
        failure = next(f for f in facts if f.kind == "failure_mode")
        assert not failure.quote_verified
        assert "verbatim" in failure.reject_reason
        assert rejected >= 1

    def test_a_field_with_no_evidence_is_rejected(self):
        import json

        payload = json.loads(self._response())
        payload["evidence"] = []
        facts, rejected = _validate_response(
            json.dumps(payload), source_text=WORK_ORDER_TEXT, model="test"
        )
        assert rejected == len(facts)
        assert all("no evidence quote" in f.reject_reason for f in facts)

    def test_a_failure_mode_outside_the_vocabulary_is_rejected(self):
        facts, rejected = _validate_response(
            self._response(failure_mode="SEAL_EXPLODED"),
            source_text=WORK_ORDER_TEXT,
            model="test",
        )
        failure = next(f for f in facts if f.kind == "failure_mode")
        assert not failure.quote_verified
        assert "outside the allowed vocabulary" in failure.reject_reason

    def test_null_fields_are_not_asserted(self):
        # Null is a correct answer and must not become a fact.
        facts, _ = _validate_response(
            self._response(suspected_cause=None), source_text=WORK_ORDER_TEXT, model="test"
        )
        assert "suspected_cause" not in {f.kind for f in facts}

    def test_invalid_json_is_recorded_as_a_rejection(self):
        facts, rejected = _validate_response(
            "I think the seal failed.", source_text=WORK_ORDER_TEXT, model="test"
        )
        assert rejected == 1
        assert "not valid JSON" in facts[0].reject_reason

    def test_a_markdown_fence_is_tolerated(self):
        fenced = f"```json\n{self._response()}\n```"
        facts, rejected = _validate_response(fenced, source_text=WORK_ORDER_TEXT, model="test")
        assert rejected == 0 and facts

    def test_rejections_are_kept_so_the_rate_is_measurable(self):
        # Discarding them would make the denominator unknowable.
        facts, _ = _validate_response(
            self._response(equipment_tag_raw="P-999Z"), source_text=WORK_ORDER_TEXT, model="test"
        )
        assert facts, "a rejected extraction must still be recorded"
        assert facts[0].reject_reason


class TestLLMGating:
    def test_disabled_by_default(self):
        from services.common.config import get_settings

        if get_settings().llm_provider != "none":
            pytest.skip("an LLM provider is configured on this machine")
        state, detail, required_env = extraction_capability()
        assert state is CapabilityState.NOT_CONFIGURED
        assert "LLM_PROVIDER" in required_env
        assert "unaffected" in detail

    def test_a_table_row_is_not_worth_a_model_call(self):
        assert not is_worth_extracting("table_row", "x" * 400, has_failure_vocabulary=True)

    def test_short_text_is_not_worth_a_model_call(self):
        assert not is_worth_extracting("prose", "seal leak", has_failure_vocabulary=True)

    def test_text_without_failure_vocabulary_is_not_worth_a_model_call(self):
        assert not is_worth_extracting("prose", "x" * 400, has_failure_vocabulary=False)

    def test_a_plausible_narrative_is_worth_a_model_call(self):
        assert is_worth_extracting(
            "incident_section", WORK_ORDER_TEXT * 2, has_failure_vocabulary=True
        )


class TestDeduplication:
    def test_identical_bytes_produce_one_document_id(self):
        content = b"%PDF-1.7 identical content"
        assert document_id(sha256_bytes(content)) == document_id(sha256_bytes(content))

    def test_a_single_byte_difference_produces_a_different_id(self):
        assert document_id(sha256_bytes(b"content a")) != document_id(sha256_bytes(b"content b"))

    def test_the_hash_is_sha256(self):
        # NIST vector for the empty string, so the algorithm cannot silently change.
        assert sha256_bytes(b"") == (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )

    def test_filename_does_not_affect_identity(self):
        # The same document arriving from two systems under two names is one
        # document. Deduplicating on name instead would double-count it.
        content = b"the same bytes"
        assert document_id(sha256_bytes(content)) == document_id(sha256_bytes(content))

    def test_the_document_id_is_derived_from_the_hash(self):
        digest = sha256_bytes(b"anything")
        assert document_id(digest).endswith(digest[:24])
