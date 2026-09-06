"""Tag grammar, normalisation and match scoring.

These are the highest-value tests in the repository. If entity resolution is
wrong the knowledge graph is a set of disconnected islands and every headline
claim is false — and the failure is silent, which is why it needs exhaustive
coverage rather than a smoke test.
"""

from __future__ import annotations

import pytest

from services.common.tags import (
    EQUIPMENT_CLASS_CODES,
    TagKind,
    TagRelation,
    blocking_key,
    decide,
    decode_instrument_function,
    normalise,
    parse,
    score_pair,
)

#: The six spellings of one pump, exactly as they appear across real systems.
#: The fifth uses U+2011 NON-BREAKING HYPHEN, which a Word autocorrect leaves
#: behind and which no human reviewer will spot.
P101B_VARIANTS = [
    "P-101B",
    "P101B",
    "P 101 B",
    "10-P-101-B",
    "P-101-B",
    "P‑101‑B",
    "p-101b",
    "  P-101B  ",
    "P_101_B",
    "P/101/B",
    "CDU1-PUMP-101B",
    "Pump 101 B",
    "P-0101B",
]


class TestNormalisation:
    @pytest.mark.parametrize("raw", P101B_VARIANTS)
    def test_all_variants_normalise_to_the_same_canonical_tag(self, raw):
        assert parse(raw).canonical == "P-101B"

    def test_unicode_hyphen_is_folded(self):
        assert normalise("P‑101‑B") == normalise("P-101-B")

    def test_leading_zeros_are_stripped_from_numeric_segments(self):
        assert parse("P-0101B").canonical == parse("P-101B").canonical

    def test_word_abbreviations_map_to_class_codes(self):
        assert parse("PUMP-101B").canonical == "P-101B"
        assert parse("HX-204").canonical == "E-204"

    def test_empty_and_garbage_input_is_survivable(self):
        for value in ("", "   ", "---", "‑"):
            result = parse(value)
            assert result.kind is TagKind.UNPARSED
            assert result.canonical == result.normalised


class TestEquipmentParsing:
    def test_segments_are_recovered(self):
        parsed = parse("10-P-101-B")
        assert parsed.kind is TagKind.EQUIPMENT
        assert (parsed.unit, parsed.cls, parsed.seq, parsed.suffix) == ("10", "P", "101", "B")

    def test_unit_prefix_is_excluded_from_the_canonical_form(self):
        # A missing plant prefix is missing information, not conflicting
        # information, so it must not partition the same asset in two.
        assert parse("10-P-101-B").canonical == parse("P-101B").canonical

    @pytest.mark.parametrize("code", sorted(EQUIPMENT_CLASS_CODES))
    def test_every_declared_class_code_parses(self, code):
        parsed = parse(f"{code}-201")
        assert parsed.kind is TagKind.EQUIPMENT
        assert parsed.cls == code

    def test_unknown_class_code_does_not_become_equipment(self):
        assert parse("QQ-101").kind is not TagKind.EQUIPMENT


class TestInstrumentParsing:
    @pytest.mark.parametrize(
        "raw,cls",
        [("PIC-101", "PIC"), ("FT-201", "FT"), ("LSHH-305", "LSHH"), ("PI-101", "PI")],
    )
    def test_isa_instrument_tags(self, raw, cls):
        parsed = parse(raw)
        assert parsed.kind is TagKind.INSTRUMENT
        assert parsed.cls == cls

    def test_instrument_is_tried_before_equipment(self):
        # "PIC101" would otherwise read as unit "PI" + class "C" (compressor).
        assert parse("PIC-101").kind is TagKind.INSTRUMENT

    def test_equipment_class_codes_win_over_isa_decoding(self):
        # "TK" decodes under ISA as Temperature/Time, but TK is a tank.
        assert parse("TK-201").kind is TagKind.EQUIPMENT
        assert parse("TK-201").cls == "TK"

    def test_isa_function_decoding(self):
        assert decode_instrument_function("PIC") == "Pressure Indicate Control"
        assert decode_instrument_function("LSHH") == "Level Switch High High"
        assert decode_instrument_function("XQ9") is None


class TestOtherGrammars:
    def test_line_number_requires_a_size_marker(self):
        parsed = parse('6"-P-1501-A1A')
        assert parsed.kind is TagKind.LINE
        assert parsed.extra["size"] == "6"
        assert parsed.seq == "1501"

    def test_without_a_size_marker_it_is_equipment_not_a_line(self):
        assert parse("P-1501").kind is TagKind.EQUIPMENT

    def test_kks_reference_designation(self):
        parsed = parse("10LAC20AP001")
        assert parsed.kind is TagKind.KKS
        assert parsed.extra["system"] == "LAC"
        assert parsed.extra["equipment_type"] == "AP"
        assert parsed.unit == "10"

    def test_technician_shorthand_is_retained_not_discarded(self):
        parsed = parse("the B pump")
        assert parsed.kind is TagKind.UNPARSED
        assert parsed.raw == "the B pump"


class TestMatchScoring:
    """The rules that protect every downstream statistic."""

    @pytest.mark.parametrize("raw", P101B_VARIANTS)
    def test_every_variant_matches_the_canonical_tag(self, raw):
        result = score_pair("P-101B", raw)
        assert result.relation is TagRelation.SAME
        assert result.score >= 0.85

    def test_siblings_are_never_merged(self):
        # THE test. Naive fuzzy matching merges these -- one character in six --
        # and corrupts every failure statistic invisibly.
        result = score_pair("P-101A", "P-101B")
        assert result.relation is TagRelation.SIBLING
        assert result.relation is not TagRelation.SAME
        assert "suffix" in result.reason

    def test_sibling_relation_is_symmetric(self):
        assert score_pair("P-101A", "P-101B").relation is score_pair("P-101B", "P-101A").relation

    def test_different_sequence_is_a_different_asset(self):
        assert score_pair("P-101B", "P-102B").relation is TagRelation.DIFFERENT

    def test_conflicting_unit_prefix_is_a_different_asset(self):
        result = score_pair("10-P-101-B", "20-P-101-B")
        assert result.relation is TagRelation.DIFFERENT
        assert "unit" in result.reason

    def test_missing_unit_prefix_is_not_conflicting(self):
        result = score_pair("10-P-101-B", "P-101B")
        assert result.relation is TagRelation.SAME
        assert 0.8 <= result.score < 0.97

    def test_different_tag_kinds_never_match(self):
        assert score_pair("P-101B", "PIC-101").relation is TagRelation.DIFFERENT

    def test_confirmed_alias_is_decisive(self):
        result = score_pair("P-101B", "the B pump", known_alias=True)
        assert result.score == 1.0
        assert result.method == "alias_table"

    def test_context_cannot_override_a_parse_level_contradiction(self):
        # A context hint is a hint; a suffix mismatch is evidence.
        result = score_pair("P-101A", "P-101B", context_similarity=1.0)
        assert result.relation is TagRelation.SIBLING


class TestBlockingAndDecisions:
    def test_variants_share_a_blocking_key(self):
        keys = {blocking_key(v) for v in P101B_VARIANTS}
        assert len(keys) == 1

    def test_siblings_share_a_blocking_key_so_they_can_be_linked(self):
        # They must land in the same bucket to be *compared* -- that is how the
        # sibling edge gets created instead of the pair being silently missed.
        assert blocking_key("P-101A") == blocking_key("P-101B")

    def test_unrelated_assets_do_not_share_a_blocking_key(self):
        assert blocking_key("P-101B") != blocking_key("E-104")

    @pytest.mark.parametrize(
        "a,b,expected",
        [
            ("P-101B", "P101B", "merge"),
            ("P-101A", "P-101B", "link_sibling"),
            ("P-101B", "P-102B", "separate"),
            # A tag missing the plant prefix scores 0.85, which sits in the
            # ambiguous middle band below the 0.90 auto-merge threshold. That is
            # deliberate: a plant with both unit 10 and unit 20 has two assets
            # whose tags differ only by the prefix the document omitted, so the
            # link is created and flagged rather than silently asserted.
            ("P-101B", "10-P-101-B", "needs_review"),
        ],
    )
    def test_decision_mapping(self, a, b, expected):
        action = decide(score_pair(a, b), auto_merge_threshold=0.90, review_threshold=0.60)
        assert action == expected

    def test_unit_prefixed_variants_unify_by_canonical_form(self):
        """The path the resolver actually takes for a missing unit prefix.

        ``score_pair`` never sees this pair in practice: the canonical form
        excludes the unit prefix, so ``10-P-101-B`` and ``P-101B`` are already
        the same key by the time candidates are looked up. The middle-band
        behaviour above only applies when a *different* canonical form has to be
        compared.
        """
        assert parse("10-P-101-B").canonical == parse("P-101B").canonical == "P-101B"

    def test_a_lower_auto_merge_threshold_admits_the_missing_prefix_case(self):
        # The threshold is configuration (ER_AUTO_MERGE_THRESHOLD), so a plant
        # with a single unit can safely lower it.
        result = score_pair("P-101B", "10-P-101-B")
        assert decide(result, auto_merge_threshold=0.80, review_threshold=0.60) == "merge"

    def test_decide_is_deterministic(self):
        result = score_pair("P-101B", "P101B")
        actions = {
            decide(result, auto_merge_threshold=0.9, review_threshold=0.6) for _ in range(20)
        }
        assert actions == {"merge"}
