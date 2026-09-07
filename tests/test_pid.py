"""P&ID digitisation.

The temptation this module has to resist is different from the agents'. There,
the cheap wrong version was a fluent LLM narrative. Here it is a box with a
confident label on it: symbol detection *looks* like it works long before it
does, because a rectangle drawn on a drawing is convincing whatever produced it.

So the tests below pin down two things: that the detectors which need no
training data behave correctly, and that the one which does need it stays
declared-unimplemented rather than quietly starting to emit guesses.
"""

from __future__ import annotations

import math

import pytest

from services.ingest import pid


def word(text: str, x0: float, top: float, width: float = 20.0, height: float = 8.0):
    """A pdfplumber word, with the geometry the detector reads."""
    return {
        "text": text,
        "x0": x0,
        "x1": x0 + width,
        "top": top,
        "bottom": top + height,
    }


def detection(kind: str, x0: float, y0: float, x1: float, y1: float, **kwargs):
    return pid.Detection(
        kind=kind,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        method=kwargs.pop("method", "test"),
        confidence=kwargs.pop("confidence", 1.0),
        **kwargs,
    )


class TestTagLocalisation:
    def test_finds_a_tag_written_as_one_word(self) -> None:
        found = pid._detect_tags([word("P-101B", 100, 200)])
        assert [d.normalised for d in found] == ["P-101B"]
        assert found[0].x0 == 100
        assert found[0].confidence == 1.0

    def test_joins_words_a_pdf_split_on_kerning(self) -> None:
        """"P", "-", "101B" is one tag the extractor happened to emit as three.

        PDF text extraction splits on typographic spacing, not on meaning. A
        detector that only parses single words misses most tags on a drawing,
        because drawings letter-space their tags.
        """
        words = [word("P", 100, 200, width=8), word("-", 108, 200, width=4),
                 word("101B", 112, 200, width=22)]
        found = pid._detect_tags(words)
        assert "P-101B" in [d.normalised for d in found]

    def test_assembled_tags_are_marked_less_certain(self) -> None:
        """The grammar is deterministic; the *assembly* is the inference.

        A tag read from one word is certain. One glued from three words rests on
        a guess about kerning, and the score should say so.
        """
        single = pid._detect_tags([word("P-101B", 100, 200)])[0]
        joined = [
            d for d in pid._detect_tags([
                word("P", 100, 200, width=8),
                word("-", 108, 200, width=4),
                word("101B", 112, 200, width=22),
            ]) if d.normalised == "P-101B"
        ][0]
        assert joined.confidence < single.confidence

    def test_words_far_apart_are_not_joined(self) -> None:
        """Two tags side by side must not become one.

        "P-101A" at one end of a line and "P-101B" at the other are two pumps,
        and joining them would invent a tag that is on no drawing.
        """
        words = [word("P-101A", 100, 200), word("P-101B", 400, 200)]
        found = {d.normalised for d in pid._detect_tags(words)}
        assert found == {"P-101A", "P-101B"}

    def test_words_on_different_lines_are_not_joined(self) -> None:
        words = [word("P-101", 100, 200), word("B", 100, 260, width=8)]
        assert "P-101B" not in {d.normalised for d in pid._detect_tags(words)}

    def test_prose_is_not_read_as_tags(self) -> None:
        found = pid._detect_tags([
            word("PIPING", 10, 10), word("AND", 60, 10), word("INSTRUMENTATION", 90, 10)
        ])
        assert found == []


class TestBubbleNaming:
    def test_a_bubble_takes_the_tag_drawn_inside_it(self) -> None:
        """An unnamed circle is nearly useless.

        "There is an instrument here" is worth much less than "PIC-101 is here",
        and on a P&ID the tag is always inside the bubble — so containment is the
        whole association rule.
        """
        bubble = detection("instrument_bubble", 100, 100, 140, 140)
        tag = detection(
            "tag", 108, 112, 132, 124, normalised="PIC-101", text="PIC-101",
            properties={"tag_kind": "instrument"},
        )
        pid._name_bubbles([bubble], [tag])
        assert bubble.normalised == "PIC-101"
        assert bubble.properties["named_by"] == "PIC-101"

    def test_a_bubble_with_no_tag_inside_stays_unnamed(self) -> None:
        bubble = detection("instrument_bubble", 100, 100, 140, 140)
        outside = detection("tag", 400, 400, 430, 412, normalised="P-101B")
        pid._name_bubbles([bubble], [outside])
        assert bubble.normalised is None

    def test_instrument_tags_win_over_line_numbers(self) -> None:
        bubble = detection("instrument_bubble", 100, 100, 160, 160)
        line = detection(
            "tag", 105, 105, 150, 115, normalised='8"-P-1501-A1A',
            properties={"tag_kind": "line"},
        )
        instrument = detection(
            "tag", 110, 130, 150, 142, normalised="PIC-101",
            properties={"tag_kind": "instrument"},
        )
        pid._name_bubbles([bubble], [line, instrument])
        assert bubble.normalised == "PIC-101"


class TestLineMerging:
    def test_both_edges_of_one_stroke_become_one_run(self) -> None:
        """Canny reports two gradient edges per drawn line.

        Regression test: the raw detector returned 857 segments for one A3
        sheet, which is not 857 pipes.
        """
        edges = [
            detection("line_segment", 100, 200, 400, 200,
                      properties={"orientation": "horizontal", "length_px": 300}),
            detection("line_segment", 100, 202, 400, 202,
                      properties={"orientation": "horizontal", "length_px": 300}),
        ]
        assert len(pid._merge_duplicates(edges)) == 1

    def test_collinear_fragments_are_rejoined(self) -> None:
        fragments = [
            detection("line_segment", 100, 200, 200, 200,
                      properties={"orientation": "horizontal", "length_px": 100}),
            detection("line_segment", 203, 200, 300, 200,
                      properties={"orientation": "horizontal", "length_px": 97}),
        ]
        merged = pid._merge_duplicates(fragments)
        assert len(merged) == 1
        assert merged[0].x0 == 100 and merged[0].x1 == 300
        assert merged[0].properties["fragments_merged"] == 2

    def test_pipes_separated_by_a_symbol_stay_separate(self) -> None:
        """A gap wider than the join tolerance is a real gap.

        Bridging it would invent connectivity straight through whatever sits in
        the gap — which on a P&ID is usually a valve.
        """
        segments = [
            detection("line_segment", 100, 200, 200, 200,
                      properties={"orientation": "horizontal", "length_px": 100}),
            detection("line_segment", 260, 200, 360, 200,
                      properties={"orientation": "horizontal", "length_px": 100}),
        ]
        assert len(pid._merge_duplicates(segments)) == 2

    def test_parallel_pipes_are_not_merged(self) -> None:
        segments = [
            detection("line_segment", 100, 200, 400, 200,
                      properties={"orientation": "horizontal", "length_px": 300}),
            detection("line_segment", 100, 240, 400, 240,
                      properties={"orientation": "horizontal", "length_px": 300}),
        ]
        assert len(pid._merge_duplicates(segments)) == 2


class TestTopology:
    def test_connects_symbols_a_segment_touches_at_both_ends(self) -> None:
        pump = detection("tag", 100, 100, 140, 112, normalised="P-101B")
        exchanger = detection("tag", 300, 100, 340, 112, normalised="E-104")
        line = detection("line_segment", 140, 104, 300, 104,
                         properties={"orientation": "horizontal"})
        connections = pid._connect([pump, exchanger], [line], symbol_offset=0, line_offset=2)
        assert len(connections) == 1
        assert {connections[0].from_index, connections[0].to_index} == {0, 1}

    def test_a_segment_touching_one_symbol_connects_nothing(self) -> None:
        """A line running off the edge of the sheet is not a connection."""
        pump = detection("tag", 100, 100, 140, 112, normalised="P-101B")
        line = detection("line_segment", 140, 104, 600, 104,
                         properties={"orientation": "horizontal"})
        assert pid._connect([pump], [line], symbol_offset=0, line_offset=1) == []

    def test_tolerance_is_bounded(self) -> None:
        """A line that stops well short of a symbol does not reach it."""
        pump = detection("tag", 100, 100, 140, 112, normalised="P-101B")
        exchanger = detection("tag", 300, 100, 340, 112, normalised="E-104")
        short = detection("line_segment", 200, 104, 260, 104,
                          properties={"orientation": "horizontal"})
        assert pid._connect([pump, exchanger], [short], symbol_offset=0, line_offset=2) == []


class TestHonesty:
    def test_equipment_symbol_detection_is_declared_unimplemented(self) -> None:
        """The test that matters most in this file.

        Symbol classification needs a trained model and there is none. If this
        ever starts reporting `available`, either a model was added — in which
        case this test should be updated deliberately — or something began
        emitting guessed labels, which is the failure this whole project is
        built to avoid.
        """
        capability = pid.capability()
        assert capability["equipment_symbols"]["state"] == "not_implemented"
        assert "trained" in capability["equipment_symbols"]["detail"].lower()

    def test_no_accuracy_figure_is_claimed_anywhere(self) -> None:
        """No detector reports an accuracy, because none has been measured."""
        for name, entry in pid.capability().items():
            detail = (entry.get("detail") or "").lower()
            assert "accuracy" not in detail or "no accuracy" in detail, name
            assert "%" not in detail, name

    def test_confidence_is_documented_as_not_a_probability(self) -> None:
        assert "not" in pid.__doc__.lower() and "probabilit" in pid.__doc__.lower()

    @pytest.mark.parametrize(
        ("angle_deg", "expected"),
        [(0.0, True), (90.0, True), (2.0, True), (45.0, False), (30.0, False)],
    )
    def test_only_orthogonal_segments_count_as_pipe_runs(
        self, angle_deg: float, expected: bool
    ) -> None:
        """Diagonals on a P&ID are leaders and symbol internals, not pipes."""
        angle = angle_deg % 180.0
        orthogonal = min(angle, abs(angle - 90.0), abs(angle - 180.0))
        assert (orthogonal <= pid.ORTHOGONAL_TOLERANCE_DEG) is expected
        assert math.isfinite(orthogonal)
