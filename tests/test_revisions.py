"""Document numbering and revision lineage.

The case that matters most here is the one where the system declines to answer.
Ordering revisions wrongly is worse than not ordering them: an engineer told
"this is the current procedure" acts on it, whereas one told "these two cannot be
separated" goes and checks. Several of these tests exist specifically to pin down
that refusal so a later "improvement" cannot quietly turn it into a guess.
"""

from __future__ import annotations

from datetime import date

import pytest

from services.common.schemas import DocumentType
from services.ingest.pipeline import _derive_revision
from services.ingest.revisions import derive_doc_number, order_revisions, revision_rank


def doc(doc_id: str, **fields: object) -> dict[str, object]:
    """A document row with only the fields the ordering logic reads."""
    base: dict[str, object] = {
        "doc_id": doc_id,
        "title": doc_id,
        "revision": None,
        "revised_on": None,
        "issued_on": None,
    }
    base.update(fields)
    return base


class TestDeriveDocNumber:
    @pytest.mark.parametrize(
        ("title", "expected"),
        [
            ("SOP-4412 Crude Charge Pump Startup", "SOP-4412"),
            ("sop 4412 crude charge pump startup", "SOP-4412"),
            ("SOP4412 startup", "SOP-4412"),
            ("INC-2019-07 Incident Investigation Report", "INC-2019-07"),
            ("UT-2025-004 Ultrasonic Thickness Survey", "UT-2025-004"),
            ("MOC-2023-07 Impeller Trim", "MOC-2023-07"),
        ],
    )
    def test_canonicalises_separator_variants(self, title: str, expected: str) -> None:
        """The same series must produce the same key however it was typed.

        This is the whole mechanism: two revisions only group if their numbers
        canonicalise identically, so separator handling *is* the feature.
        """
        found = derive_doc_number(title=title, filename="", head_text="", doc_type=DocumentType.SOP)
        assert found is not None
        assert found.value == expected

    def test_asset_tags_are_not_document_numbers(self) -> None:
        """P-101B must never be read as a document number.

        A generic letters-dash-digits pattern would match it, and the result
        would be a revision series containing every document that mentions the
        pump -- each one "superseding" the others.
        """
        assert (
            derive_doc_number(
                title="P-101B Centrifugal Pump Datasheet",
                filename="p-101b.pdf",
                head_text="P-101B rated head 145 m",
                doc_type=DocumentType.DATASHEET,
            )
            is None
        )

    def test_title_wins_over_body(self) -> None:
        """A document that cites another is not that other document."""
        found = derive_doc_number(
            title="SOP-4412 Crude Charge Pump Startup",
            filename="startup.md",
            head_text="This procedure supersedes SOP-9999 and refers to MOC-2023-07.",
            doc_type=DocumentType.SOP,
        )
        assert found is not None
        assert found.value == "SOP-4412"
        assert found.method.startswith("title:")

    def test_collection_does_not_take_a_row_identifier(self) -> None:
        """A CMMS export is not the first work order it contains.

        Regression test. The body search adopted "WO-2101" -- row one of a
        hundreds-row table -- as the export's own identity. Next month's export
        would have taken a different row, and the two would never have grouped.
        """
        assert (
            derive_doc_number(
                title="work orders cmms export",
                filename="work_orders_cmms_export.csv",
                head_text="WO-2101 | P-101B | seal replacement | 2022-11-04",
                doc_type=DocumentType.WORK_ORDER,
            )
            is None
        )

    def test_absent_number_is_a_valid_answer(self) -> None:
        assert (
            derive_doc_number(
                title="Ultrasonic thickness readings",
                filename="readings.csv",
                head_text="cml_id,thickness_mm",
                doc_type=DocumentType.INSPECTION_REPORT,
            )
            is None
        )


class TestRevisionRank:
    def test_numeric_revisions_sort_numerically(self) -> None:
        """Rev 10 comes after Rev 9. String comparison gets this backwards."""
        assert revision_rank("10") > revision_rank("9")  # type: ignore[operator]

    def test_alphabetic_revisions_sort_alphabetically(self) -> None:
        assert revision_rank("B") > revision_rank("A")  # type: ignore[operator]

    def test_numeric_and_alphabetic_are_different_schemes(self) -> None:
        """A series uses one scheme or the other; comparing across is meaningless."""
        numeric = revision_rank("2")
        alphabetic = revision_rank("B")
        assert numeric is not None and alphabetic is not None
        assert numeric[0] != alphabetic[0]

    @pytest.mark.parametrize("label", [None, "", "DRAFT", "2019-07", "REV-A-2"])
    def test_unorderable_labels_return_none(self, label: str | None) -> None:
        """A label that carries no ordering must say so rather than sort somewhere."""
        assert revision_rank(label) is None


class TestOrderRevisions:
    def test_orders_by_revision_label(self) -> None:
        ordering = order_revisions(
            [doc("b", revision="4"), doc("a", revision="3")],
        )
        assert not ordering.conflict
        assert ordering.basis == "revision_label"
        assert [d["doc_id"] for level in ordering.levels for d in level] == ["a", "b"]
        assert [d["doc_id"] for d in ordering.current] == ["b"]

    def test_same_revision_is_one_level_not_a_sequence(self) -> None:
        """Two formats of Rev 3 are renditions; neither supersedes the other.

        The Markdown source and the scanned signed copy of one procedure are the
        same revision. Ordering them would assert a supersession that does not
        exist and mark a perfectly current document superseded.
        """
        ordering = order_revisions([doc("md", revision="3"), doc("pdf", revision="3")])
        assert not ordering.conflict
        assert len(ordering.levels) == 1
        assert len(ordering.current) == 2
        assert ordering.superseded == []
        assert ordering.has_renditions

    def test_renditions_and_supersession_together(self) -> None:
        """Rev 3 in two formats, superseded by Rev 4. The real SOP-4412 case.

        Both Rev 3 documents must be superseded -- superseding only one leaves
        the other looking current to anything that reaches it directly.
        """
        ordering = order_revisions(
            [
                doc("md3", revision="3"),
                doc("pdf3", revision="3"),
                doc("md4", revision="4"),
            ]
        )
        assert not ordering.conflict
        assert len(ordering.levels) == 2
        assert {d["doc_id"] for d in ordering.superseded} == {"md3", "pdf3"}
        assert [d["doc_id"] for d in ordering.current] == ["md4"]

    def test_falls_back_to_effective_dates(self) -> None:
        ordering = order_revisions(
            [
                doc("new", revised_on=date(2025, 2, 1)),
                doc("old", revised_on=date(2024, 3, 1)),
            ]
        )
        assert not ordering.conflict
        assert ordering.basis == "revised_on"
        assert [d["doc_id"] for d in ordering.current] == ["new"]

    def test_refuses_to_order_without_evidence(self) -> None:
        """No labels, no dates: a conflict, not a guess.

        The tempting fallback is ingestion order. It is available, total, and
        completely wrong -- it records who uploaded what first. This test exists
        to make adding it a visible decision rather than a quiet convenience.
        """
        ordering = order_revisions([doc("a"), doc("b")])
        assert ordering.conflict
        assert ordering.basis == "none"
        assert ordering.note and "rather than guessing" in ordering.note
        # Everything stays current: nothing is marked superseded on a guess.
        assert len(ordering.current) == 2

    def test_mixed_revision_schemes_do_not_order_by_label(self) -> None:
        """ "A" and "2" cannot be compared, so the label basis must not be used."""
        ordering = order_revisions([doc("a", revision="A"), doc("b", revision="2")])
        assert ordering.basis != "revision_label"

    def test_single_document_is_current(self) -> None:
        ordering = order_revisions([doc("only", revision="1")])
        assert not ordering.conflict
        assert [d["doc_id"] for d in ordering.current] == ["only"]
        assert ordering.superseded == []

    def test_ordering_is_independent_of_input_order(self) -> None:
        """A revision can arrive before the one it replaces.

        Rev 4 is often scanned and uploaded before someone finds Rev 3 in a
        filing cabinet. The result must not depend on which arrived first.
        """
        forward = order_revisions([doc("a", revision="3"), doc("b", revision="4")])
        backward = order_revisions([doc("b", revision="4"), doc("a", revision="3")])
        assert [d["doc_id"] for d in forward.current] == [d["doc_id"] for d in backward.current]
        assert [d["doc_id"] for d in forward.superseded] == [
            d["doc_id"] for d in backward.superseded
        ]


class TestRevisionFieldExtraction:
    """A document's revision is the one printed on it, not one it mentions.

    The bug this pins down shipped for four days and was invisible until
    citations started carrying revision standing: `_derive_revision` searched the
    whole header for the first occurrence of the word "revision" and took the
    number after it. Every incident report that referred to the SOP it was about
    inherited that SOP's revision number.

    That is not cosmetic. `order_revisions` treats the revision label as ordering
    evidence, so a fabricated label can declare a real document superseded.
    """

    @pytest.mark.parametrize(
        ("name", "head_text", "expected"),
        [
            (
                "own header field",
                "Document: SOP-4412\nRevision: 4\nEffective from: 2025-02-01\n",
                "4",
            ),
            (
                # pdfplumber collapses a header block onto a single line. The
                # first version of this fix anchored to the start of a line and
                # silently lost the revision of the one PDF SOP in the corpus;
                # ordering fell back to effective dates and still reached the
                # right answer, which is how a regression like that survives a
                # test suite.
                "a PDF header collapsed onto one line",
                "SOP-4412 Crude Charge Pump Startup\n"
                "Document: SOP-4412 Revision: 3 Effective from: 2024-03-01\n"
                "Supersedes: SOP-4412 revision 2, effective 2018-06-01\n",
                "3",
            ),
            ("a two-part label", "Revision: 2A\n", "2A"),
            (
                # The colon alone is not enough: the value has to look like a
                # revision label, or "Revision: this document was superseded"
                # parses as revision "this".
                "a sentence after the colon",
                "Revision: this document was superseded in 2024\n",
                None,
            ),
            (
                "a supersedes line names another revision",
                "Document: SOP-4412\nSupersedes: SOP-4412 revision 3, effective 2024-03-01\n",
                None,
            ),
            (
                "prose reference wrapped onto its own line",
                "The startup procedure had been revised to SOP-4412\n"
                "revision 3 but still does not require confirmation of suction pressure\n",
                None,
            ),
            (
                "a table cell mentioning another document's revision",
                "| SOP-4412 | Review discharge pressure limit | DONE in revision 3 |\n",
                None,
            ),
            ("its own revision in a table", "| Revision | 4 |\n", "4"),
            ("abbreviated with a full stop", "Rev. 3\n", "3"),
            ("alphabetic scheme", "Revision: B\n", "B"),
            ("issue rather than revision", "Issue 2\n", "2"),
            (
                "an incident report has no revision",
                "Site: HALDIA   Plant: CDU-1\nDate of event: 2022-08-04\nSeverity: Moderate\n",
                None,
            ),
        ],
    )
    def test_reads_only_its_own_revision_field(
        self, name: str, head_text: str, expected: str | None
    ) -> None:
        assert _derive_revision(head_text) == expected, name

    def test_a_reference_far_into_the_body_is_ignored(self) -> None:
        # Even correctly formatted, a revision field two thousand characters in
        # belongs to something quoted, not to this document.
        head = "Site: HALDIA\n" + ("filler line\n" * 200) + "Revision: 9\n"
        assert _derive_revision(head) is None

    def test_no_revision_is_a_valid_answer(self) -> None:
        # The counterpart to the ordering tests above: absent evidence must stay
        # absent rather than become a guess.
        assert _derive_revision("A work order with no version information.\n") is None
