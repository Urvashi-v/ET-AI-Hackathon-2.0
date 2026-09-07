"""The three intelligence agents.

The tests that matter most here are the ones pinning down *refusal*. Each agent
has a cheap, plausible, wrong version of itself — an RCA that ranks a cause from
one data point, a lessons panel that matches any two pump failures, a compliance
dashboard that counts unexamined requirements as passing — and each is one
well-meaning change away. Several tests below exist purely to make that change
fail loudly.
"""

from __future__ import annotations

from datetime import date

import pytest

from services.agents import rca as rca_agent
from services.agents.compliance import _RECORD_CONCEPTS, RequirementFinding, _add_months
from services.ingest import records


# ---------------------------------------------------------------------------
# Record extraction
# ---------------------------------------------------------------------------


def chunk(chunk_id: str, text: str, section: str | None = None, page: int | None = 1):
    return {"chunk_id": chunk_id, "text": text, "section_path": section, "page_from": page}


INCIDENT_CHUNKS = [
    chunk(
        "c0",
        "Incident Investigation Report INC-2022-19\n"
        "Site: HALDIA   Plant: CDU-1   System: Crude Charge Pumping\n"
        "Equipment involved: P-101A (crude charge pump, duty)\n"
        "Functional location: CDU1-PUMP-101\n"
        "Date of event: 2022-08-04\n"
        "Severity: Moderate -- 22 hours unplanned downtime, no injury\n"
        "Investigation status: CLOSED\n"
        "On 4 August the duty pump was restarted following a short unit trip.",
        "Incident Investigation Report INC-2022-19",
    ),
    chunk("c1", "Mechanical seal failure by face damage.", "INC-2022-19 > Immediate cause"),
    chunk(
        "c2",
        "Startup was performed with the suction valve throttled, causing the seal faces "
        "to run without liquid film. Same mechanism as INC-2019-07. SOP-4412 revision 3 "
        "still does not require confirmation of suction pressure.",
        "INC-2022-19 > Root cause",
    ),
    chunk(
        "c3",
        "| ID | Action | Owner | Due | Status |\n"
        "|---|---|---|---|---|\n"
        "| CAPA-87 | Replace seal and shaft sleeve | Rotating Equipment | 2022-08-08 | CLOSED |\n"
        "| CAPA-88 | Install suction pressure interlock | Reliability | 2023-03-31 | OPEN |",
        "INC-2022-19 > Corrective and preventive actions",
    ),
]


class TestIncidentExtraction:
    def test_extracts_the_fields_a_report_states(self) -> None:
        record = records.extract_incident(
            doc_id="doc_x", title="incident 2022", doc_number="INC-2022-19", chunks=INCIDENT_CHUNKS
        )
        assert record is not None
        assert record.incident_id == "INC-2022-19"
        assert record.occurred_on == date(2022, 8, 4)
        assert record.investigation_status == "CLOSED"
        assert record.functional_location == "CDU1-PUMP-101"
        assert "without liquid film" in str(record.root_cause)
        assert "INC-2019-07" in record.referenced_incidents
        assert "SOP-4412" in record.referenced_procedures

    def test_every_field_carries_the_chunk_that_asserted_it(self) -> None:
        """Provenance is the point. A root cause with no chunk cannot be opened."""
        record = records.extract_incident(
            doc_id="doc_x", title="t", doc_number="INC-2022-19", chunks=INCIDENT_CHUNKS
        )
        assert record is not None
        assert record.root_cause is not None and record.root_cause.chunk_id == "c2"
        assert record.immediate_cause is not None and record.immediate_cause.chunk_id == "c1"
        assert all(a.chunk_id == "c3" for a in record.corrective_actions)

    def test_corrective_action_status_is_parsed(self) -> None:
        record = records.extract_incident(
            doc_id="doc_x", title="t", doc_number="INC-2022-19", chunks=INCIDENT_CHUNKS
        )
        assert record is not None
        by_id = {a.action_id: a for a in record.corrective_actions}
        assert by_id["CAPA-87"].is_open is False
        assert by_id["CAPA-88"].is_open is True
        assert by_id["CAPA-88"].owner == "Reliability"

    def test_unknown_status_counts_as_open(self) -> None:
        """Pessimistic by design: a missed open action is the expensive error."""
        row = chunk("c9", "| CAPA-90 | Do the thing | Ops | 2024-01-01 | |", "x > Actions")
        actions = records._parse_actions(row)
        assert actions and actions[0].is_open is True

    def test_ocr_flattened_action_rows_are_recovered(self) -> None:
        """OCR of a printed table loses the rules; the action must survive it.

        Regression test. The scanned copy of INC-2019-07 comes back as running
        text, and a pipe-only parser silently produced zero corrective actions
        for a report that lists two.
        """
        row = chunk(
            "c9",
            "CORRECTIVE ACTIONS\n"
            "CAPA-41 Replace outboard mechanical seal Rotating Equipment 2019-04-05 CLOSED",
            "x > corrective_action",
        )
        actions = records._parse_actions(row)
        assert [a.action_id for a in actions] == ["CAPA-41"]
        assert actions[0].status == "CLOSED"
        assert actions[0].due_on == date(2019, 4, 5)

    def test_document_without_structure_yields_no_record(self) -> None:
        """A guessed incident is worse than a missing one.

        A missing incident is a visible gap. A wrong one becomes a node, gets
        cited by RCA, and is believed.
        """
        assert (
            records.extract_incident(
                doc_id="d",
                title="Some memo",
                doc_number=None,
                chunks=[chunk("c0", "A memo about nothing in particular.", "Memo")],
            )
            is None
        )

    def test_document_numbers_are_not_read_as_asset_tags(self) -> None:
        """MOC-2023-07 and WO-3502 are references, not equipment.

        Regression test: a letters-dash-digits regex put both into the asset
        list, which would have created equipment nodes for document numbers.
        """
        assert records._tags_in("MOC-2023-07 changes P-101B and closes WO-3502") == ["P-101B"]


# ---------------------------------------------------------------------------
# RCA
# ---------------------------------------------------------------------------


def incident(id_: str, root: str, when: date, tag: str = "P-101B"):
    return {"incident_id": id_, "root_cause": root, "occurred_on": when, "raw_asset_tag": tag}


def work_order(id_: str, found: str, when: date, tag: str = "P-101B"):
    return {"wo_id": id_, "as_found": found, "opened_on": when, "raw_asset_tag": tag}


class TestRCARanking:
    def test_ranks_the_mechanism_with_the_most_independent_evidence(self) -> None:
        analysis = rca_agent.analyse(
            asset_tag="P-101B",
            incidents=[incident("INC-1", "ran without liquid film after startup", date(2022, 1, 1))],
            work_orders=[
                work_order("WO-1", "seal faces scored, evidence of dry running", date(2023, 1, 1)),
                work_order("WO-2", "bearing wear noted", date(2021, 1, 1)),
            ],
            sibling_events=[],
            today=date(2024, 1, 1),
        )
        assert not analysis.abstained
        assert analysis.leading is not None
        assert analysis.leading.key == "dry_running"
        assert analysis.leading.occurrences == 2

    def test_abstains_below_the_evidence_floor(self) -> None:
        """One recorded failure is not a cause.

        The most important test in this file. A single data point presented as
        "leading candidate" is the LLM failure mode reproduced with arithmetic.
        """
        analysis = rca_agent.analyse(
            asset_tag="P-101B",
            incidents=[incident("INC-1", "ran dry", date(2023, 1, 1))],
            work_orders=[],
            sibling_events=[],
            today=date(2024, 1, 1),
        )
        assert analysis.abstained
        assert analysis.candidates == []
        assert analysis.abstain_reason and "single data point" in analysis.abstain_reason

    def test_abstains_when_no_mechanism_is_recognised(self) -> None:
        analysis = rca_agent.analyse(
            asset_tag="P-101B",
            incidents=[incident("INC-1", "something went wrong", date(2023, 1, 1))],
            work_orders=[work_order("WO-1", "it broke again", date(2023, 6, 1))],
            sibling_events=[],
            today=date(2024, 1, 1),
        )
        assert analysis.abstained
        assert "none names a failure mechanism" in (analysis.abstain_reason or "")

    def test_sibling_evidence_counts_but_counts_less(self) -> None:
        own = rca_agent.analyse(
            asset_tag="P-101B",
            incidents=[incident("INC-1", "dry running", date(2023, 1, 1))],
            work_orders=[work_order("WO-1", "ran dry", date(2023, 2, 1))],
            sibling_events=[],
            today=date(2024, 1, 1),
        )
        sibling = rca_agent.analyse(
            asset_tag="P-101B",
            incidents=[incident("INC-1", "dry running", date(2023, 1, 1))],
            work_orders=[],
            sibling_events=[
                {
                    "id": "WO-9",
                    "kind": "work_order",
                    "as_found": "ran dry",
                    "date": date(2023, 2, 1),
                    "asset_tag": "P-101A",
                }
            ],
            today=date(2024, 1, 1),
        )
        assert own.leading is not None and sibling.leading is not None
        assert own.leading.score > sibling.leading.score
        assert sibling.leading.on_this_asset == 1

    def test_recent_evidence_outweighs_old_evidence(self) -> None:
        recent = rca_agent._recency_factor(date(2023, 6, 1), date(2024, 1, 1))
        old = rca_agent._recency_factor(date(2004, 1, 1), date(2024, 1, 1))
        assert recent > old
        assert 0.0 < old < 0.1

    def test_every_candidate_names_the_documents_behind_it(self) -> None:
        """A ranking nobody can check is a ranking nobody should trust."""
        analysis = rca_agent.analyse(
            asset_tag="P-101B",
            incidents=[incident("INC-1", "ran without liquid film", date(2022, 1, 1))],
            work_orders=[work_order("WO-1", "evidence of dry running", date(2023, 1, 1))],
            sibling_events=[],
            today=date(2024, 1, 1),
        )
        assert analysis.leading is not None
        assert {e.ref_id for e in analysis.leading.evidence} == {"INC-1", "WO-1"}
        assert "INC-1" in analysis.leading.rationale

    def test_as_left_is_not_treated_as_cause_evidence(self) -> None:
        """What was done about it is a remedy, not a mechanism."""
        analysis = rca_agent.analyse(
            asset_tag="P-101B",
            incidents=[],
            work_orders=[
                {
                    "wo_id": "WO-1",
                    "as_found": "",
                    "as_left": "seal ran dry, replaced with new cartridge",
                    "opened_on": date(2023, 1, 1),
                }
            ],
            sibling_events=[],
            today=date(2024, 1, 1),
        )
        assert analysis.evidence_count == 0


# ---------------------------------------------------------------------------
# Compliance
# ---------------------------------------------------------------------------


class TestComplianceEvaluation:
    def test_only_decidable_findings_count_towards_coverage(self) -> None:
        """The distinction the whole dashboard rests on.

        A requirement needing a permit system that is not connected is neither
        satisfied nor breached. Counting it either way produces a percentage
        that describes nothing.
        """
        states = ["satisfied", "gap", "needs_verification", "not_evaluable"]
        findings = [
            RequirementFinding(
                req_id=f"R-{i}",
                source_standard="S",
                clause=None,
                obligation_text="x",
                modality="shall",
                testable_by="evidence_document",
                text_status="paraphrase_for_demo",
                state=state,
                reason="",
            )
            for i, state in enumerate(states)
        ]
        decidable = [f for f in findings if f.decidable]
        assert {f.state for f in decidable} == {"satisfied", "gap"}

    def test_obligations_with_no_matching_record_type_are_unexamined(self) -> None:
        """A pump without a thickness survey has not breached a guarding rule.

        Regression test: absent inspection records were reported as gaps against
        every requirement, including "every dangerous part of machinery shall be
        securely fenced". A compliance report that cries breach on category
        errors is one nobody reads.
        """
        assert not _RECORD_CONCEPTS.search(
            "Every dangerous part of machinery shall be securely fenced"
        )
        assert _RECORD_CONCEPTS.search(
            "Pressure vessels shall undergo external visual inspection every 12 months"
        )

    @pytest.mark.parametrize(
        ("start", "months", "expected"),
        [
            (date(2025, 5, 27), 12, date(2026, 5, 27)),
            (date(2025, 1, 31), 1, date(2025, 2, 28)),
            (date(2024, 1, 31), 1, date(2024, 2, 29)),
            (date(2025, 5, 27), 48, date(2029, 5, 27)),
        ],
    )
    def test_interval_arithmetic_clamps_to_short_months(
        self, start: date, months: int, expected: date
    ) -> None:
        assert _add_months(start, months) == expected


# ---------------------------------------------------------------------------
# Lessons learned
# ---------------------------------------------------------------------------


class TestLessonsSignals:
    def test_structural_signal_ranks_same_asset_above_sibling_above_class(self) -> None:
        from services.agents.lessons import _structural

        context = {
            "tag": "P-101B",
            "siblings": {"P-101A"},
            "class_code": "P",
            "functional_location": "CDU1-PUMP-101",
        }
        same, _ = _structural({"asset_tag": "P-101B"}, context)
        sibling, _ = _structural({"asset_tag": "P-101A"}, context)
        same_class, _ = _structural({"asset_tag": "P-205", "class_code": "P"}, context)
        unrelated, _ = _structural({"asset_tag": "V-102", "class_code": "V"}, context)
        assert same > sibling > same_class > unrelated == 0.0

    def test_signal_weights_sum_to_one(self) -> None:
        from services.agents.lessons import WEIGHTS

        assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

    def test_open_actions_are_listed_before_closed_ones(self) -> None:
        """The usual finding is "we decided and never did it"; bury it and the
        panel loses its point."""
        from services.agents.lessons import _actions_of

        ordered = _actions_of(
            {
                "corrective_actions": [
                    {"action_id": "CAPA-1", "is_open": False},
                    {"action_id": "CAPA-2", "is_open": True},
                ]
            }
        )
        assert [a["action_id"] for a in ordered] == ["CAPA-2", "CAPA-1"]


# ---------------------------------------------------------------------------
# Proactive
# ---------------------------------------------------------------------------


class TestProactiveThresholds:
    def test_notification_threshold_is_stricter_than_browsing(self) -> None:
        """Interrupting someone needs a higher bar than answering a question.

        Browsing weak matches is fine when a person went looking. Pushing them is
        how alerting gets muted.
        """
        from services.agents.lessons import MIN_SIMILARITY
        from services.agents.proactive import MIN_PRECEDENT_SIMILARITY

        assert MIN_PRECEDENT_SIMILARITY > MIN_SIMILARITY

    def test_every_candidate_kind_has_an_audience(self) -> None:
        from services.agents.proactive import _AUDIENCE

        assert set(_AUDIENCE) == {
            "recurrence",
            "open_action",
            "compliance",
            "procedure_superseded",
        }
        assert all(_AUDIENCE.values())
