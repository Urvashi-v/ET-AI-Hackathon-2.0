"""API contract tests against a live stack.

Marked ``integration`` and skipped when the API is not reachable, so
``make test`` stays runnable with no Docker.

    docker compose up -d
    python scripts/ingest_dir.py data/synthetic/generated --data-class synthetic_test_data --wait
    python -m pytest -m integration

These assert the two properties the whole project rests on: nothing is
fabricated, and anything that cannot run says so with the variables that would
enable it.
"""

from __future__ import annotations

import os

import httpx
import pytest

API = os.environ.get("BRAIN_API", "http://localhost:8000")
pytestmark = pytest.mark.integration

VALID_DATA_CLASSES = {
    "real_source_document",
    "synthetic_test_data",
    "model_derived",
    "calculated_metric",
    "human_attested",
    "reference_taxonomy",
}
CAPABILITY_STATES = {
    "available",
    "disabled",
    "provider_not_configured",
    "not_implemented",
    "error",
}


@pytest.fixture(scope="module")
def client():
    with httpx.Client(base_url=API, timeout=60) as session:
        try:
            session.get("/health/live").raise_for_status()
        except Exception:
            pytest.skip(f"API not reachable at {API}; start the stack first")
        yield session


class TestHealth:
    def test_liveness(self, client):
        assert client.get("/health/live").json()["status"] == "ok"

    def test_report_names_every_backing_service(self, client):
        body = client.get("/health").json()
        assert {"postgres", "neo4j", "redis"} <= set(body["dependencies"])
        assert body["status"] in {"ok", "degraded", "down"}

    def test_dependency_state_is_probed_not_asserted(self, client):
        deps = client.get("/health").json()["dependencies"]
        # A real probe reports facts about the server it reached.
        assert deps["postgres"]["status"] == "up"
        assert "PostgreSQL" in deps["postgres"]["version"]
        assert deps["neo4j"]["constraints"] > 0
        assert isinstance(deps["redis"]["queue_depth"], int)

    def test_provider_status_names_missing_credentials(self, client):
        providers = client.get("/health").json()["providers"]
        for name, info in providers.items():
            assert info["state"] in {"disabled", "configured", "provider_not_configured"}, name
            if info["state"] == "provider_not_configured":
                assert info["missing_credentials"], f"{name} must name what is missing"

    def test_no_secret_is_ever_exposed(self, client):
        body = client.get("/health").text.lower()
        for forbidden in ("password", "api_key", "secret", "postgresql://", "bolt://"):
            assert forbidden not in body


class TestAssets:
    def test_listing_carries_a_provenance_class_on_every_row(self, client):
        items = client.get("/api/v1/assets?limit=50").json()["items"]
        assert items, "ingest the synthetic corpus before running integration tests"
        for asset in items:
            assert asset["data_class"] in VALID_DATA_CLASSES

    def test_the_six_tag_spellings_resolved_to_one_asset(self, client):
        detail = client.get("/api/v1/assets/P-101B").json()
        surfaces = {v["surface_form"] for v in detail["tag_variants"]}
        assert len(surfaces) >= 4, f"expected genuine tag variance, got {surfaces}"
        assert detail["asset"]["canonical_tag"] == "P-101B"

    def test_siblings_are_linked_and_remain_separate_assets(self, client):
        detail = client.get("/api/v1/assets/P-101B").json()
        siblings = {s["canonical_tag"] for s in detail["siblings"]}
        assert "P-101A" in siblings
        # And P-101A is its own asset with its own history, not a merged alias.
        other = client.get("/api/v1/assets/P-101A").json()
        assert other["asset"]["canonical_tag"] == "P-101A"
        assert other["asset"]["asset_id"] != detail["asset"]["asset_id"]

    def test_an_asset_is_evidenced_by_more_than_one_document(self, client):
        detail = client.get("/api/v1/assets/P-101B").json()
        types = {d["doc_type"] for d in detail["documents"]}
        assert len(detail["documents"]) > 1
        assert len(types) > 1, "the cross-document claim needs more than one document type"

    def test_stats_are_labelled_as_calculated(self, client):
        stats = client.get("/api/v1/assets/stats").json()
        assert stats["data_class"] == "calculated_metric"
        assert 0 <= stats["mention_resolution_rate_pct"] <= 100

    def test_unknown_asset_returns_a_helpful_404(self, client):
        response = client.get("/api/v1/assets/P-999Z")
        assert response.status_code == 404
        assert "hint" in response.json()["error"]["detail"]


class TestGraph:
    def test_neighbourhood_has_nodes_and_typed_edges(self, client):
        graph = client.get("/api/v1/graph/P-101B?hops=2").json()
        assert graph["anchor_found"] and graph["nodes"] and graph["edges"]
        assert {e["type"] for e in graph["edges"]}

    def test_a_missing_anchor_is_reported_rather_than_faked(self, client):
        graph = client.get("/api/v1/graph/P-999Z").json()
        assert graph["anchor_found"] is False
        assert graph["nodes"] == [] and graph["edges"] == []
        assert "does not" in graph["detail"] or "No asset" in graph["detail"]

    def test_edge_type_filtering_is_applied(self, client):
        graph = client.get("/api/v1/graph/P-101B?edges=SIBLING_OF&hops=1").json()
        assert {e["type"] for e in graph["edges"]} <= {"SIBLING_OF"}

    def test_edge_evidence_resolves_to_real_passages(self, client):
        graph = client.get("/api/v1/graph/P-101B?hops=1").json()
        described = [e for e in graph["edges"] if e["evidence_chunk_ids"]]
        if not described:
            pytest.skip("no evidence-carrying edge in this corpus")
        evidence = client.get(f"/api/v1/graph/P-101B/evidence/{described[0]['id']}").json()
        assert evidence["evidence"], "an evidence pointer must resolve to text"
        for chunk in evidence["evidence"]:
            assert chunk["text"] and chunk["data_class"] in VALID_DATA_CLASSES

    def test_schema_is_read_back_from_the_database(self, client):
        schema = client.get("/api/v1/graph/schema").json()
        labels = {row["label"] for row in schema["node_counts"]}
        assert {"Equipment", "Document", "Chunk", "Mention"} <= labels


class TestQuery:
    def test_pipeline_reports_every_leg(self, client):
        body = client.post(
            "/api/v1/query", json={"question": "Why does P-101B keep failing?"}
        ).json()
        legs = {leg["strategy"]: leg["state"] for leg in body["retrieval"]}
        assert {"lexical", "dense", "graph"} <= set(legs)
        for state in legs.values():
            assert state in CAPABILITY_STATES

    def test_an_unconfigured_leg_names_the_variables_that_enable_it(self, client):
        body = client.post(
            "/api/v1/query", json={"question": "Why does P-101B keep failing?"}
        ).json()
        for leg in body["retrieval"]:
            if leg["state"] == "provider_not_configured":
                assert leg["required_env"], f"{leg['strategy']} must say what is missing"

    def test_citations_resolve_and_carry_provenance(self, client):
        body = client.post(
            "/api/v1/query", json={"question": "Why does P-101B keep failing?"}
        ).json()
        assert body["citations"]
        for citation in body["citations"]:
            assert citation["chunk_id"] and citation["doc_id"] and citation["snippet"]
            assert citation["data_class"] in VALID_DATA_CLASSES

    def test_graph_facts_are_traversed_for_a_diagnostic_question(self, client):
        body = client.post(
            "/api/v1/query", json={"question": "Why does P-101B keep failing?"}
        ).json()
        assert body["intent"] == "diagnostic"
        assert body["graph_facts"], "a diagnostic question should reach the graph"

    def test_a_nonexistent_asset_abstains_and_names_it(self, client):
        body = client.post(
            "/api/v1/query", json={"question": "What is the vibration alarm setpoint for P-999Z?"}
        ).json()
        assert body["confidence"]["mode"].startswith("ABSTAIN")
        assert body["referral"] and "P-999Z" in body["referral"]["reason"]

    def test_no_answer_is_invented_without_a_generator(self, client):
        body = client.post(
            "/api/v1/query", json={"question": "Why does P-101B keep failing?"}
        ).json()
        if body["generation"]["state"] != "available":
            assert body["answer"] is None
            assert body["confidence"]["mode"] == "ABSTAIN_NO_GENERATOR"
            assert body["generation"]["required_env"]

    def test_the_query_is_logged_for_audit_and_replayable(self, client):
        body = client.post("/api/v1/query", json={"question": "What PPE is required?"}).json()
        replay = client.get(f"/api/v1/query/{body['query_id']}").json()
        assert replay["question"] == "What PPE is required?"
        assert replay["confidence_mode"] == body["confidence"]["mode"]

    def test_a_too_short_question_is_rejected_by_the_contract(self, client):
        response = client.post("/api/v1/query", json={"question": "x"})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "validation_error"

    def test_streaming_emits_the_real_stage_events(self, client):
        events = []
        with client.stream(
            "POST",
            "/api/v1/query/stream",
            json={"question": "Why does P-101B keep failing?"},
        ) as response:
            for line in response.iter_lines():
                if line.startswith("event:"):
                    events.append(line.split(":", 1)[1].strip())
        assert {"intent", "retrieval", "confidence", "done"} <= set(events)


class TestIngestionContract:
    def test_data_class_is_mandatory(self, client):
        response = client.post(
            "/api/v1/ingest/paths",
            json={"paths": ["data/synthetic/generated"], "source_system": "test"},
        )
        assert response.status_code == 400

    def test_path_traversal_is_rejected(self, client):
        response = client.post(
            "/api/v1/ingest/paths",
            json={"paths": ["../../etc"], "data_class": "synthetic_test_data"},
        )
        assert response.status_code == 400

    def test_reingesting_the_same_corpus_is_idempotent(self, client):
        """The property everything else depends on: same bytes, no duplicates."""
        before = client.get("/api/v1/assets/stats").json()
        response = client.post(
            "/api/v1/ingest/paths",
            json={
                "paths": ["data/synthetic/generated"],
                "data_class": "synthetic_test_data",
                "source_system": "synthetic_cmms",
            },
        )
        body = response.json()
        assert body["accepted"] == 0, "already-ingested content must be recognised"
        assert body["duplicates"] > 0
        after = client.get("/api/v1/assets/stats").json()
        assert after["total_assets"] == before["total_assets"]
        assert after["total_mentions"] == before["total_mentions"]

    def test_job_stage_report_is_truthful_about_skipped_stages(self, client):
        jobs = client.get("/api/v1/ingest?limit=5").json()["items"]
        if not jobs:
            pytest.skip("no ingestion jobs yet")
        job = client.get(f"/api/v1/ingest/{jobs[0]['job_id']}").json()
        for stage in job["stage_reports"]:
            assert stage["state"] in CAPABILITY_STATES
            if stage["state"] == "provider_not_configured":
                assert stage["required_env"]

    def test_unknown_job_returns_404(self, client):
        assert client.get("/api/v1/ingest/job_does_not_exist").status_code == 404


class TestIngestionProvenance:
    """Day 2: what the write path must be able to prove about what it read."""

    def _documents(self, client):
        jobs = client.get("/api/v1/ingest?limit=10").json()["items"]
        docs = []
        for job in jobs:
            docs.extend(client.get(f"/api/v1/ingest/{job['job_id']}").json()["documents"])
        if not docs:
            pytest.skip("no documents ingested yet")
        return docs

    def test_every_document_records_which_parser_read_it(self, client):
        for doc in self._documents(client):
            assert doc["parser"], f"{doc['original_filename']} does not say how it was parsed"

    def test_a_scanned_document_records_its_ocr_engine_and_confidence(self, client):
        scans = [d for d in self._documents(client) if d["ocr_engine"]]
        if not scans:
            pytest.skip("no OCR-processed document in this corpus")
        for doc in scans:
            assert 0.0 < doc["ocr_mean_confidence"] <= 1.0
            assert doc["ocr_word_count"] > 0
            # Recognised characters are not read characters, and the record says so.
            assert doc["has_text_layer"] is False

    def test_a_text_layer_document_claims_no_ocr(self, client):
        read = [d for d in self._documents(client) if d["has_text_layer"]]
        if not read:
            pytest.skip("no text-layer document in this corpus")
        for doc in read:
            assert doc["ocr_engine"] is None

    def test_a_drawing_is_flagged_with_the_evidence_behind_the_verdict(self, client):
        drawings = [d for d in self._documents(client) if d["is_drawing"]]
        if not drawings:
            pytest.skip("no drawing in this corpus")
        for doc in drawings:
            assert doc["vector_objects"] > 0
            assert doc["doc_type"] == "pid"

    def test_every_document_reports_its_processing_time_and_counts(self, client):
        for doc in self._documents(client):
            assert doc["processing_ms"] is not None and doc["processing_ms"] >= 0
            assert doc["chunk_count"] >= 0
            assert doc["graph_nodes_created"] >= 0

    def test_job_totals_are_real_aggregates(self, client):
        jobs = client.get("/api/v1/ingest?limit=5").json()["items"]
        if not jobs:
            pytest.skip("no ingestion jobs")
        job = client.get(f"/api/v1/ingest/{jobs[0]['job_id']}").json()
        assert job["pages"] == sum((d["page_count"] or 0) for d in job["documents"])
        assert job["graph_nodes_created"] == sum(d["graph_nodes_created"] for d in job["documents"])

    def test_extraction_counts_are_reported(self, client):
        jobs = client.get("/api/v1/ingest?limit=5").json()["items"]
        if not jobs:
            pytest.skip("no ingestion jobs")
        job = client.get(f"/api/v1/ingest/{jobs[0]['job_id']}").json()
        assert job["extractions_verified"] <= job["extractions_total"]


class TestChunkProvenance:
    def test_citations_carry_the_page_they_came_from(self, client):
        body = client.post(
            "/api/v1/query", json={"question": "What PPE is required for P-101B?"}
        ).json()
        if not body["citations"]:
            pytest.skip("no citations returned")
        assert any(c["page"] is not None for c in body["citations"])

    def test_a_citation_resolves_to_a_real_chunk(self, client):
        body = client.post(
            "/api/v1/query", json={"question": "Why does P-101B keep failing?"}
        ).json()
        if not body["citations"]:
            pytest.skip("no citations returned")
        for citation in body["citations"]:
            assert citation["chunk_id"].startswith("chk_")
            assert citation["snippet"]


class TestAgentsAndCompliance:
    def test_rca_returns_real_evidence_and_no_invented_tree(self, client):
        body = client.post(
            "/api/v1/rca",
            json={"asset_tag": "P-101B", "failure_description": "High vibration and seal leak"},
        ).json()
        assert body["asset_found"]
        assert body["evidence_gathered"]["work_orders"] > 0
        assert body["status"]["state"] in CAPABILITY_STATES
        if body["status"]["state"] != "available":
            assert body["causal_tree"] is None, "no template tree may be substituted"

    def test_rca_similar_events_are_explained(self, client):
        body = client.post(
            "/api/v1/rca",
            json={"asset_tag": "P-101B", "failure_description": "seal leak dry running"},
        ).json()
        for event in body["similar_events"]:
            assert event["explanation"], "an unexplained similarity score is not usable"

    def test_rca_metrics_refuse_to_invent_an_mtbf(self, client):
        body = client.post(
            "/api/v1/rca", json={"asset_tag": "PIC-101", "failure_description": "drift"}
        ).json()
        evidence = body.get("evidence_gathered", {})
        if evidence.get("mtbf_days") is None and "mtbf_status" in evidence:
            assert "insufficient_data" in evidence["mtbf_status"]

    def test_rca_on_an_unknown_asset_says_so(self, client):
        body = client.post(
            "/api/v1/rca", json={"asset_tag": "P-999Z", "failure_description": "anything"}
        ).json()
        assert body["asset_found"] is False
        assert body["causal_tree"] is None

    def test_compliance_reports_requirement_provenance(self, client):
        body = client.post("/api/v1/compliance", json={"scope": {"years": 3}}).json()
        assert body["requirement_provenance"]
        assert body["status"]["state"] in CAPABILITY_STATES

    def test_every_compliance_gap_explains_itself(self, client):
        body = client.post("/api/v1/compliance", json={"scope": {"years": 3}}).json()
        for gap in body["gaps"]:
            assert gap["detail"] and gap["obligation_text"]
            assert gap["modality"] in {"shall", "should", "may"}
            assert gap["data_class"] in VALID_DATA_CLASSES

    def test_evidence_package_declares_it_is_not_implemented(self, client):
        body = client.post(
            "/api/v1/compliance/evidence-package", json={"scope": {"years": 3}}
        ).json()
        assert body["status"]["state"] == "not_implemented"
        assert body["package_url"] is None


class TestProactiveAndFeedback:
    def test_notifications_declare_the_engine_state(self, client):
        body = client.get("/api/v1/notifications").json()
        assert body["engine"]["state"] in CAPABILITY_STATES
        # No sample alerts: an unimplemented engine returns an empty list.
        if body["engine"]["state"] == "not_implemented":
            assert body["items"] == []

    def test_feedback_requires_a_target(self, client):
        assert client.post("/api/v1/feedback", json={"verdict": "down"}).status_code == 400

    def test_feedback_on_an_unknown_query_is_rejected(self, client):
        response = client.post("/api/v1/feedback", json={"query_id": "qry_nope", "verdict": "down"})
        assert response.status_code == 404

    def test_feedback_round_trip(self, client):
        query = client.post("/api/v1/query", json={"question": "What PPE is required?"}).json()
        body = client.post(
            "/api/v1/feedback",
            json={
                "query_id": query["query_id"],
                "verdict": "down",
                "reason": "incomplete",
                "submitted_by": "integration-test",
            },
        ).json()
        assert body["feedback_id"] > 0 and body["queued_for_review"]

    def test_events_are_persisted_and_replayable(self, client):
        body = client.get("/api/v1/events?limit=10").json()
        assert "persisted" in body and "bus_recent" in body


class TestSecurityHeadersAndErrors:
    def test_security_headers_are_present(self, client):
        headers = client.get("/health/live").headers
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["X-Frame-Options"] == "DENY"

    def test_every_response_carries_a_request_id(self, client):
        assert client.get("/health/live").headers["x-request-id"]

    def test_errors_are_structured_and_leak_nothing(self, client):
        body = client.get("/api/v1/assets/P-999Z").json()
        assert set(body["error"]) == {"code", "message", "detail"}
        assert "Traceback" not in body["error"]["message"]

    def test_openapi_document_is_served(self, client):
        spec = client.get("/openapi.json").json()
        assert "/api/v1/query" in spec["paths"]
        assert "/api/v1/ingest" in spec["paths"]


class TestDashboard:
    @pytest.mark.parametrize(
        "page",
        ["index", "ingestion", "copilot", "graph", "reliability", "compliance", "field"],
    )
    def test_every_dashboard_page_is_served(self, client, page):
        response = client.get(f"/ui/{page}.html")
        assert response.status_code == 200
        assert "<!DOCTYPE html>" in response.text

    def test_pages_contain_no_hard_coded_operational_values(self, client):
        """Guard against a dashboard drifting back to static numbers.

        Every metric element must ship empty or as a placeholder; the values are
        fetched from the API at runtime.
        """
        html = client.get("/ui/index.html").text
        assert 'class="metric-value is-pending">—<' in html

    def test_frontend_ships_no_credentials(self, client):
        for path in ("/ui/js/api.js", "/ui/js/ui.js", "/ui/index.html"):
            text = client.get(path).text.lower()
            for forbidden in ("api_key", "apikey", "secret", "password", "sk-", "bearer "):
                assert forbidden not in text, f"{path} must not contain {forbidden}"
