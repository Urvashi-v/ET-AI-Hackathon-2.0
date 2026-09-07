"""Retrieval components that need no database: tokenizer, fusion, confidence.

The tokenizer is the one with teeth. A general-purpose analyser turns ``P-101B``
into ``{p, 101b}``, which destroys the exact-match object that matters most in
this domain — and since embeddings also blur ``P-101B`` against ``P-101A``,
losing it lexically means losing it entirely.
"""

from __future__ import annotations

import pytest

from services.common.schemas import ConfidenceMode, QueryIntent, UserContext
from services.retrieval.confidence import ConfidenceInputs, build_referral
from services.retrieval.confidence import score as confidence_score
from services.retrieval.fusion import reciprocal_rank_fusion, weights_for
from services.retrieval.generate import ContextPassage, verify_claims, verify_quote
from services.retrieval.intent import understand
from services.retrieval.lexical import tokenize, tokenize_query


class TestTokenizer:
    @pytest.mark.parametrize(
        "raw", ["P-101B", "P101B", "P 101 B", "10-P-101-B", "P-101-B", "P‑101‑B"]
    )
    def test_every_tag_variant_produces_one_identical_token(self, raw):
        assert "p-101b" in tokenize(raw)

    def test_a_tag_is_never_split_into_pieces(self):
        tokens = tokenize("Pump P-101B tripped")
        assert "p-101b" in tokens
        assert "101b" not in tokens and "p" not in tokens

    def test_sibling_tags_remain_distinct_tokens(self):
        # If these collapsed, no lexical query could tell the pair apart.
        assert tokenize("P-101A")[0] != tokenize("P-101B")[0]

    def test_standard_references_survive_as_one_token(self):
        assert "oisd-std-105" in tokenize("Refer to OISD-STD-105 for permits")

    def test_index_and_query_tokenizers_agree_on_tags(self):
        assert set(tokenize("P-101B")) <= set(tokenize_query("what about P-101B?"))

    def test_query_side_joins_a_spaced_tag(self):
        assert "p-101b" in tokenize_query("tell me about P 101 B")

    def test_negations_are_not_stripped(self):
        # "no", "not" and "before" carry real meaning in a procedure.
        tokens = tokenize("do not start before confirming suction pressure")
        assert "not" in tokens and "before" in tokens

    def test_light_stemming_only(self):
        assert tokenize("bearings")[0] == "bearing"
        assert tokenize("bearing")[0] == "bearing"
        # Aggressive stemming would conflate "bearing" with "bear".
        assert tokenize("bearing")[0] != tokenize("bear")[0]

    def test_deterministic(self):
        text = "Seal failure on P-101B per OISD-STD-105 clause 4.2"
        assert len({tuple(tokenize(text)) for _ in range(50)}) == 1


class TestIntentRouting:
    @pytest.mark.parametrize(
        "question,expected",
        [
            ("How many seal failures were there last year?", QueryIntent.AGGREGATE),
            ("What are the steps to isolate P-101B?", QueryIntent.PROCEDURAL),
            ("Why does P-101B keep failing?", QueryIntent.DIAGNOSTIC),
            ("What is the spare for P-101B?", QueryIntent.MULTI_HOP),
            ("Compare P-101A versus P-101B downtime", QueryIntent.COMPARATIVE),
            ("What is the design discharge pressure of P-101B?", QueryIntent.LOOKUP),
        ],
    )
    def test_routes_by_lexical_marker(self, question, expected):
        assert understand(question).intent is expected

    def test_aggregate_questions_are_not_sent_to_top_k_retrieval(self):
        # This single routing step is what fixes "RAG can't count".
        result = understand("How many corrective work orders on P-101B?")
        assert result.intent is QueryIntent.AGGREGATE
        assert result.aggregation == "count"
        assert result.requires_graph

    def test_entities_are_linked_from_the_question(self):
        assert understand("Why does P-101B keep failing?").entity_tags() == ["P-101B"]

    def test_informal_reference_resolves_from_user_context(self):
        result = understand("why does the B pump keep failing?", UserContext(asset_tag="P-101A"))
        assert "P-101B" in result.entity_tags()

    def test_informal_reference_without_context_is_reported_unresolved(self):
        # A wrong asset is worse than no asset: never guess a sequence number.
        result = understand("why does the B pump keep failing?")
        assert not result.entity_tags()
        assert result.unresolved_references

    def test_standby_resolves_to_the_sibling(self):
        result = understand("what did we do to the standby pump?", UserContext(asset_tag="P-101A"))
        assert "P-101B" in result.entity_tags()

    def test_abbreviations_are_expanded(self):
        assert "vibration" in understand("high vib on P-101B").normalised

    def test_time_windows_are_parsed(self):
        assert understand("failures in the last 5 years").time_window == {
            "kind": "relative",
            "n": 5,
            "unit": "year",
        }

    def test_intent_is_deterministic(self):
        modes = {understand("Why does P-101B keep failing?").intent for _ in range(30)}
        assert len(modes) == 1


class TestReciprocalRankFusion:
    def test_agreement_between_retrievers_outranks_a_single_strong_hit(self):
        fused = reciprocal_rank_fusion(
            {
                "lexical": [{"chunk_id": "a"}, {"chunk_id": "b"}],
                "graph": [{"chunk_id": "b"}, {"chunk_id": "c"}],
            }
        )
        assert fused[0].chunk_id == "b"
        assert set(fused[0].contributions) == {"lexical", "graph"}

    def test_needs_no_score_normalisation(self):
        # Inputs carry no scores at all -- fusion is rank-based by construction,
        # which is why incomparable BM25 and cosine scores can be combined.
        fused = reciprocal_rank_fusion(
            {"lexical": [{"chunk_id": "x"}], "dense": [{"chunk_id": "x"}]}
        )
        assert fused[0].chunk_id == "x"

    def test_an_unavailable_retriever_degrades_rather_than_breaks(self):
        fused = reciprocal_rank_fusion({"lexical": [{"chunk_id": "a"}], "dense": []})
        assert [f.chunk_id for f in fused] == ["a"]

    def test_intent_weights_favour_the_right_leg(self):
        assert (
            weights_for(QueryIntent.MULTI_HOP)["graph"]
            > weights_for(QueryIntent.MULTI_HOP)["lexical"]
        )
        assert weights_for(QueryIntent.LOOKUP)["lexical"] > weights_for(QueryIntent.LOOKUP)["graph"]

    def test_weighting_changes_the_winner(self):
        lists = {"lexical": [{"chunk_id": "L"}], "graph": [{"chunk_id": "G"}]}
        assert reciprocal_rank_fusion(lists, weights={"graph": 5.0})[0].chunk_id == "G"
        assert reciprocal_rank_fusion(lists, weights={"lexical": 5.0})[0].chunk_id == "L"

    def test_payload_is_preserved(self):
        fused = reciprocal_rank_fusion({"lexical": [{"chunk_id": "a", "text": "hello"}]})
        assert fused[0].payload["text"] == "hello"

    def test_deterministic(self):
        lists = {
            "lexical": [{"chunk_id": c} for c in "abcde"],
            "graph": [{"chunk_id": c} for c in "ecadb"],
        }
        orders = {tuple(f.chunk_id for f in reciprocal_rank_fusion(lists)) for _ in range(30)}
        assert len(orders) == 1


class TestConfidenceAndAbstention:
    def _inputs(self, **overrides):
        base = {
            "top_score": 8.0,
            "distinct_documents": 3,
            "distinct_source_systems": 2,
            "current_documents": 3,
            "total_documents": 3,
            "graph_facts": 12,
            "total_claims": 4,
            "verified_claims": 4,
            "answerer_available": True,
        }
        base.update(overrides)
        return ConfidenceInputs(**base)

    def test_strong_corroborated_evidence_answers(self):
        assert confidence_score(self._inputs()).mode is ConfidenceMode.ANSWER

    def test_a_missing_asset_forces_abstention_regardless_of_other_evidence(self):
        # Never answer about a similarly-named asset.
        report = confidence_score(self._inputs(anchors_missing=["P-999Z"]))
        assert report.mode is ConfidenceMode.ABSTAIN_AND_ROUTE
        assert report.score <= 0.35

    def test_single_source_lowers_confidence(self):
        corroborated = confidence_score(self._inputs()).score
        alone = confidence_score(
            self._inputs(distinct_documents=1, distinct_source_systems=1)
        ).score
        assert alone < corroborated

    def test_unverified_claims_lower_confidence(self):
        assert (
            confidence_score(self._inputs(verified_claims=1)).score
            < confidence_score(self._inputs()).score
        )

    def test_superseded_evidence_lowers_confidence(self):
        assert (
            confidence_score(self._inputs(current_documents=0)).score
            < confidence_score(self._inputs()).score
        )

    def test_no_composable_answer_is_distinguished_from_weak_evidence(self):
        """"Nothing in the evidence answers this" is not "the evidence is weak".

        The distinction used to be about a missing LLM. It no longer is: the
        extractive answerer needs no credential, so reaching this mode means
        retrieval returned passages and none of them contained a sentence that
        addressed the question. That is a different problem from low-scoring
        evidence and gets a different explanation, because the operator's next
        move differs -- rephrase, versus go and find the document.
        """
        report = confidence_score(self._inputs(answerer_available=False))
        assert report.mode is ConfidenceMode.ABSTAIN_NO_ANSWER
        # Strong signals, yet still no answer: it is the absence of an answerable
        # sentence that decides this mode, not the confidence score.
        assert report.score > 0.5
        assert "no sentence" in report.explanation.lower()
        assert "no answer was composed" in report.explanation.lower()

    def test_weak_evidence_abstains(self):
        report = confidence_score(
            self._inputs(
                top_score=0.2,
                distinct_documents=1,
                distinct_source_systems=1,
                graph_facts=0,
                total_claims=4,
                verified_claims=0,
            )
        )
        assert report.mode is ConfidenceMode.ABSTAIN_AND_ROUTE

    def test_explanation_names_the_weakest_signal(self):
        report = confidence_score(
            self._inputs(top_score=6.0, distinct_documents=1, distinct_source_systems=1)
        )
        assert "corroborat" in report.explanation or "one source" in report.explanation

    def test_referral_is_never_a_bare_refusal(self):
        referral = build_referral(anchors_missing=["P-999Z"], intent="lookup")
        assert "P-999Z" in referral["reason"]
        assert referral["expected_document"] and referral["owner_role"] and referral["next_step"]

    def test_referral_names_the_document_class_by_intent(self):
        assert (
            "SOP" in build_referral(anchors_missing=[], intent="procedural")["expected_document"]
            or "procedure"
            in build_referral(anchors_missing=[], intent="procedural")["expected_document"]
        )

    def test_deterministic(self):
        scores = {confidence_score(self._inputs()).score for _ in range(30)}
        assert len(scores) == 1


class TestClaimVerification:
    def _passage(self, marker="C1", text="The discharge pressure limit is 10 barg."):
        return ContextPassage(
            marker=marker,
            chunk_id="c",
            doc_id="d",
            doc_title="SOP-4412",
            doc_type="sop",
            data_class="synthetic_test_data",
            page=1,
            section_path=None,
            text=text,
            retriever="lexical",
            rank=1,
            score=1.0,
        )

    def test_a_cited_claim_counts_as_verified(self):
        result = verify_claims("The discharge pressure limit is 10 barg [C1].", [self._passage()])
        assert result["total_claims"] == 1 and result["verified_claims"] == 1

    def test_an_uncited_claim_is_counted_as_unsupported(self):
        result = verify_claims(
            "The pump was replaced in 2019 by the night shift crew.", [self._passage()]
        )
        assert result["verified_claims"] == 0 and result["unsupported_claims"]

    def test_a_citation_that_does_not_resolve_is_not_credited(self):
        result = verify_claims(
            "Some long enough factual claim about the pump [C9].", [self._passage()]
        )
        assert result["verified_claims"] == 0

    def test_headings_are_not_counted_as_claims(self):
        assert verify_claims("Caveats:", [self._passage()])["total_claims"] == 0

    def test_verbatim_quote_check(self):
        passage = self._passage().text
        assert verify_quote("discharge pressure limit is 10 barg", passage)
        assert verify_quote(
            "The  discharge   pressure limit is 10 barg", passage
        ), "whitespace-insensitive"
        assert not verify_quote("the limit is 12 barg", passage)
        assert not verify_quote("", passage)
