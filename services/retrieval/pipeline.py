"""The read path: question in, grounded and cited answer out.

    understand -> decompose -> parallel retrieve (lexical | dense | graph,
        for the question and each sub-question) -> RRF fusion -> cross-encoder
        rerank -> context assembly -> grounded answer -> citation binding
        -> claim verification -> confidence -> answer or abstain

Every stage reports its own state, so the response says which legs ran, which
were unavailable and why, and how each contributed. That is what makes the
system inspectable rather than a black box, and it is what the evaluation
harness measures.

**Answering never depends on a credential.** The default answerer is extractive:
it selects verbatim sentences from retrieved passages, so it is incapable of
answering a plant-specific question from general knowledge. An LLM, when
``LLM_PROVIDER`` is configured, replaces it with abstractive prose under citation
binding and verbatim verification. ``answer_method`` on the response says which
one produced the text, because the two carry different risks.

Capability-gated stages report their state rather than being faked: ``dense``
needs an embedding provider, ``rerank`` needs a reranker model, ``generation``
needs an LLM. With none of them configured the endpoint still performs real query
understanding, real BM25, real graph traversal, real fusion, real extraction and
real citation binding.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

from services.common import db
from services.common.config import get_settings
from services.common.ids import request_id
from services.common.logging import get_logger
from services.common.schemas import (
    AnswerClaim,
    CapabilityState,
    CapabilityStatus,
    Citation,
    ConfidenceMode,
    DataClass,
    DocumentType,
    QueryIntent,
    QueryResponse,
    RetrievalLeg,
    SuggestedAction,
    UserContext,
)
from services.retrieval import compose as compose_mod
from services.retrieval import confidence as confidence_mod
from services.retrieval import dense, fusion, generate, graph_retrieval, lexical, rerank
from services.retrieval.intent import QueryUnderstanding, understand

log = get_logger(__name__)


async def answer_question(
    question: str,
    ctx: UserContext | None = None,
    *,
    mode: str = "auto",
    top_k: int | None = None,
) -> QueryResponse:
    settings = get_settings()
    ctx = ctx or UserContext()
    started = time.perf_counter()
    query_id = f"qry_{request_id()[:20]}"

    # --- 1-2. intent + query understanding --------------------------------
    understanding = understand(question, ctx)

    # --- 3. parallel retrieval --------------------------------------------
    legs: list[RetrievalLeg] = []
    ranked_lists: dict[str, list[dict[str, Any]]] = {}

    lexical_task = (
        _run_lexical(question, understanding, settings.retrieval_top_k_lexical)
        if mode != "graph_only"
        else _skipped_leg("lexical", "mode=graph_only")
    )
    dense_task = (
        _run_dense(question, settings.retrieval_top_k_dense)
        if mode == "auto"
        else _skipped_leg("dense", f"mode={mode}")
    )
    graph_task = (
        _run_graph(understanding, settings.retrieval_top_k_graph)
        if mode != "lexical_only"
        else _skipped_leg("graph", "mode=lexical_only")
    )

    (
        (lex_leg, lex_rows),
        (den_leg, den_rows),
        (gra_leg, gra_rows, gra_result),
    ) = await asyncio.gather(lexical_task, dense_task, graph_task)
    legs.extend([lex_leg, den_leg, gra_leg])
    if lex_rows:
        ranked_lists["lexical"] = lex_rows
    if den_rows:
        ranked_lists["dense"] = den_rows
    if gra_rows:
        ranked_lists["graph"] = gra_rows

    # --- 3b. sub-question retrieval ----------------------------------------
    # A compound question retrieved as one string can land between its two
    # answers and reach neither. Each sub-question gets its own lexical and dense
    # pass, folded into the same fusion at a discount.
    if understanding.sub_questions:
        sub_legs, sub_lists = await _run_sub_questions(understanding, settings)
        legs.extend(sub_legs)
        ranked_lists.update(sub_lists)

    # Terms the corpus has never recorded. Same class of problem as an asset tag
    # with no node -- the question names something the system has never heard of
    # -- but reached through the inverted index, because a site or plant name is
    # not an equipment tag and the entity layer never sees it.
    unknown_terms = await lexical.unknown_proper_nouns(question)

    # --- 4. fusion ---------------------------------------------------------
    fused = fusion.reciprocal_rank_fusion(
        ranked_lists,
        k=settings.retrieval_rrf_k,
        weights=fusion.weights_for(understanding.intent),
    )

    # --- 5. rerank ---------------------------------------------------------
    # RRF fuses on rank, which knows nothing about what the passages say. A
    # cross-encoder reads the query and passage together and is the highest-ROI
    # quality step available -- but it is expensive, so it runs over the fused
    # shortlist rather than the corpus. With RERANKER_PROVIDER=none the fused
    # order is used unchanged and the response says so.
    final_k = top_k or settings.retrieval_final_k
    shortlist = fused[: settings.rerank_candidates]
    hydrated = await _hydrate(shortlist)
    rerank_leg, passages = await _rerank(question, hydrated, final_k)
    legs.append(rerank_leg)
    citations = [
        Citation(
            marker=p.marker,
            chunk_id=p.chunk_id,
            doc_id=p.doc_id,
            doc_title=p.doc_title,
            doc_type=DocumentType(p.doc_type),
            data_class=DataClass(p.data_class),
            page=p.page,
            section_path=p.section_path,
            snippet=p.text[:600],
            quote_verified=False,
            retriever=p.retriever,
            rank=p.rank,
            score=round(p.score, 6),
            is_current=p.is_current,
            revision=p.revision,
        )
        for p in passages
    ]

    # --- 7-8. answering + verification -------------------------------------
    answer = await _answer(
        question=question,
        understanding=understanding,
        passages=passages,
        graph_facts=gra_result.facts if gra_result else [],
        ctx=ctx,
    )
    for citation in citations:
        if citation.marker in answer.used_markers:
            citation.quote_verified = True

    # --- 9. confidence and routing ----------------------------------------
    report = confidence_mod.score(
        confidence_mod.ConfidenceInputs(
            top_score=max((p.score for p in passages), default=0.0),
            scores=[p.score for p in passages],
            distinct_documents=len({p.doc_id for p in passages}),
            distinct_source_systems=len({p.source_system for p in passages}),
            current_documents=sum(1 for p in passages if p.is_current),
            total_documents=len(passages),
            graph_facts=len(gra_result.facts) if gra_result else 0,
            total_claims=answer.total_claims,
            verified_claims=answer.verified_claims,
            anchors_missing=gra_result.anchors_missing if gra_result else [],
            answerer_available=answer.text is not None,
            answer_method=answer.method,
            answer_relevance=answer.relevance,
            missing_terms=answer.missing_terms,
            unknown_terms=unknown_terms,
            wants_live_state=understanding.wants_live_state,
            score_scale=("reranker" if rerank_leg.state is CapabilityState.AVAILABLE else "fusion"),
        )
    )

    abstained = report.mode in (ConfidenceMode.ABSTAIN_AND_ROUTE, ConfidenceMode.ABSTAIN_NO_ANSWER)
    referral = None
    if abstained:
        referral = confidence_mod.build_referral(
            anchors_missing=gra_result.anchors_missing if gra_result else [],
            intent=understanding.intent.value,
        )

    latency_ms = int((time.perf_counter() - started) * 1000)
    response = QueryResponse(
        query_id=query_id,
        question=question,
        intent=understanding.intent,
        intent_confidence=understanding.intent_confidence,
        intent_method=understanding.method,
        sub_questions=understanding.sub_questions,
        resolved_entities=understanding.entities,
        # An abstention withholds the answer text itself, not just a warning
        # beside it. Showing a low-confidence answer and labelling it low
        # confidence is how people end up acting on it anyway.
        answer=None if abstained else answer.text,
        answer_data_class=(
            None if abstained or not answer.text else _answer_data_class(answer.method)
        ),
        answer_method=answer.method,
        claims=[] if abstained else answer.claims,
        abstained=abstained,
        generation=answer.status,
        citations=citations,
        graph_facts=(gra_result.facts[:40] if gra_result else []),
        graph_entities=(gra_result.entity_refs() if gra_result else []),
        retrieval=legs,
        retrieval_sources=sorted({r for p in passages for r in p.retriever.split("+")} - {"none"}),
        confidence=report,
        referral=referral,
        actions=_suggest_actions(understanding, passages),
        latency_ms=latency_ms,
    )

    await _log_query(response, understanding, ranked_lists)
    # One structured line per query, carrying everything needed to diagnose it
    # without reproducing it. The request id is bound by middleware, so this line
    # joins to the HTTP access log and to every other line the request emitted.
    #
    # Nothing here is a secret. Provider *names* are logged and credentials never
    # are -- the question "which model answered this?" is operationally essential
    # and "with which key?" is never asked of a log.
    log.info(
        "query.answered",
        query_id=query_id,
        # --- what was asked -------------------------------------------------
        intent=understanding.intent.value,
        intent_confidence=round(understanding.intent_confidence, 3),
        intent_method=understanding.method,
        sub_questions=len(understanding.sub_questions),
        entities=[e.get("canonical_tag") for e in understanding.entities],
        # --- what retrieval did ---------------------------------------------
        retrieval_sources=response.retrieval_sources,
        candidates={(leg.leg_id or leg.strategy): leg.candidates for leg in legs},
        leg_state={(leg.leg_id or leg.strategy): leg.state.value for leg in legs},
        leg_latency_ms={(leg.leg_id or leg.strategy): round(leg.elapsed_ms, 1) for leg in legs},
        rerank_state=rerank_leg.state.value,
        rerank_candidates=rerank_leg.candidates,
        rerank_ms=round(rerank_leg.elapsed_ms, 1),
        # --- what was answered ----------------------------------------------
        answer_method=answer.method,
        answer_provider=_answer_provider(answer.method),
        model_latency_ms=round(answer.elapsed_ms, 1),
        citations=len(citations),
        claims=len(response.claims),
        verified_claims=answer.verified_claims,
        graph_facts=len(response.graph_facts),
        # --- how it was judged ----------------------------------------------
        confidence=report.score,
        confidence_mode=report.mode.value,
        abstained=abstained,
        signals=report.signals,
        # --- what went wrong -------------------------------------------------
        errors=[
            {"stage": leg.leg_id or leg.strategy, "detail": (leg.detail or "")[:160]}
            for leg in legs
            if leg.state is CapabilityState.ERROR
        ],
        latency_ms=latency_ms,
    )
    return response


def _answer_provider(method: str | None) -> str:
    """Which component produced the answer text.

    The provider *name*, never its credential. Distinguishing an extractive
    answer from an LLM one is the first question asked when an answer is
    disputed, and it cannot be recovered from the text after the fact.
    """
    if method == "abstractive":
        return f"llm:{get_settings().llm_provider}"
    if method == "extractive":
        return "extractive:deterministic"
    return "none"


# ---------------------------------------------------------------------------
# Answering
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AnswerOutcome:
    """The answer, however it was produced, in one shape.

    Extraction and generation are interchangeable at this boundary, which is what
    lets the LLM be genuinely optional rather than a hole in the product.
    """

    text: str | None
    method: str | None
    status: CapabilityStatus
    claims: list[AnswerClaim] = field(default_factory=list)
    used_markers: set[str] = field(default_factory=set)
    total_claims: int = 0
    verified_claims: int = 0
    #: Time spent producing the answer text, separate from retrieval. The figure
    #: that answers "is the model slow, or is retrieval slow?" -- which is
    #: otherwise guesswork from a single end-to-end number.
    elapsed_ms: float = 0.0
    #: How much of the question the answer covers. 1.0 for abstractive answers,
    #: which are not measured this way -- an LLM rephrases rather than reusing the
    #: question's vocabulary, so term coverage would penalise good prose.
    relevance: float = 1.0
    missing_terms: list[str] = field(default_factory=list)


async def _answer(
    *,
    question: str,
    understanding: QueryUnderstanding,
    passages: list[generate.ContextPassage],
    graph_facts: list[Any],
    ctx: UserContext,
) -> AnswerOutcome:
    """Produce an answer, preferring the LLM when one is configured.

    When it is not -- the default -- extraction answers instead. This ordering is
    deliberate: an LLM synthesises across passages better than sentence selection
    can, and it is fenced by citation binding and verbatim verification. But its
    absence must not turn the copilot into a search box, so extraction is a real
    answerer rather than a placeholder.
    """
    started = time.perf_counter()
    llm_status = generate.generation_capability()
    if llm_status.state is CapabilityState.AVAILABLE:
        result = await generate.generate(
            question=question,
            passages=passages,
            graph_facts=graph_facts,
            role=ctx.role,
            site=ctx.site,
            work_order=ctx.work_order,
        )
        if result.status.state is CapabilityState.AVAILABLE and result.answer:
            return AnswerOutcome(
                text=result.answer,
                method="abstractive",
                elapsed_ms=(time.perf_counter() - started) * 1000,
                status=result.status,
                claims=_claims_from_generation(result, passages),
                used_markers=result.used_markers,
                total_claims=result.total_claims,
                verified_claims=result.verified_claims,
            )
        # The provider is configured but the call failed. Fall through to
        # extraction rather than returning nothing: degraded is better than dead,
        # and the status still reports the failure.
        log.warning("answer.generation_unavailable", detail=result.status.detail)

    # Term rarity, so relevance weights the word the question turns on above the
    # words that merely carry it. One indexed lookup, on terms already tokenized.
    frequencies = await lexical.document_frequencies(lexical.tokenize_query(question))
    composed = compose_mod.compose(
        question=question,
        intent=understanding.intent,
        passages=passages,
        entity_tags=understanding.entity_tags(),
        doc_frequencies=frequencies,
    )
    verification = compose_mod.verify_composed(composed, passages)
    state = CapabilityState.AVAILABLE if composed.text else CapabilityState.NOT_CONFIGURED
    detail = composed.detail or ""
    if llm_status.state is not CapabilityState.AVAILABLE:
        detail += (
            " Abstractive generation is not configured, so the answer is extracted "
            "verbatim rather than written."
        )
    return AnswerOutcome(
        text=composed.text,
        method="extractive" if composed.text else None,
        elapsed_ms=(time.perf_counter() - started) * 1000,
        status=CapabilityStatus(
            capability="grounded_answer",
            state=state,
            detail=detail.strip(),
            required_env=(
                llm_status.required_env if llm_status.state is not CapabilityState.AVAILABLE else []
            ),
        ),
        claims=[
            AnswerClaim(
                text=c.text,
                marker=c.marker,
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                doc_title=c.doc_title,
                page=c.page,
                char_start=c.char_start,
                char_end=c.char_end,
                verbatim=True,
                score=round(c.score, 4),
            )
            for c in composed.claims
        ],
        used_markers=composed.used_markers,
        total_claims=int(verification["total_claims"]),
        verified_claims=int(verification["verified_claims"]),
        relevance=composed.relevance,
        missing_terms=composed.missing_terms,
    )


def _claims_from_generation(
    result: generate.GenerationResult, passages: list[generate.ContextPassage]
) -> list[AnswerClaim]:
    """Split LLM prose into cited claims for the same UI treatment as extraction.

    ``verbatim`` is False here and that is the honest value: the sentence was
    written by the model, not copied from the passage, so it can be attributed to
    a source but not located inside one.
    """
    if not result.answer:
        return []
    by_marker = {p.marker: p for p in passages}
    claims: list[AnswerClaim] = []
    for sentence in generate._SENTENCE_SPLIT.split(result.answer):
        sentence = sentence.strip()
        if not sentence:
            continue
        markers = generate._CITATION_MARKER.findall(sentence)
        for number in markers or []:
            passage = by_marker.get(f"C{number}")
            if not passage:
                continue
            claims.append(
                AnswerClaim(
                    text=sentence,
                    marker=passage.marker,
                    chunk_id=passage.chunk_id,
                    doc_id=passage.doc_id,
                    doc_title=passage.doc_title,
                    page=passage.page,
                    verbatim=False,
                )
            )
    return claims


def _answer_data_class(method: str | None) -> DataClass:
    """Extraction copies source text; generation writes new text.

    Labelling extracted sentences MODEL_DERIVED would overstate what happened to
    them -- they are the source document, selected.
    """
    return DataClass.REAL_SOURCE_DOCUMENT if method == "extractive" else DataClass.MODEL_DERIVED


# ---------------------------------------------------------------------------
# Retrieval legs
# ---------------------------------------------------------------------------


async def _run_lexical(
    question: str,
    understanding: QueryUnderstanding,
    top_k: int,
    *,
    question_override: str | None = None,
) -> tuple[RetrievalLeg, list[dict[str, Any]]]:
    """BM25 over the abbreviation-expanded question, or over a sub-question."""
    started = time.perf_counter()
    text = question_override or understanding.normalised
    try:
        rows = await lexical.search(text, top_k=top_k)
        elapsed = (time.perf_counter() - started) * 1000
        return (
            RetrievalLeg(
                strategy="lexical",
                state=CapabilityState.AVAILABLE,
                candidates=len(rows),
                elapsed_ms=round(elapsed, 2),
                detail="Okapi BM25 over a tag-preserving tokenizer",
            ),
            rows,
        )
    except Exception as exc:
        log.error("retrieval.lexical_failed", error=str(exc))
        return (
            RetrievalLeg(
                strategy="lexical",
                state=CapabilityState.ERROR,
                candidates=0,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                detail=f"{type(exc).__name__}: {str(exc)[:200]}",
            ),
            [],
        )


async def _run_dense(question: str, top_k: int) -> tuple[RetrievalLeg, list[dict[str, Any]]]:
    result = await dense.search(question, top_k=top_k)
    return (
        RetrievalLeg(
            strategy="dense",
            state=result.state,
            candidates=len(result.rows),
            elapsed_ms=round(result.elapsed_ms, 2),
            detail=result.detail,
            required_env=result.required_env,
        ),
        result.rows,
    )


async def _run_graph(
    understanding: QueryUnderstanding, top_k: int
) -> tuple[RetrievalLeg, list[dict[str, Any]], graph_retrieval.GraphRetrievalResult | None]:
    tags = understanding.entity_tags()
    if not tags:
        return (
            RetrievalLeg(
                strategy="graph",
                state=CapabilityState.AVAILABLE,
                candidates=0,
                elapsed_ms=0.0,
                detail="No asset tag was resolved from the question, so there is no anchor "
                "to traverse from.",
            ),
            [],
            None,
        )
    try:
        result = await graph_retrieval.retrieve(
            tags=tags, intent=understanding.intent, hops=2, limit=top_k
        )
        detail = (
            f"{len(result.facts)} facts from anchors {', '.join(result.anchors_found) or 'none'}"
        )
        if result.anchors_missing:
            detail += f"; not in corpus: {', '.join(result.anchors_missing)}"
        return (
            RetrievalLeg(
                strategy="graph",
                state=CapabilityState.AVAILABLE,
                candidates=len(result.evidence_chunk_ids),
                elapsed_ms=round(result.elapsed_ms, 2),
                detail=detail,
            ),
            result.as_ranked_chunks(),
            result,
        )
    except Exception as exc:
        log.error("retrieval.graph_failed", error=str(exc))
        return (
            RetrievalLeg(
                strategy="graph",
                state=CapabilityState.ERROR,
                candidates=0,
                elapsed_ms=0.0,
                detail=f"{type(exc).__name__}: {str(exc)[:200]}",
            ),
            [],
            None,
        )


async def _run_sub_questions(
    understanding: QueryUnderstanding, settings: Any
) -> tuple[list[RetrievalLeg], dict[str, list[dict[str, Any]]]]:
    """Retrieve for each decomposed sub-question, in parallel with the others.

    Only the lexical and dense legs are re-run. The graph leg is anchored on the
    resolved entities, which are already the union across all sub-questions, so
    traversing again per clause would return the same neighbourhood at the cost
    of another round trip.
    """
    subs = understanding.sub_questions
    # Half depth: these are supporting evidence, and a full-depth list per
    # sub-question would let two fragments out-vote the whole question in fusion.
    depth = max(5, settings.retrieval_top_k_lexical // 2)

    tasks: list[Any] = []
    for sub in subs:
        tasks.append(_run_lexical(sub, understanding, depth, question_override=sub))
        tasks.append(_run_dense(sub, depth))
    outcomes = await asyncio.gather(*tasks)

    legs: list[RetrievalLeg] = []
    lists: dict[str, list[dict[str, Any]]] = {}
    for index, sub in enumerate(subs):
        for offset, strategy in enumerate(("lexical", "dense")):
            leg, rows = outcomes[index * 2 + offset]
            key = f"{strategy}:sub{index + 1}"
            leg.leg_id = key
            leg.sub_question = sub
            legs.append(leg)
            if rows:
                lists[key] = rows
    return legs, lists


async def _rerank(
    question: str, passages: list[generate.ContextPassage], final_k: int
) -> tuple[RetrievalLeg, list[generate.ContextPassage]]:
    """Score the shortlist with a cross-encoder and cut it to ``final_k``.

    Markers are assigned *after* reordering, so ``C1`` is always the passage the
    reranker put first. Assigning them before would make the citation numbering
    disagree with the displayed order.
    """
    if not passages:
        return (
            RetrievalLeg(
                strategy="rerank",
                state=CapabilityState.AVAILABLE,
                candidates=0,
                elapsed_ms=0.0,
                detail="Nothing was retrieved, so there was nothing to rerank.",
            ),
            [],
        )

    result = await rerank.get_reranker().rerank(question, [p.text for p in passages])

    if result.state is CapabilityState.AVAILABLE:
        ordered = [passages[i] for i in result.order]
        for position, index in enumerate(result.order):
            ordered[position].score = result.normalised(index)
        detail = (
            f"model={result.model}; reordered {len(passages)} candidate(s), "
            f"kept top {min(final_k, len(ordered))}"
        )
    else:
        # Fused order preserved. Reported, not silently skipped.
        ordered = passages
        detail = result.detail or "Reranking did not run; the fused order is unchanged."

    kept = ordered[:final_k]
    for position, passage in enumerate(kept, start=1):
        passage.marker = f"C{position}"
        passage.rank = position

    return (
        RetrievalLeg(
            strategy="rerank",
            state=result.state,
            candidates=len(passages),
            elapsed_ms=round(result.elapsed_ms, 2),
            detail=detail,
            required_env=result.required_env,
        ),
        kept,
    )


async def _skipped_leg(strategy: str, reason: str) -> Any:
    leg = RetrievalLeg(
        strategy=strategy,  # type: ignore[arg-type]
        state=CapabilityState.DISABLED,
        candidates=0,
        elapsed_ms=0.0,
        detail=reason,
    )
    return (leg, [], None) if strategy == "graph" else (leg, [])


# ---------------------------------------------------------------------------
# Context hydration
# ---------------------------------------------------------------------------


async def _hydrate(selected: list[fusion.FusedResult]) -> list[generate.ContextPassage]:
    """Load full chunk text for the fused winners.

    The graph leg contributes chunk ids without text, so hydration happens once,
    here, rather than each retriever carrying its own copy.
    """
    if not selected:
        return []
    chunk_ids = [r.chunk_id for r in selected]
    rows = await db.fetch_all(
        """
        SELECT c.chunk_id, c.doc_id, c.text, c.section_path, c.page_from,
               c.data_class::text AS chunk_data_class,
               d.title, d.doc_type::text AS doc_type, d.source_system, d.is_current,
               d.revision
          FROM document_chunks c
          JOIN documents d ON d.doc_id = c.doc_id
         WHERE c.chunk_id = ANY(%s)
        """,
        (chunk_ids,),
    )
    by_id = {row["chunk_id"]: row for row in rows}

    passages: list[generate.ContextPassage] = []
    for index, fused_row in enumerate(selected, start=1):
        row = by_id.get(fused_row.chunk_id)
        if not row:
            # A chunk id from the graph with no row in Postgres means the two
            # stores have drifted. Skip it rather than emitting a citation that
            # cannot be opened.
            log.warning("retrieval.chunk_missing_in_postgres", chunk_id=fused_row.chunk_id)
            continue
        passages.append(
            generate.ContextPassage(
                marker=f"C{index}",
                chunk_id=row["chunk_id"],
                doc_id=row["doc_id"],
                doc_title=row["title"],
                doc_type=row["doc_type"],
                data_class=row["chunk_data_class"],
                page=row["page_from"],
                section_path=row["section_path"],
                text=row["text"],
                retriever=fused_row.retriever_summary,
                rank=index,
                score=fused_row.score,
                is_current=bool(row["is_current"]),
                revision=row["revision"],
                source_system=row["source_system"],
            )
        )
    return passages


def _suggest_actions(
    understanding: QueryUnderstanding, passages: list[generate.ContextPassage]
) -> list[SuggestedAction]:
    """An answer that ends in a full stop is information; one that ends in a
    button is a product."""
    actions: list[SuggestedAction] = []
    tags = understanding.entity_tags()
    if tags:
        actions.append(
            SuggestedAction(
                label=f"Open {tags[0]} in the graph",
                action_type="open_graph",
                payload={"asset_tag": tags[0]},
            )
        )
        actions.append(
            SuggestedAction(
                label="Run root cause analysis",
                action_type="open_rca",
                payload={"asset_tag": tags[0]},
            )
        )
    if passages:
        actions.append(
            SuggestedAction(
                label="Open the cited source",
                action_type="open_document",
                payload={"doc_id": passages[0].doc_id, "page": passages[0].page},
            )
        )
    actions.append(
        SuggestedAction(label="This answer was wrong", action_type="feedback", payload={})
    )
    return actions


async def _log_query(
    response: QueryResponse,
    understanding: QueryUnderstanding,
    ranked_lists: dict[str, list[dict[str, Any]]],
) -> None:
    """Audit trail. Every machine assertion is logged with what produced it."""
    try:
        async with db.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO query_log (
                    query_id, question, normalised_question, intent, intent_confidence,
                    entities, retrieval_stats, answer_text, confidence_score,
                    confidence_mode, abstained, generator_provider, latency_ms
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (query_id) DO NOTHING
                """,
                (
                    response.query_id,
                    response.question,
                    understanding.normalised,
                    response.intent.value,
                    response.intent_confidence,
                    json.dumps(response.resolved_entities),
                    json.dumps({k: len(v) for k, v in ranked_lists.items()}),
                    response.answer,
                    response.confidence.score,
                    response.confidence.mode.value,
                    response.confidence.mode
                    in (ConfidenceMode.ABSTAIN_AND_ROUTE, ConfidenceMode.ABSTAIN_NO_ANSWER),
                    get_settings().llm_provider,
                    response.latency_ms,
                ),
            )
            for citation in response.citations:
                await cur.execute(
                    """
                    INSERT INTO citations (
                        query_id, marker, chunk_id, doc_id, page, quote,
                        quote_verified, retriever, rank, score
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        response.query_id,
                        citation.marker,
                        citation.chunk_id,
                        citation.doc_id,
                        citation.page,
                        citation.snippet[:1000],
                        citation.quote_verified,
                        citation.retriever,
                        citation.rank,
                        citation.score,
                    ),
                )
    except Exception as exc:  # logging must never break answering
        log.error("query.audit_log_failed", error=str(exc))


async def retrieval_health() -> dict[str, Any]:
    """What the dashboard shows on its retrieval panel."""
    from services.ingest.embeddings import embedding_capability

    stats = await lexical.corpus_stats()
    dense_state, dense_detail, dense_env = embedding_capability()
    return {
        # The reranker cache's hit rate, exposed so it is observable. A cache
        # nobody can see is a cache nobody can tell has stopped working -- and a
        # hit rate that silently falls to zero looks exactly like a slow host.
        "rerank_cache": rerank.cache_stats(),
        "lexical": {"state": "available", **stats},
        "dense": {
            "state": dense_state.value,
            "detail": dense_detail,
            "required_env": dense_env,
            "indexed_chunks": await dense.indexed_chunk_count(),
        },
        "generation": generate.generation_capability().model_dump(),
        "intents": [i.value for i in QueryIntent],
    }
