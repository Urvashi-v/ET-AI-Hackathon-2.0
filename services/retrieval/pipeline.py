"""The read path: question in, grounded and cited answer out.

    understand -> parallel retrieve (lexical | dense | graph) -> RRF fusion
        -> rerank -> context assembly -> grounded generation -> citation binding
        -> verification -> confidence -> answer or abstain

Every stage reports its own state, so the response says which legs ran, which
were unavailable and why, and how each contributed. That is what makes the
system inspectable rather than a black box, and it is what the evaluation
harness measures.

Two stages are capability-gated and say so rather than being faked:
``dense`` (needs an embedding provider) and ``generation`` (needs an LLM
provider). With neither configured the endpoint still performs real query
understanding, real BM25, real graph traversal, real fusion and real citation
binding, and returns the evidence with ``ABSTAIN_NO_GENERATOR``.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from services.common import db
from services.common.config import get_settings
from services.common.ids import request_id
from services.common.logging import get_logger
from services.common.schemas import (
    CapabilityState,
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
from services.retrieval import confidence as confidence_mod
from services.retrieval import dense, fusion, generate, graph_retrieval, lexical
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

    # --- 4. fusion ---------------------------------------------------------
    fused = fusion.reciprocal_rank_fusion(
        ranked_lists,
        k=settings.retrieval_rrf_k,
        weights=fusion.weights_for(understanding.intent),
    )

    # --- 5. rerank ---------------------------------------------------------
    # A cross-encoder reranker is the highest-ROI quality intervention available,
    # and it is a capability boundary like the others: with RERANKER_PROVIDER=none
    # the fused order is used unchanged and the response says so, rather than
    # claiming a rerank happened.
    final_k = top_k or settings.retrieval_final_k
    if settings.reranker_provider == "none":
        legs.append(
            RetrievalLeg(
                strategy="rerank",
                state=CapabilityState.NOT_CONFIGURED,
                candidates=len(fused),
                elapsed_ms=0.0,
                detail="Cross-encoder reranking is not configured; the fused RRF order is used "
                "unchanged. This is reported, not silently skipped.",
                required_env=["RERANKER_PROVIDER"],
            )
        )
    selected = fused[:final_k]

    # --- 6. context assembly ----------------------------------------------
    passages = await _hydrate(selected)
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
        )
        for p in passages
    ]

    # --- 7-8. generation + verification ------------------------------------
    generation = await generate.generate(
        question=question,
        passages=passages,
        graph_facts=gra_result.facts if gra_result else [],
        role=ctx.role,
        site=ctx.site,
        work_order=ctx.work_order,
    )
    for citation in citations:
        if citation.marker in generation.used_markers:
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
            total_claims=generation.total_claims,
            verified_claims=generation.verified_claims,
            anchors_missing=gra_result.anchors_missing if gra_result else [],
            generator_available=generation.status.state is CapabilityState.AVAILABLE,
        )
    )

    referral = None
    if report.mode in (ConfidenceMode.ABSTAIN_AND_ROUTE, ConfidenceMode.ABSTAIN_NO_GENERATOR):
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
        resolved_entities=understanding.entities,
        answer=generation.answer,
        answer_data_class=DataClass.MODEL_DERIVED if generation.answer else None,
        generation=generation.status,
        citations=citations,
        graph_facts=(gra_result.facts[:40] if gra_result else []),
        retrieval=legs,
        confidence=report,
        referral=referral,
        actions=_suggest_actions(understanding, passages),
        latency_ms=latency_ms,
    )

    await _log_query(response, understanding, ranked_lists)
    log.info(
        "query.answered",
        query_id=query_id,
        intent=understanding.intent.value,
        mode=report.mode.value,
        confidence=report.score,
        citations=len(citations),
        graph_facts=len(response.graph_facts),
        latency_ms=latency_ms,
    )
    return response


# ---------------------------------------------------------------------------
# Retrieval legs
# ---------------------------------------------------------------------------


async def _run_lexical(
    question: str, understanding: QueryUnderstanding, top_k: int
) -> tuple[RetrievalLeg, list[dict[str, Any]]]:
    started = time.perf_counter()
    try:
        rows = await lexical.search(understanding.normalised, top_k=top_k)
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
               d.title, d.doc_type::text AS doc_type, d.source_system, d.is_current
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
                    in (ConfidenceMode.ABSTAIN_AND_ROUTE, ConfidenceMode.ABSTAIN_NO_GENERATOR),
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
