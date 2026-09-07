#!/usr/bin/env python3
"""Evaluation harness.

This exists from day one, before the numbers are good, because a quality claim
you cannot measure is an adjective. Running it after every change and keeping
the history is what turns "we improved retrieval" into a chart.

It measures the six things the brief's evaluation focus names, and it is honest
about which of them are measurable in the current configuration:

======================================  ==================================================
entity extraction / linkage             measured -- resolution rate and cross-document
                                        linkage, read from the live entity layer
retrieval quality on expert questions   measured -- context recall and precision against
                                        the documents each golden question should reach
answer correctness                      **not measurable without a generation provider.**
                                        Reported as ``not_measurable``, never as zero and
                                        never as a guess
abstention behaviour                    measured -- the rate on deliberately unanswerable
                                        questions, and the false-abstention rate
time to answer                          measured -- p50 and p95 end to end
intent routing                          measured -- accuracy against the labelled category
======================================  ==================================================

Every run is written to ``eval/results/`` with a timestamp and the full
configuration it ran under, so a later comparison is meaningful.

Usage::

    python eval/run_eval.py                         # against localhost:8000
    python eval/run_eval.py --api http://host:8000
    python eval/run_eval.py --tag "with-reranker"
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from eval import metrics

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN = REPO_ROOT / "eval" / "golden.jsonl"
RESULTS_DIR = REPO_ROOT / "eval" / "results"

#: Golden categories that must produce an abstention.
UNANSWERABLE = {"unanswerable"}

#: Which intent the router should choose for each golden category. ``lookup`` is
#: the correct fallback for an unanswerable question: the router's job is to
#: classify the question, not to know the answer is absent.
EXPECTED_INTENT = {
    "lookup": {"lookup"},
    "multi_hop": {"multi_hop", "lookup"},
    "aggregate": {"aggregate"},
    "procedural": {"procedural"},
    "diagnostic": {"diagnostic"},
    "comparative": {"comparative"},
    # Compliance and drawing questions are *topics*, not intents. "When is the
    # next inspection due?" is a lookup that happens to be about an obligation,
    # and the router is right to classify it as one. Scoring them against a
    # "compliance" intent that does not exist measured the harness, not the
    # system -- both categories read 0.000 until this was fixed.
    "compliance": {"lookup", "aggregate", "multi_hop", "comparative", "procedural"},
    "drawing": {"lookup", "multi_hop"},
    "unanswerable": {"lookup", "aggregate", "diagnostic", "multi_hop", "procedural", "comparative"},
}


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            cases.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_no}: {exc.msg}") from exc
    return cases


def normalise_title(title: str) -> str:
    return " ".join(str(title or "").lower().replace("_", " ").replace("-", " ").split())


def evaluate_case(client: httpx.Client, case: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response = client.post(
            "/api/v1/query",
            json={
                "question": case["question"],
                "user_ctx": {"role": "reliability_engineer"},
                "mode": "auto",
            },
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return {
            "id": case["id"],
            "category": case["category"],
            "error": f"{type(exc).__name__}: {str(exc)[:200]}",
            "latency_s": time.perf_counter() - started,
        }

    latency = time.perf_counter() - started
    retrieved_titles = {normalise_title(c["doc_title"]) for c in payload.get("citations", [])}
    expected_titles = {normalise_title(t) for t in case.get("expected_docs", [])}

    context_recall = (
        len(retrieved_titles & expected_titles) / len(expected_titles) if expected_titles else None
    )
    context_precision = (
        len(retrieved_titles & expected_titles) / len(retrieved_titles) if retrieved_titles else 0.0
    )

    resolved = {e.get("canonical_tag") for e in payload.get("resolved_entities", [])}
    expected_entities = set(case.get("expected_entities", []))
    entity_recall = (
        len(resolved & expected_entities) / len(expected_entities) if expected_entities else None
    )

    mode = payload.get("confidence", {}).get("mode")
    abstained = mode in ("ABSTAIN_AND_ROUTE", "ABSTAIN_NO_ANSWER")
    should_abstain = case["category"] in UNANSWERABLE

    # An abstention on an unanswerable question is only *correct* if it names
    # what is missing. A bare refusal is not the behaviour being claimed.
    referral = payload.get("referral")
    abstention_is_routed = bool(referral and referral.get("expected_document"))

    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "latency_s": latency,
        "intent": payload.get("intent"),
        "intent_correct": payload.get("intent") in EXPECTED_INTENT.get(case["category"], set()),
        "context_recall": context_recall,
        "context_precision": context_precision,
        "entity_recall": entity_recall,
        "citations": len(payload.get("citations", [])),
        "graph_facts": len(payload.get("graph_facts", [])),
        "confidence": payload.get("confidence", {}).get("score"),
        "confidence_mode": mode,
        "abstained": abstained,
        "should_abstain": should_abstain,
        "abstention_is_routed": abstention_is_routed,
        "generation_state": payload.get("generation", {}).get("state"),
        "answer_present": bool(payload.get("answer")),
        "answer_method": payload.get("answer_method"),
        "retrieval_legs": {leg["strategy"]: leg["state"] for leg in payload.get("retrieval", [])},
        # Per-leg timings, so a latency regression can be attributed to a stage
        # rather than merely observed at the endpoint.
        "leg_latency_ms": {
            (leg.get("leg_id") or leg["strategy"]): leg["elapsed_ms"]
            for leg in payload.get("retrieval", [])
        },
        "claim_count": len(payload.get("claims", [])),
        "verbatim_claims": sum(1 for c in payload.get("claims", []) if c.get("verbatim")),
        # Kept so citation validity can re-fetch each one and confirm it opens.
        "citation_refs": [
            {
                "marker": c.get("marker"),
                "doc_id": c.get("doc_id"),
                "chunk_id": c.get("chunk_id"),
                "page": c.get("page"),
                "snippet": c.get("snippet"),
            }
            for c in payload.get("citations", [])
        ],
    }


def mean(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None]
    return round(statistics.fmean(clean), 4) if clean else None


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1))))
    return round(ordered[index], 4)


def summarise(results: list[dict[str, Any]], system: dict[str, Any]) -> dict[str, Any]:
    ok = [r for r in results if "error" not in r]
    errored = [r for r in results if "error" in r]
    answerable = [r for r in ok if not r["should_abstain"]]
    unanswerable = [r for r in ok if r["should_abstain"]]
    latencies = [r["latency_s"] for r in ok]

    generation_available = system.get("generation", {}).get("state") == "available"

    by_category: dict[str, Any] = {}
    for result in ok:
        bucket = by_category.setdefault(
            result["category"],
            {"n": 0, "context_recall": [], "intent_correct": 0, "abstained": 0, "latency": []},
        )
        bucket["n"] += 1
        bucket["context_recall"].append(result["context_recall"])
        bucket["intent_correct"] += int(result["intent_correct"])
        bucket["abstained"] += int(result["abstained"])
        bucket["latency"].append(result["latency_s"])
    for bucket in by_category.values():
        bucket["context_recall"] = mean(bucket["context_recall"])
        bucket["intent_accuracy"] = round(bucket["intent_correct"] / bucket["n"], 4)
        bucket["abstention_rate"] = round(bucket["abstained"] / bucket["n"], 4)
        bucket["p50_latency_s"] = percentile(bucket["latency"], 50)
        del bucket["latency"], bucket["intent_correct"], bucket["abstained"]

    return {
        "n_cases": len(results),
        "n_errored": len(errored),
        "retrieval": {
            "context_recall": mean([r["context_recall"] for r in answerable]),
            "context_precision": mean([r["context_precision"] for r in answerable]),
            "entity_recall": mean([r["entity_recall"] for r in ok]),
            "mean_citations": mean([float(r["citations"]) for r in ok]),
            "mean_graph_facts": mean([float(r["graph_facts"]) for r in ok]),
        },
        "intent_routing": {
            "accuracy": mean([float(r["intent_correct"]) for r in ok]),
        },
        "abstention": {
            "recall_on_unanswerable": mean([float(r["abstained"]) for r in unanswerable]),
            "routed_referral_rate": mean(
                [float(r["abstention_is_routed"]) for r in unanswerable if r["abstained"]]
            ),
            # Measured unconditionally. This used to be reported as "not
            # meaningful without a generator", which was true when the only
            # answerer was an LLM and every query therefore abstained. The
            # extractive answerer needs no credential, so abstention is now a
            # real decision on every query and the rate at which it fires on
            # answerable questions is a real cost -- a useful answer withheld.
            #
            # It is the counterweight to recall_on_unanswerable: either number
            # alone can be driven to 1.0 by a system that always abstains or
            # never does, and only the pair says anything.
            "false_abstention_rate": mean([float(r["abstained"]) for r in answerable]),
            "false_abstention_note": None,
        },
        "answer_quality": (
            {
                "state": "not_measurable",
                "reason": "No generation provider is configured, so no answers were produced. "
                "Correctness is reported as not_measurable rather than as zero.",
            }
            if not generation_available
            else {
                "state": "measurable",
                "answers_produced": sum(1 for r in answerable if r["answer_present"]),
            }
        ),
        "latency": {
            "p50_s": percentile(latencies, 50),
            "p95_s": percentile(latencies, 95),
            "max_s": round(max(latencies), 4) if latencies else None,
        },
        "by_category": by_category,
    }


def leg_latency_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Where the time goes, per retrieval leg.

    A p95 at the endpoint says the system is slow; a p95 per leg says which part.
    Without this, every latency investigation starts by re-deriving the same
    breakdown by hand.
    """
    by_leg: dict[str, list[float]] = {}
    for result in results:
        for leg, ms in (result.get("leg_latency_ms") or {}).items():
            by_leg.setdefault(leg, []).append(float(ms))
    return {
        leg: {
            "p50_ms": percentile(values, 50),
            "p95_ms": percentile(values, 95),
            "max_ms": max(values),
            "samples": len(values),
        }
        for leg, values in sorted(by_leg.items())
    }


def collect_system_state(client: httpx.Client) -> dict[str, Any]:
    """Snapshot the configuration a run was measured under.

    Includes the reranker cache state, which matters for reading the latency
    figures: a second run over the same golden set is served from cache and is
    not comparable with a cold one. Recording the hit rate makes a warm run
    identifiable rather than mistakable for a fast system.
    """
    state: dict[str, Any] = {}
    for key, path in (
        ("health", "/health"),
        ("retrieval", "/api/v1/query/health"),
        ("assets", "/api/v1/assets/stats"),
    ):
        try:
            state[key] = client.get(path).json()
        except Exception as exc:
            state[key] = {"error": f"{type(exc).__name__}: {exc}"}

    retrieval = state.get("retrieval", {})
    return {
        "providers": state.get("health", {}).get("providers", {}),
        "generation": retrieval.get("generation", {}),
        "dense": retrieval.get("dense", {}),
        "lexical": retrieval.get("lexical", {}),
        "entity_layer": state.get("assets", {}),
        # Cache state at the start of the run. A second pass over the same golden
        # set is served from cache and its latency figures are not comparable
        # with a cold one; recording this makes a warm run identifiable.
        "rerank_cache_at_start": retrieval.get("rerank_cache", {}),
        "app_version": state.get("health", {}).get("version"),
    }


def print_report(summary: dict[str, Any], system: dict[str, Any], results: list[dict]) -> None:
    def fmt(value: Any) -> str:
        if value is None:
            return "n/a"
        return f"{value:.3f}" if isinstance(value, float) else str(value)

    print("\n" + "=" * 78)
    print("EVALUATION REPORT")
    print("=" * 78)

    print("\nConfiguration under test")
    print(f"  generation      : {system.get('generation', {}).get('state', 'unknown')}")
    print(f"  dense retrieval : {system.get('dense', {}).get('state', 'unknown')}")
    print(f"  lexical index   : {system.get('lexical', {}).get('chunk_count', 0)} chunks")
    entity = system.get("entity_layer", {})
    print(
        f"  entity layer    : {entity.get('total_assets', 0)} assets, "
        f"{entity.get('mention_resolution_rate_pct', 0)}% mention resolution, "
        f"{entity.get('multi_document_pct', 0)}% multi-document"
    )

    print(f"\nCases: {summary['n_cases']} ({summary['n_errored']} errored)")

    print("\nRetrieval quality (answerable questions)")
    for label, key in (
        ("context recall", "context_recall"),
        ("context precision", "context_precision"),
        ("entity recall", "entity_recall"),
        ("mean citations", "mean_citations"),
        ("mean graph facts", "mean_graph_facts"),
    ):
        print(f"  {label:22} {fmt(summary['retrieval'][key])}")

    print("\nIntent routing")
    print(f"  {'accuracy':22} {fmt(summary['intent_routing']['accuracy'])}")

    print("\nAbstention")
    print(f"  {'recall on unanswerable':22} {fmt(summary['abstention']['recall_on_unanswerable'])}")
    print(f"  {'referral named':22} {fmt(summary['abstention']['routed_referral_rate'])}")
    print(f"  {'false abstention':22} {fmt(summary['abstention']['false_abstention_rate'])}")
    if summary["abstention"]["false_abstention_note"]:
        print(f"    -> {summary['abstention']['false_abstention_note']}")

    print("\nAnswer quality")
    quality = summary["answer_quality"]
    print(f"  {'state':22} {quality['state']}")
    if quality.get("reason"):
        print(f"    -> {quality['reason']}")

    print("\nLatency")
    for label, key in (("p50", "p50_s"), ("p95", "p95_s"), ("max", "max_s")):
        print(f"  {label:22} {fmt(summary['latency'][key])} s")

    print("\nBy category")
    print(
        f"  {'category':16} {'n':>3} {'ctx recall':>11} {'intent':>8} {'abstain':>8} {'p50 s':>8}"
    )
    for name, bucket in sorted(summary["by_category"].items()):
        print(
            f"  {name:16} {bucket['n']:>3} {fmt(bucket['context_recall']):>11} "
            f"{fmt(bucket['intent_accuracy']):>8} {fmt(bucket['abstention_rate']):>8} "
            f"{fmt(bucket['p50_latency_s']):>8}"
        )

    entity = summary.get("entity_extraction", {})
    if entity.get("state") == "measured":
        print("\nEntity extraction (hand-labelled reference set)")
        print(f"  {'precision (micro)':22} {fmt(entity['precision_micro'])}")
        print(f"  {'recall (micro)':22} {fmt(entity['recall_micro'])}")
        print(f"  {'F1 (micro)':22} {fmt(entity['f1_micro'])}")
        print(
            f"  {'tp / fp / fn':22} "
            f"{entity['true_positives']} / {entity['false_positives']} / {entity['false_negatives']}"
        )
        weakest = sorted(
            (d for d in entity["per_document"] if d.get("recall") is not None),
            key=lambda d: d["recall"],
        )[:3]
        for doc in weakest:
            if doc["missed"] or doc["spurious"]:
                print(
                    f"    {doc['doc_title'][:44]:46} recall {fmt(doc['recall'])}"
                    + (f" missed {', '.join(doc['missed'])}" if doc["missed"] else "")
                    + (f" spurious {', '.join(doc['spurious'])}" if doc["spurious"] else "")
                )
    elif entity:
        print(f"\nEntity extraction: {entity.get('state')} — {entity.get('reason', '')}")

    linkage = summary.get("linkage", {})
    if linkage.get("state") == "measured":
        print("\nLinkage completeness")
        for label, key in (
            ("mention resolution", "mention_resolution_rate"),
            ("multi-document assets", "assets_with_documents_pct"),
            ("cross-system assets", "cross_system_pct"),
            ("drawing tags linked", "drawing_linkage_rate"),
        ):
            print(f"  {label:22} {fmt(linkage.get(key))}")
        print(f"  {'review queue open':22} {linkage.get('review_queue_open')}")

    citations = summary.get("citation_validity", {})
    if citations.get("state") == "measured":
        print("\nCitation validity")
        print(
            f"  {'valid / checked':22} "
            f"{citations['citations_valid']} / {citations['citations_checked']}"
        )
        print(f"  {'validity rate':22} {fmt(citations['validity_rate'])}")
        for failure in citations["failures"][:5]:
            print(f"    {failure['case']} {failure['marker']}: {'; '.join(failure['problems'])}")
    elif citations:
        print(f"\nCitation validity: {citations.get('state')} — {citations.get('reason', '')}")

    grounded = summary.get("groundedness", {})
    if grounded.get("state") == "measured":
        print("\nGroundedness")
        print(f"  {'claims verbatim':22} {grounded['claims_verbatim']} / {grounded['claims_total']}")
        print(f"  {'groundedness':22} {fmt(grounded['groundedness'])}")
        print(f"  {'answers with citations':22} {fmt(grounded['cited_answer_rate'])}")
        print(f"  {'answer correctness':22} {grounded['answer_correctness']['state']}")
        print(f"    -> {grounded['answer_correctness']['reason']}")

    compliance = summary.get("compliance_detection", {})
    if compliance.get("state") == "measured":
        print("\nCompliance gap detection")
        print(f"  {'correct / scored':22} {compliance['correct']} / {compliance['pairs_scored']}")
        print(f"  {'accuracy':22} {fmt(compliance['accuracy'])}")
        for mismatch in compliance["mismatches"][:5]:
            print(
                f"    {mismatch['req_id']} on {mismatch['asset_tag']}: "
                f"expected {mismatch['expected']}, got {mismatch['actual']}"
            )
    elif compliance:
        print(f"\nCompliance detection: {compliance.get('state')} — {compliance.get('reason', '')}")

    legs = summary.get("leg_latency", {})
    if legs:
        print("\nLatency by retrieval leg")
        print(f"  {'leg':16} {'p50 ms':>9} {'p95 ms':>9} {'max ms':>9}")
        for leg, stats in sorted(legs.items(), key=lambda kv: -(kv[1]["p95_ms"] or 0)):
            print(
                f"  {leg:16} {stats['p50_ms']:>9.0f} {stats['p95_ms']:>9.0f} "
                f"{stats['max_ms']:>9.0f}"
            )

    failures = [r for r in results if "error" in r]
    if failures:
        print("\nErrored cases")
        for failure in failures:
            print(f"  {failure['id']}: {failure['error']}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--golden", default=str(GOLDEN))
    parser.add_argument("--tag", default="", help="label recorded with this run")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)

    golden_path = Path(args.golden)
    if not golden_path.is_file():
        print(f"No golden set at {golden_path}", file=sys.stderr)
        return 1
    cases = load_cases(golden_path)
    if not cases:
        print("Golden set is empty.", file=sys.stderr)
        return 1

    with httpx.Client(base_url=args.api, timeout=120) as client:
        try:
            client.get("/health/live").raise_for_status()
        except Exception:
            print(f"Cannot reach the API at {args.api}. Is the stack up?", file=sys.stderr)
            return 1

        system = collect_system_state(client)
        print(f"Running {len(cases)} golden question(s) against {args.api}…")
        results = []
        for index, case in enumerate(cases, start=1):
            result = evaluate_case(client, case)
            results.append(result)
            mark = "!" if "error" in result else ("A" if result.get("abstained") else ".")
            print(
                f"  [{index:>2}/{len(cases)}] {mark} {case['id']} {case['category']:<13} "
                f"{case['question'][:56]}"
            )

        # The golden set measures retrieval and abstention. These four measure
        # things it structurally cannot, each against its own reference set.
        print("\nMeasuring entity extraction, linkage, citations and compliance…")
        extra = {
            "entity_extraction": metrics.entity_metrics(client),
            "linkage": metrics.linkage_metrics(client),
            "citation_validity": metrics.citation_metrics(client, results),
            "groundedness": metrics.groundedness_metrics(results),
            "compliance_detection": metrics.compliance_metrics(client),
        }

    summary = summarise(results, system)
    summary.update(extra)
    summary["leg_latency"] = leg_latency_summary(results)
    print_report(summary, system, results)

    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out = RESULTS_DIR / f"{stamp}{'-' + args.tag if args.tag else ''}.json"
        out.write_text(
            json.dumps(
                {
                    "run_at": datetime.now(UTC).isoformat(),
                    "tag": args.tag,
                    "api": args.api,
                    "golden_set": golden_path.name,
                    "system": system,
                    "summary": summary,
                    "results": results,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"Saved: {out.relative_to(REPO_ROOT)}")

    # A harness that fails the build when accuracy is low would be useless on day
    # one. It fails only when the system could not be exercised at all.
    return 1 if summary["n_errored"] == summary["n_cases"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
