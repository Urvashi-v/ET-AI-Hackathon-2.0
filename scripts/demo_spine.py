#!/usr/bin/env python3
"""One asset, through every capability, against the live stack.

    ingestion -> graph -> copilot -> RCA -> compliance -> lessons learned
                                        -> proactive notification

Every figure printed is fetched from the running API at the moment it is
printed. Nothing is cached, precomputed or hard-coded, so a stage that has
stopped working shows up here as a smaller number rather than as the same
reassuring output.

    docker compose exec api python scripts/demo_spine.py
    docker compose exec api python scripts/demo_spine.py --asset P-101B
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import httpx  # noqa: E402

BASE = "http://127.0.0.1:8000/api/v1"


def rule(title: str) -> None:
    print(f"\n\033[36m{'─' * 3} {title} {'─' * max(3, 66 - len(title))}\033[0m")


def bullet(label: str, value: object) -> None:
    print(f"  {label:<34} {value}")


async def main(asset: str, question: str) -> int:
    async with httpx.AsyncClient(timeout=120) as client:
        # --- 1. ingestion -----------------------------------------------------
        rule(f"1. INGESTION — what {asset} is built from")
        docs = (await client.get(f"{BASE}/documents", params={"limit": 100})).json()
        assets = (await client.get(f"{BASE}/assets", params={"limit": 300})).json()
        target = next(
            (a for a in assets["items"] if a["canonical_tag"].upper() == asset.upper()), None
        )
        if target is None:
            print(f"  {asset} is not in the corpus. Ingest documents mentioning it first.")
            return 1
        bullet("documents in corpus", docs["total"])
        bullet("current revisions", sum(1 for d in docs["items"] if d["is_current"]))
        bullet("superseded", sum(1 for d in docs["items"] if not d["is_current"]))
        bullet(f"{asset} mentions / documents", f"{target['mention_count']} / {target['document_count']}")
        bullet("source systems", target["source_system_count"])

        # --- 2. graph ---------------------------------------------------------
        rule(f"2. GRAPH — what {asset} is connected to")
        neighbourhood = (
            await client.get(f"{BASE}/graph/{target['asset_id']}", params={"hops": 2})
        ).json()
        bullet("nodes in 2-hop neighbourhood", len(neighbourhood.get("nodes", [])))
        bullet("edges", len(neighbourhood.get("edges", [])))
        # Neo4j returns every label on a node; the most specific one is last and
        # is the one worth counting.
        labels: dict[str, int] = {}
        for node in neighbourhood.get("nodes", []):
            names = node.get("labels") or ["?"]
            key = names[-1]
            labels[key] = labels.get(key, 0) + 1
        bullet("node kinds", ", ".join(f"{k}x{v}" for k, v in sorted(labels.items())))

        # --- 3. copilot -------------------------------------------------------
        rule("3. COPILOT — a grounded, cited answer")
        answer = (await client.post(f"{BASE}/query", json={"question": question})).json()
        bullet("question", question)
        bullet("intent", f"{answer['intent']} ({answer['intent_confidence']:.2f})")
        bullet("answer method", answer["answer_method"])
        bullet("confidence", f"{answer['confidence']['score']:.3f} {answer['confidence']['mode']}")
        bullet("retrieval legs", ", ".join(answer["retrieval_sources"]))
        bullet("citations / claims", f"{len(answer['citations'])} / {len(answer['claims'])}")
        bullet("latency", f"{answer['latency_ms']} ms")
        if answer.get("answer"):
            print(f"\n    {answer['answer'][:300]}")

        # --- 4. RCA -----------------------------------------------------------
        rule("4. RCA — candidate causes ranked from recorded evidence")
        rca = (
            await client.post(
                f"{BASE}/rca",
                json={"asset_tag": asset, "failure_description": question},
            )
        ).json()
        bullet("evidence gathered", rca["evidence_gathered"])
        bullet("MTBF (days)", rca["reliability_metrics"].get("mtbf_days", "insufficient_data"))
        bullet("overall confidence", rca["overall_confidence"])
        if rca["causal_analysis_abstained"]:
            bullet("ABSTAINED", rca["causal_analysis_reason"])
        for i, cause in enumerate(rca["candidate_causes"][:3], start=1):
            print(
                f"    {i}. {cause['label'][:52]:54} "
                f"{cause['occurrences']} record(s), self {cause['on_this_asset']}"
                f"/sibling {cause['on_siblings']}"
            )
        for action in rca["open_corrective_actions"][:3]:
            print(
                f"       open: {action['capa_id']} ({action['owner'] or 'unassigned'}) "
                f"raised against {action['raised_against']}"
            )

        # --- 5. compliance ----------------------------------------------------
        rule("5. COMPLIANCE — obligations decided against stored evidence")
        compliance = (
            await client.get(f"{BASE}/compliance/evaluate", params={"asset_tag": asset})
        ).json()
        bullet("requirements loaded", compliance["requirements_loaded"])
        bullet("satisfied", compliance["satisfied"])
        bullet("gaps", compliance["gaps"])
        bullet("needs verification", compliance["needs_verification"])
        bullet("not evaluable", compliance["not_evaluable"])
        bullet(
            "coverage of decidable",
            f"{compliance['coverage_pct_of_decidable']}% of {compliance['decidable_count']}",
        )
        bullet("requirement text provenance", compliance["requirement_provenance"])
        for finding in [f for f in compliance["findings"] if f["state"] == "gap"][:3]:
            print(f"    gap: {finding['req_id']:18} {finding['reason'][:70]}")

        # --- 6. lessons learned -----------------------------------------------
        rule("6. LESSONS LEARNED — has this happened before?")
        lessons = (
            await client.post(
                f"{BASE}/lessons", json={"description": question, "asset_tag": asset}
            )
        ).json()
        bullet("incidents compared", lessons["incidents_considered"])
        bullet("semantic matching", lessons["semantic_matching"]["state"])
        for match in lessons["matches"]:
            open_actions = [a for a in match["corrective_actions"] if a.get("is_open")]
            print(
                f"    {match['incident_id']:14} {match['similarity']:.2f} "
                f"[{match['asset_tag']}] {match['occurred_on']} "
                f"— {len(open_actions)} open action(s)"
            )
            for reason in match["match_reasons"][:2]:
                print(f"       why: {reason[:78]}")

        # --- 7. proactive -----------------------------------------------------
        rule("7. PROACTIVE — what the system would say unprompted")
        evaluated = (
            await client.post(
                f"{BASE}/notifications/evaluate",
                json={
                    "event_type": "demo.walkthrough",
                    "asset_tag": asset,
                    "description": question,
                    "dry_run": True,
                },
            )
        ).json()
        bullet("candidates", len(evaluated["candidates"]))
        for candidate in evaluated["candidates"]:
            print(f"    [{candidate['severity']:6}] {candidate['title'][:62]}")
            print(f"             -> {candidate['audience_role']}")

        rule("SPINE COMPLETE")
        print(
            f"  {asset} traversed ingestion, graph, copilot, RCA, compliance, lessons\n"
            "  and the proactive path — one asset, one substrate, seven capabilities."
        )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", default="P-101B")
    parser.add_argument(
        "--question", default="Why did the mechanical seal on P-101B fail after startup?"
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.asset, args.question)))
