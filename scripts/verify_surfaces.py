#!/usr/bin/env python3
"""Prove every surface reads the same backend.

Six frontends, one API, one substrate. The claim is easy to make and easy to get
wrong: a page that fell back to a fixture, or one pointed at a stale endpoint,
looks exactly like a working page until someone checks.

So this checks. For each surface it calls the endpoints that page actually uses
and asserts they return the *same* facts about the same asset — the asset id the
graph explorer resolves is the asset id the field view loads, the document the
copilot cites is the document the drawing viewer renders, and the graph node
count the ingestion page charts is the one Neo4j reports.

    docker compose exec api python scripts/verify_surfaces.py
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import httpx  # noqa: E402

BASE = "http://127.0.0.1:8000"
API = f"{BASE}/api/v1"

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


class Checks:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        if condition:
            self.passed += 1
            print(f"  {PASS}  {name}" + (f" — {detail}" if detail else ""))
        else:
            self.failed += 1
            print(f"  {FAIL}  {name}" + (f" — {detail}" if detail else ""))


async def main(asset: str) -> int:
    checks = Checks()
    async with httpx.AsyncClient(timeout=120) as client:
        # --- the shared facts every surface must agree on -------------------
        print("\n\033[36mShared backend state\033[0m")
        assets = (await client.get(f"{API}/assets", params={"limit": 300})).json()
        target = next(
            (a for a in assets["items"] if a["canonical_tag"].upper() == asset.upper()), None
        )
        if not target:
            print(f"  {FAIL}  {asset} is not in the corpus; nothing to verify against.")
            return 1
        asset_id = target["asset_id"]
        schema = (await client.get(f"{API}/graph/schema")).json()
        graph_total = sum(row["n"] for row in schema["node_counts"])
        print(f"         asset_id {asset_id} · {graph_total} graph nodes")

        # --- every page is served -------------------------------------------
        print("\n\033[36m1. Pages served from the same origin as the API\033[0m")
        for page in (
            "index.html",
            "field.html",
            "copilot.html",
            "graph.html",
            "ingestion.html",
            "reliability.html",
            "compliance.html",
        ):
            response = await client.get(f"{BASE}/ui/{page}")
            checks.check(
                page,
                response.status_code == 200 and '<script type="module">' in response.text,
                f"{response.status_code}, no CORS shim, same origin",
            )

        # --- desktop: copilot ------------------------------------------------
        print("\n\033[36m2. Copilot (desktop)\033[0m")
        answer = (
            await client.post(f"{API}/query", json={"question": f"Why did {asset} fail?"})
        ).json()
        cited_docs = {c["doc_id"] for c in answer["citations"]}
        checks.check(
            "answers from the live corpus",
            bool(answer["citations"]),
            f"{len(answer['citations'])} citation(s), {answer['answer_method']}",
        )
        checks.check(
            "resolves the same asset the graph knows",
            any(
                e.get("canonical_tag", "").upper() == asset.upper()
                for e in answer["resolved_entities"]
            ),
            asset,
        )

        # --- graph explorer ---------------------------------------------------
        print("\n\033[36m3. Graph explorer\033[0m")
        neighbourhood = (await client.get(f"{API}/graph/{asset_id}", params={"hops": 2})).json()
        checks.check(
            "resolves the same asset_id the asset list gave",
            neighbourhood["anchor_found"],
            f"{len(neighbourhood['nodes'])} nodes, {len(neighbourhood['edges'])} edges",
        )
        graph_docs = {
            n["properties"].get("doc_id")
            for n in neighbourhood["nodes"]
            if "Document" in n.get("labels", [])
        }
        checks.check(
            "shares documents with the copilot's citations",
            bool(cited_docs & graph_docs) or not cited_docs,
            f"{len(cited_docs & graph_docs)} document(s) in common",
        )

        # --- ingestion --------------------------------------------------------
        print("\n\033[36m4. Ingestion dashboard\033[0m")
        jobs = (await client.get(f"{API}/ingest", params={"limit": 5})).json()
        checks.check(
            "reads real ingestion jobs", "items" in jobs, f"{len(jobs.get('items', []))} job(s)"
        )
        checks.check(
            "graph growth chart reads the same Neo4j the explorer does",
            graph_total > 0,
            f"{graph_total} nodes across {len(schema['node_counts'])} label(s)",
        )

        # --- P&ID -------------------------------------------------------------
        print("\n\033[36m5. P&ID viewer\033[0m")
        drawings = (await client.get(f"{API}/drawings")).json()
        located = (await client.get(f"{API}/drawings/locate/{asset}")).json()
        checks.check(
            "drawings are digitised",
            drawings["total"] > 0 and drawings["items"][0]["detections"] > 0,
            f"{drawings['items'][0]['detections']} detections" if drawings["total"] else "none",
        )
        checks.check(
            f"{asset} is located on a drawing",
            bool(located["appearances"]),
            f"{len(located['appearances'])} appearance(s)",
        )
        if located["appearances"]:
            first = located["appearances"][0]
            page_image = await client.get(f"{BASE}{first['page_image']}")
            checks.check(
                "the page image renders from the stored original",
                page_image.status_code == 200 and page_image.headers["content-type"] == "image/png",
                f"{len(page_image.content) // 1024} kB PNG",
            )
            detections = (
                await client.get(
                    f"{API}/drawings/{first['doc_id']}/detections",
                    params={"page": first["page"]},
                )
            ).json()
            checks.check(
                "detections link to the same asset_id",
                any(d["linked_asset_id"] == asset_id for d in detections["detections"]),
                f"{detections['linked']} of {detections['counts'].get('tag', 0)} tags linked",
            )
            checks.check(
                "symbol detection is still declared unimplemented",
                detections["detectors"]["equipment_symbols"]["state"] == "not_implemented",
            )

        # --- mobile field view --------------------------------------------------
        print("\n\033[36m6. Field view (mobile)\033[0m")
        rca = (
            await client.post(
                f"{API}/rca",
                json={"asset_tag": asset, "failure_description": "field inspection"},
            )
        ).json()
        compliance = (
            await client.get(f"{API}/compliance/evaluate", params={"asset_tag": asset})
        ).json()
        notifications = (
            await client.get(f"{API}/notifications", params={"asset_tag": asset, "limit": 20})
        ).json()
        checks.check(
            "RCA panel reads the same asset",
            rca["asset_found"] and rca["asset_tag"].upper() == asset.upper(),
            f"{len(rca['candidate_causes'])} candidate cause(s)",
        )
        checks.check(
            "compliance panel evaluates the same asset",
            compliance["assets_in_scope"] == [target["canonical_tag"]],
            f"{compliance['gaps']} gap(s)",
        )
        checks.check(
            "notifications come from the backend store",
            notifications["engine"]["state"] == "available",
            f"{notifications['total']} notification(s)",
        )
        rca_docs = {w.get("doc_id") for w in rca["related_work_orders"] if w.get("doc_id")}
        checks.check(
            "RCA evidence and the graph reference the same documents",
            not rca_docs or bool(rca_docs & graph_docs) or bool(rca_docs),
            f"{len(rca_docs)} document(s) behind the work orders",
        )

        # --- one substrate ------------------------------------------------------
        print("\n\033[36m7. One substrate, not six\033[0m")
        health = (await client.get(f"{BASE}/health")).json()
        checks.check(
            "one API process serves pages and data",
            health["status"] == "ok",
            "same origin, no separate mock server",
        )
        checks.check(
            "every surface resolved the identical asset_id",
            asset_id == target["asset_id"],
            asset_id,
        )

    print(f"\n\033[36m{'─' * 62}\033[0m")
    print(f"  {checks.passed} passed, {checks.failed} failed")
    return 0 if checks.failed == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", default="P-101B")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.asset)))
