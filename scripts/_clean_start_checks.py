#!/usr/bin/env python3
"""Assertions for the clean-room startup test.

Split out of `clean_start_test.sh` because the checks were inline `python -c`
strings inside double-quoted shell, and the escaping defeated itself: Python 3.12
rejects a backslash-escaped quote inside an f-string expression, so three checks
were syntax errors that the harness reported as failures of the *system*. A test
that fails because of its own quoting is worse than no test — it teaches you to
distrust the output.

Each check prints what it found and exits 0 or 1.

    python scripts/_clean_start_checks.py health
"""

from __future__ import annotations

import json
import sys
import urllib.request

BASE = "http://localhost:8000"


def get(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE}{path}", timeout=60) as response:
        return json.load(response)


def post(path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_health() -> bool:
    status = get("/health")["status"]
    print(f"  health: {status}")
    return status == "ok"


def check_dependencies() -> bool:
    dependencies = get("/health")["dependencies"]
    for name, value in dependencies.items():
        print(f"  {name:10} {value.get('status')}")
    return all(v.get("status") in ("up", "ok") for v in dependencies.values())


def check_documents() -> bool:
    total = get("/api/v1/documents?limit=200")["total"]
    print(f"  {total} document(s)")
    return total >= 10


def check_entities() -> bool:
    stats = get("/api/v1/assets/stats")
    print(
        f"  {stats['total_assets']} assets, {stats['total_mentions']} mentions, "
        f"{stats['mention_resolution_rate_pct']}% resolved"
    )
    return stats["total_assets"] >= 10


def check_drawings() -> bool:
    items = get("/api/v1/drawings")["items"]
    if not items:
        print("  no drawings digitised")
        return False
    first = items[0]
    print(f"  {first['detections']} detections, {first['linked']} linked to assets")
    return first["detections"] > 0


def check_incidents() -> bool:
    total = get("/api/v1/lessons/incidents")["total"]
    print(f"  {total} incident record(s)")
    return total >= 1


def check_answer() -> bool:
    payload = post("/api/v1/query", {"question": "Why did the mechanical seal on P-101B fail?"})
    confidence = payload["confidence"]
    print(
        f"  {confidence['mode']} conf={confidence['score']:.2f} "
        f"{len(payload['citations'])} citations via {payload['retrieval_sources']}"
    )
    return bool(payload["citations"])


def check_abstains() -> bool:
    payload = post(
        "/api/v1/query", {"question": "What is the vibration alarm setpoint for P-999Z?"}
    )
    mode = payload["confidence"]["mode"]
    print(f"  {mode}")
    return "ABSTAIN" in mode


def check_rca() -> bool:
    payload = post(
        "/api/v1/rca",
        {"asset_tag": "P-101B", "failure_description": "mechanical seal failure"},
    )
    print(
        f"  {len(payload['candidate_causes'])} candidate cause(s), "
        f"{len(payload['open_corrective_actions'])} open action(s)"
    )
    return bool(payload["candidate_causes"])


def check_compliance() -> bool:
    payload = get("/api/v1/compliance/evaluate?asset_tag=V-102")
    print(
        f"  {payload['satisfied']} satisfied, {payload['gaps']} gaps, "
        f"{payload['not_evaluable']} not evaluable, "
        f"{payload['requirements_loaded']} requirements loaded"
    )
    return payload["requirements_loaded"] > 0


CHECKS = {
    "health": check_health,
    "dependencies": check_dependencies,
    "documents": check_documents,
    "entities": check_entities,
    "drawings": check_drawings,
    "incidents": check_incidents,
    "answer": check_answer,
    "abstains": check_abstains,
    "rca": check_rca,
    "compliance": check_compliance,
}


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in CHECKS:
        print(f"usage: {sys.argv[0]} <{'|'.join(CHECKS)}>", file=sys.stderr)
        return 2
    try:
        return 0 if CHECKS[sys.argv[1]]() else 1
    except Exception as exc:
        print(f"  check raised {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
