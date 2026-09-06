#!/usr/bin/env python3
"""Submit a directory to the running ingestion pipeline and follow the job.

Talks to the real API over HTTP -- it does not import the pipeline and run it
in-process, so what this script exercises is exactly what the dashboard
exercises.

``--data-class`` is mandatory and has no default. Every document must declare
whether it is a real source document or clearly-labelled test data, and the
person running the command is the one who knows.

Usage::

    python scripts/ingest_dir.py data/synthetic/generated --data-class synthetic_test_data
    python scripts/ingest_dir.py data/corpus --data-class real_source_document --wait
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
VALID_CLASSES = ("real_source_document", "synthetic_test_data")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="directory or file, relative to the repository root")
    parser.add_argument(
        "--data-class",
        required=True,
        choices=VALID_CLASSES,
        help="Provenance class of these documents. Required -- there is no default.",
    )
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument(
        "--source-system", default=None, help="Origin label used for cross-system linkage metrics"
    )
    parser.add_argument("--wait", action="store_true", help="poll until the job finishes")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args(argv)

    source_system = args.source_system or (
        "synthetic_cmms" if args.data_class == "synthetic_test_data" else "corpus"
    )

    target = Path(args.path)
    if not target.is_absolute():
        target = REPO_ROOT / args.path
    if not target.exists():
        print(f"No such path: {target}", file=sys.stderr)
        return 1

    # The API resolves paths relative to its own repository root, which inside
    # the container is /app. Send the repo-relative form.
    try:
        relative = target.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        print(
            f"Path must be inside the repository ({REPO_ROOT}); the API only ingests "
            "from permitted roots.",
            file=sys.stderr,
        )
        return 1

    with httpx.Client(base_url=args.api, timeout=120) as client:
        try:
            response = client.post(
                "/api/v1/ingest/paths",
                json={
                    "source": "filesystem",
                    "paths": [relative],
                    "data_class": args.data_class,
                    "source_system": source_system,
                    "recursive": True,
                    "submitted_by": "scripts/ingest_dir.py",
                },
            )
        except httpx.ConnectError:
            print(f"Cannot reach the API at {args.api}. Is the stack up?", file=sys.stderr)
            return 1

        if response.status_code >= 400:
            print(f"Ingestion rejected ({response.status_code}): {response.text}", file=sys.stderr)
            return 1

        body = response.json()
        print(
            f"Job {body['job_id']}: {body['accepted']} accepted, "
            f"{body['rejected']} rejected, {body['duplicates']} duplicate(s)"
        )
        for entry in body["files"]:
            mark = "+" if entry["accepted"] else "-"
            reason = f"  ({entry['reason']})" if entry.get("reason") else ""
            print(f"  {mark} {entry['filename']}{reason}")

        if not args.wait:
            print(f"\nPoll: {args.api}{body['poll']}")
            return 0

        deadline = time.time() + args.timeout
        last_stage = None
        while time.time() < deadline:
            job = client.get(f"/api/v1/ingest/{body['job_id']}").json()
            if job.get("stage") and job["stage"] != last_stage:
                last_stage = job["stage"]
                print(f"  ... {last_stage}")
            if job["status"] in ("succeeded", "failed", "partial", "cancelled"):
                print(f"\nStatus: {job['status']}")
                print(f"  documents processed : {job['processed']}")
                print(f"  duplicates skipped  : {job['skipped_duplicates']}")
                print(f"  failed              : {job['failed']}")
                print(f"  chunks created      : {job['chunks_created']}")
                print(f"  mentions created    : {job['mentions_created']}")
                print(f"  new assets          : {job['entities_created']}")
                print(f"  graph edges         : {job['edges_created']}")
                if job.get("stage_reports"):
                    print("\n  Stage report:")
                    for stage in job["stage_reports"]:
                        detail = f" -- {stage['detail']}" if stage.get("detail") else ""
                        print(f"    {stage['state']:26} {stage['stage']:16}{detail}")
                        if stage.get("required_env"):
                            print(f"      requires: {', '.join(stage['required_env'])}")
                if job.get("review_queue"):
                    print(f"\n  {len(job['review_queue'])} item(s) flagged for human review.")
                if job.get("error"):
                    print(f"\n  Error: {job['error']}", file=sys.stderr)
                return 0 if job["status"] in ("succeeded", "partial") else 1
            time.sleep(1.0)

        print(f"Timed out after {args.timeout}s waiting for the job.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
