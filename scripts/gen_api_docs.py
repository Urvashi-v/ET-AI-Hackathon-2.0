#!/usr/bin/env python3
"""Generate `docs/api.md` from the application's own OpenAPI schema.

Hand-written endpoint tables drift. They drift quietly, and the reader has no
way to tell which half of the document is still true. This script builds the
reference section from the schema FastAPI derives from the route signatures, so
the only way for the table to be wrong is for the API itself to be wrong.

The prose is written here rather than in the output file, because the output is
overwritten wholesale on every run.

    python scripts/gen_api_docs.py                     # against the running stack
    python scripts/gen_api_docs.py --offline           # import the app directly
    python scripts/gen_api_docs.py --check             # fail if the file is stale
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "docs" / "api.md"
DEFAULT_BASE = "http://localhost:8000"

# Ordered so the document reads as a walk through the system rather than
# alphabetically. A tag missing from here still appears, at the end.
TAG_ORDER = [
    "health",
    "ingestion",
    "documents",
    "drawings",
    "assets",
    "knowledge graph",
    "copilot",
    "reliability",
    "compliance",
    "proactive",
    "events",
    "feedback",
]

TAG_NOTES = {
    "health": "Liveness, readiness and a full dependency report. `/health` probes every "
    "store with a real query and reports counts, not a hard-coded `ok`.",
    "ingestion": "Upload or enqueue paths. Ingestion is asynchronous: the POST returns a "
    "job id, and `GET /api/v1/ingest/{job_id}` carries a per-stage report including "
    "the stages that could not run and why.",
    "documents": "The corpus, its revision lineage, and the evidence behind a citation — "
    "the chunk text, the rendered source page, and the original file as ingested.",
    "drawings": "P&ID digitisation output: detected tags, ISA instrument bubbles and pipe "
    "runs, with the page geometry each was found at.",
    "assets": "Canonical equipment after entity resolution, plus the dossier that the "
    "field and reliability surfaces are built on.",
    "knowledge graph": "The neighbourhood around an asset, the ontology as installed in "
    "the running database, and the evidence behind any single edge.",
    "copilot": "The read path. `POST /api/v1/query` runs intent → lexical ‖ dense ‖ graph → "
    "RRF → rerank → assemble → compose → verify → confidence, and returns an answer "
    "with citations or an abstention with a referral.",
    "reliability": "Root cause analysis over the graph, and precedent matching against the "
    "incident register.",
    "compliance": "Requirements loaded from source material, evaluated against stored "
    "evidence, with a verdict per requirement and a coverage figure over the "
    "decidable subset.",
    "proactive": "Notifications raised by the matchers, and the endpoint that evaluates an "
    "event against them.",
    "events": "The system event bus, as a list and as a Server-Sent Events stream.",
    "feedback": "Thumbs and corrections against a logged query.",
}

PREAMBLE = """# API reference

The dashboard talks to this API and nothing else. There is no fixture mode, no
mock server and no second code path for the demo — every number on every screen
arrives through one of the endpoints below.

Base URL `http://localhost:8000`. Interactive docs at `/docs`, the raw schema at
`/openapi.json`. Everything under `/api/v1`; health probes are unversioned
because orchestrators expect them at a fixed path.

**This file is generated** by `python scripts/gen_api_docs.py` from the schema
FastAPI derives from the route signatures. Edit the script, not the output.

## Conventions

### One error envelope

Every failure — validation, not-found, an unconfigured capability, an
unreachable store, an unhandled exception — returns the same shape:

```json
{
  "error": {
    "code": "not_found",
    "message": "No document 9f2c...",
    "detail": {}
  }
}
```

`code` is a stable identifier a client may branch on; `message` is for a human;
`detail` carries structure where there is any (the offending fields for a
validation error, the missing environment variables for an unconfigured
capability). Stack traces, connection strings and file paths never cross the
boundary.

| `code` | Status | Raised when |
|---|---|---|
| `validation_error` | 400 | the request did not match the contract; `detail.fields` names each one |
| `file_validation_error` | 400 | an uploaded file was rejected — wrong type, too large, or empty |
| `not_found` | 404 | the resource does not exist; `detail.hint` suggests what to try |
| `unsupported_media_type` | 415 | e.g. asking for a rendered page image of a Markdown file |
| `conflict` | 409 | the request contradicts current state |
| `rate_limited` | 429 | the upload limiter rejected the request |
| `capability_not_configured` | 503 | a stage cannot run; `detail.required_env` names the variables |
| `dependency_unavailable` | 503 | Postgres, Neo4j or Redis is unreachable |
| `internal_error` | 500 | unhandled; `detail.request_id` is the log correlation id |

### Request ids

Every response carries `x-request-id`. Send your own and it is honoured;
otherwise one is generated. It appears on every structured log line for that
request, and the dashboard prints it in its error state so a screenshot is
enough to find the logs.

### Provenance is part of the contract

Any endpoint returning a displayable operational value returns a `data_class`
alongside it: `real_source_document`, `synthetic_test_data`, `model_derived`,
`calculated_metric`, `human_attested` or `reference_taxonomy`. The frontend
renders it as a badge and never infers it. A value that arrives without one is
rendered "Unlabelled" rather than quietly as real.

### Capability status instead of silence

A stage that cannot run reports itself rather than being skipped. The shape is
the same everywhere:

```json
{
  "capability": "generation",
  "state": "provider_not_configured",
  "detail": "No LLM provider is configured; the extractive composer was used.",
  "required_env": ["LLM_PROVIDER", "ANTHROPIC_API_KEY"]
}
```

`state` is one of `available`, `disabled`, `provider_not_configured`,
`not_implemented`, `error`.

### Authentication

**There is none, and this is a real limitation rather than an oversight.** The
system is single-tenant and bound to localhost by the compose file. A `role`
field is accepted on queries and shapes retrieval and phrasing, but it is a
personalisation input, not an authorisation boundary — nothing is withheld from
a caller who claims a different role. See `docs/security.md` for what is and is
not enforced today, and what deploying this outside a workstation would require.

### Streaming

Two endpoints are Server-Sent Events rather than JSON: `POST /api/v1/query/stream`
emits each retrieval stage as it completes, and `GET /api/v1/events/stream`
carries the system event bus. Both send `event:` and `data:` lines; the browser's
native `EventSource` is used for the second and a streamed `fetch` for the first,
because `EventSource` cannot issue a POST.

"""

FOOTER = """
## What the schema does not yet cover

{modelled} of {total} operations declare a Pydantic response model, so those
appear in `/openapi.json` with a full schema for their success payload. The
remainder return `dict[str, Any]` and are documented by their summary and their
tests rather than by a generated schema. That is a genuine gap in the contract:
a client generator pointed at this API today produces typed request models and
untyped responses for {untyped} of the {total} operations.

It is listed here rather than papered over with models written to match whatever
the handler happens to return — a response model that is not enforced by the
handler's own types is a second source of truth, and the wrong one to trust.

## Related

* `docs/architecture.md` — the seven layers and the three paths
* `docs/retrieval.md` — what the copilot endpoints actually run
* `docs/security.md` — input validation, upload limits, and the missing auth layer
* `docs/evaluation.md` — how the numbers these endpoints report are measured
"""


def load_spec(base: str, offline: bool) -> dict[str, Any]:
    if offline:
        sys.path.insert(0, str(REPO_ROOT))
        from services.api.main import app  # noqa: PLC0415

        return app.openapi()
    with urllib.request.urlopen(f"{base}/openapi.json", timeout=30) as response:
        return json.load(response)


def parameters_for(op: dict[str, Any]) -> str:
    parts = []
    for param in op.get("parameters", []):
        name = param["name"]
        schema = param.get("schema", {})
        bounds = []
        if "minimum" in schema:
            bounds.append(f"≥{schema['minimum']}")
        if "maximum" in schema:
            bounds.append(f"≤{schema['maximum']}")
        if "maxLength" in schema:
            bounds.append(f"≤{schema['maxLength']} chars")
        suffix = f" ({', '.join(bounds)})" if bounds else ""
        marker = "**" if param.get("required") else ""
        parts.append(f"{marker}`{name}`{marker}{suffix}")
    if op.get("requestBody"):
        parts.append("_body_")
    return ", ".join(parts) or "—"


def render(spec: dict[str, Any]) -> str:
    by_tag: dict[str, list[tuple[str, str, dict[str, Any]]]] = {}
    total = modelled = 0
    for path, operations in spec["paths"].items():
        for method, op in operations.items():
            if method not in ("get", "post", "put", "patch", "delete"):
                continue
            total += 1
            success = op.get("responses", {}).get("200", {})
            if "$ref" in json.dumps(success):
                modelled += 1
            tag = (op.get("tags") or ["other"])[0]
            by_tag.setdefault(tag, []).append((method.upper(), path, op))

    order = [t for t in TAG_ORDER if t in by_tag]
    order += sorted(t for t in by_tag if t not in TAG_ORDER)

    out = [PREAMBLE, "## Endpoints\n"]
    for tag in order:
        out.append(f"### {tag}\n")
        if tag in TAG_NOTES:
            out.append(TAG_NOTES[tag] + "\n")
        out.append("| Method | Path | Parameters | Purpose |")
        out.append("|---|---|---|---|")
        for method, path, op in sorted(by_tag[tag], key=lambda r: (r[1], r[0])):
            summary = op.get("summary", "").replace("|", "\\|")
            out.append(f"| `{method}` | `{path}` | {parameters_for(op)} | {summary} |")
        out.append("")

    out.append(FOOTER.format(modelled=modelled, total=total, untyped=total - modelled))
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--offline", action="store_true", help="import the app instead of HTTP")
    parser.add_argument("--check", action="store_true", help="exit 1 if docs/api.md is stale")
    args = parser.parse_args()

    try:
        spec = load_spec(args.base, args.offline)
    except Exception as exc:
        print(f"could not load the OpenAPI schema: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"is the stack running? try: {args.base}/openapi.json", file=sys.stderr)
        return 2

    rendered = render(spec)

    if args.check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        if current != rendered:
            print("docs/api.md is stale; run: python scripts/gen_api_docs.py", file=sys.stderr)
            subprocess.run(["git", "--no-pager", "diff", "--stat", str(OUTPUT)], check=False)
            return 1
        print("docs/api.md is up to date")
        return 0

    # newline="\n" because the document id of anything generated on this repo is
    # a hash over bytes, and platform newline translation has already produced
    # one duplicate-document incident here.
    OUTPUT.write_text(rendered, encoding="utf-8", newline="\n")
    operations = sum(
        1
        for ops in spec["paths"].values()
        for m in ops
        if m in ("get", "post", "put", "patch", "delete")
    )
    print(f"wrote {OUTPUT.relative_to(REPO_ROOT)} — {operations} operations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
