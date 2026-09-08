# API reference

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


## Endpoints

### health

Liveness, readiness and a full dependency report. `/health` probes every store with a real query and reports counts, not a hard-coded `ok`.

| Method | Path | Parameters | Purpose |
|---|---|---|---|
| `GET` | `/health` | — | Full system report |
| `GET` | `/health/live` | — | Liveness probe |
| `GET` | `/health/ready` | — | Readiness probe |

### ingestion

Upload or enqueue paths. Ingestion is asynchronous: the POST returns a job id, and `GET /api/v1/ingest/{job_id}` carries a per-stage report including the stages that could not run and why.

| Method | Path | Parameters | Purpose |
|---|---|---|---|
| `GET` | `/api/v1/ingest` | `limit` (≥1, ≤100) | Recent ingestion jobs |
| `POST` | `/api/v1/ingest` | _body_ | Submit documents for ingestion (multipart upload) |
| `POST` | `/api/v1/ingest/paths` | _body_ | Submit documents already present on a readable path |
| `GET` | `/api/v1/ingest/{job_id}` | **`job_id`** | Ingestion job progress |

### documents

The corpus, its revision lineage, and the evidence behind a citation — the chunk text, the rendered source page, and the original file as ingested.

| Method | Path | Parameters | Purpose |
|---|---|---|---|
| `GET` | `/api/v1/documents` | `limit` (≥1, ≤200), `offset` (≥0), `doc_type`, `current_only` | Documents in the corpus |
| `GET` | `/api/v1/documents/{doc_id}` | **`doc_id`** | One document, with its revision lineage |
| `GET` | `/api/v1/documents/{doc_id}/chunks/{chunk_id}` | **`doc_id`**, **`chunk_id`** | One extracted span, in full |
| `GET` | `/api/v1/documents/{doc_id}/page/{page}.png` | **`doc_id`**, **`page`** (≥1) | Rendered source page |
| `GET` | `/api/v1/documents/{doc_id}/raw` | **`doc_id`** | The original file as ingested |

### drawings

P&ID digitisation output: detected tags, ISA instrument bubbles and pipe runs, with the page geometry each was found at.

| Method | Path | Parameters | Purpose |
|---|---|---|---|
| `GET` | `/api/v1/drawings` | — | Documents with digitised drawings |
| `GET` | `/api/v1/drawings/locate/{asset_tag}` | **`asset_tag`** | Where an asset appears on any drawing |
| `GET` | `/api/v1/drawings/{doc_id}/detections` | **`doc_id`**, `page` (≥1), `kind`, `min_confidence` (≥0.0, ≤1.0) | Everything detected on a drawing page |

### assets

Canonical equipment after entity resolution, plus the dossier that the field and reliability surfaces are built on.

| Method | Path | Parameters | Purpose |
|---|---|---|---|
| `GET` | `/api/v1/assets` | `q`, `class_code`, `tag_kind`, `multi_document_only`, `limit` (≥1, ≤500), `offset` (≥0) | List canonical assets |
| `GET` | `/api/v1/assets/stats` | — | Entity-layer statistics (calculated metrics) |
| `GET` | `/api/v1/assets/{asset_id}` | **`asset_id`** | Full asset dossier |

### knowledge graph

The neighbourhood around an asset, the ontology as installed in the running database, and the evidence behind any single edge.

| Method | Path | Parameters | Purpose |
|---|---|---|---|
| `GET` | `/api/v1/graph/schema` | — | Ontology as installed in the running database |
| `GET` | `/api/v1/graph/{asset_id}` | **`asset_id`**, `hops` (≥1, ≤3), `edges`, `as_of`, `limit` (≥10, ≤1000) | Asset neighbourhood |
| `GET` | `/api/v1/graph/{asset_id}/evidence/{edge_id}` | **`asset_id`**, **`edge_id`** | Evidence behind one graph edge |

### copilot

The read path. `POST /api/v1/query` runs intent → lexical ‖ dense ‖ graph → RRF → rerank → assemble → compose → verify → confidence, and returns an answer with citations or an abstention with a referral.

| Method | Path | Parameters | Purpose |
|---|---|---|---|
| `POST` | `/api/v1/query` | _body_ | Ask the copilot |
| `GET` | `/api/v1/query/health` | — | Retrieval subsystem status |
| `POST` | `/api/v1/query/stream` | _body_ | Ask the copilot (Server-Sent Events) |
| `GET` | `/api/v1/query/{query_id}` | **`query_id`** | Replay a logged query |

### reliability

Root cause analysis over the graph, and precedent matching against the incident register.

| Method | Path | Parameters | Purpose |
|---|---|---|---|
| `POST` | `/api/v1/lessons` | _body_ | Find historical incidents resembling a new event |
| `GET` | `/api/v1/lessons/incidents` | `asset_tag`, `limit` (≥1, ≤200) | The incident register |
| `GET` | `/api/v1/lessons/incidents/{incident_id}` | **`incident_id`** | One incident, in full |
| `POST` | `/api/v1/rca` | _body_ | Root cause analysis for a failure event |

### compliance

Requirements loaded from source material, evaluated against stored evidence, with a verdict per requirement and a coverage figure over the decidable subset.

| Method | Path | Parameters | Purpose |
|---|---|---|---|
| `POST` | `/api/v1/compliance` | _body_ | Scan a scope for compliance gaps |
| `GET` | `/api/v1/compliance/evaluate` | `asset_tag`, `standard` | Evaluate requirements against stored evidence |
| `POST` | `/api/v1/compliance/evidence-package` | _body_ | Generate an audit evidence package |
| `GET` | `/api/v1/compliance/requirements` | `standard`, `limit` (≥1, ≤1000) | Loaded requirements with their provenance |
| `GET` | `/api/v1/compliance/requirements/{req_id}` | **`req_id`** | One requirement, with its provenance |

### proactive

Notifications raised by the matchers, and the endpoint that evaluates an event against them.

| Method | Path | Parameters | Purpose |
|---|---|---|---|
| `GET` | `/api/v1/notifications` | `asset_tag`, `role`, `unacknowledged_only`, `limit` (≥1, ≤200) | Proactive notifications |
| `POST` | `/api/v1/notifications/evaluate` | _body_ | Run the proactive matchers against an event |
| `POST` | `/api/v1/notifications/{notification_id}/acknowledge` | **`notification_id`** | Acknowledge a notification |

### events

The system event bus, as a list and as a Server-Sent Events stream.

| Method | Path | Parameters | Purpose |
|---|---|---|---|
| `GET` | `/api/v1/events` | `limit` (≥1, ≤200) | Recent system events |
| `GET` | `/api/v1/events/stream` | `replay` (≥0, ≤200) | Live event stream (Server-Sent Events) |

### feedback

Thumbs and corrections against a logged query.

| Method | Path | Parameters | Purpose |
|---|---|---|---|
| `GET` | `/api/v1/feedback` | — | Feedback summary |
| `POST` | `/api/v1/feedback` | _body_ | Submit feedback on an answer or a notification |


## What the schema does not yet cover

10 of 41 operations declare a Pydantic response model, so those
appear in `/openapi.json` with a full schema for their success payload. The
remainder return `dict[str, Any]` and are documented by their summary and their
tests rather than by a generated schema. That is a genuine gap in the contract:
a client generator pointed at this API today produces typed request models and
untyped responses for 31 of the 41 operations.

It is listed here rather than papered over with models written to match whatever
the handler happens to return — a response model that is not enforced by the
handler's own types is a second source of truth, and the wrong one to trust.

## Related

* `docs/architecture.md` — the seven layers and the three paths
* `docs/retrieval.md` — what the copilot endpoints actually run
* `docs/security.md` — input validation, upload limits, and the missing auth layer
* `docs/evaluation.md` — how the numbers these endpoints report are measured
