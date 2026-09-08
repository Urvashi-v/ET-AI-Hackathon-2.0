"""Metric families the golden set alone cannot measure.

The question-answering harness measures retrieval and abstention. Four other
things this system claims are only measurable against different evidence, and
each has its own reference set or its own definition of correct:

``entity``      precision, recall and F1 against hand-labelled documents
``linkage``     how completely the graph connects what was extracted
``citation``    whether every citation resolves to text that exists
``compliance``  whether requirement verdicts match what the records support

Every function here returns ``None`` rather than a number when the evidence to
compute it is absent. A metric with no reference set is not zero — it is
unmeasured, and the report says so.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

EVAL_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------


def entity_metrics(client: httpx.Client) -> dict[str, Any]:
    """Precision, recall and F1 of asset extraction, per document and overall.

    Scored against ``eval/entities.jsonl``, which lists the tags each document
    genuinely names. Both halves matter and they fail differently: a missed tag
    means a document is invisible to questions about that asset, while a spurious
    tag creates an equipment node for something that does not exist. Reporting
    only recall would hide the second entirely.

    Micro-averaged rather than macro: every mention counts once, so a document
    with thirteen tags is not weighted the same as one with two. Macro is also
    reported, because a system that does well on the easy documents and badly on
    the drawing should not be able to hide behind the average.
    """
    reference = _load_entity_reference()
    if not reference:
        return {"state": "not_measured", "reason": "eval/entities.jsonl is absent or empty."}

    try:
        documents = client.get("/api/v1/documents", params={"limit": 200}).json()["items"]
    except Exception as exc:
        return {"state": "error", "reason": f"{type(exc).__name__}: {exc}"}

    by_title = {d["title"]: d["doc_id"] for d in documents}
    per_document: list[dict[str, Any]] = []
    tp = fp = fn = 0

    for row in reference:
        title = row["doc_title"]
        doc_id = by_title.get(title)
        if not doc_id:
            per_document.append({"doc_title": title, "state": "document_not_ingested"})
            continue

        expected = {t.upper() for t in row["expected_tags"]}
        found = _extracted_tags(client, doc_id)

        hits = expected & found
        spurious = found - expected
        missed = expected - found
        tp += len(hits)
        fp += len(spurious)
        fn += len(missed)

        per_document.append(
            {
                "doc_title": title,
                "expected": sorted(expected),
                "found": sorted(found),
                "missed": sorted(missed),
                "spurious": sorted(spurious),
                "precision": _ratio(len(hits), len(found)),
                "recall": _ratio(len(hits), len(expected)),
            }
        )

    scored = [d for d in per_document if "precision" in d]
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    return {
        "state": "measured",
        "documents_scored": len(scored),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision_micro": precision,
        "recall_micro": recall,
        "f1_micro": _f1(precision, recall),
        "precision_macro": _mean([d["precision"] for d in scored if d["precision"] is not None]),
        "recall_macro": _mean([d["recall"] for d in scored if d["recall"] is not None]),
        "per_document": per_document,
        "method": "hand-labelled reference set, eval/entities.jsonl",
    }


def _extracted_tags(client: httpx.Client, doc_id: str) -> set[str]:
    """Every asset tag the system associates with a document.

    Unions mentions in the text with detections on the drawing. Both are ways the
    system claims "this document names this asset", and scoring only one would
    under-report the P&ID -- where the tags are found by the drawing path and
    would otherwise all read as misses.
    """
    tags: set[str] = set()
    try:
        detail = client.get(f"/api/v1/documents/{doc_id}").json()
        for chunk in detail.get("chunks", []):
            payload = client.get(f"/api/v1/documents/{doc_id}/chunks/{chunk['chunk_id']}").json()
            for mention in payload.get("mentions", []):
                if mention.get("tag_kind") == "line":
                    continue
                tag = mention.get("canonical_tag") or mention.get("normalised")
                if tag:
                    tags.add(tag.upper())
    except Exception:
        pass

    try:
        detections = client.get(f"/api/v1/drawings/{doc_id}/detections", params={"page": 1}).json()
        for detection in detections.get("detections", []):
            if detection["kind"] != "tag":
                continue
            # Line numbers are excluded to match the reference set, which lists
            # equipment and instruments. A line is a real detection and a correct
            # one -- scoring it against a reference that never claimed to cover
            # lines counted four correct detections as errors and dragged P&ID
            # precision down by a quarter.
            if (detection.get("properties") or {}).get("tag_kind") == "line":
                continue
            tag = detection.get("canonical_tag") or detection.get("normalised")
            if tag:
                tags.add(tag.upper())
    except Exception:
        pass
    return tags


# ---------------------------------------------------------------------------
# Linkage completeness
# ---------------------------------------------------------------------------


def linkage_metrics(client: httpx.Client) -> dict[str, Any]:
    """How completely the graph connects what ingestion extracted.

    Extraction finding a tag is worth nothing if the tag never becomes an edge.
    Four ratios, each answering a question that a broken pipeline answers badly:

    * **mention resolution** -- of the tags found in text, how many reached a
      canonical asset rather than the review queue;
    * **asset documentation** -- of the assets that exist, how many are evidenced
      by at least one document;
    * **cross-system corroboration** -- how many are evidenced by more than one
      *source system*, which is the platform's actual claim;
    * **drawing linkage** -- of the tags detected on drawings, how many resolved.
    """
    try:
        stats = client.get("/api/v1/assets/stats").json()
        drawings = client.get("/api/v1/drawings").json()
    except Exception as exc:
        return {"state": "error", "reason": f"{type(exc).__name__}: {exc}"}

    # Counted per drawing from the detections endpoint rather than from the
    # listing's summary columns. The listing counts *distinct tag texts* against
    # *all linked detections* including instrument bubbles, so the ratio came out
    # at 1.083 -- a linkage rate above 1 is arithmetically impossible and was a
    # sign the two numbers were not counting the same things.
    detections = linked = 0
    for drawing in drawings.get("items", []):
        try:
            page = client.get(
                f"/api/v1/drawings/{drawing['doc_id']}/detections", params={"page": 1}
            ).json()
        except Exception:
            continue
        tags = [d for d in page.get("detections", []) if d["kind"] == "tag"]
        detections += len(tags)
        linked += sum(1 for d in tags if d.get("linked_asset_id"))

    return {
        "state": "measured",
        # Key names taken from the endpoint's actual response. They read as
        # percentages there, so they are divided rather than reported as ratios
        # alongside figures that are.
        "mention_resolution_rate": _fraction(stats.get("mention_resolution_rate_pct")),
        "assets_total": stats.get("total_assets"),
        "mentions_total": stats.get("total_mentions"),
        "assets_with_documents_pct": _fraction(stats.get("multi_document_pct")),
        "cross_system_pct": _fraction(stats.get("cross_system_pct")),
        "drawing_tags_detected": detections,
        "drawing_tags_linked": linked,
        "drawing_linkage_rate": _ratio(linked, detections) if detections else None,
        "review_queue_open": stats.get("mentions_needing_review"),
        "method": "graph and asset statistics from the live stores",
    }


# ---------------------------------------------------------------------------
# Citation validity
# ---------------------------------------------------------------------------


def citation_metrics(client: httpx.Client, results: list[dict[str, Any]]) -> dict[str, Any]:
    """Does every citation resolve to text that actually exists?

    The claim this project rests on is that a citation can be opened. That is
    checkable rather than assertable: fetch the chunk each citation names, and
    confirm the document, the chunk and the quoted snippet are all really there.

    A citation that cannot be opened is worse than no citation, because it
    carries the authority of evidence without the substance, so this is scored
    strictly -- a single unresolvable field fails the whole citation.
    """
    checked = valid = 0
    failures: list[dict[str, Any]] = []

    for result in results:
        for citation in result.get("citation_refs", []):
            checked += 1
            problems = []
            try:
                payload = client.get(
                    f"/api/v1/documents/{citation['doc_id']}/chunks/{citation['chunk_id']}"
                )
                if payload.status_code != 200:
                    problems.append(f"chunk unreachable ({payload.status_code})")
                else:
                    chunk = payload.json()["chunk"]
                    snippet = (citation.get("snippet") or "").strip()
                    if snippet and _squash(snippet[:120]) not in _squash(chunk["text"]):
                        problems.append("snippet not found in the chunk it cites")
                    if citation.get("page") and chunk.get("page_from") != citation["page"]:
                        problems.append(
                            f"page {citation['page']} but chunk starts on {chunk.get('page_from')}"
                        )
            except Exception as exc:
                problems.append(f"{type(exc).__name__}: {exc}")

            if problems:
                failures.append(
                    {
                        "case": result["id"],
                        "marker": citation.get("marker"),
                        "doc_id": citation["doc_id"],
                        "problems": problems,
                    }
                )
            else:
                valid += 1

    if not checked:
        return {"state": "not_measured", "reason": "No citations were produced to check."}
    return {
        "state": "measured",
        "citations_checked": checked,
        "citations_valid": valid,
        "validity_rate": _ratio(valid, checked),
        "failures": failures[:20],
        "method": "every cited chunk re-fetched and its snippet verified against stored text",
    }


# ---------------------------------------------------------------------------
# Groundedness
# ---------------------------------------------------------------------------


def groundedness_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """What fraction of answer claims are verbatim spans of a cited passage.

    For the extractive answerer this should be 1.0 by construction, and checking
    it anyway is the point: a guarantee that is never verified is a comment.
    A drop here means an offset bug or a passage swapped after composition.

    Answer *correctness* is a different question and is not measured. Grading an
    answer against a reference needs a judge this project does not have, and the
    honest report is ``not_measured`` rather than a number derived from string
    overlap and presented as accuracy.
    """
    answered = [r for r in results if r.get("answer_present")]
    if not answered:
        return {
            "state": "not_measured",
            "reason": "No answers were produced, so groundedness has nothing to score.",
        }

    total_claims = sum(r.get("claim_count", 0) for r in answered)
    verbatim = sum(r.get("verbatim_claims", 0) for r in answered)
    cited = sum(1 for r in answered if r.get("citations", 0) > 0)

    return {
        "state": "measured",
        "answers": len(answered),
        "answers_with_citations": cited,
        "cited_answer_rate": _ratio(cited, len(answered)),
        "claims_total": total_claims,
        "claims_verbatim": verbatim,
        "groundedness": _ratio(verbatim, total_claims) if total_claims else None,
        "answer_correctness": {
            "state": "not_measured",
            "reason": (
                "Grading an answer against a reference needs a judge (a human or a "
                "capable LLM). Neither is configured, and a string-overlap score "
                "presented as correctness would be worse than no number."
            ),
        },
        "method": "claims re-checked for verbatim containment in the passage they cite",
    }


# ---------------------------------------------------------------------------
# Compliance gap detection
# ---------------------------------------------------------------------------


def compliance_metrics(client: httpx.Client) -> dict[str, Any]:
    """Do requirement verdicts match what the records support?

    Scored against ``eval/compliance_expectations.jsonl``, which states the
    expected verdict for requirement/asset pairs whose answer is determinable by
    hand from the stored records -- interval arithmetic and graph state. The
    procedure-text and permit-record modes are excluded from scoring on purpose:
    their correct verdict is "a human must decide", and grading the system on
    agreeing with a judgement nobody has made would be circular.
    """
    expectations = _load_jsonl(EVAL_DIR / "compliance_expectations.jsonl")
    if not expectations:
        return {
            "state": "not_measured",
            "reason": "eval/compliance_expectations.jsonl is absent or empty.",
        }

    by_asset: dict[str, dict[str, Any]] = {}
    correct = 0
    mismatches: list[dict[str, Any]] = []

    for row in expectations:
        asset = row["asset_tag"]
        if asset not in by_asset:
            try:
                by_asset[asset] = client.get(
                    "/api/v1/compliance/evaluate", params={"asset_tag": asset}
                ).json()
            except Exception as exc:
                return {"state": "error", "reason": f"{type(exc).__name__}: {exc}"}

        findings = {f["req_id"]: f for f in by_asset[asset]["findings"]}
        finding = findings.get(row["req_id"])
        actual = finding["state"] if finding else "absent"
        if actual == row["expected_state"]:
            correct += 1
        else:
            mismatches.append(
                {
                    "req_id": row["req_id"],
                    "asset_tag": asset,
                    "expected": row["expected_state"],
                    "actual": actual,
                    "reason_given": (finding or {}).get("reason", ""),
                    "why_expected": row.get("notes", ""),
                }
            )

    return {
        "state": "measured",
        "pairs_scored": len(expectations),
        "correct": correct,
        "accuracy": _ratio(correct, len(expectations)),
        "mismatches": mismatches,
        "method": (
            "hand-determined verdicts for evidence-document and graph-state requirements, "
            "eval/compliance_expectations.jsonl. Modes whose correct answer is 'a human "
            "must decide' are excluded rather than graded."
        ),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_entity_reference() -> list[dict[str, Any]]:
    return [row for row in _load_jsonl(EVAL_DIR / "entities.jsonl") if "doc_title" in row]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        # Comment rows document the file for a reader; they are not data.
        if "_comment" in row:
            continue
        rows.append(row)
    return rows


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if not precision or not recall:
        return 0.0 if (precision is not None and recall is not None) else None
    return round(2 * precision * recall / (precision + recall), 4)


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _fraction(value: Any) -> float | None:
    """A percentage from the API, as a fraction, so it sits beside real ratios."""
    return round(float(value) / 100.0, 4) if isinstance(value, (int, float)) else None


def _squash(text: str) -> str:
    return " ".join(text.split()).lower()
