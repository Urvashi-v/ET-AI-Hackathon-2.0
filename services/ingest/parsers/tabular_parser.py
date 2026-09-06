"""Tabular and structured-record parser (CSV, JSON).

CMMS exports and inspection logs are not documents and must not be chunked as
prose. They are records, and they are handled as records:

* structured fields are mapped to schema columns using a **declared mapping per
  source system**, applied deterministically -- not by asking a model to read
  every row, which is neither reproducible nor affordable;
* the free-text field (the CMMS "long text", where the diagnostic information
  actually lives) is kept whole and goes through the normal NLP path;
* a **natural-language summary is composed per record** from its own field
  values, so the record is reachable by semantic search. The summary is a
  deterministic template over real values -- it states only what the row
  contains and adds nothing.

Unmapped columns are preserved in ``extra`` rather than dropped, and reported,
because a column nobody mapped is usually the interesting one.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import date
from typing import Any

from services.common.logging import get_logger
from services.ingest.parsers.base import ParsedDocument, TextBlock

log = get_logger(__name__)

#: Column-name synonyms per canonical field. Extending this dictionary is how a
#: new source system is onboarded -- a one-off, reviewable act, after which
#: every row of that system is parsed deterministically.
COLUMN_SYNONYMS: dict[str, tuple[str, ...]] = {
    "wo_id": ("wo_id", "work_order", "work_order_no", "order", "notification", "wo"),
    "asset_tag": ("asset_tag", "equipment", "equipment_tag", "tag", "tag_no", "equip"),
    "functional_location": ("functional_location", "fl", "fl_tag", "floc", "location"),
    "wo_type": ("wo_type", "order_type", "type", "maintenance_type"),
    "status": ("status", "wo_status", "state"),
    "priority": ("priority", "prio"),
    "description": ("description", "short_text", "title", "summary"),
    "long_text": (
        "long_text",
        "longtext",
        "notes",
        "technician_notes",
        "work_performed",
        "remarks",
    ),
    "as_found": ("as_found", "asfound", "found_condition", "condition_on_arrival"),
    "as_left": ("as_left", "asleft", "left_condition", "condition_on_departure"),
    "coded_failure_mode": (
        "failure_code",
        "failure_mode_code",
        "damage_code",
        "coded_failure_mode",
    ),
    "opened_on": ("opened_on", "created_on", "open_date", "start_date", "reported_on"),
    "closed_on": ("closed_on", "close_date", "completion_date", "completed_on"),
    "downtime_hours": ("downtime_hours", "downtime_hrs", "downtime"),
    "cost": ("cost", "actual_cost", "total_cost"),
    # inspection
    "cml_id": ("cml_id", "cml", "location_id", "point"),
    "method": ("method", "ndt_method", "technique"),
    "inspected_on": ("inspected_on", "inspection_date", "survey_date", "date"),
    "thickness_mm": ("thickness_mm", "thickness", "reading_mm", "measured_thickness"),
    "min_required_mm": ("min_required_mm", "t_min", "minimum_thickness", "min_thickness"),
    "inspector": ("inspector", "inspected_by", "technician"),
    "finding": ("finding", "observation", "result", "comment"),
    # incident
    # `report_no` is deliberately NOT an incident synonym: it is the report
    # number on an inspection survey far more often than an incident id, and
    # mapping it here filed every UT reading as an incident.
    "incident_id": ("incident_id", "inc_id", "event_id"),
    "report_no": ("report_no", "report_number", "survey_no"),
    "occurred_on": ("occurred_on", "event_date", "incident_date", "date_of_event"),
    "severity": ("severity", "sev", "consequence"),
    "narrative": ("narrative", "what_happened", "description_of_event", "details"),
    "immediate_cause": ("immediate_cause", "direct_cause"),
    "root_cause": ("root_cause", "basic_cause", "underlying_cause"),
}

_REVERSE_SYNONYMS: dict[str, str] = {
    syn: canonical for canonical, syns in COLUMN_SYNONYMS.items() for syn in syns
}


class TabularParser:
    name = "tabular"

    def can_parse(self, extension: str) -> bool:
        return extension.lower() in {".csv", ".json"}

    def parse(self, data: bytes, *, filename: str) -> ParsedDocument:
        text = _decode(data)
        if filename.lower().endswith(".json"):
            rows, warnings = _load_json_rows(text)
        else:
            rows, warnings = _load_csv_rows(text)

        blocks: list[TextBlock] = []
        records: list[dict[str, Any]] = []
        unmapped: set[str] = set()
        offset = 0

        for index, row in enumerate(rows):
            mapped, extra = _map_row(row)
            unmapped.update(extra.keys())
            summary = compose_record_summary(mapped, extra)
            record = {**mapped, "extra": extra, "row_index": index, "summary": summary}
            records.append(record)

            blocks.append(
                TextBlock(
                    text=summary,
                    page=None,
                    char_start=offset,
                    char_end=offset + len(summary),
                    kind="record",
                    section_path=f"record {index + 1}",
                    metadata={"row_index": index},
                )
            )
            offset += len(summary) + 2

        if unmapped:
            warnings.append(
                "Unmapped columns retained under 'extra': " + ", ".join(sorted(unmapped)[:20])
            )
        if not rows:
            warnings.append("No rows were parsed from the file.")

        return ParsedDocument(
            blocks=blocks,
            page_count=None,
            has_text_layer=True,
            parser=self.name,
            warnings=warnings,
            records=records,
            metadata={"row_count": len(rows), "unmapped_columns": sorted(unmapped)},
        )


def compose_record_summary(mapped: dict[str, Any], extra: dict[str, Any]) -> str:
    """Compose a retrievable sentence from a record's own values.

    Deterministic and purely descriptive: every clause is emitted only if the
    corresponding field is present, and the values are reproduced verbatim. It
    adds no interpretation, so the resulting chunk is safe to cite.
    """
    parts: list[str] = []

    if mapped.get("wo_id"):
        head = f"Work order {mapped['wo_id']}"
        if mapped.get("asset_tag"):
            head += f" on {mapped['asset_tag']}"
        if mapped.get("functional_location"):
            head += f" (functional location {mapped['functional_location']})"
        parts.append(head)
    elif mapped.get("incident_id"):
        head = f"Incident {mapped['incident_id']}"
        if mapped.get("asset_tag"):
            head += f" involving {mapped['asset_tag']}"
        parts.append(head)
    elif mapped.get("cml_id"):
        head = f"Inspection at CML {mapped['cml_id']}"
        if mapped.get("asset_tag"):
            head += f" on {mapped['asset_tag']}"
        parts.append(head)
    elif mapped.get("asset_tag"):
        parts.append(f"Record for {mapped['asset_tag']}")

    for label, key in (
        ("type", "wo_type"),
        ("status", "status"),
        ("priority", "priority"),
        ("severity", "severity"),
        ("method", "method"),
        ("inspector", "inspector"),
    ):
        if mapped.get(key):
            parts.append(f"{label} {mapped[key]}")

    for label, key in (
        ("opened", "opened_on"),
        ("closed", "closed_on"),
        ("occurred", "occurred_on"),
        ("inspected", "inspected_on"),
    ):
        if mapped.get(key):
            parts.append(f"{label} {mapped[key]}")

    if mapped.get("downtime_hours") is not None:
        parts.append(f"downtime {mapped['downtime_hours']} h")
    if mapped.get("cost") is not None:
        parts.append(f"cost {mapped['cost']}")
    if mapped.get("thickness_mm") is not None:
        clause = f"measured thickness {mapped['thickness_mm']} mm"
        if mapped.get("min_required_mm") is not None:
            clause += f" against minimum {mapped['min_required_mm']} mm"
        parts.append(clause)
    if mapped.get("coded_failure_mode"):
        parts.append(f"coded failure mode {mapped['coded_failure_mode']}")

    summary = "; ".join(parts) if parts else "Record"
    summary += "."

    for label, key in (
        ("Description", "description"),
        ("As found", "as_found"),
        ("As left", "as_left"),
        ("Notes", "long_text"),
        ("Narrative", "narrative"),
        ("Immediate cause", "immediate_cause"),
        ("Root cause", "root_cause"),
        ("Finding", "finding"),
    ):
        value = mapped.get(key)
        if value:
            summary += f" {label}: {value}"
            if not str(value).rstrip().endswith((".", "!", "?")):
                summary += "."

    return summary.strip()


def _decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _strip_leading_comments(text: str) -> tuple[str, list[str]]:
    """Drop leading `#` lines so the header row is read as the header.

    Real exports carry banners, provenance lines and generator stamps above the
    header. Without this, `csv.DictReader` takes the banner as the column names
    and every subsequent row maps into a single garbage field -- a failure that
    is silent, because parsing "succeeds".
    """
    lines = text.splitlines(keepends=True)
    dropped: list[str] = []
    index = 0
    while index < len(lines) and lines[index].lstrip().startswith("#"):
        dropped.append(lines[index].strip().lstrip("#").strip())
        index += 1
    return "".join(lines[index:]), dropped


def _load_csv_rows(text: str) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    text, banners = _strip_leading_comments(text)
    if banners:
        warnings.append("Leading comment line(s) skipped before the header: " + " | ".join(banners))
    sample = text[:8192]
    try:
        dialect: Any = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
        warnings.append("CSV dialect could not be sniffed; assumed comma-separated.")
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    rows = [{(k or "").strip(): v for k, v in row.items()} for row in reader]
    return rows, warnings


def _load_json_rows(text: str) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return [], [f"JSON could not be parsed: {exc.msg} at line {exc.lineno}"]
    if isinstance(payload, dict):
        for key in ("records", "items", "rows", "data"):
            if isinstance(payload.get(key), list):
                return payload[key], warnings
        return [payload], warnings
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)], warnings
    return [], ["JSON root is neither an object nor an array of objects."]


def _map_row(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    mapped: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    for raw_key, value in row.items():
        if raw_key is None:
            continue
        key = str(raw_key).strip().lower().replace(" ", "_").replace("-", "_")
        canonical = _REVERSE_SYNONYMS.get(key)
        cleaned = _clean(value)
        if cleaned is None:
            continue
        if canonical:
            mapped.setdefault(canonical, _coerce(canonical, cleaned))
        else:
            extra[key] = cleaned
    return mapped, extra


def _clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return value


def _coerce(field: str, value: Any) -> Any:
    if field in {"downtime_hours", "cost", "thickness_mm", "min_required_mm"}:
        try:
            return float(str(value).replace(",", ""))
        except (TypeError, ValueError):
            return None
    if field in {"opened_on", "closed_on", "occurred_on", "inspected_on"}:
        return _parse_date(str(value))
    return value


def _parse_date(value: str) -> str | None:
    value = value.strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y", "%Y/%m/%d", "%d-%b-%Y", "%d %b %Y"):
        try:
            from datetime import datetime

            parsed: date = datetime.strptime(value, fmt).date()
            return parsed.isoformat()
        except ValueError:
            continue
    return None
