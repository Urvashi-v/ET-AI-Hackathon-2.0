"""Structured records from semi-structured documents.

An incident report is not free prose. It has a document number, a date, an
equipment reference, an immediate cause, a root cause and a table of corrective
actions — all under headings, in a layout that barely varies between reports
because a form template produced it. The same is true of MOCs and inspection
reports.

That regularity is worth exploiting, and this module exploits it deterministically:
field labels and section headings, matched case-insensitively, against chunks the
structure-aware chunker has already split by section. No model, no inference, and
**every extracted field carries the chunk it came from**, so an ``Incident`` node
in the graph can be opened back to the sentence that asserted it.

Why this matters beyond convenience
-----------------------------------
Until now the corpus held incident *documents* but no incident *records*. RCA
reported ``incidents: 0`` for a pump with two investigated seal failures on file,
because nothing had turned the prose into nodes. Lessons-learned had nothing to
compare against. The documents were retrievable and the facts inside them were
invisible to every query that did not go through text search.

What this deliberately does not do
----------------------------------
It does not attempt narrative documents with no structure. A report that buries
its root cause in the ninth paragraph of an essay is a job for the LLM extractor
(``llm_extract.py``), which is credential-gated and says so. Here, a document
that does not present the expected headings yields **no record** rather than a
guessed one — a missing incident is a visible gap, while a wrong one is a false
fact that will be cited with confidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from services.common.ids import deterministic_uuid
from services.common.logging import get_logger
from services.common.schemas import DataClass, DocumentType
from services.ingest.extract import extract_all, extract_tags

log = get_logger(__name__)


@dataclass(slots=True)
class FieldValue:
    """One extracted value, with the chunk that asserted it."""

    value: str
    chunk_id: str
    page: int | None = None

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value


@dataclass(slots=True)
class CorrectiveAction:
    action_id: str
    description: str
    owner: str | None
    due_on: date | None
    status: str | None
    chunk_id: str
    page: int | None = None

    @property
    def is_open(self) -> bool:
        """Anything not explicitly closed is open.

        Deliberately pessimistic. A CAPA whose status column is blank, or says
        something the parser does not recognise, is treated as outstanding: the
        cost of chasing a closed action is a wasted phone call, and the cost of
        missing an open one is the corrective action nobody did.
        """
        return (self.status or "").strip().upper() not in {"CLOSED", "COMPLETE", "COMPLETED", "DONE"}


@dataclass(slots=True)
class IncidentRecord:
    incident_id: str
    doc_id: str
    title: str
    asset_tags: list[str] = field(default_factory=list)
    functional_location: str | None = None
    occurred_on: date | None = None
    severity: str | None = None
    investigation_status: str | None = None
    narrative: FieldValue | None = None
    immediate_cause: FieldValue | None = None
    root_cause: FieldValue | None = None
    recurrence_note: FieldValue | None = None
    corrective_actions: list[CorrectiveAction] = field(default_factory=list)
    referenced_incidents: list[str] = field(default_factory=list)
    referenced_procedures: list[str] = field(default_factory=list)
    #: Which of the expected fields were found, so completeness is measurable
    #: rather than assumed.
    fields_found: list[str] = field(default_factory=list)
    fields_missing: list[str] = field(default_factory=list)

    @property
    def has_cause(self) -> bool:
        return self.root_cause is not None or self.immediate_cause is not None


#: Front-matter labels. The value runs to the end of the line, or to the next
#: label on the same line -- these reports pack several onto one row
#: ("Site: HALDIA   Plant: CDU-1   System: Crude Charge Pumping").
_LABELS: dict[str, tuple[str, ...]] = {
    "site": ("site",),
    "plant": ("plant", "unit"),
    "system": ("system",),
    "equipment": ("equipment involved", "equipment", "asset", "tag"),
    "functional_location": ("functional location", "func loc", "floc"),
    "occurred_on": ("date of event", "event date", "date of occurrence", "date"),
    "severity": ("severity", "consequence"),
    "investigation_status": ("investigation status", "status", "report status"),
    "change_type": ("type of change", "change type"),
    "requested_by": ("requested by", "originator"),
}

#: Section headings, matched on the *last* segment of the chunk's section path.
#: Several spellings per concept because a scanned report is OCR'd in upper case
#: and a Markdown source is title case.
#: Both spellings of every heading appear in this corpus, for a structural
#: reason: a Markdown source keeps its literal heading text ("Root cause"), while
#: the OCR path labels sections by the role it inferred ("root_cause"). Matching
#: only one form silently drops half the corpus -- which is exactly what happened
#: to the scanned copy of INC-2019-07 on the first run.
_SECTIONS: dict[str, tuple[str, ...]] = {
    "narrative": (
        "what happened", "description of event", "event description", "summary",
        "narrative",
    ),
    "immediate_cause": ("immediate cause", "direct cause", "apparent cause", "immediate_cause"),
    "root_cause": ("root cause", "underlying cause", "basic cause", "root_cause"),
    "recurrence": (
        "analysis of recurrence", "recurrence", "previous occurrences", "recurrence_analysis",
    ),
    "actions": (
        "corrective and preventive actions",
        "corrective actions",
        "corrective_action",
        "corrective_actions",
        "capa",
        "actions",
        "recommendations",
    ),
}

_LABEL_LINE = re.compile(r"([A-Za-z][A-Za-z /_-]{2,28}?)\s*:\s*(.+?)(?=\s{2,}[A-Za-z][A-Za-z /_-]{2,28}?\s*:|$)")

#: A CAPA/action table row: | ID | Action | Owner | Due | Status |
_TABLE_ROW = re.compile(r"^\s*\|(.+)\|\s*$")
_ACTION_ID = re.compile(r"\b((?:CAPA|CA|AR|ACT)[-\s]?\d{1,6})\b", re.I)
_INCIDENT_REF = re.compile(r"\b(INC[-\s]?\d{4}[-\s]?\d{1,3})\b", re.I)
_PROCEDURE_REF = re.compile(r"\b(SOP[-\s]?\d{3,6})\b", re.I)
_TAG_IN_TEXT = re.compile(r"\b([A-Z]{1,4}-\d{2,4}[A-Z]?)\b")

#: An ISO or slashed date, used to find the "due" column in a row OCR has
#: flattened into running text.
_DATE_TOKEN = re.compile(r"^\d{4}-\d{2}-\d{2}$|^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$")

#: Status words a CAPA table ends with. Anything else is treated as part of the
#: description rather than misread as a status.
_STATUS_WORDS = frozenset(
    {"OPEN", "CLOSED", "COMPLETE", "COMPLETED", "DONE", "PENDING", "OVERDUE", "IN-PROGRESS"}
)

#: Fields an incident report is expected to carry. Absences are recorded rather
#: than filled in, because "the report does not state a root cause" is itself a
#: finding -- and a common one in real plants.
_EXPECTED = ("occurred_on", "immediate_cause", "root_cause", "narrative", "corrective_actions")


def extract_incident(
    *, doc_id: str, title: str, doc_number: str | None, chunks: list[dict[str, Any]]
) -> IncidentRecord | None:
    """Build an incident record from an already-chunked report.

    Returns ``None`` when the document presents none of the expected structure.
    That is the honest outcome: a document that does not look like an incident
    report should not become an incident node, and a half-invented one would be
    worse than none.
    """
    if not chunks:
        return None

    sections = _by_section(chunks)
    front = chunks[0]
    labels = _labels_in(front.get("text", ""))

    incident_id = _canonical_ref(doc_number) if doc_number else None
    if not incident_id:
        found = _INCIDENT_REF.search(title) or _INCIDENT_REF.search(front.get("text", ""))
        incident_id = _canonical_ref(found.group(1)) if found else None
    if not incident_id:
        # No identifier of its own. Derived from the document so the node is
        # still addressable and still idempotent across re-ingestion.
        incident_id = f"INC-{str(deterministic_uuid(doc_id))[:8].upper()}"

    record = IncidentRecord(incident_id=incident_id, doc_id=doc_id, title=title)

    record.occurred_on = _first_date(labels.get("occurred_on"), front.get("text", ""))
    record.severity = labels.get("severity")
    record.investigation_status = _normalise_status(labels.get("investigation_status"))
    record.functional_location = labels.get("functional_location")

    equipment = labels.get("equipment") or ""
    record.asset_tags = _tags_in(equipment) or _tags_in(front.get("text", ""))[:2]

    for name, key in (
        ("narrative", "narrative"),
        ("immediate_cause", "immediate_cause"),
        ("root_cause", "root_cause"),
        ("recurrence", "recurrence_note"),
    ):
        chunk = sections.get(name)
        if chunk:
            setattr(record, key, _field_from(chunk))

    # Markdown reports open with a chunk whose section path is the document title,
    # holding the front matter and the "what happened" prose together. There is no
    # separate narrative section to find, but the narrative is right there.
    if record.narrative is None and len(front.get("text", "")) > 200:
        record.narrative = _field_from(front)

    actions_chunk = sections.get("actions")
    if actions_chunk:
        record.corrective_actions = _parse_actions(actions_chunk)

    body = "\n".join(c.get("text", "") for c in chunks)
    record.referenced_incidents = sorted(
        {_canonical_ref(m) for m in _INCIDENT_REF.findall(body)} - {incident_id}
    )
    record.referenced_procedures = sorted({_canonical_ref(m) for m in _PROCEDURE_REF.findall(body)})

    present = {
        "occurred_on": record.occurred_on is not None,
        "immediate_cause": record.immediate_cause is not None,
        "root_cause": record.root_cause is not None,
        "narrative": record.narrative is not None,
        "corrective_actions": bool(record.corrective_actions),
    }
    record.fields_found = sorted(k for k, v in present.items() if v)
    record.fields_missing = sorted(k for k, v in present.items() if not v)

    # A document with a date and nothing else is not an incident report. Requiring
    # at least one cause statement or one action keeps stray documents that happen
    # to carry an INC- reference from becoming incident records.
    if not (record.has_cause or record.corrective_actions):
        log.info(
            "records.incident_rejected",
            doc_id=doc_id,
            reason="no cause statement or corrective action found",
        )
        return None
    return record


@dataclass(slots=True)
class ChangeRecord:
    """A management-of-change record. Same treatment as an incident."""

    moc_id: str
    doc_id: str
    title: str
    asset_tags: list[str] = field(default_factory=list)
    raised_on: date | None = None
    change_type: str | None = None
    status: str | None = None
    description: FieldValue | None = None
    outstanding_actions: list[CorrectiveAction] = field(default_factory=list)
    referenced_procedures: list[str] = field(default_factory=list)


_MOC_REF = re.compile(r"\b(MOC[-\s]?\d{4}[-\s]?\d{1,3})\b", re.I)


def extract_change(
    *, doc_id: str, title: str, doc_number: str | None, chunks: list[dict[str, Any]]
) -> ChangeRecord | None:
    """Build an MOC record.

    MOCs matter to RCA out of proportion to their number: a change to an asset is
    the single most common reason its failure behaviour changes, and "what was
    modified before this started happening?" is the question a reliability
    engineer asks first.
    """
    if not chunks:
        return None
    front = chunks[0]
    labels = _labels_in(front.get("text", ""))

    moc_id = _canonical_ref(doc_number) if doc_number else None
    if not moc_id:
        found = _MOC_REF.search(title) or _MOC_REF.search(front.get("text", ""))
        moc_id = _canonical_ref(found.group(1)) if found else None
    if not moc_id:
        return None

    body = "\n".join(c.get("text", "") for c in chunks)
    sections = _by_section(chunks)
    record = ChangeRecord(moc_id=moc_id, doc_id=doc_id, title=title)
    record.raised_on = _first_date(labels.get("occurred_on"), front.get("text", ""))
    record.change_type = labels.get("change_type")
    record.status = _normalise_status(labels.get("investigation_status"))
    record.asset_tags = _tags_in(labels.get("equipment") or "") or _tags_in(body)[:3]
    record.description = _field_from(sections.get("narrative") or front)
    record.referenced_procedures = sorted({_canonical_ref(m) for m in _PROCEDURE_REF.findall(body)})

    # An MOC's follow-up table is where the "we changed it but never updated the
    # datasheet" gap lives, which is exactly what compliance needs to see.
    actions_chunk = sections.get("actions")
    if actions_chunk:
        record.outstanding_actions = [a for a in _parse_actions(actions_chunk) if a.is_open]
    return record


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _by_section(chunks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Map each known section name to the chunk that holds it.

    Matched on the last segment of ``section_path``, which the chunker builds as
    "Document title > Heading". Case-insensitive, because the scanned copy of the
    same report comes back from OCR in upper case.
    """
    found: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        path = (chunk.get("section_path") or "").strip()
        heading = path.split(">")[-1].strip().lower() if path else ""
        if not heading:
            continue
        for name, spellings in _SECTIONS.items():
            if name in found:
                continue
            if any(heading == s or heading.startswith(s) for s in spellings):
                found[name] = chunk
    return found


def _labels_in(text: str) -> dict[str, str]:
    """Pull ``Label: value`` pairs out of front matter."""
    values: dict[str, str] = {}
    for line in text.splitlines()[:20]:
        for raw_label, raw_value in _LABEL_LINE.findall(line):
            label = raw_label.strip().lower()
            value = raw_value.strip()
            if not value:
                continue
            for name, spellings in _LABELS.items():
                if name in values:
                    continue
                if label in spellings:
                    values[name] = value
                    break
    return values


def _field_from(chunk: dict[str, Any]) -> FieldValue | None:
    text = (chunk.get("text") or "").strip()
    if not text:
        return None
    # Strip the repeated heading the chunker prepends, so the stored value is the
    # statement itself rather than "Root cause Root cause was...".
    heading = (chunk.get("section_path") or "").split(">")[-1].strip()
    if heading and text.lower().startswith(heading.lower()):
        text = text[len(heading) :].strip(" :\n")
    return FieldValue(value=text, chunk_id=chunk["chunk_id"], page=chunk.get("page_from"))


def _parse_actions(chunk: dict[str, Any]) -> list[CorrectiveAction]:
    """Parse the corrective-action table.

    Pipe-delimited rows, because that is what the Markdown source and the
    table-aware PDF parser both produce. Rows without an action identifier are
    skipped -- the header and the ``|---|---|`` separator are rows too.
    """
    actions: list[CorrectiveAction] = []
    for line in (chunk.get("text") or "").splitlines():
        cells = _row_cells(line)
        if not cells:
            continue
        id_match = _ACTION_ID.search(cells[0])
        if not id_match:
            continue
        actions.append(
            CorrectiveAction(
                action_id=_canonical_ref(id_match.group(1)),
                description=cells[1] if len(cells) > 1 else "",
                owner=cells[2] if len(cells) > 2 and cells[2] else None,
                due_on=_first_date(cells[3] if len(cells) > 3 else None, ""),
                status=_normalise_status(cells[4] if len(cells) > 4 else None),
                chunk_id=chunk["chunk_id"],
                page=chunk.get("page_from"),
            )
        )
    return actions


def _row_cells(line: str) -> list[str] | None:
    """Split one action row into cells, whichever way the row was rendered.

    Pipe-delimited when the source was Markdown or the PDF table extractor found
    ruling lines. But OCR of a printed table loses the rules and returns the row
    as running text -- "CAPA-41 Replace outboard mechanical seal Rotating
    Equipment 2019-04-05 CLOSED" -- so the same action has to be recovered
    positionally: identifier, then description, then the trailing owner, date and
    status that a CAPA table always ends with.
    """
    match = _TABLE_ROW.match(line)
    if match:
        cells = [c.strip() for c in match.group(1).split("|")]
        if len(cells) < 2 or set("".join(cells)) <= {"-", ":", " "}:
            return None
        return cells

    id_match = _ACTION_ID.match(line.strip())
    if not id_match:
        return None
    remainder = line.strip()[id_match.end() :].strip()

    # Peel the fixed trailing fields off the end, so whatever is left is the
    # description however many words it ran to.
    status = None
    date_text = None
    tokens = remainder.split()
    if tokens and tokens[-1].upper() in _STATUS_WORDS:
        status = tokens.pop()
    if tokens and _DATE_TOKEN.match(tokens[-1]):
        date_text = tokens.pop()
    # The owner is a short capitalised phrase before the date; without a date to
    # anchor it there is no reliable boundary, so it is left unset rather than
    # guessed out of the description.
    owner = None
    if date_text and len(tokens) > 2:
        trailing = [t for t in tokens[-3:] if t[:1].isupper()]
        if trailing and len(trailing) >= 2:
            owner = " ".join(tokens[-len(trailing) :])
            tokens = tokens[: -len(trailing)]
    return [id_match.group(1), " ".join(tokens), owner or "", date_text or "", status or ""]


def _first_date(*candidates: str | None) -> date | None:
    for text in candidates:
        if not text:
            continue
        found = extract_all(text).dates
        if found:
            return date.fromisoformat(found[0]["date"])
    return None


def _normalise_status(value: str | None) -> str | None:
    if not value:
        return None
    # Severity lines read "Moderate -- 22 hours downtime"; keep the leading token
    # so a status column and a severity line both reduce to something comparable.
    token = re.split(r"\s+--\s+|\s+-\s+|,", value.strip())[0].strip().upper()
    return token[:40] or None


def _tags_in(text: str) -> list[str]:
    """Asset tags, using the project's own tag grammar rather than a regex.

    A naive "letters-dash-digits" pattern reads MOC-2023, WO-3502 and SOP-4412 as
    equipment, which puts document numbers into the asset list and then into the
    graph as equipment nodes. ``extract_tags`` already masks document references
    and validates against the ISA/IEC grammars, so it is the right tool -- the
    regex was a shortcut past work that was already done.
    """
    if not text:
        return []
    seen: list[str] = []
    for mention in extract_tags(text):
        tag = mention.normalised or mention.surface_form
        if tag and tag not in seen:
            seen.append(tag)
    return seen


def _canonical_ref(raw: str) -> str:
    """``INC 2019 07`` / ``inc-2019-07`` -> ``INC-2019-07``."""
    collapsed = re.sub(r"[\s_]+", "-", raw.strip().upper())
    collapsed = re.sub(r"^([A-Z]+?)-?(\d)", r"\1-\2", collapsed)
    return re.sub(r"-{2,}", "-", collapsed)


def is_extractable(doc_type: DocumentType) -> bool:
    """Which document types this module attempts at all."""
    return doc_type in (DocumentType.INCIDENT_REPORT, DocumentType.MOC)


def data_class_for(doc_data_class: str | DataClass) -> DataClass:
    """A record inherits the provenance of the document it came from.

    Extraction is deterministic and verbatim, so nothing here is model-derived:
    a root cause pulled out of a synthetic report is still synthetic test data,
    and out of a real report is still a real source document.
    """
    return DataClass(doc_data_class) if isinstance(doc_data_class, str) else doc_data_class
