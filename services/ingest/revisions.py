"""Document numbering and revision lineage.

The problem this solves is narrow and consequential. Ingestion is idempotent
because ``doc_id`` is a content hash — which means every revision of SOP-4412
lands as a *different document* with no relationship to the last one. Retrieval
then treats Rev 2 and Rev 3 as two independent sources, ranks whichever matches
the query wording better, and can hand a technician a superseded procedure with
full confidence. In a plant that is not a ranking defect, it is a safety one.

So documents are grouped by ``doc_number`` — the identifier printed on the
document itself, which is stable across revisions — and ordered within the group
to establish who supersedes whom.

**The system refuses to guess.** Ordering requires evidence: a revision label, a
revision date, or an issue date. When two documents share a number and none of
those can separate them, both stay current, ``revision_conflict`` is set, and the
pair is surfaced for a human. Silently picking one would be the single most
dangerous thing this module could do, and picking "the one ingested most
recently" is exactly that dressed up as a heuristic — ingestion order reflects
who uploaded what first, nothing more.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from services.common import db
from services.common.logging import get_logger
from services.common.schemas import DocumentType

log = get_logger(__name__)


@dataclass(slots=True)
class DocNumber:
    value: str
    method: str


#: Document-number grammars, most specific first. Each is anchored on a prefix
#: that identifies a document *class*, because a bare "4412" is not a document
#: number -- it is a number that happens to appear in a document.
#:
#: Deliberately not a generic "letters-dash-digits" pattern: that matches asset
#: tags (P-101B), line numbers and half the contents of a P&ID, and a wrong
#: grouping here merges the revision histories of unrelated documents.
_NUMBER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("sop", re.compile(r"\b(SOP[-\s]?\d{3,6})\b", re.I)),
    ("incident", re.compile(r"\b(INC[-\s]?\d{4}[-\s]?\d{1,3})\b", re.I)),
    ("inspection", re.compile(r"\b(UT[-\s]?\d{4}[-\s]?\d{1,4})\b", re.I)),
    ("moc", re.compile(r"\b(MOC[-\s]?\d{4}[-\s]?\d{1,3})\b", re.I)),
    ("capa", re.compile(r"\b(CAPA[-\s]?\d{1,6})\b", re.I)),
    ("procedure", re.compile(r"\b(PRO?C[-\s]?\d{3,6})\b", re.I)),
    ("drawing", re.compile(r"\b(PID[-\s]?\d{3,6}|P&ID[-\s]?\d{3,6})\b", re.I)),
    ("workorder", re.compile(r"\b(WO[-\s]?\d{4,8})\b", re.I)),
)

#: Where a document number is looked for, in order of trustworthiness. The title
#: and filename are chosen by whoever issued or filed the document; body text is
#: a last resort because a procedure that *references* SOP-4412 is not SOP-4412.
_SEARCH_ORDER = ("title", "filename", "head")

#: Only the first part of the body is considered, and only when title and
#: filename yield nothing. Beyond the front matter, a document number is far more
#: likely to be a cross-reference than the document's own identity.
_HEAD_WINDOW = 400

#: Document types whose body text may be searched for their own number. These are
#: *single-instance* documents: one incident report is about one incident, so a
#: number in its front matter identifies it.
#:
#: Everything else is excluded, and the reason is a bug this rule exists to
#: prevent. A CMMS export is a table of hundreds of work orders; searching its
#: body found "WO-2101" -- the first row -- and adopted it as the export's own
#: identity. The next month's export would have taken a different row's number,
#: and the two would never group. A collection is not an instance of the thing it
#: collects, so its number can only come from its title or filename.
_BODY_SEARCHABLE_TYPES: frozenset[DocumentType] = frozenset(
    {
        DocumentType.SOP,
        DocumentType.INCIDENT_REPORT,
        DocumentType.INSPECTION_REPORT,
        DocumentType.MOC,
    }
)

#: Revision labels sort numerically when they are numbers and alphabetically when
#: they are letters, and the two schemes never mix within one document series.
#: "10" must sort after "9", which a string comparison gets wrong.
_NUMERIC_REVISION = re.compile(r"^\d+$")


def derive_doc_number(
    *, title: str, filename: str, head_text: str, doc_type: DocumentType | None = None
) -> DocNumber | None:
    """Find the document's own identifier, or return None.

    None is a perfectly good answer: work-order exports and CSV extracts have no
    document number, and inventing one would create revision groups that do not
    exist.
    """
    body_searchable = doc_type in _BODY_SEARCHABLE_TYPES
    sources = {
        "title": title or "",
        "filename": Path(filename or "").stem.replace("_", " "),
        "head": (head_text or "")[:_HEAD_WINDOW] if body_searchable else "",
    }
    for source_name in _SEARCH_ORDER:
        text = sources[source_name]
        if not text:
            continue
        for kind, pattern in _NUMBER_PATTERNS:
            match = pattern.search(text)
            if match:
                return DocNumber(
                    value=_canonical_number(match.group(1)),
                    method=f"{source_name}:{kind}",
                )
    return None


def _canonical_number(raw: str) -> str:
    """``sop 4412`` / ``SOP-4412`` / ``Sop4412`` all denote the same series."""
    collapsed = re.sub(r"[\s_]+", "-", raw.strip().upper())
    collapsed = re.sub(r"^([A-Z&]+?)-?(\d)", r"\1-\2", collapsed)
    return re.sub(r"-{2,}", "-", collapsed)


def revision_rank(revision: str | None) -> tuple[int, float] | None:
    """Order a revision label. Returns None when it carries no ordering.

    Two schemes, kept apart: numeric revisions ("2", "3", "10") and alphabetic
    ones ("A", "B"). A series uses one or the other; comparing across them is
    meaningless, so the scheme is part of the key and mixed groups fall through
    to date ordering instead.
    """
    if not revision:
        return None
    label = revision.strip().upper()
    if _NUMERIC_REVISION.match(label):
        return (0, float(label))
    if len(label) <= 2 and label.isalpha():
        # A=1, B=2, ... AA sorts after Z.
        value = 0.0
        for char in label:
            value = value * 26 + (ord(char) - ord("A") + 1)
        return (1, value)
    return None


@dataclass(slots=True)
class RevisionOrdering:
    """The outcome of trying to order one document-number series.

    ``levels`` is a list of *revision levels*, oldest first. Each level holds the
    one-or-more documents that assert that revision. The nesting is what makes
    the common real case expressible: SOP-4412 exists as Rev 3 in two formats
    (Markdown source and scanned PDF) and Rev 4 in one, which is three documents
    at two levels. A flat ordering has to call that either a sequence of three --
    inventing an order between two identical revisions -- or unorderable, and
    both are wrong.
    """

    levels: list[list[dict[str, Any]]]
    basis: str
    conflict: bool
    note: str | None = None

    @property
    def current(self) -> list[dict[str, Any]]:
        """Every document at the newest revision level."""
        return self.levels[-1] if self.levels else []

    @property
    def superseded(self) -> list[dict[str, Any]]:
        return [d for level in self.levels[:-1] for d in level]

    @property
    def has_renditions(self) -> bool:
        return any(len(level) > 1 for level in self.levels)


def order_revisions(documents: list[dict[str, Any]]) -> RevisionOrdering:
    """Group documents into revision levels and order the levels.

    Two bases are tried, in order of how directly they express the issuer's
    intent:

    1. **revision label** -- what was printed on the document;
    2. **revision date**, then **issue date** -- when it took effect.

    If neither separates the documents, the series is a conflict rather than a
    guess. Ingestion timestamp is deliberately not a fallback: it records who
    uploaded what first, which has no relationship to which revision is current,
    and using it would produce a confident, arbitrary, and unfalsifiable answer.
    """
    if len(documents) < 2:
        return RevisionOrdering(levels=[list(documents)], basis="single", conflict=False)

    ranks = [revision_rank(d.get("revision")) for d in documents]
    schemes = {r[0] for r in ranks if r is not None}
    if all(r is not None for r in ranks) and len(schemes) == 1:
        return RevisionOrdering(
            levels=_group_by(documents, [r[1] for r in ranks]),  # type: ignore[index]
            basis="revision_label",
            conflict=False,
        )

    # No usable labels: fall back to effective dates. Documents sharing a date
    # form one level, exactly as documents sharing a revision label do.
    for field, basis in (("revised_on", "revised_on"), ("issued_on", "issued_on")):
        values = [d.get(field) for d in documents]
        if all(isinstance(value, date) for value in values):
            return RevisionOrdering(
                levels=_group_by(documents, [v.toordinal() for v in values]),  # type: ignore[union-attr]
                basis=basis,
                conflict=False,
            )

    return RevisionOrdering(
        levels=[list(documents)],
        basis="none",
        conflict=True,
        note=(
            "These documents share a document number but carry no revision label or date "
            "that separates them. All are left current rather than guessing which one an "
            "engineer should follow."
        ),
    )


def _group_by(documents: list[dict[str, Any]], keys: list[float]) -> list[list[dict[str, Any]]]:
    """Bucket documents by ordering key, returning buckets oldest first."""
    buckets: dict[float, list[dict[str, Any]]] = {}
    for key, document in zip(keys, documents, strict=True):
        buckets.setdefault(key, []).append(document)
    return [buckets[key] for key in sorted(buckets)]


async def reconcile(doc_number: str) -> dict[str, Any]:
    """Recompute supersession for one document-number series.

    Run after each ingest rather than incrementally, because a revision can
    arrive out of order -- Rev 3 scanned and uploaded before someone finds Rev 2
    in a filing cabinet -- and an incremental update would leave the chain wrong.
    Recomputing a handful of rows is cheap and always converges to the same state
    regardless of arrival order, which is the property that matters.
    """
    rows = await db.fetch_all(
        """
        SELECT doc_id, title, revision, revised_on, issued_on, created_at,
               superseded_by, is_current, valid_from, valid_to
          FROM documents
         WHERE doc_number = %s
         ORDER BY created_at
        """,
        (doc_number,),
    )
    if not rows:
        return {"doc_number": doc_number, "documents": 0, "status": "absent"}

    ordering = order_revisions(rows)

    if ordering.conflict:
        # Every member stays current and is flagged. This branch is what keeps
        # the module honest: an unresolvable series produces a visible problem
        # rather than an invisible wrong answer.
        await db.execute(
            """
            UPDATE documents
               SET superseded_by = NULL, valid_to = NULL, is_current = true,
                   revision_conflict = true, revision_note = %s
             WHERE doc_number = %s
            """,
            (ordering.note, doc_number),
        )
        log.warning(
            "revisions.conflict", doc_number=doc_number, documents=len(rows), basis=ordering.basis
        )
        return {
            "doc_number": doc_number,
            "documents": len(rows),
            "status": "conflict",
            "basis": ordering.basis,
            "note": ordering.note,
        }

    levels = ordering.levels
    current_level = levels[-1]
    # Every document at the newest level is current. More than one means the
    # revision is held in several formats, which is normal and not a problem.
    current_ids = [d["doc_id"] for d in current_level]
    # The effective date of the replacement, used to close out the ones it
    # replaced. Null when the successor carries no date: "superseded, date
    # unknown" is true, and a fabricated date would flow into compliance
    # reporting as though it were established.
    effective = current_level[0].get("revised_on") or current_level[0].get("issued_on")

    async with db.connection() as conn, conn.cursor() as cur:
        for index, level in enumerate(levels):
            successor = levels[index + 1] if index + 1 < len(levels) else None
            is_current = successor is None
            # A superseded document is closed out when its replacement took
            # effect, not when it was ingested.
            successor_date = None
            if successor is not None:
                successor_date = successor[0].get("revised_on") or successor[0].get("issued_on")
            note = _level_note(level, successor, ordering)
            await cur.execute(
                """
                UPDATE documents
                   SET superseded_by = %s,
                       is_current = %s,
                       valid_from = COALESCE(valid_from, revised_on, issued_on),
                       valid_to = %s,
                       revision_conflict = false,
                       revision_note = %s
                 WHERE doc_id = ANY(%s)
                """,
                (
                    # Points at one representative of the succeeding level. The
                    # full picture lives in the graph, where SUPERSEDES links
                    # every pair; this column can only hold one reference.
                    successor[0]["doc_id"] if successor else None,
                    is_current,
                    successor_date,
                    note,
                    [d["doc_id"] for d in level],
                ),
            )

    superseded = len(ordering.superseded)
    if superseded or ordering.has_renditions:
        log.info(
            "revisions.reconciled",
            doc_number=doc_number,
            documents=len(rows),
            levels=len(levels),
            superseded=superseded,
            basis=ordering.basis,
            current=current_ids,
        )
    return {
        "doc_number": doc_number,
        "documents": len(rows),
        "status": "ok",
        "basis": ordering.basis,
        "levels": len(levels),
        "superseded": superseded,
        "current_doc_ids": current_ids,
        "effective_from": effective.isoformat() if effective else None,
        "chain": [
            {
                "revision": level[0].get("revision"),
                "documents": [
                    {"doc_id": d["doc_id"], "title": d["title"]} for d in level
                ],
            }
            for level in levels
        ],
    }


def _level_note(
    level: list[dict[str, Any]],
    successor: list[dict[str, Any]] | None,
    ordering: RevisionOrdering,
) -> str | None:
    """The sentence shown beside a document explaining its standing."""
    rendition_note = (
        f"Held in {len(level)} formats at this revision; none supersedes another."
        if len(level) > 1
        else ""
    )
    if successor is None:
        return rendition_note or None
    replacement = successor[0]
    label = replacement.get("revision")
    superseded_note = (
        f"Superseded by revision {label}." if label else f"Superseded by {replacement['title']}."
    )
    superseded_note += f" Ordered by {ordering.basis}."
    return f"{superseded_note} {rendition_note}".strip()


async def conflicts() -> list[dict[str, Any]]:
    """Revision groups a human needs to resolve. Surfaced in the UI, not buried."""
    return await db.fetch_all(
        """
        SELECT doc_number, revision_note,
               count(*)::int AS documents,
               array_agg(doc_id ORDER BY created_at) AS doc_ids,
               array_agg(title ORDER BY created_at) AS titles
          FROM documents
         WHERE revision_conflict
         GROUP BY doc_number, revision_note
         ORDER BY doc_number
        """
    )
