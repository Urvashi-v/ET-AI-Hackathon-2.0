"""Entity extraction from chunk text.

A hybrid extractor, because neither half works alone: pure regex misses
everything expressed in prose, and pure LLM extraction is expensive,
non-deterministic and hallucinates tags that are not in the source.

Implemented and running:

``regex``
    Tag grammars (equipment, instrument, line, KKS), document references, clause
    references, dates and quantities-with-units. High precision, zero cost,
    fully deterministic, and it recovers the large majority of mentions in
    industrial text.

``gazetteer``
    Multi-pattern matching against the canonical tags and confirmed aliases
    already in the graph, so extraction improves as the corpus grows. This is
    what catches technician shorthand once a human has confirmed it once.

Defined but not running without a credential:

``llm``
    Schema-constrained extraction of failure modes, causes, actions and
    obligations -- the things expressed as language. The contract is enforced
    here: :func:`validate_extraction` rejects any extraction whose tag or
    evidence quote does not appear verbatim in the source chunk. That check is
    the cheapest hallucination defence available, and it runs regardless of
    which model produced the extraction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from services.common.tags import ParsedTag, TagKind, parse

# ---------------------------------------------------------------------------
# Candidate patterns
# ---------------------------------------------------------------------------

#: Candidate tag-shaped strings. Deliberately permissive: everything it finds is
#: then handed to the tag grammar in services/common/tags.py, which is what
#: actually decides whether the string is a tag. Recall here, precision there.
_TAG_CANDIDATE = re.compile(
    r"""
    (?<![A-Za-z0-9])
    (?:
        \d{1,3}[\s\-‐-―_/]{0,2}                      # optional unit prefix
    )?
    [A-Z]{1,5}                                                  # class / ISA letters
    [\s\-‐-―_/]{0,2}
    \d{2,5}                                                     # sequence
    (?:[\s\-‐-―_/]{0,2}[A-Z])?                        # optional item suffix
    (?![A-Za-z0-9])
    """,
    re.VERBOSE,
)

_KKS_CANDIDATE = re.compile(r"(?<![A-Za-z0-9])\d{1,2}[A-Z]{3}\d{2}[A-Z]{2}\d{3}(?![A-Za-z0-9])")

_LINE_CANDIDATE = re.compile(
    r'(?<![A-Za-z0-9])\d{1,2}(?:\.\d)?\s*(?:"|IN)\s*[-‐-―]?\s*'
    r"[A-Z]{1,3}\s*[-‐-―]?\s*\d{3,5}(?:\s*[-‐-―]?\s*[A-Z0-9]{2,6})?"
)

#: Document and standard references: OISD-STD-105, SOP-4412, WO-4471, CAPA-88,
#: MOC-2023-07, API 610, ISO 14224.
#:
#: Note what is deliberately absent: PSV. A pressure safety valve is a physical
#: asset with an ISA-style tag, so PSV-204 must reach the tag grammar and become
#: an :Instrument node, not be swallowed as a document reference. Prefixes are
#: matched case-sensitively -- lower-casing "IS" or "WO" would match ordinary
#: English words followed by a number.
_DOC_REFERENCE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?P<kind>OISD(?:-STD)?|SOP|WO|MOC|CAPA|NCR|INC|API|ISO|IEC|PTW|JSA|HAZOP)"
    r"[\s\-]?(?P<number>\d{2,6}(?:[-/]\d{1,4})?)"
    r"(?![A-Za-z])"
)

#: Clause references inside a standard: "clause 4.2", "§7.3.1", "para 5(a)".
_CLAUSE_REFERENCE = re.compile(
    r"(?:clause|section|para(?:graph)?|§)\s*(\d+(?:\.\d+){0,3}(?:\([a-z]\))?)", re.I
)

_DATE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"), "%Y-%m-%d"),
    (re.compile(r"\b(\d{2}/\d{2}/\d{4})\b"), "%d/%m/%Y"),
    (re.compile(r"\b(\d{1,2}\s+[A-Z][a-z]{2}\s+\d{4})\b"), "%d %b %Y"),
    (re.compile(r"\b(\d{1,2}-[A-Z][a-z]{2}-\d{4})\b"), "%d-%b-%Y"),
)

#: Quantities with units -- a bare number is never a fact in this domain.
_QUANTITY = re.compile(
    r"(?<![A-Za-z0-9.])(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>barg?|kPa|MPa|psi[ga]?|mm|cm|m|km|kg|t|degC|°C|K|rpm|Hz|kW|MW|A|V|"
    r"m3/h|m³/h|LPM|GPM|hrs?|hours?|mm/yr|micron|µm)"
    r"(?![A-Za-z])",
    re.I,
)

#: Degradation language -- the leading indicator mined from maintenance text.
#: Weights are the documented judgement of how strongly each phrase signals an
#: developing failure; they are configuration, not measurement, and the API
#: labels any score derived from them as model_derived.
DEGRADATION_CUES: dict[str, float] = {
    "seepage": 0.5,
    "weeping": 0.5,
    "slight leak": 0.6,
    "minor leak": 0.6,
    "intermittent": 0.4,
    "unusual noise": 0.7,
    "abnormal sound": 0.7,
    "abnormal noise": 0.7,
    "vibration increased": 0.8,
    "vibration high": 0.8,
    "running hot": 0.7,
    "temperature rising": 0.7,
    "topped up": 0.5,
    "refilled oil": 0.5,
    "tightened gland": 0.6,
    "monitor closely": 0.6,
    "keep watch": 0.3,
    "temporary repair": 0.9,
    "clamp fitted": 0.9,
    "will attend during shutdown": 0.7,
    "deferred to shutdown": 0.7,
}


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TagMention:
    surface_form: str
    normalised: str
    canonical: str
    tag_kind: str
    char_start: int
    char_end: int
    extractor: str
    confidence: float
    parsed: ParsedTag

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface_form": self.surface_form,
            "normalised": self.normalised,
            "canonical": self.canonical,
            "tag_kind": self.tag_kind,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "extractor": self.extractor,
            "confidence": round(self.confidence, 3),
        }


@dataclass(slots=True)
class ExtractionResult:
    tags: list[TagMention] = field(default_factory=list)
    document_refs: list[dict[str, Any]] = field(default_factory=list)
    clause_refs: list[dict[str, Any]] = field(default_factory=list)
    dates: list[dict[str, Any]] = field(default_factory=list)
    quantities: list[dict[str, Any]] = field(default_factory=list)
    degradation_cues: list[dict[str, Any]] = field(default_factory=list)
    functional_locations: list[dict[str, Any]] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return {
            "tags": len(self.tags),
            "document_refs": len(self.document_refs),
            "clause_refs": len(self.clause_refs),
            "dates": len(self.dates),
            "quantities": len(self.quantities),
            "degradation_cues": len(self.degradation_cues),
            "functional_locations": len(self.functional_locations),
        }


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


#: A tag introduced as a functional location is a *position* in the process, not
#: the machine occupying it. ``CDU1-PUMP-101`` parses perfectly well as pump
#: P-101, and left alone the extractor creates a phantom Equipment node sitting
#: alongside the real P-101A and P-101B. That phantom then accumulates evidence
#: and shows up in every count. Functional locations are recovered separately,
#: from the record's own FL field, and modelled as :FunctionalLocation.
_FL_CONTEXT = re.compile(
    r"(?:functional\s+location|floc|fl\s*tag)\s*[:\-]?\s*(?P<tag>[A-Z0-9][A-Z0-9\-‐-―_/ ]{2,30})",
    re.I,
)


def find_functional_location_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of tags introduced as functional locations."""
    return [(m.start("tag"), m.end("tag")) for m in _FL_CONTEXT.finditer(text)]


def find_functional_locations(text: str) -> list[dict[str, Any]]:
    """Functional-location tags with their surface form and span."""
    out: list[dict[str, Any]] = []
    for match in _FL_CONTEXT.finditer(text):
        surface = match.group("tag").strip()
        if not surface:
            continue
        parsed = parse(surface)
        out.append(
            {
                "surface_form": surface,
                "normalised": parsed.normalised,
                "char_start": match.start("tag"),
                "char_end": match.start("tag") + len(surface),
            }
        )
    return out


def find_document_reference_spans(text: str) -> list[tuple[int, int]]:
    """Character spans occupied by document/standard references.

    These are masked before tag extraction. Without the mask, ``SOP-4412``
    parses cleanly as pump P-4412 and ``CAPA-88`` as agitator A-88 -- inventing
    assets that do not exist and attaching real evidence to them. Suppressing
    that is not cosmetic: a phantom asset corrupts every count computed over the
    graph.
    """
    return [(m.start(), m.end()) for m in _DOC_REFERENCE.finditer(text)]


def extract_tags(
    text: str,
    *,
    gazetteer: dict[str, str] | None = None,
    exclude_spans: list[tuple[int, int]] | None = None,
) -> list[TagMention]:
    """Find tag mentions with their exact character offsets.

    ``gazetteer`` maps a normalised surface form to a canonical tag; entries come
    from assets and confirmed aliases already in the store, which is how the
    system acquires plant-specific vocabulary as it ingests.

    ``exclude_spans`` suppresses candidates inside document references; when it
    is not supplied the references are computed here, so calling this function
    directly is as safe as going through :func:`extract_all`.
    """
    found: dict[tuple[int, int], TagMention] = {}
    blocked = (
        exclude_spans
        if exclude_spans is not None
        else find_document_reference_spans(text) + find_functional_location_spans(text)
    )

    for pattern, extractor in (
        (_LINE_CANDIDATE, "regex:line"),
        (_KKS_CANDIDATE, "regex:kks"),
        (_TAG_CANDIDATE, "regex:tag"),
    ):
        for match in pattern.finditer(text):
            surface = match.group(0).strip()
            parsed = parse(surface)
            if parsed.kind is TagKind.UNPARSED:
                continue
            span = (match.start(), match.start() + len(surface))
            if any(_overlaps(span, existing) for existing in found):
                continue
            if any(_overlaps(span, blocked_span) for blocked_span in blocked):
                continue
            found[span] = TagMention(
                surface_form=surface,
                normalised=parsed.normalised,
                canonical=parsed.canonical,
                tag_kind=parsed.kind.value,
                char_start=span[0],
                char_end=span[1],
                extractor=extractor,
                # Parsed against a known grammar: high precision by construction.
                confidence=0.95,
                parsed=parsed,
            )

    if gazetteer:
        lowered = text.lower()
        for surface_lower, canonical in gazetteer.items():
            if not surface_lower or len(surface_lower) < 3:
                continue
            start = 0
            while (idx := lowered.find(surface_lower, start)) != -1:
                span = (idx, idx + len(surface_lower))
                start = idx + len(surface_lower)
                if any(_overlaps(span, existing) for existing in found):
                    continue
                surface = text[span[0] : span[1]]
                parsed = parse(canonical)
                found[span] = TagMention(
                    surface_form=surface,
                    normalised=parsed.normalised,
                    canonical=canonical,
                    tag_kind=parsed.kind.value,
                    char_start=span[0],
                    char_end=span[1],
                    extractor="gazetteer",
                    confidence=0.9,
                    parsed=parsed,
                )

    return sorted(found.values(), key=lambda m: m.char_start)


def extract_all(text: str, *, gazetteer: dict[str, str] | None = None) -> ExtractionResult:
    """Run every deterministic extractor over one chunk.

    Order matters: document references are resolved first so their spans can be
    masked out of tag extraction.
    """
    result = ExtractionResult()

    reference_spans: list[tuple[int, int]] = []
    for match in _DOC_REFERENCE.finditer(text):
        reference_spans.append((match.start(), match.end()))
        result.document_refs.append(
            {
                "kind": match.group("kind").upper().replace("-STD", ""),
                "number": match.group("number"),
                "surface_form": match.group(0),
                "char_start": match.start(),
                "char_end": match.end(),
            }
        )

    result.functional_locations = find_functional_locations(text)
    masked = reference_spans + find_functional_location_spans(text)
    result.tags = extract_tags(text, gazetteer=gazetteer, exclude_spans=masked)

    for match in _CLAUSE_REFERENCE.finditer(text):
        result.clause_refs.append(
            {"clause": match.group(1), "surface_form": match.group(0), "char_start": match.start()}
        )

    for pattern, fmt in _DATE_PATTERNS:
        for match in pattern.finditer(text):
            parsed_date = _try_date(match.group(1), fmt)
            if parsed_date:
                result.dates.append(
                    {
                        "date": parsed_date.isoformat(),
                        "surface_form": match.group(1),
                        "char_start": match.start(),
                    }
                )

    for match in _QUANTITY.finditer(text):
        result.quantities.append(
            {
                "value": float(match.group("value")),
                "unit": match.group("unit"),
                "surface_form": match.group(0),
                "char_start": match.start(),
            }
        )

    lowered = text.lower()
    for cue, weight in DEGRADATION_CUES.items():
        idx = lowered.find(cue)
        if idx != -1:
            result.degradation_cues.append(
                {
                    "cue": cue,
                    "weight": weight,
                    "char_start": idx,
                    "quote": text[idx : idx + len(cue)],
                }
            )

    return result


def validate_extraction(
    *, source_text: str, asserted_tag: str | None, evidence_quotes: list[str]
) -> tuple[bool, str | None]:
    """The verbatim-evidence check.

    Any extraction -- from any extractor, including an LLM -- must be supported
    by spans that literally occur in the source chunk. One string containment
    test per claim, no second model, deterministic. An extraction that fails this
    is rejected outright rather than stored with a lower confidence.
    """
    if asserted_tag and asserted_tag not in source_text:
        return False, f"asserted tag {asserted_tag!r} does not occur in the source text"
    for quote in evidence_quotes:
        if quote and quote not in source_text:
            return (
                False,
                f"evidence quote {quote[:60]!r} does not occur verbatim in the source text",
            )
    return True, None


def build_gazetteer(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Build the lower-cased surface-form -> canonical-tag map from stored
    assets and their confirmed aliases."""
    gazetteer: dict[str, str] = {}
    for row in rows:
        canonical = row.get("canonical_tag")
        surface = row.get("surface_form") or canonical
        if canonical and surface:
            gazetteer[str(surface).lower()] = str(canonical)
    return gazetteer


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _try_date(value: str, fmt: str) -> date | None:
    try:
        return datetime.strptime(value, fmt).date()
    except ValueError:
        return None
