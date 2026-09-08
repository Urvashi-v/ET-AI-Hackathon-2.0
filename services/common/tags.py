"""Industrial tag anatomy: normalisation, parsing, blocking and match scoring.

This module is the load-bearing piece of entity resolution. The same physical
pump is written six different ways across a plant's systems, and if these are
not unified the knowledge graph is a set of disconnected islands.

The approach is **parse-based, not fuzzy-first**. Industrial tags are grammatical
identifiers (IEC 81346 for equipment, ISA 5.1 for instruments), so they can be
decomposed into ``{unit, class, sequence, suffix}``. Two strings that decompose
identically are the same asset with far more confidence than any edit distance
can justify. Fuzzy similarity is the fallback, never the primary signal.

The single most important rule encoded here::

    P-101A and P-101B are NOT the same asset.

They are a duty/standby pair -- siblings. A naive edit-distance matcher merges
them (one character in six) and silently corrupts every downstream failure
statistic. :func:`score_pair` returns ``TagRelation.SIBLING`` for that case, which
the graph writer turns into a ``SIBLING_OF`` edge rather than a merge.

No I/O, no configuration, no side effects -- so it is exhaustively unit-testable
and its behaviour is identical in the API, the worker and the test suite.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import StrEnum

from rapidfuzz.distance import JaroWinkler

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

#: Equipment class codes and their meaning. Kept here as the parser's validation
#: set; the same table is seeded into Neo4j as :EquipmentClass nodes so the
#: ontology is data, not code (see database/cypher/002_ontology_seed.cypher).
EQUIPMENT_CLASS_CODES: dict[str, str] = {
    "P": "Pump",
    "C": "Compressor",
    "K": "Blower",
    "E": "Heat exchanger",
    "V": "Vessel / drum",
    "D": "Drum",
    "T": "Tower / column",
    "TK": "Tank",
    "M": "Motor",
    "F": "Furnace / filter",
    "B": "Boiler",
    "R": "Reactor",
    "G": "Generator",
    "A": "Agitator",
    "H": "Heater",
    "S": "Separator",
    "X": "Miscellaneous / package",
    "FN": "Fan",
    "AG": "Air cooler",
    "CV": "Conveyor",
}

#: Word-level abbreviations seen in functional-location tags and CMMS exports.
#: Applied per separator-delimited segment, so ``CDU1-PUMP-101B`` becomes
#: ``CDU1-P-101B`` before compaction.
SEGMENT_ABBREVIATIONS: dict[str, str] = {
    "PUMP": "P",
    "PMP": "P",
    "PU": "P",
    "COMP": "C",
    "COMPRESSOR": "C",
    "BLOWER": "K",
    "EXCH": "E",
    "EXCHANGER": "E",
    "HX": "E",
    "HTX": "E",
    "VES": "V",
    "VESSEL": "V",
    "DRUM": "D",
    "COLUMN": "T",
    "TOWER": "T",
    "TANK": "TK",
    "MTR": "M",
    "MOTOR": "M",
    "FURN": "F",
    "FURNACE": "F",
    "FILTER": "F",
    "BOILER": "B",
    "REACTOR": "R",
    "GEN": "G",
    "FAN": "FN",
}

#: ISA 5.1 first letters (measured variable).
ISA_FIRST_LETTERS: dict[str, str] = {
    "A": "Analysis",
    "B": "Burner / combustion",
    "C": "Conductivity",
    "D": "Density",
    "E": "Voltage",
    "F": "Flow",
    "H": "Hand (manual)",
    "I": "Current",
    "J": "Power",
    "K": "Time / schedule",
    "L": "Level",
    "M": "Moisture",
    "P": "Pressure",
    "Q": "Quantity / totalised",
    "R": "Radiation",
    "S": "Speed / frequency",
    "T": "Temperature",
    "V": "Vibration",
    "W": "Weight / force",
    "Z": "Position",
}

#: ISA 5.1 succeeding letters (readout / output function).
ISA_SUCCEEDING_LETTERS: dict[str, str] = {
    "A": "Alarm",
    "C": "Control",
    "E": "Element (primary sensor)",
    "G": "Glass / gauge",
    "I": "Indicate",
    "L": "Low",
    "H": "High",
    "R": "Record",
    "S": "Switch",
    "T": "Transmit",
    "V": "Valve",
    "Y": "Relay / compute",
    "Z": "Driver / actuator",
}

#: All Unicode separator characters that appear in real tags. The non-ASCII
#: hyphens are what a Word autocorrect leaves behind and they are invisible to a
#: human reviewer -- U+2011 NON-BREAKING HYPHEN is the classic offender.
_SEPARATOR_CHARS = "\\s\\-\\u2010\\u2011\\u2012\\u2013\\u2014\\u2015\\u2212_/\\.:"
_SEPARATOR_RUN = re.compile(f"[{_SEPARATOR_CHARS}]+")
#: The inch mark is retained: it is the only unambiguous discriminator between a
#: line number (``6"-P-1501-A1A``) and an equipment tag (``10-P-1501``).
_NON_TAG_CHARS = re.compile(r'[^A-Z0-9\-"]')

# Equipment patterns, most specific first. ``unit`` is an optional plant/area
# prefix -- numeric (``10-P-101-B``) or alphanumeric (``CDU1-PUMP-101B``).
_EQUIPMENT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(?P<unit>\d{1,3})(?P<cls>[A-Z]{1,2})(?P<seq>\d{2,5})(?P<sfx>[A-Z])?$"),
    re.compile(r"^(?P<unit>[A-Z]{2,6}\d{0,2})(?P<cls>[A-Z]{1,2})(?P<seq>\d{2,5})(?P<sfx>[A-Z])?$"),
    re.compile(r"^(?P<unit>)(?P<cls>[A-Z]{1,2})(?P<seq>\d{2,5})(?P<sfx>[A-Z])?$"),
)

_INSTRUMENT_PATTERN = re.compile(r"^(?P<unit>\d{0,3})(?P<cls>[A-Z]{2,5})(?P<seq>\d{2,5})$")

#: Line numbers look like ``6"-P-1501-A1A`` (size, service, sequence, pipe spec).
#: The size marker is mandatory -- without it the string is an equipment tag.
_LINE_PATTERN = re.compile(
    r'^(?P<size>\d{1,2}(?:\.\d)?)(?:"|IN)'
    r"(?P<service>[A-Z]{1,3})(?P<seq>\d{3,5})(?P<spec>[A-Z0-9]{2,6})?$"
)

#: IEC 81346 / KKS reference designation used in Indian and European power
#: plants, e.g. ``10LAC20AP001``: unit 10, system LAC (feedwater), subsystem 20,
#: equipment type AP (pump), sequence 001. Decomposed but deliberately NOT
#: mapped onto process-plant class codes -- that mapping is plant-specific and
#: inventing one would assert a fact we cannot evidence.
_KKS_PATTERN = re.compile(
    r"^(?P<unit>\d{1,2})(?P<system>[A-Z]{3})(?P<subsystem>\d{2})"
    r"(?P<eqtype>[A-Z]{2})(?P<seq>\d{3})$"
)


class TagKind(StrEnum):
    EQUIPMENT = "equipment"
    INSTRUMENT = "instrument"
    LINE = "line"
    KKS = "kks"
    UNPARSED = "unparsed"


class TagRelation(StrEnum):
    """How two tag strings relate. Drives what the graph writer does."""

    SAME = "same"  # merge into one canonical asset
    SIBLING = "sibling"  # duty/standby pair -> SIBLING_OF edge, never a merge
    DIFFERENT = "different"  # leave separate
    UNKNOWN = "unknown"  # below thresholds -> human review queue


@dataclass(frozen=True, slots=True)
class ParsedTag:
    """A decomposed industrial tag. ``raw`` is retained as provenance."""

    raw: str
    normalised: str
    compact: str
    kind: TagKind
    unit: str | None = None
    cls: str | None = None
    seq: str | None = None
    suffix: str | None = None
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def parsed(self) -> bool:
        return self.kind is not TagKind.UNPARSED

    @property
    def canonical(self) -> str:
        """The canonical string form used as the graph's unique key.

        Deliberately excludes the unit prefix: ``10-P-101-B`` and ``P-101B`` are
        the same pump written by two systems, and a missing prefix is missing
        information rather than conflicting information. The unit, when present,
        is kept as a property and used as a *disambiguating* signal in
        :func:`score_pair`.
        """
        if self.kind is TagKind.EQUIPMENT and self.cls and self.seq:
            return f"{self.cls}-{self.seq}{self.suffix or ''}"
        if self.kind is TagKind.INSTRUMENT and self.cls and self.seq:
            return f"{self.cls}-{self.seq}"
        if self.kind is TagKind.LINE:
            size = self.extra.get("size", "")
            service = self.extra.get("service", "")
            return f'{size}"-{service}-{self.seq}' if self.seq else self.normalised
        if self.kind is TagKind.KKS:
            return self.compact
        return self.normalised

    @property
    def class_label(self) -> str | None:
        if self.kind is TagKind.EQUIPMENT and self.cls:
            return EQUIPMENT_CLASS_CODES.get(self.cls)
        if self.kind is TagKind.INSTRUMENT and self.cls:
            return decode_instrument_function(self.cls)
        return None

    def to_dict(self) -> dict[str, object]:
        return {
            "raw": self.raw,
            "normalised": self.normalised,
            "canonical": self.canonical,
            "kind": self.kind.value,
            "unit": self.unit,
            "class_code": self.cls,
            "class_label": self.class_label,
            "sequence": self.seq,
            "suffix": self.suffix,
            "extra": dict(self.extra),
        }


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def normalise(raw: str) -> str:
    """Canonicalise a raw tag string to ``UPPER-CASE-HYPHEN-SEPARATED`` form.

    NFKC folds compatibility characters; every separator variant (including the
    non-ASCII hyphens that Word autocorrect inserts) collapses to a single
    ``-``; known word abbreviations are mapped to class codes segment by
    segment; and leading zeros are stripped from purely numeric segments so
    ``P-0101B`` and ``P-101B`` agree.
    """
    if not raw:
        return ""
    s = unicodedata.normalize("NFKC", raw).upper().strip()
    s = _SEPARATOR_RUN.sub("-", s)
    s = _NON_TAG_CHARS.sub("", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    if not s:
        return ""

    out: list[str] = []
    for seg in s.split("-"):
        if not seg:
            continue
        seg = SEGMENT_ABBREVIATIONS.get(seg, seg)
        if seg.isdigit():
            seg = str(int(seg)) if seg != "0" * len(seg) else "0"
        out.append(seg)
    return "-".join(out)


def compact(normalised: str) -> str:
    """Separator-free form used for pattern matching (``P-101-B`` -> ``P101B``)."""
    return normalised.replace("-", "")


def decode_instrument_function(letters: str) -> str | None:
    """Decode an ISA 5.1 instrument letter group, e.g. ``PIC`` or ``LSHH``."""
    if not letters:
        return None
    first = ISA_FIRST_LETTERS.get(letters[0])
    if first is None:
        return None
    parts = [first]
    for ch in letters[1:]:
        succ = ISA_SUCCEEDING_LETTERS.get(ch)
        if succ is None:
            return None
        parts.append(succ)
    return " ".join(parts)


def _split_leading_zeros(seq: str) -> str:
    return str(int(seq)) if seq.isdigit() else seq


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse(raw: str) -> ParsedTag:
    """Decompose a tag string. Always returns a :class:`ParsedTag`.

    An unparseable string yields ``kind=UNPARSED`` rather than ``None`` so that
    callers keep the provenance of what they could not understand -- technician
    shorthand such as "the B pump" must be retained and resolved by context,
    never discarded.
    """
    norm = normalise(raw)
    comp = compact(norm)
    if not comp:
        return ParsedTag(raw=raw, normalised=norm, compact=comp, kind=TagKind.UNPARSED)

    # 1. Line numbers. The mandatory size marker makes these unambiguous, so
    #    they are tested first.
    m = _LINE_PATTERN.match(comp)
    if m:
        gd = m.groupdict()
        return ParsedTag(
            raw=raw,
            normalised=norm,
            compact=comp,
            kind=TagKind.LINE,
            seq=_split_leading_zeros(gd["seq"]),
            extra={k: v for k, v in gd.items() if v and k in ("size", "service", "spec")},
        )

    # 2. KKS reference designations (power plants).
    m = _KKS_PATTERN.match(comp)
    if m:
        gd = m.groupdict()
        return ParsedTag(
            raw=raw,
            normalised=norm,
            compact=comp,
            kind=TagKind.KKS,
            unit=gd["unit"],
            seq=gd["seq"],
            extra={
                "scheme": "KKS",
                "system": gd["system"],
                "subsystem": gd["subsystem"],
                "equipment_type": gd["eqtype"],
            },
        )

    # 3. Instruments before equipment. ``PIC-101`` would otherwise be read as
    #    unit "PI" + class "C" (compressor) by the alphanumeric-prefix pattern.
    #    The guard is that the letter group must decode fully under ISA 5.1 AND
    #    must not itself be a known equipment class code -- which is what keeps
    #    ``TK-201`` a tank rather than a Temperature/Time instrument.
    m = _INSTRUMENT_PATTERN.match(comp)
    if m:
        cls = m.group("cls")
        if cls not in EQUIPMENT_CLASS_CODES and decode_instrument_function(cls):
            return ParsedTag(
                raw=raw,
                normalised=norm,
                compact=comp,
                kind=TagKind.INSTRUMENT,
                unit=m.group("unit") or None,
                cls=cls,
                seq=_split_leading_zeros(m.group("seq")),
            )

    # 4. Equipment.
    for pattern in _EQUIPMENT_PATTERNS:
        m = pattern.match(comp)
        if not m:
            continue
        gd = m.groupdict()
        cls = gd.get("cls") or ""
        if cls not in EQUIPMENT_CLASS_CODES:
            continue
        return ParsedTag(
            raw=raw,
            normalised=norm,
            compact=comp,
            kind=TagKind.EQUIPMENT,
            unit=(gd.get("unit") or None),
            cls=cls,
            seq=_split_leading_zeros(gd["seq"]),
            suffix=gd.get("sfx") or None,
        )

    return ParsedTag(raw=raw, normalised=norm, compact=comp, kind=TagKind.UNPARSED)


def blocking_key(tag: ParsedTag | str) -> str:
    """Cheap key that co-locates all plausible matches.

    Comparing every mention against every other is O(n^2) and will not finish on
    a real corpus. Candidates are only ever compared inside a block. The key
    deliberately excludes the A/B suffix so that siblings land in the same block
    and can be *linked* rather than silently missed.
    """
    parsed = parse(tag) if isinstance(tag, str) else tag
    if parsed.kind in (TagKind.EQUIPMENT, TagKind.INSTRUMENT) and parsed.cls and parsed.seq:
        return f"{parsed.kind.value}|{parsed.cls}|{parsed.seq}"
    if parsed.kind is TagKind.LINE and parsed.seq:
        return f"line|{parsed.extra.get('service', '')}|{parsed.seq}"
    if parsed.kind is TagKind.KKS:
        return f"kks|{parsed.extra.get('system', '')}|{parsed.seq}"
    return f"unparsed|{parsed.compact[:8]}"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MatchResult:
    score: float
    relation: TagRelation
    method: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "score": round(self.score, 4),
            "relation": self.relation.value,
            "method": self.method,
            "reason": self.reason,
        }


def score_pair(
    a_raw: str,
    b_raw: str,
    *,
    context_similarity: float = 0.0,
    known_alias: bool = False,
) -> MatchResult:
    """Score whether two tag strings denote the same physical asset.

    ``context_similarity`` in [0, 1] expresses whether the surrounding evidence
    agrees (same document family, same unit, same equipment class co-mentioned).
    It can only *raise* a fuzzy score; it can never override a parse-level
    contradiction, because a contradiction is evidence and a context hint is not.
    """
    if known_alias:
        return MatchResult(1.0, TagRelation.SAME, "alias_table", "human-confirmed alias")

    pa, pb = parse(a_raw), parse(b_raw)

    if pa.parsed and pb.parsed:
        if pa.kind is not pb.kind:
            return MatchResult(
                0.05,
                TagRelation.DIFFERENT,
                "parse",
                f"different tag kinds ({pa.kind.value} vs {pb.kind.value})",
            )

        core_match = pa.cls == pb.cls and pa.seq == pb.seq
        if core_match:
            if pa.suffix == pb.suffix:
                if pa.unit and pb.unit and pa.unit != pb.unit:
                    return MatchResult(
                        0.15,
                        TagRelation.DIFFERENT,
                        "parse",
                        f"conflicting unit prefix ({pa.unit} vs {pb.unit})",
                    )
                if pa.unit == pb.unit:
                    return MatchResult(
                        0.97, TagRelation.SAME, "parse", "all tag segments match exactly"
                    )
                # Missing is not conflicting: one system omits the plant prefix.
                return MatchResult(
                    0.85,
                    TagRelation.SAME,
                    "parse",
                    "class/sequence/suffix match; unit prefix present on one side only",
                )
            # THE rule that protects every downstream statistic.
            return MatchResult(
                0.30,
                TagRelation.SIBLING,
                "parse",
                (
                    f"identical class and sequence but different item suffix "
                    f"({pa.suffix or '-'} vs {pb.suffix or '-'}): parallel train / "
                    f"duty-standby pair, not the same asset"
                ),
            )
        return MatchResult(
            0.10,
            TagRelation.DIFFERENT,
            "parse",
            "class or sequence differs",
        )

    # At least one side did not parse -- fall back to string similarity, which
    # can never on its own justify an automatic merge.
    similarity = (
        JaroWinkler.similarity(pa.compact, pb.compact) if pa.compact and pb.compact else 0.0
    )
    score = min(1.0, similarity * 0.6 + 0.2 * max(0.0, min(1.0, context_similarity)))
    relation = TagRelation.UNKNOWN if score >= 0.45 else TagRelation.DIFFERENT
    return MatchResult(
        score,
        relation,
        "fuzzy",
        f"unparsed tag; jaro-winkler={similarity:.3f}, context={context_similarity:.2f}",
    )


def decide(
    result: MatchResult,
    *,
    auto_merge_threshold: float,
    review_threshold: float,
) -> str:
    """Map a score to an action. Three outcomes, never two.

    The middle band matters: an ambiguous link is created and flagged
    ``needs_review`` rather than being silently asserted or silently dropped.
    """
    if result.relation is TagRelation.SIBLING:
        return "link_sibling"
    if result.relation is TagRelation.SAME and result.score >= auto_merge_threshold:
        return "merge"
    if result.score >= review_threshold:
        return "needs_review"
    return "separate"
