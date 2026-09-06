"""Query understanding: intent classification and entity linking.

This one cheap step fixes most of the failure modes people blame on "RAG": an
aggregate question ("how many seal failures last year?") routed to top-k
retrieval returns k documents rather than a count, and no amount of reranking
fixes that. Routing by intent sends it to a graph aggregation instead.

The classifier is **deterministic and rule-based**, and that is a deliberate
choice rather than a placeholder. The intents are distinguished by a small set of
lexical markers ("how many", "why does", "what is the spare for") that are stable
across phrasings, so rules are accurate here, cost nothing, and -- crucially --
give the same answer every time, which the evaluation harness depends on. The
returned ``method`` records which rule fired, so a misrouted question is
diagnosable.

Entity linking resolves informal reference ("the B pump", "that standby") using
the user's own context, exactly as a colleague standing next to them would.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from services.common.schemas import QueryIntent, UserContext
from services.common.tags import TagKind, parse
from services.ingest.extract import extract_tags

# Ordered: the first pattern that matches wins, so the more specific intents are
# tested before the general ones.
_INTENT_RULES: tuple[tuple[QueryIntent, float, str, re.Pattern[str]], ...] = (
    (
        QueryIntent.AGGREGATE,
        0.9,
        "counting/ranking marker",
        re.compile(
            r"\b(how many|how much|count|total|sum|average|mean|median|top\s+\d+|"
            r"most\s+\w+|worst|bad actor|rank|trend|per year|per month|breakdown by)\b",
            re.I,
        ),
    ),
    (
        QueryIntent.PROCEDURAL,
        0.88,
        "procedure marker",
        re.compile(
            r"\b(how do i|how to|what are the steps|steps to|procedure for|"
            r"how should|safely (isolate|start|stop)|what permit|do i need a permit|"
            r"what must be (confirmed|checked|done)|before start\w*|prior to start\w*|"
            r"preconditions?|prerequisites?)\b",
            re.I,
        ),
    ),
    (
        QueryIntent.DIAGNOSTIC,
        0.88,
        "causal marker",
        re.compile(
            r"\b(why (does|is|did|do)|root cause|what caused|keeps? (failing|tripping|"
            r"leaking|vibrating)|reason for|diagnos)\w*\b",
            re.I,
        ),
    ),
    (
        QueryIntent.COMPARATIVE,
        0.85,
        "comparison marker",
        re.compile(r"\b(compare|versus|vs\.?|difference between|better than|which of)\b", re.I),
    ),
    (
        QueryIntent.MULTI_HOP,
        0.82,
        "relational marker",
        re.compile(
            r"\b(spare for|standby for|downstream of|upstream of|feeds? into|fed by|"
            r"affected if|impact of (taking|isolating)|connected to|isolat\w+ (points?|valves?)|"
            r"what else)\b",
            re.I,
        ),
    ),
)

#: Domain shorthand expanded before matching, so "vib" reaches the diagnostic
#: rule and BM25 sees the full token.
ABBREVIATIONS: dict[str, str] = {
    "vib": "vibration",
    "temp": "temperature",
    "press": "pressure",
    "disch": "discharge",
    "suct": "suction",
    "mech": "mechanical",
    "bearg": "bearing",
    "brg": "bearing",
    "pm": "preventive maintenance",
    "cm": "corrective maintenance",
    "wo": "work order",
    "sd": "shutdown",
    "s/d": "shutdown",
    "ptw": "permit to work",
    "loto": "lock out tag out",
    "npsh": "net positive suction head",
    "rca": "root cause analysis",
    "moc": "management of change",
}

#: Informal references a technician actually uses. Resolved against user context.
_INFORMAL = re.compile(
    r"\b(the|that|this)\s+(?P<suffix>[ab])\s+(pump|compressor|motor|blower|fan)\b|"
    r"\b(std\s*by|standby|stand-by|spare)\s+(pump|compressor|motor)\b",
    re.I,
)

_TIME_WINDOW = re.compile(
    r"\b(last|past|previous)\s+(?P<n>\d+|one|two|three|five|ten)\s+(?P<unit>day|week|month|year)s?\b"
    r"|\b(this|current)\s+(?P<unit2>year|month|quarter)\b"
    r"|\b(?P<year>19\d{2}|20\d{2})\b",
    re.I,
)

_WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "five": 5, "ten": 10}


@dataclass(slots=True)
class QueryUnderstanding:
    original: str
    normalised: str
    intent: QueryIntent
    intent_confidence: float
    method: str
    entities: list[dict[str, Any]] = field(default_factory=list)
    unresolved_references: list[str] = field(default_factory=list)
    time_window: dict[str, Any] | None = None
    requires_graph: bool = False
    aggregation: str | None = None

    def entity_tags(self) -> list[str]:
        return [e["canonical_tag"] for e in self.entities if e.get("canonical_tag")]

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.value,
            "confidence": round(self.intent_confidence, 3),
            "method": self.method,
            "entities": self.entities,
            "unresolved_references": self.unresolved_references,
            "time_window": self.time_window,
            "requires_graph": self.requires_graph,
            "aggregation": self.aggregation,
        }


def expand_abbreviations(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        word = match.group(0)
        expansion = ABBREVIATIONS.get(word.lower())
        return expansion if expansion else word

    return re.sub(r"\b[a-zA-Z/]{2,5}\b", replace, text)


def understand(question: str, ctx: UserContext | None = None) -> QueryUnderstanding:
    ctx = ctx or UserContext()
    normalised = expand_abbreviations(question.strip())

    intent = QueryIntent.LOOKUP
    confidence = 0.55
    method = "default:lookup"
    for candidate, weight, description, pattern in _INTENT_RULES:
        if pattern.search(normalised):
            intent, confidence, method = candidate, weight, f"rule:{description}"
            break

    entities, unresolved = _link_entities(normalised, ctx)

    # A question naming no asset the corpus knows, and no informal reference we
    # can anchor, is a candidate for abstention. The decision is not made here --
    # retrieval still runs, and the confidence layer decides on real evidence.
    if not entities and unresolved:
        method += "; unresolved asset reference"

    aggregation = _aggregation_of(normalised) if intent is QueryIntent.AGGREGATE else None
    requires_graph = intent in (
        QueryIntent.MULTI_HOP,
        QueryIntent.DIAGNOSTIC,
        QueryIntent.AGGREGATE,
        QueryIntent.COMPARATIVE,
    )

    return QueryUnderstanding(
        original=question,
        normalised=normalised,
        intent=intent,
        intent_confidence=confidence,
        method=method,
        entities=entities,
        unresolved_references=unresolved,
        time_window=_time_window_of(normalised),
        requires_graph=requires_graph,
        aggregation=aggregation,
    )


def _link_entities(text: str, ctx: UserContext) -> tuple[list[dict[str, Any]], list[str]]:
    entities: list[dict[str, Any]] = []
    seen: set[str] = set()

    for mention in extract_tags(text):
        if mention.canonical in seen:
            continue
        seen.add(mention.canonical)
        entities.append(
            {
                "surface_form": mention.surface_form,
                "canonical_tag": mention.canonical,
                "tag_kind": mention.tag_kind,
                "method": "tag_grammar",
                "confidence": mention.confidence,
            }
        )

    unresolved: list[str] = []
    for match in _INFORMAL.finditer(text):
        surface = match.group(0)
        resolved = _resolve_informal(surface, match.groupdict().get("suffix"), ctx)
        if resolved and resolved not in seen:
            seen.add(resolved)
            entities.append(
                {
                    "surface_form": surface,
                    "canonical_tag": resolved,
                    "tag_kind": "equipment",
                    "method": "coreference_from_user_context",
                    "confidence": 0.7,
                }
            )
        elif not resolved:
            unresolved.append(surface)

    # The user's own context is itself an anchor: a technician assigned to a work
    # order on P-101B asking "why does it keep failing" means that pump.
    for source, value in (("user_ctx.asset_tag", ctx.asset_tag), ("user_ctx.work_order", None)):
        if value:
            parsed = parse(value)
            if parsed.kind is not TagKind.UNPARSED and parsed.canonical not in seen:
                seen.add(parsed.canonical)
                entities.append(
                    {
                        "surface_form": value,
                        "canonical_tag": parsed.canonical,
                        "tag_kind": parsed.kind.value,
                        "method": source,
                        "confidence": 0.8,
                    }
                )

    return entities, unresolved


def _resolve_informal(surface: str, suffix: str | None, ctx: UserContext) -> str | None:
    """Resolve "the B pump" against the asset the user is currently working on.

    Only ever produces a tag by substituting the suffix into a *known* anchor
    tag. It never guesses a sequence number, because a wrong asset is worse than
    no asset.
    """
    anchor = ctx.asset_tag
    if not anchor:
        return None
    parsed = parse(anchor)
    if parsed.kind is not TagKind.EQUIPMENT or not parsed.cls or not parsed.seq:
        return None
    if suffix:
        return f"{parsed.cls}-{parsed.seq}{suffix.upper()}"
    if re.search(r"std\s*by|standby|stand-by|spare", surface, re.I) and parsed.suffix:
        # The standby is the sibling: same class and sequence, other suffix.
        other = "B" if parsed.suffix.upper() == "A" else "A"
        return f"{parsed.cls}-{parsed.seq}{other}"
    return None


def _time_window_of(text: str) -> dict[str, Any] | None:
    match = _TIME_WINDOW.search(text)
    if not match:
        return None
    groups = match.groupdict()
    if groups.get("year"):
        return {"kind": "year", "year": int(groups["year"])}
    if groups.get("unit2"):
        return {"kind": "current", "unit": groups["unit2"].lower()}
    raw_n = (groups.get("n") or "1").lower()
    n = _WORD_NUMBERS.get(raw_n)
    if n is None:
        n = int(raw_n) if raw_n.isdigit() else 1
    return {"kind": "relative", "n": n, "unit": (groups.get("unit") or "year").lower()}


def _aggregation_of(text: str) -> str:
    lowered = text.lower()
    if re.search(r"\b(how many|count|number of)\b", lowered):
        return "count"
    if re.search(r"\b(total|sum)\b", lowered):
        return "sum"
    if re.search(r"\b(top|rank|worst|bad actor|most)\b", lowered):
        return "rank"
    if re.search(r"\b(trend|over time|per (year|month))\b", lowered):
        return "trend"
    if re.search(r"\b(average|mean|median)\b", lowered):
        return "average"
    return "count"
