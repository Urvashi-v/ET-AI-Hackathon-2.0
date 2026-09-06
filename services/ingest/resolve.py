"""Entity resolution: mentions become canonical assets.

The four-stage pipeline from the blueprint, implemented against the store:

1. **Normalise** -- Unicode folding, separator collapsing, abbreviation
   expansion (``services.common.tags.normalise``).
2. **Parse** -- decompose into ``{unit, class, sequence, suffix}`` using the tag
   grammars.
3. **Block** -- generate a blocking key and only ever compare candidates inside
   the same block. Comparing all pairs is O(n^2) and does not finish.
4. **Score and decide** -- combine parse agreement, string similarity and alias
   evidence into one confidence, then choose one of four actions.

The decision is deliberately four-way, not two-way:

``merge``          resolve the mention onto an existing asset
``link_sibling``   P-101A / P-101B -- record a SIBLING_OF edge, never merge
``needs_review``   ambiguous: create the link and flag it for a human
``separate``       create a new canonical asset

Nothing here writes to Neo4j; it returns decisions, and ``graph_writer`` applies
them. That keeps resolution pure enough to test exhaustively against fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from services.common.config import get_settings
from services.common.ids import asset_id as make_asset_id
from services.common.logging import get_logger
from services.common.tags import (
    MatchResult,
    TagKind,
    TagRelation,
    blocking_key,
    decide,
    parse,
    score_pair,
)
from services.ingest.extract import TagMention

log = get_logger(__name__)


@dataclass(slots=True)
class ResolutionDecision:
    mention: TagMention
    action: str  # merge | link_sibling | needs_review | separate
    asset_id: str
    canonical_tag: str
    score: float
    method: str
    reason: str
    needs_review: bool
    sibling_of: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface_form": self.mention.surface_form,
            "action": self.action,
            "canonical_tag": self.canonical_tag,
            "score": round(self.score, 3),
            "method": self.method,
            "reason": self.reason,
            "needs_review": self.needs_review,
            "sibling_of": self.sibling_of,
        }


@dataclass(slots=True)
class ResolutionOutcome:
    decisions: list[ResolutionDecision] = field(default_factory=list)
    new_assets: dict[str, dict[str, Any]] = field(default_factory=dict)
    sibling_pairs: set[tuple[str, str]] = field(default_factory=set)
    review_items: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> dict[str, int]:
        return {
            "mentions": len(self.decisions),
            "resolved": sum(1 for d in self.decisions if d.action in ("merge", "separate")),
            "needs_review": sum(1 for d in self.decisions if d.needs_review),
            "new_assets": len(self.new_assets),
            "sibling_links": len(self.sibling_pairs),
        }


class AssetIndex:
    """In-memory blocked index of the assets already known to the system.

    Loaded once per ingestion job from Postgres and updated as new assets are
    created, so a document that introduces P-101B in chunk 3 resolves chunk 40's
    mention of "10-P-101-B" onto the same asset without a round trip.
    """

    def __init__(self, existing: list[dict[str, Any]] | None = None) -> None:
        self._by_canonical: dict[str, dict[str, Any]] = {}
        self._blocks: dict[str, list[str]] = {}
        self._aliases: dict[str, str] = {}
        for row in existing or []:
            self.add(row)

    def add(self, asset: dict[str, Any]) -> None:
        canonical = asset["canonical_tag"]
        self._by_canonical[canonical] = asset
        self._blocks.setdefault(blocking_key(canonical), []).append(canonical)

    def add_alias(self, surface_form: str, canonical_tag: str) -> None:
        self._aliases[surface_form.lower()] = canonical_tag

    def candidates(self, canonical: str) -> list[dict[str, Any]]:
        return [
            self._by_canonical[tag]
            for tag in self._blocks.get(blocking_key(canonical), [])
            if tag in self._by_canonical
        ]

    def alias_hit(self, surface_form: str) -> str | None:
        return self._aliases.get(surface_form.lower())

    def get(self, canonical: str) -> dict[str, Any] | None:
        return self._by_canonical.get(canonical)

    def __len__(self) -> int:
        return len(self._by_canonical)


def resolve_mentions(
    mentions: list[TagMention],
    index: AssetIndex,
    *,
    doc_id: str,
    data_class: str,
    site: str | None = None,
) -> ResolutionOutcome:
    """Resolve a document's mentions against the known asset index."""
    settings = get_settings()
    outcome = ResolutionOutcome()

    for mention in mentions:
        canonical = mention.canonical
        if not canonical:
            continue

        alias_target = index.alias_hit(mention.surface_form)
        candidates = index.candidates(canonical)

        # An unsuffixed tag sitting in a bucket that already holds suffixed
        # siblings is genuinely ambiguous. "P-101" alongside P-101A and P-101B
        # is usually the position or the pair, not a third machine -- but a
        # document could also be naming a single unsuffixed pump. Creating it
        # silently puts a phantom peer in the asset list that then accumulates
        # evidence and distorts every count, so it goes to the review queue
        # instead. This is the middle band working as designed.
        if _is_ambiguous_unsuffixed(mention, candidates):
            decision = _create_new(mention, index, doc_id=doc_id, data_class=data_class, site=site)
            decision.needs_review = True
            decision.method = "ambiguous_unsuffixed"
            siblings = ", ".join(
                sorted(c["canonical_tag"] for c in candidates if c.get("item_suffix"))
            )
            decision.reason = (
                f"tag has no item suffix but suffixed assets exist in the same block "
                f"({siblings}); it may denote the functional location or the pair rather "
                f"than a distinct machine"
            )
            outcome.decisions.append(decision)
            outcome.new_assets[decision.canonical_tag] = index.get(decision.canonical_tag) or {}
            outcome.review_items.append(
                {
                    "kind": "entity_resolution",
                    "subject": f"{mention.surface_form} -> new asset {decision.canonical_tag}",
                    "confidence": 0.5,
                    "detail": {
                        "surface_form": mention.surface_form,
                        "created_as": decision.canonical_tag,
                        "existing_siblings": siblings,
                        "reason": decision.reason,
                        "doc_id": doc_id,
                    },
                }
            )
            continue

        best: tuple[MatchResult, dict[str, Any]] | None = None
        for candidate in candidates:
            result = score_pair(
                canonical,
                candidate["canonical_tag"],
                known_alias=(alias_target == candidate["canonical_tag"]),
            )
            if best is None or result.score > best[0].score:
                best = (result, candidate)

        if best is None:
            decision = _create_new(mention, index, doc_id=doc_id, data_class=data_class, site=site)
            outcome.decisions.append(decision)
            outcome.new_assets[decision.canonical_tag] = index.get(decision.canonical_tag) or {}
            continue

        result, candidate = best
        action = decide(
            result,
            auto_merge_threshold=settings.er_auto_merge_threshold,
            review_threshold=settings.er_review_threshold,
        )

        if action == "merge":
            outcome.decisions.append(
                ResolutionDecision(
                    mention=mention,
                    action="merge",
                    asset_id=candidate["asset_id"],
                    canonical_tag=candidate["canonical_tag"],
                    score=result.score,
                    method=result.method,
                    reason=result.reason,
                    needs_review=False,
                )
            )
            continue

        if action == "link_sibling":
            # A sibling is a *different* asset that must also exist. Create it,
            # then record the pair -- merging here would corrupt every downstream
            # failure statistic silently.
            decision = _create_new(mention, index, doc_id=doc_id, data_class=data_class, site=site)
            decision.action = "link_sibling"
            decision.sibling_of = candidate["canonical_tag"]
            decision.score = result.score
            decision.method = result.method
            decision.reason = result.reason
            outcome.decisions.append(decision)
            outcome.new_assets[decision.canonical_tag] = index.get(decision.canonical_tag) or {}
            pair = tuple(sorted((decision.canonical_tag, candidate["canonical_tag"])))
            outcome.sibling_pairs.add(pair)  # type: ignore[arg-type]
            continue

        if action == "needs_review":
            outcome.decisions.append(
                ResolutionDecision(
                    mention=mention,
                    action="needs_review",
                    asset_id=candidate["asset_id"],
                    canonical_tag=candidate["canonical_tag"],
                    score=result.score,
                    method=result.method,
                    reason=result.reason,
                    needs_review=True,
                )
            )
            outcome.review_items.append(
                {
                    "kind": "entity_resolution",
                    "subject": f"{mention.surface_form} -> {candidate['canonical_tag']}",
                    "confidence": result.score,
                    "detail": {
                        "surface_form": mention.surface_form,
                        "candidate": candidate["canonical_tag"],
                        "reason": result.reason,
                        "method": result.method,
                        "doc_id": doc_id,
                    },
                }
            )
            continue

        decision = _create_new(mention, index, doc_id=doc_id, data_class=data_class, site=site)
        outcome.decisions.append(decision)
        outcome.new_assets[decision.canonical_tag] = index.get(decision.canonical_tag) or {}

    return outcome


def _create_new(
    mention: TagMention,
    index: AssetIndex,
    *,
    doc_id: str,
    data_class: str,
    site: str | None,
) -> ResolutionDecision:
    canonical = mention.canonical
    existing = index.get(canonical)
    if existing:
        return ResolutionDecision(
            mention=mention,
            action="merge",
            asset_id=existing["asset_id"],
            canonical_tag=canonical,
            score=1.0,
            method="exact_canonical",
            reason="canonical form already present in the index",
            needs_review=False,
        )

    parsed = parse(canonical)
    asset_identifier = make_asset_id(canonical)
    asset: dict[str, Any] = {
        "asset_id": asset_identifier,
        "canonical_tag": canonical,
        "tag_kind": parsed.kind.value,
        "class_code": parsed.cls,
        "class_label": parsed.class_label,
        "unit_prefix": mention.parsed.unit,
        "sequence_no": parsed.seq,
        "item_suffix": parsed.suffix,
        "site": site,
        "data_class": data_class,
        "first_seen_doc": doc_id,
    }
    index.add(asset)
    return ResolutionDecision(
        mention=mention,
        action="separate",
        asset_id=asset_identifier,
        canonical_tag=canonical,
        score=1.0,
        method="new_entity",
        reason="no candidate in the blocking bucket; created a new canonical asset",
        needs_review=parsed.kind is TagKind.UNPARSED,
    )


def _is_ambiguous_unsuffixed(mention: TagMention, candidates: list[dict[str, Any]]) -> bool:
    """True when an unsuffixed equipment tag collides with suffixed siblings."""
    parsed = mention.parsed
    if parsed.kind is not TagKind.EQUIPMENT or parsed.suffix:
        return False
    if any(c["canonical_tag"] == mention.canonical for c in candidates):
        return False  # already resolved once; not ambiguous any more
    return any(c.get("item_suffix") for c in candidates)


def sibling_relation_check(tag_a: str, tag_b: str) -> bool:
    """Public helper used by tests and the graph writer."""
    return score_pair(tag_a, tag_b).relation is TagRelation.SIBLING
