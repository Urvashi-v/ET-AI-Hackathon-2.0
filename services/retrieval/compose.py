"""Grounded answer composition.

Two ways to turn retrieved evidence into an answer, and this module implements
the one that works without a credential.

**Extractive composition (this module, always available).** The answer is built
*only* from sentences that occur verbatim in retrieved passages. Nothing is
paraphrased, nothing is summarised, and no sentence can enter the answer without
the citation of the chunk it came from. The property that matters:

    The copilot cannot answer from general knowledge, because the only
    strings it is able to emit are strings that exist in the corpus.

That is a structural guarantee, not a prompt instruction — and prompt
instructions are exactly what fails under pressure in a safety-critical domain.
It also makes claim verification exact rather than approximate: every claim *is*
an evidence span, so the check is a containment test that cannot be fooled.

**Abstractive generation (``services/retrieval/generate.py``, credential-gated).**
Fluent prose, better at synthesising across passages, and correspondingly harder
to constrain — which is why it carries citation binding, claim verification and
the verbatim guard. It runs only when ``LLM_PROVIDER`` is configured.

The honest trade-off: extraction reads less smoothly and cannot combine two
half-answers into one sentence. In exchange it cannot hallucinate a torque
figure, and in this domain that is the better default.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from services.common.schemas import QueryIntent
from services.retrieval.generate import ContextPassage
from services.retrieval.lexical import tokenize_query

#: Sentences in the composed answer. Beyond this it stops being an answer and
#: becomes the evidence panel, which is displayed separately anyway.
MAX_SENTENCES = 4

#: A fragment shorter than this is a heading or a stray label, not a claim.
MIN_SENTENCE_CHARS = 30
MAX_SENTENCE_CHARS = 420

#: Split on sentence punctuation, or on a line break that is a real boundary.
#:
#: Extracted documents are full of lines that end without punctuation --
#: headings, procedure steps, table rows -- so a newline often *is* a boundary,
#: and treating it as one keeps "Immediate cause" from being glued to the
#: sentence beneath it. But PDFs also soft-wrap prose mid-sentence, and splitting
#: there truncates the claim. ``_unwrap`` rejoins the soft wraps first, so by the
#: time this pattern runs every remaining newline is a genuine break.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])|\n+")

#: A line continues the one above it when the line above did not finish a
#: thought: no terminal punctuation, and the next line opens lowercase (or with a
#: closing bracket or conjunction). "found" / "scored, with a witness pattern" is
#: one sentence the parser wrapped; "Immediate cause" / "The seal failed." is not.
_SOFT_WRAP = re.compile(r"(?<![.!?:;])\n(?=[a-z),;]|and\b|or\b|with\b)")

#: Sentence-initial noise from parsed documents: table scaffolding, banners.
_NOISE = re.compile(r"^(SYNTHETIC TEST DATA|Page \d+|Table \d+|\|)", re.I)

#: A quantity with a unit. Weighted heavily for lookup questions, where the
#: whole point of the question is usually a number.
_QUANTITY = re.compile(
    r"\d+(?:\.\d+)?\s*(?:barg?|kPa|MPa|psi|mm|cm|m|kg|t|degC|°C|K|rpm|Hz|kW|MW|"
    r"hrs?|hours?|mm/yr|months?|years?|seconds?|minutes?)\b",
    re.I,
)

#: Question words that signal the shape of the answer being asked for, mapped to
#: the sentence features that would satisfy them.
_INTENT_CUES: dict[QueryIntent, tuple[str, ...]] = {
    QueryIntent.PROCEDURAL: (
        "shall", "must", "confirm", "open", "close", "apply", "stop", "start",
        "required", "before", "permit", "isolat", "lock-out", "ppe",
    ),
    QueryIntent.DIAGNOSTIC: (
        "cause", "caused", "because", "due to", "failed", "failure", "found",
        "evidence", "scored", "dry running", "root cause",
    ),
    QueryIntent.LOOKUP: ("is", "limit", "setpoint", "rated", "design", "specified"),
    QueryIntent.MULTI_HOP: ("spare", "standby", "sibling", "feeds", "downstream", "isolat"),
    QueryIntent.AGGREGATE: ("total", "count", "each", "per"),
    QueryIntent.COMPARATIVE: ("than", "versus", "compared", "both", "while"),
    QueryIntent.UNANSWERABLE: (),
}


@dataclass(slots=True)
class ComposedClaim:
    """One sentence of the answer, with the evidence it was taken from."""

    text: str
    marker: str
    chunk_id: str
    doc_id: str
    doc_title: str
    page: int | None
    char_start: int
    char_end: int
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "marker": self.marker,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "page": self.page,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "score": round(self.score, 4),
            # True by construction: the text was copied out of the chunk, so the
            # containment check cannot fail. It is still asserted, not assumed --
            # see verify_composed().
            "verbatim": True,
        }


@dataclass(slots=True)
class ComposedAnswer:
    text: str | None
    claims: list[ComposedClaim] = field(default_factory=list)
    method: str = "extractive"
    detail: str | None = None
    used_markers: set[str] = field(default_factory=set)
    #: Fraction of the question's content terms that appear in the answer.
    #:
    #: The signal that tells "answered" from "retrieved something adjacent". Ask
    #: for the NPSH of P-101B and retrieval will happily return well-ranked
    #: passages about P-101B -- they are about the right pump, they simply do not
    #: mention NPSH. Verbatim verification cannot catch that (every sentence is
    #: genuinely from the corpus) and neither can the reranker (the passages
    #: really are related). Term coverage can: if the word the question turns on
    #: never appears in the answer, the question was not answered.
    relevance: float = 0.0
    #: The question terms that no selected sentence contains. Displayed on an
    #: abstention so the operator sees *what* was missing, not merely that
    #: something was.
    missing_terms: list[str] = field(default_factory=list)

    @property
    def total_claims(self) -> int:
        return len(self.claims)

    @property
    def verified_claims(self) -> int:
        return sum(1 for c in self.claims if c.text)


def compose(
    *,
    question: str,
    intent: QueryIntent,
    passages: list[ContextPassage],
    entity_tags: list[str],
    max_sentences: int = MAX_SENTENCES,
    doc_frequencies: dict[str, int] | None = None,
) -> ComposedAnswer:
    """Select the sentences that answer the question, in evidence order.

    Scoring combines four signals, none of which is a model:

    * **query-term overlap** — how much of the question's vocabulary the sentence
      actually contains, normalised by sentence length so a long paragraph does
      not win by covering everything;
    * **passage rank** — the reranked position of the passage the sentence came
      from, so a sentence from the best passage starts ahead;
    * **intent cues** — a procedural question wants imperatives, a diagnostic one
      wants causal language, a lookup usually wants a number;
    * **entity presence** — a sentence naming the asset the question is about is
      more likely to be about it.
    """
    if not passages:
        return ComposedAnswer(
            text=None,
            detail="No passages were retrieved, so there is nothing to compose an answer from.",
        )

    query_terms = set(tokenize_query(question))
    cues = _INTENT_CUES.get(intent, ())
    tags_lower = {t.lower() for t in entity_tags}

    candidates: list[ComposedClaim] = []
    for rank, passage in enumerate(passages):
        for sentence, start, end in _sentences(passage.text):
            score = _score_sentence(
                sentence=sentence,
                query_terms=query_terms,
                cues=cues,
                tags=tags_lower,
                passage_rank=rank,
                intent=intent,
            )
            if score <= 0:
                continue
            candidates.append(
                ComposedClaim(
                    text=sentence,
                    marker=passage.marker,
                    chunk_id=passage.chunk_id,
                    doc_id=passage.doc_id,
                    doc_title=passage.doc_title,
                    page=passage.page,
                    char_start=start,
                    char_end=end,
                    score=score,
                )
            )

    if not candidates:
        return ComposedAnswer(
            text=None,
            detail=(
                "Passages were retrieved, but none contains a sentence that addresses the "
                "question. Nothing is composed rather than returning the nearest paragraph."
            ),
        )

    candidates.sort(key=lambda c: -c.score)
    selected = _select(candidates, max_sentences)
    # Presented in evidence order rather than score order: a procedure read out
    # of sequence is worse than useless.
    selected.sort(key=lambda c: (c.marker_index, c.char_start))

    text = " ".join(f"{c.text} [{c.marker}]" for c in selected)
    relevance, missing = _relevance(query_terms, selected, tags_lower, doc_frequencies)
    return ComposedAnswer(
        text=text,
        claims=selected,
        method="extractive",
        detail=(
            f"Composed from {len(selected)} verbatim sentence(s) across "
            f"{len({c.chunk_id for c in selected})} passage(s). No text was generated."
        ),
        used_markers={c.marker for c in selected},
        relevance=relevance,
        missing_terms=missing,
    )


def _relevance(
    query_terms: set[str],
    claims: list[ComposedClaim],
    tags: set[str],
    doc_frequencies: dict[str, int] | None = None,
) -> tuple[float, list[str]]:
    """How much of what was asked the answer actually covers, weighted by IDF.

    Two refinements over plain term counting, both of which matter.

    **The asset tag is excluded.** Every passage retrieved for a question about
    P-101B mentions P-101B, so counting it rewards the answer for restating the
    subject -- precisely the failure being measured.

    **Remaining terms are weighted by inverse document frequency.** "What is the
    NPSH required for P-101B at its current impeller diameter?" has five content
    terms; an answer about impeller diameter covers two of them and scores 0.4
    unweighted, comfortably above any sane floor. But "NPSH" occurs in one chunk
    of the corpus and "required" in dozens: the rare term is the question, and
    the common ones are grammar. Weighting by IDF puts most of the mass on the
    word that actually had to be answered, and coverage falls to where it
    belongs.

    Without frequencies (no index, or none of the terms are known) this degrades
    to uniform weighting rather than failing.
    """
    content = {
        term
        for term in query_terms
        if term not in _FUNCTION_WORDS
        and not any(term in tag or tag in term for tag in tags)
    }

    # Drop terms the corpus contains nowhere at all. Two reasons, and the second
    # is the one that bit.
    #
    # Logically: an extractive answer is built from corpus text, so a term that
    # occurs in no chunk cannot possibly appear in it. Keeping such terms in the
    # denominator imposes a penalty no correct answer can avoid.
    #
    # Practically: they were getting the *maximum* IDF weight. "Why does P-101B
    # keep failing?" carries "doe" and "keep", ordinary English the stopword list
    # does not catch and this corpus never uses; at df=0 they outweighed "seal"
    # and "failing" combined, and a question the system answered correctly scored
    # 0.07 relevance and abstained.
    #
    # A genuinely significant unknown word -- a site the corpus has never heard
    # of -- is caught by the unknown-term gate instead, which says so explicitly
    # rather than expressing it as a low score.
    if doc_frequencies:
        answerable = {term for term in content if doc_frequencies.get(term, 0) > 0}
        # Unless that leaves nothing: a question composed entirely of vocabulary
        # the corpus lacks is better described by the gate than by a 1.0 here.
        if answerable:
            content = answerable

    if not content:
        # The question was nothing but an asset tag ("P-101B?"). Coverage is
        # undefined rather than zero, and 1.0 lets the other signals decide.
        return 1.0, []

    answer_terms: set[str] = set()
    for claim in claims:
        answer_terms |= set(tokenize_query(claim.text))
    covered = {term for term in content if _covers(term, answer_terms)}

    weights = {term: _idf(term, doc_frequencies) for term in content}
    total = sum(weights.values())
    if total <= 0:
        return 0.0, sorted(content - covered)
    score = sum(weights[term] for term in covered) / total

    # Ordered by how much each missing term narrowed the question, so the
    # explanation names the word that mattered rather than the first one
    # alphabetically.
    missing = sorted(content - covered, key=lambda t: -weights[t])
    return score, missing


#: Shortest prefix treated as evidence of a shared stem. Four characters keeps
#: "seal"/"sealing" and "fail"/"failure" together without collapsing "port" into
#: "portable".
_STEM_PREFIX = 4


def _covers(term: str, answer_terms: set[str]) -> bool:
    """Is this question term present in the answer, allowing for inflection?

    The BM25 tokenizer applies only a light plural strip, so "fail" and "failed"
    are distinct index terms -- correct for retrieval, where over-stemming
    conflates genuinely different words, but wrong here. Asked why a seal failed
    and answered "the seal failed", coverage should not report "fail" missing;
    with IDF weighting that one miss was enough to abstain on a question the
    system had answered correctly.

    Prefix matching rather than a real stemmer: it needs no dependency, is
    symmetric, and errs toward counting a term as covered -- which is the safe
    direction, since the cost of a false abstention is a useful answer withheld.
    """
    if term in answer_terms:
        return True
    if len(term) < _STEM_PREFIX:
        return False
    return any(
        len(other) >= _STEM_PREFIX
        and (other.startswith(term[:_STEM_PREFIX]) and (other.startswith(term) or term.startswith(other)))
        for other in answer_terms
    )

#: Function words that survive the BM25 stopword list.
#:
#: The retrieval tokenizer keeps these deliberately -- BM25 weights them to
#: nothing anyway, and stripping more aggressively risks losing a term that
#: matters in a tag. For *relevance* they are actively harmful: "Why does P-101B
#: keep failing?" carries "doe" and "keep", an answer will never repeat them, and
#: counting them as unanswered demands drags a correct answer below the floor.
#:
#: Strictly auxiliaries, quantifiers and question words. Anything that could
#: carry plant meaning stays out of this set -- "describe" and "change" are
#: content words even though they often appear in question framing.
_FUNCTION_WORDS = frozenset(
    {
        "doe", "does", "did", "do", "done", "keep", "keeps", "kept",
        "many", "much", "any", "some", "there", "here", "what", "which",
        "who", "whose", "when", "where", "why", "how", "been", "being",
        "have", "has", "had", "will", "would", "should", "could", "can",
        "may", "might", "must", "shall", "get", "got", "make", "made",
        "give", "given", "take", "taken", "know", "need", "want", "like",
        "tell", "say", "said", "show", "look", "come", "go", "put",
    }
)

#: Cap on a single term's weight. A term the corpus has never seen has unbounded
#: IDF, and one unknown word should not be able to make every other term in the
#: question irrelevant to the score -- the unknown-term gate handles that case
#: directly and more honestly.
_MAX_IDF = 4.0


def _idf(term: str, doc_frequencies: dict[str, int] | None) -> float:
    if not doc_frequencies:
        return 1.0
    df = doc_frequencies.get(term)
    if df is None:
        return 1.0
    if df <= 0:
        return _MAX_IDF
    # log(N/df) with N approximated by the largest df seen; the absolute scale is
    # irrelevant because the result is normalised by the total.
    corpus = max(max(doc_frequencies.values()), 1)
    return min(_MAX_IDF, 1.0 + math.log((corpus + 1) / df))


def verify_composed(answer: ComposedAnswer, passages: list[ContextPassage]) -> dict[str, Any]:
    """Check every composed claim against the passage it claims to come from.

    Verification should pass trivially — the sentences were copied out of those
    passages. It is run anyway, because a guarantee that is never checked is a
    comment, and this one catches real regressions: an offset bug, a passage
    swapped after composition, a marker pointing at the wrong chunk.
    """
    by_marker = {p.marker: p for p in passages}
    verified = 0
    unsupported: list[str] = []

    for claim in answer.claims:
        passage = by_marker.get(claim.marker)
        if passage is not None and _squash(claim.text) in _squash(passage.text):
            verified += 1
        else:
            unsupported.append(claim.text[:200])

    return {
        "used_markers": answer.used_markers,
        "total_claims": len(answer.claims),
        "verified_claims": verified,
        "unsupported_claims": unsupported,
    }


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _sentences(text: str) -> list[tuple[str, int, int]]:
    """Split into sentences, keeping each one's offsets in the chunk.

    Offsets are what let the UI highlight the exact sentence inside the source
    passage, so they are tracked rather than recomputed by searching later --
    a repeated sentence would otherwise anchor to the wrong occurrence.
    """
    # Rejoin soft wraps before splitting. Substituting a single space for a single
    # newline preserves every character offset, so the spans returned here still
    # index correctly into the *original* chunk text that the UI highlights.
    text = _SOFT_WRAP.sub(" ", text)

    out: list[tuple[str, int, int]] = []
    cursor = 0
    for raw in _SENTENCE_SPLIT.split(text):
        sentence = raw.strip()
        if not sentence:
            cursor += len(raw) + 1
            continue
        start = text.find(sentence, cursor)
        if start == -1:
            start = cursor
        cursor = start + len(sentence)
        if MIN_SENTENCE_CHARS <= len(sentence) <= MAX_SENTENCE_CHARS and not _NOISE.match(sentence):
            out.append((sentence, start, start + len(sentence)))
    return out


def _score_sentence(
    *,
    sentence: str,
    query_terms: set[str],
    cues: tuple[str, ...],
    tags: set[str],
    passage_rank: int,
    intent: QueryIntent,
) -> float:
    lowered = sentence.lower()
    sentence_terms = set(tokenize_query(sentence))
    if not sentence_terms:
        return 0.0

    overlap = query_terms & sentence_terms
    if not overlap:
        return 0.0

    # Coverage of the question, tempered by how much else the sentence carries.
    coverage = len(overlap) / max(len(query_terms), 1)
    density = len(overlap) / (len(sentence_terms) ** 0.5)
    score = 2.0 * coverage + 0.6 * density

    # The best passage starts ahead, but not so far ahead that a better sentence
    # further down can never win.
    score += 1.2 / (1 + passage_rank)

    if cues:
        hits = sum(1 for cue in cues if cue in lowered)
        score += min(0.9, 0.3 * hits)

    if tags and any(tag in lowered for tag in tags):
        score += 0.5

    if intent in (QueryIntent.LOOKUP, QueryIntent.AGGREGATE) and _QUANTITY.search(sentence):
        score += 0.7

    return score


def _select(candidates: list[ComposedClaim], limit: int) -> list[ComposedClaim]:
    """Take the best sentences, avoiding near-duplicates.

    The same statement often appears in two revisions of a document and in the
    record that quotes it. Repeating it three times reads as padding and crowds
    out the sentence that would have added something.
    """
    selected: list[ComposedClaim] = []
    seen_terms: list[set[str]] = []

    for candidate in candidates:
        if len(selected) >= limit:
            break
        terms = set(tokenize_query(candidate.text))
        if any(_jaccard(terms, previous) > 0.6 for previous in seen_terms):
            continue
        selected.append(candidate)
        seen_terms.append(terms)

    return selected


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


# ``marker_index`` sorts C1, C2, ... C10 numerically rather than as strings.
def _marker_index(self: ComposedClaim) -> int:
    match = re.search(r"\d+", self.marker)
    return int(match.group()) if match else 0


ComposedClaim.marker_index = property(_marker_index)  # type: ignore[attr-defined]
