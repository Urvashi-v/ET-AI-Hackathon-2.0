"""Lexical retrieval: tokenizer and Okapi BM25.

The tokenizer is the interesting part. A general-purpose analyser destroys
exactly the tokens that matter most here: ``to_tsvector('english', 'P-101B')``
yields ``{p, 101b}``, so the single most important exact-match object in an
industrial corpus stops being searchable as itself. Since embeddings *also*
blur ``P-101B`` against ``P-101A`` and ``P-102B``, losing it on the lexical leg
too means losing it entirely.

So this tokenizer is domain-aware:

* industrial tags are recognised by the shared grammar in
  ``services.common.tags`` and emitted **whole, in canonical form**, so
  ``P 101 B``, ``10-P-101-B`` and ``P‑101‑B`` all index and query as ``p-101b``;
* standard and document references (``OISD-STD-105``, ``SOP-4412``) survive as
  single tokens;
* everything else is lower-cased, split on non-alphanumerics, stripped of a
  small stopword list, and light-stemmed (plural ``s`` only -- aggressive
  stemming would conflate ``bearing`` and ``bear``).

Index time and query time call the same function. If they ever diverge,
retrieval degrades silently, which is the worst kind of bug to have in a system
whose whole claim is groundedness.
"""

from __future__ import annotations

import re
from typing import Any

from services.common import db
from services.common.logging import get_logger
from services.common.tags import (
    EQUIPMENT_CLASS_CODES,
    TagKind,
    decode_instrument_function,
    parse,
)

log = get_logger(__name__)

#: Deliberately short. Aggressive stopword removal hurts a technical corpus:
#: "no", "not" and "before" carry real meaning in a procedure.
_STOPWORDS = frozenset(
    """
    a an the and or of to in on at for from with by as is are was were be been being
    this that these those it its into than then there their they them we you your
    i he she his her which who whom whose what when where why how
    """.split()
)

_TOKEN_SPLIT = re.compile(r"[^A-Za-z0-9\"'’\-‐-―_/.]+")
_TRAILING_PUNCT = re.compile(r"^[\-_.'\"’]+|[\-_.'\"’]+$")

#: Reference prefixes kept whole so a clause citation is exactly matchable.
_REFERENCE = re.compile(
    r"^(?:OISD(?:-STD)?|SOP|WO|MOC|CAPA|NCR|INC|API|ISO|IEC|PTW|JSA|HAZOP)[-\s]?\d{2,6}"
    r"(?:[-/]\d{1,4})?$",
    re.I,
)


#: A tag written with spaces instead of hyphens: "P 101 B". Real work-order text
#: and real questions both contain these, and the generic splitter would break
#: the tag into fragments that are then discarded as too short.
_SPACED_TAG = re.compile(r"\b([A-Za-z]{1,4})\s+(\d{2,5})\s*([A-Za-z])?\b")


def _join_spaced_tags(text: str) -> str:
    """Rejoin space-separated tags before splitting.

    Guarded on both sides, because the pattern is otherwise happy to turn
    "the 101 b" into a heat exchanger: the letter group must be a known
    equipment class code or a valid ISA instrument group, and must not be an
    ordinary English word.
    """

    def replace(match: re.Match[str]) -> str:
        letters, sequence, suffix = match.group(1), match.group(2), match.group(3) or ""
        upper = letters.upper()
        if letters.lower() in _STOPWORDS:
            return match.group(0)
        if upper not in EQUIPMENT_CLASS_CODES and not decode_instrument_function(upper):
            return match.group(0)
        return f"{upper}-{sequence}{suffix.upper()}"

    return _SPACED_TAG.sub(replace, text)


def tokenize(text: str) -> list[str]:
    """Tokenize for both indexing and querying. Order is preserved.

    Index time and query time call this same function. If the two ever diverge
    retrieval degrades silently, which is the worst kind of bug to have in a
    system whose whole claim is groundedness.
    """
    if not text:
        return []
    tokens: list[str] = []
    for raw in _TOKEN_SPLIT.split(_join_spaced_tags(text)):
        piece = _TRAILING_PUNCT.sub("", raw)
        if not piece:
            continue

        if _REFERENCE.match(piece):
            tokens.append(re.sub(r"[\s]", "-", piece.upper()).lower())
            continue

        parsed = parse(piece)
        if parsed.kind in (TagKind.EQUIPMENT, TagKind.INSTRUMENT, TagKind.LINE, TagKind.KKS):
            tokens.append(parsed.canonical.lower())
            continue

        lowered = piece.lower()
        if lowered in _STOPWORDS or len(lowered) < 2:
            continue
        tokens.append(_light_stem(lowered))
    return tokens


def tokenize_query(text: str) -> list[str]:
    """Query-side tokenization.

    Identical to index-side by construction -- it is the same function. Kept as a
    named entry point so the call sites read clearly and so any future
    query-only expansion has one obvious home.
    """
    return tokenize(text)


def _light_stem(token: str) -> str:
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


# ---------------------------------------------------------------------------
# Index maintenance
# ---------------------------------------------------------------------------


async def index_chunk_terms(chunks: list[dict[str, Any]]) -> int:
    """Write BM25 postings and lengths for a batch of chunks. Idempotent."""
    if not chunks:
        return 0

    total_postings = 0
    async with db.connection() as conn, conn.cursor() as cur:
        for chunk in chunks:
            tokens = tokenize(chunk["text"])
            if not tokens:
                continue
            frequencies: dict[str, int] = {}
            for token in tokens:
                frequencies[token] = frequencies.get(token, 0) + 1

            await cur.execute("DELETE FROM chunk_terms WHERE chunk_id = %s", (chunk["chunk_id"],))
            await cur.executemany(
                "INSERT INTO chunk_terms (chunk_id, term, tf) VALUES (%s, %s, %s) "
                "ON CONFLICT (chunk_id, term) DO UPDATE SET tf = EXCLUDED.tf",
                [(chunk["chunk_id"], term, tf) for term, tf in frequencies.items()],
            )
            await cur.execute(
                "INSERT INTO chunk_lengths (chunk_id, length, doc_id, data_class) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (chunk_id) DO UPDATE SET length = EXCLUDED.length",
                (chunk["chunk_id"], len(tokens), chunk["doc_id"], chunk["data_class"]),
            )
            total_postings += len(frequencies)

        await cur.execute("SELECT * FROM refresh_lexical_stats()")

    return total_postings


async def corpus_stats() -> dict[str, Any]:
    row = await db.fetch_one(
        "SELECT chunk_count, total_tokens, updated_at FROM lexical_corpus_stats WHERE id"
    )
    if not row:
        return {"chunk_count": 0, "total_tokens": 0, "avgdl": 0.0}
    chunk_count = int(row["chunk_count"])
    total = int(row["total_tokens"])
    return {
        "chunk_count": chunk_count,
        "total_tokens": total,
        "avgdl": round(total / chunk_count, 2) if chunk_count else 0.0,
        "updated_at": row["updated_at"],
    }


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


async def search(
    question: str, *, top_k: int = 50, doc_filter: list[str] | None = None
) -> list[dict[str, Any]]:
    """Run BM25 and return ranked chunks with their document metadata."""
    terms = tokenize_query(question)
    if not terms:
        return []

    rows = await db.fetch_all(
        """
        SELECT b.chunk_id,
               b.doc_id,
               b.score,
               b.matched_terms,
               c.text,
               c.section_path,
               c.page_from,
               c.chunk_kind,
               c.data_class::text  AS chunk_data_class,
               d.title,
               d.doc_type::text    AS doc_type,
               d.data_class::text  AS doc_data_class,
               d.source_system,
               d.is_current
          FROM bm25_search(%(terms)s, %(top_k)s, 1.2, 0.75, %(doc_filter)s) b
          JOIN document_chunks c ON c.chunk_id = b.chunk_id
          JOIN documents d       ON d.doc_id   = b.doc_id
         ORDER BY b.score DESC
        """,
        {"terms": terms, "top_k": top_k, "doc_filter": doc_filter},
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        row["retriever"] = "lexical"
    return rows
