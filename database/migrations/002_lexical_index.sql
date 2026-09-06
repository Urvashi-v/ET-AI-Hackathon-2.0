-- =============================================================================
-- 002_lexical_index.sql -- a real Okapi BM25 index over the chunk store.
--
-- Why not Postgres full-text search?
--   `to_tsvector('english', 'P-101B')` produces the lexemes {p, 101b}. The tag
--   -- the single most important exact-match object in this domain -- is
--   destroyed by the default tokenizer, and `ts_rank_cd` is not BM25 anyway.
--   Industrial queries are full of exact identifiers (P-101B, OISD-STD-105,
--   PSV-204) that dense retrieval reliably blurs, so the lexical leg has to be
--   genuinely good rather than nominally present.
--
-- What this is instead:
--   A classic inverted index. `services/retrieval/lexical.py` owns the
--   tokenizer, which keeps industrial tags whole; this schema stores the
--   postings and the statistics (N, avgdl, df) that Okapi BM25 needs, and
--   `bm25_search` computes the standard scoring formula in SQL.
--
-- Idempotent.
-- =============================================================================

-- Postings list: one row per (chunk, term) with the in-chunk term frequency.
CREATE TABLE IF NOT EXISTS chunk_terms (
    chunk_id  TEXT    NOT NULL REFERENCES document_chunks (chunk_id) ON DELETE CASCADE,
    term      TEXT    NOT NULL,
    tf        INTEGER NOT NULL,
    PRIMARY KEY (chunk_id, term)
);

CREATE INDEX IF NOT EXISTS idx_chunk_terms_term ON chunk_terms (term);

-- Per-chunk document length in tokens (the |D| in the BM25 denominator).
CREATE TABLE IF NOT EXISTS chunk_lengths (
    chunk_id   TEXT PRIMARY KEY REFERENCES document_chunks (chunk_id) ON DELETE CASCADE,
    length     INTEGER NOT NULL,
    doc_id     TEXT NOT NULL REFERENCES documents (doc_id) ON DELETE CASCADE,
    data_class data_class NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunk_lengths_doc ON chunk_lengths (doc_id);

-- Corpus statistics, maintained incrementally by the ingestion pipeline so that
-- BM25 does not need a full scan at query time.
CREATE TABLE IF NOT EXISTS lexical_corpus_stats (
    id          BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (id),
    chunk_count BIGINT NOT NULL DEFAULT 0,
    total_tokens BIGINT NOT NULL DEFAULT 0,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO lexical_corpus_stats (id, chunk_count, total_tokens)
VALUES (TRUE, 0, 0)
ON CONFLICT (id) DO NOTHING;

-- Recompute the corpus statistics from the postings. Cheap at demo scale and
-- called after each ingestion job so avgdl is always exact rather than drifting.
CREATE OR REPLACE FUNCTION refresh_lexical_stats()
RETURNS TABLE (chunk_count BIGINT, total_tokens BIGINT)
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE lexical_corpus_stats
       SET chunk_count = COALESCE((SELECT count(*) FROM chunk_lengths), 0),
           total_tokens = COALESCE((SELECT sum(length) FROM chunk_lengths), 0),
           updated_at = now()
     WHERE id;
    RETURN QUERY SELECT s.chunk_count, s.total_tokens FROM lexical_corpus_stats s WHERE s.id;
END;
$$;

-- Okapi BM25.
--
--   score(D,Q) = sum over q in Q of
--       IDF(q) * ( tf(q,D) * (k1 + 1) )
--                / ( tf(q,D) + k1 * (1 - b + b * |D| / avgdl) )
--
--   IDF(q) = ln( 1 + (N - df(q) + 0.5) / (df(q) + 0.5) )     [Lucene variant,
--                                                             always positive]
--
-- `query_terms` is the tokenized query produced by the same tokenizer used at
-- index time -- if the two ever diverge, retrieval silently degrades, which is
-- why both call one implementation.
CREATE OR REPLACE FUNCTION bm25_search(
    query_terms TEXT[],
    top_k       INTEGER DEFAULT 50,
    k1          REAL    DEFAULT 1.2,
    b           REAL    DEFAULT 0.75,
    doc_filter  TEXT[]  DEFAULT NULL
)
RETURNS TABLE (chunk_id TEXT, doc_id TEXT, score REAL, matched_terms INTEGER)
LANGUAGE sql
STABLE
AS $$
    WITH stats AS (
        SELECT GREATEST(s.chunk_count, 1)::REAL                            AS n_docs,
               GREATEST(s.total_tokens::REAL / GREATEST(s.chunk_count, 1), 1.0) AS avgdl
          FROM lexical_corpus_stats s
         WHERE s.id
    ),
    df AS (
        SELECT ct.term, count(*)::REAL AS doc_freq
          FROM chunk_terms ct
         WHERE ct.term = ANY(query_terms)
         GROUP BY ct.term
    ),
    hits AS (
        SELECT ct.chunk_id,
               cl.doc_id,
               ct.term,
               ct.tf::REAL       AS tf,
               cl.length::REAL   AS len
          FROM chunk_terms ct
          JOIN chunk_lengths cl ON cl.chunk_id = ct.chunk_id
         WHERE ct.term = ANY(query_terms)
           AND (doc_filter IS NULL OR cl.doc_id = ANY(doc_filter))
    )
    SELECT h.chunk_id,
           h.doc_id,
           SUM(
               ln(1 + (s.n_docs - d.doc_freq + 0.5) / (d.doc_freq + 0.5))
               * (h.tf * (k1 + 1))
               / (h.tf + k1 * (1 - b + b * h.len / s.avgdl))
           )::REAL AS score,
           COUNT(DISTINCT h.term)::INTEGER AS matched_terms
      FROM hits h
      JOIN df d   ON d.term = h.term
     CROSS JOIN stats s
     GROUP BY h.chunk_id, h.doc_id
     ORDER BY score DESC
     LIMIT top_k;
$$;
