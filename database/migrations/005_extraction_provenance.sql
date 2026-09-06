-- =============================================================================
-- 005_extraction_provenance.sql
--
-- Day 2 fills in the provenance the Day 1 contract declared but could not yet
-- populate, and adds the per-document timing the ingestion dashboard reports.
--
-- The distinction these columns make possible is the important one: a character
-- *read* from an embedded text layer and a character *recognised* by OCR are not
-- the same kind of fact, and anything downstream that treats them alike is
-- making an assumption it should not. `extraction_method` and
-- `extraction_confidence` are what let a reader — or an auditor — tell them
-- apart at the point of citation.
--
-- Idempotent.
-- =============================================================================

-- --- chunk-level extraction provenance ---------------------------------------
ALTER TABLE document_chunks
    ADD COLUMN IF NOT EXISTS extraction_method     TEXT,
    -- Weakest confidence among the source blocks, not the mean: a passage is
    -- only as trustworthy as its worst line.
    ADD COLUMN IF NOT EXISTS extraction_confidence REAL NOT NULL DEFAULT 1.0;

CREATE INDEX IF NOT EXISTS idx_chunks_low_confidence
    ON document_chunks (extraction_confidence)
    WHERE extraction_confidence < 0.6;

-- --- document-level parse and OCR record --------------------------------------
ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS parser              TEXT,
    ADD COLUMN IF NOT EXISTS ocr_engine          TEXT,
    ADD COLUMN IF NOT EXISTS ocr_mean_confidence REAL,
    ADD COLUMN IF NOT EXISTS ocr_word_count      INTEGER,
    -- Vector-object count and the resulting verdict. This is the evidence
    -- behind a "this is a drawing" classification, kept so the decision can be
    -- audited rather than just trusted.
    ADD COLUMN IF NOT EXISTS vector_objects      INTEGER,
    ADD COLUMN IF NOT EXISTS is_drawing          BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS tables_found        INTEGER NOT NULL DEFAULT 0,
    -- What the dashboard reports per document.
    ADD COLUMN IF NOT EXISTS processing_ms       INTEGER,
    ADD COLUMN IF NOT EXISTS chunk_count         INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS mention_count       INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS graph_nodes_created INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS graph_edges_created INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS ingest_error        TEXT;

CREATE INDEX IF NOT EXISTS idx_documents_drawing ON documents (is_drawing) WHERE is_drawing;

-- --- job-level timing ----------------------------------------------------------
ALTER TABLE ingestion_jobs
    ADD COLUMN IF NOT EXISTS duration_ms         INTEGER,
    ADD COLUMN IF NOT EXISTS graph_nodes_created INTEGER NOT NULL DEFAULT 0;

-- --- extraction audit ----------------------------------------------------------
-- Every fact an extractor asserts, with the verbatim span it was based on and
-- whether that span was found in the source text.
--
-- The verbatim check is the cheapest hallucination defence there is: one string
-- containment test, no second model, fully deterministic. Recording the outcome
-- per extraction is what turns it from a guard into a measurable claim —
-- "0.0% of asserted facts lack a verified source span" is a number, not an
-- adjective, and this table is where it comes from.
CREATE TABLE IF NOT EXISTS extractions (
    extraction_id  BIGSERIAL PRIMARY KEY,
    doc_id         TEXT NOT NULL REFERENCES documents (doc_id) ON DELETE CASCADE,
    chunk_id       TEXT REFERENCES document_chunks (chunk_id) ON DELETE CASCADE,
    kind           TEXT NOT NULL,          -- failure_mode | cause | action | obligation | ...
    extractor      TEXT NOT NULL,          -- regex:* | gazetteer | llm:<model>
    payload        JSONB NOT NULL DEFAULT '{}'::jsonb,
    evidence_quote TEXT,
    quote_verified BOOLEAN NOT NULL DEFAULT FALSE,
    reject_reason  TEXT,                   -- non-null when the extraction was rejected
    confidence     REAL,
    char_start     INTEGER,
    char_end       INTEGER,
    page           INTEGER,
    data_class     data_class NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_extractions_doc      ON extractions (doc_id);
CREATE INDEX IF NOT EXISTS idx_extractions_kind     ON extractions (kind);
CREATE INDEX IF NOT EXISTS idx_extractions_rejected ON extractions (created_at DESC)
    WHERE NOT quote_verified;
