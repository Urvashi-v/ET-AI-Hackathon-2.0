-- Document revision lineage.
--
-- Until now every ingested document was independently "current", which is wrong
-- in a way that matters more than most data-quality problems: an engineer handed
-- SOP-4412 Rev 2 while Rev 3 exists is being told to follow a superseded
-- procedure, and the system gave no indication. Retrieval will happily return
-- both revisions and rank the older one first if it happens to match better.
--
-- The fix needs a key that survives re-issue. `doc_id` is a content hash, so
-- every revision gets a different one by construction -- that is exactly what
-- makes ingestion idempotent, and exactly why it cannot group revisions.
-- `doc_number` is the human identifier printed on the document itself
-- ("SOP-4412", "INC-2019-07"), which is stable across revisions.

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS doc_number text,
    -- How doc_number was obtained, so a wrong grouping is diagnosable rather
    -- than mysterious. Same rationale as doc_type_method.
    ADD COLUMN IF NOT EXISTS doc_number_method text,
    -- Set when two documents share a doc_number but cannot be ordered from
    -- available evidence. The system refuses to guess which is current; both
    -- stay current and the pair is surfaced for a human to resolve.
    ADD COLUMN IF NOT EXISTS revision_conflict boolean NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS revision_note text;

-- The lookup behind "show me every revision of SOP-4412", and behind the
-- reconciliation that runs after each ingest.
CREATE INDEX IF NOT EXISTS idx_documents_doc_number
    ON documents (doc_number)
    WHERE doc_number IS NOT NULL;

-- Retrieval filters on currency far more often than it filters on anything
-- else, and a partial index keeps that cheap as the superseded set grows.
CREATE INDEX IF NOT EXISTS idx_documents_current
    ON documents (doc_type, doc_number)
    WHERE is_current;

CREATE INDEX IF NOT EXISTS idx_documents_revision_conflict
    ON documents (doc_number)
    WHERE revision_conflict;

-- superseded_by already existed as free text; make it a real reference so a
-- dangling pointer to a deleted document cannot survive. ON DELETE SET NULL
-- rather than CASCADE: deleting a superseding revision must not delete the
-- history it superseded.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'documents_superseded_by_fkey'
    ) THEN
        ALTER TABLE documents
            ADD CONSTRAINT documents_superseded_by_fkey
            FOREIGN KEY (superseded_by) REFERENCES documents(doc_id) ON DELETE SET NULL;
    END IF;
END $$;

-- A document cannot supersede itself. Cheap to state, and it makes the
-- reconciliation logic's correctness enforceable rather than merely intended.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'documents_no_self_supersede'
    ) THEN
        ALTER TABLE documents
            ADD CONSTRAINT documents_no_self_supersede
            CHECK (superseded_by IS DISTINCT FROM doc_id);
    END IF;
END $$;

COMMENT ON COLUMN documents.doc_number IS
    'Human document identifier stable across revisions (e.g. SOP-4412). Groups '
    'revisions that doc_id, being a content hash, necessarily separates.';
COMMENT ON COLUMN documents.revision_conflict IS
    'Two or more documents share a doc_number but cannot be ordered from the '
    'evidence available. Both remain current; a human must resolve it. The '
    'system does not guess which procedure an engineer should follow.';
