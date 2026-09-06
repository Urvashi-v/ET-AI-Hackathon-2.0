-- =============================================================================
-- 003_vector_index.sql -- pgvector dense index.
--
-- The embedding dimension is a property of the configured provider, so the
-- column type is templated: `${EMBEDDING_DIM}` is substituted by the migration
-- runner (services/common/migrate.py) from the EMBEDDING_DIM setting. Changing
-- provider therefore means changing one environment variable and re-running
-- migrations, rather than editing SQL.
--
-- No embeddings exist until an embedding provider is configured. Until then
-- `chunk_embeddings` is simply empty and dense retrieval reports
-- `provider_not_configured` -- it does not fall back to random vectors.
-- =============================================================================

CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id    TEXT PRIMARY KEY REFERENCES document_chunks (chunk_id) ON DELETE CASCADE,
    doc_id      TEXT NOT NULL REFERENCES documents (doc_id) ON DELETE CASCADE,
    embedding   vector(${EMBEDDING_DIM}) NOT NULL,
    -- The model tag lets embeddings be migrated incrementally when the provider
    -- changes, instead of forcing a full corpus re-embed.
    model       TEXT NOT NULL,
    dim         INTEGER NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_doc   ON chunk_embeddings (doc_id);
CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_model ON chunk_embeddings (model);

-- HNSW with cosine distance. Built now so the first embedded corpus is indexed
-- from the start; on an empty table this is free.
CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_hnsw
    ON chunk_embeddings USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
