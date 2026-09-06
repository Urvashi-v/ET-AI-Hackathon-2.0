-- =============================================================================
-- 004_reference_taxonomy_class.sql
--
-- Adds `reference_taxonomy` to the data_class enum.
--
-- Curated reference data -- ISO 14224-style failure modes, equipment class
-- codes, atomised regulatory requirements -- is none of the four original
-- classes. It is not extracted from an ingested document, it is not synthetic
-- test data, it is not a model inference, and it is not a computation over
-- stored rows. It is vocabulary the system was configured with, and the
-- dashboard needs to be able to say so.
--
-- Added as its own migration rather than by editing 001_core.sql: that file has
-- already been applied and its checksum recorded, and the runner refuses to
-- silently re-run a changed migration over live data.
-- =============================================================================

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
          FROM pg_enum e
          JOIN pg_type t ON t.oid = e.enumtypid
         WHERE t.typname = 'data_class'
           AND e.enumlabel = 'reference_taxonomy'
    ) THEN
        ALTER TYPE data_class ADD VALUE 'reference_taxonomy';
    END IF;
END $$;
