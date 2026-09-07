-- P&ID digitisation: what was found on a drawing, and where.
--
-- A tag on a P&ID is different from a tag in prose. In prose the useful anchor
-- is a character offset; on a drawing it is a rectangle, because the question
-- an engineer asks is "where is P-101B on this sheet?" and the answer has to be
-- somewhere you can point at.
--
-- One row per detected thing, whatever detected it. Keeping text detections and
-- symbol detections in one table rather than two is deliberate: they share the
-- same geometry, the same provenance questions and the same consumer (the
-- drawing viewer), and splitting them would mean two queries and two code paths
-- to draw one overlay.

CREATE TABLE IF NOT EXISTS drawing_detections (
    detection_id    bigserial PRIMARY KEY,
    doc_id          text NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    page            integer NOT NULL,

    -- tag | instrument_bubble | line_segment | equipment_symbol
    kind            text NOT NULL,
    -- The recognised text, when there is any. A circle has geometry and no text
    -- until a tag is associated with it.
    text            text,
    normalised      text,

    -- PDF user-space coordinates, origin top-left, same frame the page renders
    -- in. Stored as four columns rather than a box type so the viewer can scale
    -- them without parsing, and so a partial box is impossible.
    x0              real NOT NULL,
    y0              real NOT NULL,
    x1              real NOT NULL,
    y1              real NOT NULL,
    -- Page dimensions at detection time. Without these a stored box cannot be
    -- placed on a render at any other scale, which is every render.
    page_width      real NOT NULL,
    page_height     real NOT NULL,

    -- How it was found: pdf_word_geometry, hough_circle, hough_lines_p, ...
    -- Named rather than scored alone, because "a Hough circle at accumulator
    -- threshold 30" is a claim a reader can evaluate and "0.82 confident" is not.
    method          text NOT NULL,
    -- Detector-specific quality. For text this is the tag parser's confidence;
    -- for geometric detectors it is a normalised measure of fit, and it is NOT a
    -- probability -- see services/ingest/pid.py.
    confidence      real NOT NULL DEFAULT 0.0,

    -- The canonical asset this detection resolves to, when it resolves. NULL is
    -- a real answer: a tag on a drawing that names equipment the corpus has
    -- never ingested is a finding, and inventing the asset would hide it.
    linked_asset_id text REFERENCES assets(asset_id) ON DELETE SET NULL,

    -- Free-form detector output: Hough radius, line angle, associated bubble.
    properties      jsonb NOT NULL DEFAULT '{}'::jsonb,
    data_class      data_class NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now(),

    -- Re-running the pipeline on the same drawing must update rather than
    -- duplicate. Geometry is the natural key: two detections of the same kind at
    -- the same place are the same detection.
    CONSTRAINT drawing_detections_unique
        UNIQUE (doc_id, page, kind, x0, y0, x1, y1)
);

CREATE INDEX IF NOT EXISTS idx_detections_doc_page
    ON drawing_detections (doc_id, page);

-- "Where does this asset appear on any drawing?" — the query behind clicking an
-- asset and having its location highlighted.
CREATE INDEX IF NOT EXISTS idx_detections_asset
    ON drawing_detections (linked_asset_id)
    WHERE linked_asset_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_detections_kind
    ON drawing_detections (kind);

-- Connectivity recovered from the drawing: which two detections a line joins.
-- Separate from the detections themselves because an edge is not a thing on the
-- page, it is a relationship between two things on the page, and conflating
-- them makes "how many symbols are on this sheet?" unanswerable.
CREATE TABLE IF NOT EXISTS drawing_connections (
    connection_id   bigserial PRIMARY KEY,
    doc_id          text NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    page            integer NOT NULL,
    from_detection  bigint NOT NULL REFERENCES drawing_detections(detection_id) ON DELETE CASCADE,
    to_detection    bigint NOT NULL REFERENCES drawing_detections(detection_id) ON DELETE CASCADE,
    -- The line segment(s) that justify this connection. Kept so a claimed
    -- connection can be drawn back onto the sheet and checked by eye, which is
    -- the only practical way to review topology extraction.
    via_detections  bigint[] NOT NULL DEFAULT '{}',
    method          text NOT NULL,
    confidence      real NOT NULL DEFAULT 0.0,
    properties      jsonb NOT NULL DEFAULT '{}'::jsonb,
    data_class      data_class NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT drawing_connections_unique UNIQUE (from_detection, to_detection, method),
    CONSTRAINT drawing_connections_not_self CHECK (from_detection <> to_detection)
);

CREATE INDEX IF NOT EXISTS idx_connections_doc ON drawing_connections (doc_id, page);

COMMENT ON TABLE drawing_detections IS
    'One row per thing found on a drawing page, with the rectangle it occupies. '
    'Text and symbol detections share this table because they share geometry, '
    'provenance and consumer.';
COMMENT ON COLUMN drawing_detections.confidence IS
    'Detector-specific quality measure, NOT a probability. For text detections it '
    'is the tag parser confidence; for Hough detectors it is a normalised measure '
    'of geometric fit. Comparable within a method, meaningless across methods.';
COMMENT ON COLUMN drawing_detections.linked_asset_id IS
    'NULL means the tag was read but names equipment not in the corpus. That is a '
    'finding, not a failure, and the asset is never invented to fill it.';
