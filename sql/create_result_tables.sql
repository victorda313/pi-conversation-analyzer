-- Session-level classification results
CREATE TABLE IF NOT EXISTS session_classification (
    session_id TEXT PRIMARY KEY,
    primary_category TEXT NOT NULL,
    all_categories JSONB NOT NULL,
    processed_upto TIMESTAMPTZ NOT NULL,
    run_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model TEXT NOT NULL,
    instructions_version TEXT,
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_session_classification_processed_upto
    ON session_classification (processed_upto DESC);

-- Message-level classification results
CREATE TABLE IF NOT EXISTS message_classification (
    message_id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    primary_category TEXT NOT NULL,
    all_categories JSONB NOT NULL,
    run_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model TEXT NOT NULL,
    instructions_version TEXT
);

CREATE INDEX IF NOT EXISTS idx_message_classification_session_id
    ON message_classification (session_id);
