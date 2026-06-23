-- =============================================================================
-- PDF RAG Database Schema — pgvector + chunked storage
-- Run this on a fresh database
-- =============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "vector";          -- pgvector for embeddings

-- =============================================================================
-- 1. DOCUMENT CATEGORIES
-- =============================================================================
CREATE TABLE document_categories (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    label       TEXT NOT NULL,
    description TEXT,
    created_at  TIMESTAMPTZ DEFAULT now()
);

INSERT INTO document_categories (name, label, description) VALUES
    ('treatment_protocol', 'Treatment Protocol',  'TPT treatment guidelines and protocols'),
    ('lab_report',         'Lab Report',           'Laboratory test results and reports'),
    ('patient_summary',    'Patient Summary',      'Patient case summaries and discharge notes'),
    ('policy_document',    'Policy Document',      'Ministry or institutional policy documents'),
    ('research_paper',     'Research Paper',       'Clinical research and study papers'),
    ('general',            'General',              'Uncategorised clinical documents');


-- =============================================================================
-- 2. CLINICAL DOCUMENTS
-- =============================================================================
CREATE TABLE clinical_documents (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename        TEXT NOT NULL,
    category_id     INT REFERENCES document_categories(id) ON DELETE SET NULL,
    file_data       BYTEA NOT NULL,
    page_count      INT,
    file_size_bytes INT,
    chunk_count     INT DEFAULT 0,
    metadata        JSONB DEFAULT '{}',
    uploaded_at     TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_clinical_docs_category   ON clinical_documents(category_id);
CREATE INDEX idx_clinical_docs_uploaded   ON clinical_documents(uploaded_at DESC);
CREATE INDEX idx_clinical_docs_metadata   ON clinical_documents USING GIN(metadata);


-- =============================================================================
-- 3. DOCUMENT CHUNKS
--    384-dim embeddings from all-MiniLM-L6-v2
-- =============================================================================
CREATE TABLE document_chunks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     UUID NOT NULL REFERENCES clinical_documents(id) ON DELETE CASCADE,
    chunk_index     INT  NOT NULL,
    page_number     INT,
    chunk_text      TEXT NOT NULL,
    token_count     INT,
    embedding       vector(384),
    created_at      TIMESTAMPTZ DEFAULT now(),
    UNIQUE(document_id, chunk_index)
);

CREATE INDEX idx_chunks_embedding
    ON document_chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX idx_chunks_document_id  ON document_chunks(document_id);
CREATE INDEX idx_chunks_chunk_index  ON document_chunks(document_id, chunk_index);
CREATE INDEX idx_chunks_page_number  ON document_chunks(document_id, page_number);


-- =============================================================================
-- 4. QUERY AUDIT LOG
-- =============================================================================
CREATE TABLE query_audit_log (
    id                 SERIAL PRIMARY KEY,
    user_question      TEXT NOT NULL,
    is_clinical        BOOLEAN DEFAULT TRUE,
    tool_calls         JSONB DEFAULT '[]',
    chunks_used        UUID[],
    formatted_response TEXT,
    error_message      TEXT,
    duration_ms        INT,
    queried_at         TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_audit_queried_at ON query_audit_log(queried_at DESC);


-- =============================================================================
-- 5. HELPER VIEW
-- =============================================================================
CREATE VIEW chunks_with_document AS
SELECT
    ck.id              AS chunk_id,
    ck.document_id,
    ck.chunk_index,
    ck.page_number,
    ck.chunk_text,
    ck.token_count,
    ck.embedding,
    d.filename,
    d.metadata         AS doc_metadata,
    d.uploaded_at,
    dc.label           AS category
FROM document_chunks ck
JOIN clinical_documents   d  ON d.id  = ck.document_id
LEFT JOIN document_categories dc ON dc.id = d.category_id;
