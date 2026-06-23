"""
ingest.py — PDF ingestion pipeline
===================================
Responsibilities:
  1. Extract text from PDF (PyMuPDF, page-aware)
  2. Chunk text (~300 tokens, 50-token overlap)
  3. Embed each chunk (all-MiniLM-L6-v2, local, no API cost)
  4. Store document + chunks in PostgreSQL with pgvector

Called by the /v1/upload-pdf endpoint in app.py.
Never called by the agent — this is upload-time only.
"""

import os
import json
import logging
from typing import Generator

import fitz                          # PyMuPDF
import psycopg2
import psycopg2.extras
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("pmtpt.ingest")

# ---------------------------------------------------------------------------
# Embedding model — loaded once at module import, reused for every upload.
# all-MiniLM-L6-v2: 384-dim, ~80MB, fast on CPU, good semantic quality.
# ---------------------------------------------------------------------------
_EMBED_MODEL: SentenceTransformer | None = None

def _get_embed_model() -> SentenceTransformer:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        logger.info("Loading embedding model (first call — cached after this) …")
        _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedding model ready.")
    return _EMBED_MODEL


# ---------------------------------------------------------------------------
# Chunking parameters
# ---------------------------------------------------------------------------
CHUNK_SIZE    = 300    # target tokens per chunk
CHUNK_OVERLAP = 50     # overlap between consecutive chunks

def _rough_token_count(text: str) -> int:
    """Fast approximation: 1 token ≈ 4 characters (good enough for chunking)."""
    return len(text) // 4


def _chunk_page_text(
    page_text: str,
    page_number: int,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> Generator[dict, None, None]:
    """
    Split a single page's text into overlapping word-boundary chunks.
    Yields dicts: {page_number, chunk_text, token_count}
    """
    words = page_text.split()
    if not words:
        return

    start = 0
    while start < len(words):
        end = start + chunk_size * 4  # word count approximation (4 words ≈ 1 token avg)
        chunk_words = words[start:end]
        chunk_text  = " ".join(chunk_words).strip()

        if chunk_text:
            yield {
                "page_number": page_number,
                "chunk_text":  chunk_text,
                "token_count": _rough_token_count(chunk_text),
            }

        # Slide window forward (subtract overlap)
        step = max(1, (chunk_size - overlap) * 4)
        start += step


def extract_and_chunk(pdf_bytes: bytes) -> tuple[int, list[dict]]:
    """
    Extract text from PDF bytes and split into chunks.

    Returns:
        page_count: number of pages in the PDF
        chunks: list of {page_number, chunk_text, token_count}
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_count = len(doc)
    all_chunks: list[dict] = []

    for page in doc:
        text = page.get_text("text").strip()
        if not text:
            continue  # skip blank/image-only pages
        for chunk in _chunk_page_text(text, page_number=page.number + 1):
            all_chunks.append(chunk)

    doc.close()
    logger.info("Extracted %d chunks from %d pages.", len(all_chunks), page_count)
    return page_count, all_chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Add 'embedding' key (list[float]) to each chunk dict.
    Batches all chunks in one model call for efficiency.
    """
    model  = _get_embed_model()
    texts  = [c["chunk_text"] for c in chunks]
    # encode returns a numpy array of shape (N, 384)
    vectors = model.encode(texts, batch_size=64, show_progress_bar=False)

    for chunk, vec in zip(chunks, vectors):
        chunk["embedding"] = vec.tolist()

    logger.info("Embedded %d chunks.", len(chunks))
    return chunks


def store_document(
    conn,
    filename:    str,
    pdf_bytes:   bytes,
    page_count:  int,
    chunks:      list[dict],
    category_id: int | None = None,
    metadata:    dict       = None,
) -> str:
    """
    Insert the document and all its chunks in a single transaction.
    Returns the new document UUID.
    """
    metadata = metadata or {}

    with conn:   # transaction — rolls back automatically on exception
        cur = conn.cursor()

        # Insert parent document
        cur.execute(
            """
            INSERT INTO clinical_documents
                (filename, category_id, file_data, page_count, file_size_bytes,
                 chunk_count, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                filename,
                category_id,
                psycopg2.Binary(pdf_bytes),
                page_count,
                len(pdf_bytes),
                len(chunks),
                json.dumps(metadata),
            ),
        )
        document_id = str(cur.fetchone()[0])
        logger.info("Inserted document %s (%s).", document_id, filename)

        # Bulk-insert chunks with embeddings using execute_values (fast)
        chunk_rows = [
            (
                document_id,
                idx,
                c["page_number"],
                c["chunk_text"],
                c["token_count"],
                c["embedding"],     # psycopg2 + pgvector adapter accepts list[float]
            )
            for idx, c in enumerate(chunks)
        ]

        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO document_chunks
                (document_id, chunk_index, page_number, chunk_text, token_count, embedding)
            VALUES %s
            """,
            chunk_rows,
            template="(%s, %s, %s, %s, %s, %s::vector)",
            page_size=200,
        )
        logger.info("Inserted %d chunks for document %s.", len(chunks), document_id)

        cur.close()

    return document_id


def ingest_pdf(
    conn,
    filename:    str,
    pdf_bytes:   bytes,
    category_id: int | None = None,
    metadata:    dict       = None,
) -> dict:
    """
    Full pipeline: extract → chunk → embed → store.
    Returns summary dict suitable for the API response.
    """
    page_count, chunks = extract_and_chunk(pdf_bytes)

    if not chunks:
        raise ValueError("No extractable text found in PDF. It may be a scanned image-only PDF.")

    chunks = embed_chunks(chunks)

    document_id = store_document(
        conn       = conn,
        filename   = filename,
        pdf_bytes  = pdf_bytes,
        page_count = page_count,
        chunks     = chunks,
        category_id = category_id,
        metadata   = metadata or {},
    )

    return {
        "document_id": document_id,
        "filename":    filename,
        "page_count":  page_count,
        "chunk_count": len(chunks),
        "status":      "indexed",
    }
