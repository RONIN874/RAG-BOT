import os
import json
import logging
import secrets
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

import psycopg2
from fastapi import FastAPI, Header, HTTPException, Request, UploadFile, File, Form, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from src.main import graph
from src.ingest import ingest_pdf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("pmtpt")

_REQUIRED_ENV = ["DATABASE_URL", "GROQ_API_KEY", "FRONTEND_URL"]

def _validate_env() -> None:
    missing = [k for k in _REQUIRED_ENV if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}.")

def _validate_db() -> None:
    try:
        conn = psycopg2.connect(os.getenv("DATABASE_URL", ""), connect_timeout=5)
        conn.close()
        logger.info("Database connectivity check passed.")
    except Exception as exc:
        raise RuntimeError(f"Database connectivity check failed: {exc}") from exc

def _get_db_conn():
    """Return a raw psycopg2 connection (caller must close/return it)."""
    return psycopg2.connect(os.getenv("DATABASE_URL", ""), connect_timeout=5)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting PMTPT Clinical AI Assistant …")
    _validate_env()
    _validate_db()
    logger.info("All startup checks passed. Server is ready.")
    yield
    logger.info("Server shutting down.")

FRONTEND_URL = os.getenv("FRONTEND_URL", "")
if not FRONTEND_URL:
    logger.warning("FRONTEND_URL is not set. CORS will block all cross-origin requests.")

limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])

app = FastAPI(
    title="PMTPT Clinical AI Assistant",
    version="2.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL] if FRONTEND_URL else [],
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)

_API_KEY = os.getenv("API_KEY", "")

def _require_api_key(x_api_key: str = Header(default="")) -> None:
    if not _API_KEY:
        logger.warning("API_KEY not set — request allowed without auth.")
        return
    if not secrets.compare_digest(x_api_key, _API_KEY):
        logger.warning("Rejected request with invalid or missing API key.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Pass it as the X-API-Key header.",
        )


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class Question(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)

    @field_validator("question")
    @classmethod
    def no_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Question must not be blank.")
        return stripped


class AskResponse(BaseModel):
    answer: str


class UploadResponse(BaseModel):
    document_id: str
    filename:    str
    page_count:  int
    chunk_count: int
    status:      str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post(
    "/v1/ask",
    response_model=AskResponse,
    summary="Ask a clinical question — answered from indexed PDF documents",
)
@limiter.limit("10/minute")
async def ask(
    request: Request,
    q: Question,
    x_api_key: str = Header(default=""),
) -> AskResponse:
    """
    Invoke the LangGraph RAG agent.
    The agent searches indexed PDF chunks semantically and synthesises an answer.
    Requires X-API-Key header.
    """
    _require_api_key(x_api_key)
    logger.info("Received question (length=%d)", len(q.question))

    try:
        result = graph.invoke(
            {"user_question": q.question},
            config={"recursion_limit": 3},
        )
    except Exception as exc:
        logger.exception("LangGraph invocation failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Something went wrong.")

    answer = result.get("formatted_response", "") or "Something went wrong."
    return AskResponse(answer=answer)


@app.post(
    "/v1/upload-pdf",
    response_model=UploadResponse,
    summary="Upload a PDF — extracts, chunks, embeds, and indexes it",
)
@limiter.limit("5/minute")
async def upload_pdf(
    request:     Request,
    file:        UploadFile = File(..., description="PDF file to index"),
    category:    str        = Form(default="general", description="document category name"),
    metadata:    str        = Form(default="{}", description="optional JSON metadata string"),
    x_api_key:   str        = Header(default=""),
) -> UploadResponse:
    """
    Upload and index a PDF document.

    Steps performed server-side:
      1. Extract text page-by-page (PyMuPDF)
      2. Chunk into ~300-token windows with 50-token overlap
      3. Embed each chunk (all-MiniLM-L6-v2, local)
      4. Store document + chunks in PostgreSQL with pgvector embeddings

    After upload the document is immediately searchable via /v1/ask.
    Requires X-API-Key header.
    """
    _require_api_key(x_api_key)

    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are accepted.",
        )

    # Read file bytes
    pdf_bytes = await file.read()
    if len(pdf_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # File size guard — 50 MB max
    MAX_SIZE = 50 * 1024 * 1024
    if len(pdf_bytes) > MAX_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File exceeds 50 MB limit.",
        )

    # Parse metadata JSON
    try:
        metadata_dict = json.loads(metadata)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="metadata must be a valid JSON string.",
        )

    # Resolve category_id from name
    conn = _get_db_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id FROM document_categories WHERE name = %s", (category,)
        )
        row = cur.fetchone()
        category_id = row[0] if row else None
        cur.close()

        if category_id is None:
            logger.warning("Unknown category '%s' — defaulting to 'general'.", category)
            cur = conn.cursor()
            cur.execute("SELECT id FROM document_categories WHERE name = 'general'")
            row = cur.fetchone()
            category_id = row[0] if row else None
            cur.close()

        # Run the full ingest pipeline
        try:
            result = ingest_pdf(
                conn        = conn,
                filename    = file.filename,
                pdf_bytes   = pdf_bytes,
                category_id = category_id,
                metadata    = metadata_dict,
            )
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
        except Exception as e:
            logger.exception("Ingest pipeline failed: %s", e)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Ingestion failed.")

        logger.info(
            "Indexed %s → %d chunks, %d pages.",
            file.filename, result["chunk_count"], result["page_count"]
        )
        return UploadResponse(**result)

    finally:
        conn.close()


@app.get(
    "/v1/documents",
    summary="List all indexed documents",
)
async def list_documents(x_api_key: str = Header(default="")) -> dict:
    """Returns a list of all indexed PDF documents with metadata."""
    _require_api_key(x_api_key)

    conn = _get_db_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                d.id::text,
                d.filename,
                dc.label AS category,
                d.page_count,
                d.chunk_count,
                d.file_size_bytes,
                d.metadata,
                d.uploaded_at::text
            FROM clinical_documents d
            LEFT JOIN document_categories dc ON dc.id = d.category_id
            ORDER BY d.uploaded_at DESC
            """
        )
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, row)) for row in cur.fetchall()]
        cur.close()
        return {"documents": rows, "total": len(rows)}
    finally:
        conn.close()


@app.delete(
    "/v1/documents/{document_id}",
    summary="Delete an indexed document and all its chunks",
)
async def delete_document(
    document_id: str,
    x_api_key: str = Header(default=""),
) -> dict:
    """Permanently removes a document and cascades to its chunks."""
    _require_api_key(x_api_key)

    conn = _get_db_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM clinical_documents WHERE id = %s RETURNING filename",
            (document_id,)
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Document not found.")
        conn.commit()
        cur.close()
        return {"deleted": document_id, "filename": row[0]}
    finally:
        conn.close()


@app.get("/health", summary="Health check")
async def health() -> dict:
    try:
        conn = psycopg2.connect(os.getenv("DATABASE_URL", ""), connect_timeout=3)
        conn.close()
        db_status = "ok"
    except Exception as exc:
        logger.error("Health check DB ping failed: %s", exc)
        db_status = "unavailable"

    overall     = "ok" if db_status == "ok" else "degraded"
    status_code = status.HTTP_200_OK if overall == "ok" else status.HTTP_503_SERVICE_UNAVAILABLE

    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=status_code,
        content={
            "status":   overall,
            "database": db_status,
            "service":  "PMTPT Clinical AI Assistant",
            "version":  "2.0.0",
        },
    )


@app.get("/", summary="Root status")
async def root() -> dict:
    return {
        "status":  "active",
        "service": "PMTPT Clinical AI Assistant",
        "version": "2.0.0",
        "docs":    "/docs",
        "endpoints": {
            "ask":       "POST /v1/ask",
            "upload":    "POST /v1/upload-pdf",
            "list":      "GET  /v1/documents",
            "delete":    "DELETE /v1/documents/{id}",
        },
    }
