---
title: PMTPT Clinical AI Assistant
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
---

# PMTPT Clinical AI Assistant

A RAG-based clinical AI assistant for Tuberculosis Preventive Treatment (TPT).

## Architecture
- **FastAPI** backend with LangGraph agent
- **pgvector** (PostgreSQL) for semantic chunk search
- **all-MiniLM-L6-v2** for local embeddings (pre-loaded at build time)
- **Groq / LLaMA 3.3 70B** for answer synthesis

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/ask` | Ask a clinical question |
| `POST` | `/v1/upload-pdf` | Upload & index a PDF |
| `GET`  | `/v1/documents` | List all indexed documents |
| `DELETE` | `/v1/documents/{id}` | Remove a document |
| `GET`  | `/health` | Health check |
| `GET`  | `/docs` | Swagger UI |

## Environment Variables (set in HF Space Secrets)

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | ✅ | PostgreSQL connection string (Neon recommended) |
| `GROQ_API_KEY` | ✅ | Groq API key |
| `FRONTEND_URL` | ✅ | Your frontend URL for CORS |
| `API_KEY` | ✅ | Secret key for X-API-Key header auth |
