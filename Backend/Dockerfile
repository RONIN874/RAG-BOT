FROM python:3.11-slim

RUN useradd -m -u 1000 appuser

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set cache path before download so build-time and runtime use the same location
ENV SENTENCE_TRANSFORMERS_HOME=/app/models

# Pre-download embedding model at build time — zero cold start on first request
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2')"

COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 7860

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
