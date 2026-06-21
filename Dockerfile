FROM python:3.11-slim

WORKDIR /app

# System dependencies for chromadb and audio
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# No local model download needed — embeddings use Google text-embedding-004 API

COPY . .

EXPOSE 8000

# Railway sets PORT dynamically; fall back to 8000 locally
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
