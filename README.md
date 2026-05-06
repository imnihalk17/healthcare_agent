# RAG API (minimal)

This repository contains a minimal production-ready RAG API designed for deployment (Render, Docker).

Usage

1. Set `HUGGINGFACE_TOKEN` in environment for private models (if required).
2. Build and run with Docker or deploy to Render.

Local run (after installing dependencies):

```bash
pip install -r requirements.txt
python api_production.py
```

Endpoints

- `GET /health` - health check
- `POST /query` - main query endpoint

Notes

- Replace or populate the `chroma_db` directory with your vector index for production data.
- This repo avoids embedding secrets; provide tokens via env vars.
