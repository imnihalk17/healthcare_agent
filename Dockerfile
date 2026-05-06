FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --only-binary :all: -r requirements.txt

COPY api_production.py api_production.py

RUN mkdir -p chroma_db
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV HOST=0.0.0.0

CMD ["python", "-m", "uvicorn", "api_production:app", "--host", "0.0.0.0", "--port", "8000"]
