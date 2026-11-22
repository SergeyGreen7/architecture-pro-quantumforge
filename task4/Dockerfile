FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    EMBEDDING_MODEL=BAAI/bge-m3 \
    EMBEDDING_CACHE_DIR=/app/task2/embedding_model_cache \
    RAG_INDEX_DIR=/app/task3/chroma_db \
    LOG_DIR=/tmp/quantumforge-rag-bot/logs

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir uv==0.13.7

WORKDIR /app/task4

COPY task4/pyproject.toml task4/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY task4/src ./src
COPY task3/chroma_db /app/task3/chroma_db

RUN uv sync --frozen --no-dev \
    && mkdir -p /app/task2/embedding_model_cache \
    && .venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3', cache_folder='/app/task2/embedding_model_cache', device='cpu')"

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 8080

CMD ["sh", "-c", "exec .venv/bin/uvicorn quantumforge_bot.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
