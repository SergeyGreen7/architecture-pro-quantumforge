cd ../task4
set -a
source .env
set +a

export RAG_INDEX_DIR=../task5/chroma_db
export RAG_SECURITY_MODE=off
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ./.venv/bin/uvicorn quantumforge_bot.app:app --host 127.0.0.1 --port 8080