set -a
source .env
set +a

F_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
./.venv/bin/uvicorn quantumforge_bot.app:app \
  --host 127.0.0.1 \
  --port 8080