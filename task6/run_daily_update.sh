#!/bin/sh

set -u

TASK6_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON_BIN="$TASK6_DIR/.venv/bin/python"
CRON_LOG="$TASK6_DIR/storage/logs/cron.log"
MAX_ATTEMPTS=3
RETRY_DELAY_SECONDS=60

mkdir -p "$TASK6_DIR/storage/logs"

if [ ! -x "$PYTHON_BIN" ]; then
  printf '%s ERROR task6 virtual environment is missing; run: cd task6 && uv sync\n' \
    "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$CRON_LOG"
  exit 2
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

attempt=1
while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
  printf '%s INFO daily index update attempt=%s/%s\n' \
    "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$attempt" "$MAX_ATTEMPTS" >> "$CRON_LOG"

  "$PYTHON_BIN" "$TASK6_DIR/update_index.py" >> "$CRON_LOG" 2>&1
  status=$?
  if [ "$status" -eq 0 ]; then
    printf '%s INFO daily index update completed\n' \
      "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$CRON_LOG"
    exit 0
  fi

  printf '%s ERROR daily index update failed attempt=%s status=%s\n' \
    "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$attempt" "$status" >> "$CRON_LOG"
  if [ "$attempt" -lt "$MAX_ATTEMPTS" ]; then
    sleep "$RETRY_DELAY_SECONDS"
  fi
  attempt=$((attempt + 1))
done

printf '%s ERROR daily index update exhausted retries\n' \
  "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$CRON_LOG"
exit 1
