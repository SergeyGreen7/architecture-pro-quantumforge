#!/bin/sh

set -eu

TASK6_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
MARKER="# quantumforge-task6-daily-index"
CRON_LINE="0 6 * * * /bin/sh $TASK6_DIR/run_daily_update.sh $MARKER"
TEMP_FILE=$(mktemp "${TMPDIR:-/tmp}/quantumforge-cron.XXXXXX")
trap 'rm -f "$TEMP_FILE"' EXIT HUP INT TERM

crontab -l > "$TEMP_FILE" 2>/dev/null || true
if grep -F "$MARKER" "$TEMP_FILE" >/dev/null 2>&1; then
  echo "QuantumForge daily index job is already installed:"
  grep -F "$MARKER" "$TEMP_FILE"
  exit 0
fi

printf '%s\n' "$CRON_LINE" >> "$TEMP_FILE"
crontab "$TEMP_FILE"
echo "Installed daily 06:00 cron job:"
echo "$CRON_LINE"
