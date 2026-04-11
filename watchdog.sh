#!/bin/bash
# watchdog.sh — Detect dead DRL services and auto-restart via start_services.sh.
#
# Run from cron every 2 minutes:
#   */2 * * * * /home/claude/.../drl-trading-system/watchdog.sh
#
# Reads logs/running_services.json, kill -0's every PID, and if any are dead
# (or the file is missing a service) triggers start_services.sh. Sends a
# Telegram alert when a failure is detected so you know recovery fired.
#
# Silent on the happy path — only logs + alerts when something is actually wrong.

set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDS_FILE="$REPO/logs/running_services.json"
LOCK_FILE="$REPO/logs/watchdog.lock"
LOG="$REPO/logs/watchdog.log"
TS() { date '+%Y-%m-%d %H:%M:%S'; }

mkdir -p "$REPO/logs"

# Prevent concurrent runs. If a previous watchdog is still restarting things,
# bail silently — the next cron tick will re-check.
exec 200>"$LOCK_FILE"
if ! flock -n 200; then
    exit 0
fi

# No PIDs file = services never started. Don't auto-start from nothing; require
# a human to run start_services.sh at least once so we know the initial state
# was intentional.
if [ ! -f "$PIDS_FILE" ]; then
    echo "[$(TS)] no running_services.json — skipping (run start_services.sh first)" >> "$LOG"
    exit 0
fi

# Parse expected PIDs and check liveness. Each dead service becomes
# "name(pid=N)" in DEAD_LIST.
DEAD_LIST=""
while IFS=':' read -r name pid; do
    [ -z "$name" ] && continue
    [ -z "$pid" ] && continue
    [ "$pid" = "null" ] && continue
    if ! kill -0 "$pid" 2>/dev/null; then
        DEAD_LIST="${DEAD_LIST}${name}(pid=${pid}) "
    fi
done < <(python3 -c "
import json, sys
try:
    with open('$PIDS_FILE') as f:
        data = json.load(f)
    for k, v in data.items():
        print(f'{k}:{v.get(\"pid\", \"\")}')
except Exception as e:
    print(f'ERROR:{e}', file=sys.stderr)
    sys.exit(1)
")

# Happy path — everything alive, nothing to do, no log spam.
if [ -z "$DEAD_LIST" ]; then
    exit 0
fi

echo "[$(TS)] DEAD: $DEAD_LIST" >> "$LOG"
echo "[$(TS)] triggering start_services.sh..." >> "$LOG"

# --- Telegram alert: failure detected ---
# Reuse the same env vars start_services.sh uses for its "services restarted"
# notification so we stay consistent with which chat gets pinged.
set -a
# shellcheck disable=SC1091
source "$REPO/.env" 2>/dev/null || true
set +a

if [ -n "${TELEGRAM_ALERT_BOT_TOKEN:-}" ] && [ -n "${TELEGRAM_CHAT_IDS:-}" ]; then
    _CHAT=$(echo "$TELEGRAM_CHAT_IDS" | cut -d',' -f1)
    curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_ALERT_BOT_TOKEN}/sendMessage" \
        --data-urlencode "chat_id=${_CHAT}" \
        --data-urlencode "text=⚠️ Watchdog: services down → ${DEAD_LIST}
Running start_services.sh to recover..." \
        > /dev/null 2>&1 || true
fi

# --- Restart ---
# start_services.sh is idempotent (kills existing instances before relaunching)
# and sends its own "🚀 Services restarted" Telegram ping on success, so we
# don't duplicate the success notification here.
bash "$REPO/start_services.sh" >> "$LOG" 2>&1
RC=$?
echo "[$(TS)] start_services.sh exited rc=$RC" >> "$LOG"

# Only ping Telegram on failure — success is already announced by start_services.sh.
if [ $RC -ne 0 ] && [ -n "${TELEGRAM_ALERT_BOT_TOKEN:-}" ] && [ -n "${TELEGRAM_CHAT_IDS:-}" ]; then
    curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_ALERT_BOT_TOKEN}/sendMessage" \
        --data-urlencode "chat_id=${_CHAT}" \
        --data-urlencode "text=❌ Watchdog: start_services.sh FAILED (rc=${RC}) — manual intervention needed. See logs/watchdog.log" \
        > /dev/null 2>&1 || true
fi

exit $RC
