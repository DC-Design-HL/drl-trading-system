#!/usr/bin/env bash
# One-shot: ARM the autonomous live-apply loop.
#
# Scheduled (via crontab) for 2026-05-29 20:00 UTC — just before experiment
# #2's 7-day paper window closes (20:18 UTC) — so the loop can carry exp #2
# through the canary gate to live on schedule. Year-guarded + idempotent
# (touch), so it fires exactly once and a future-year cron tick is a no-op.
#
# Arming only creates the AUTONOMY_ARMED flag. Every actual safety check
# (Risk Officer veto, monotonic-tightening guard, circuit breaker) happens
# later in the orchestrator at apply time — not here.
set -euo pipefail

REPO="/home/claude/packages/327adce6-6ec4-4402-890c-9d12c6e8a471/workspace/drl-trading-system"
cd "$REPO"

LOG="logs/self_improve_arm.log"
mkdir -p data/self_improve logs

# Guard: only 2026 (the scheduled year). Prevents an accidental re-fire.
if [ "$(date -u +%Y)" != "2026" ]; then
    echo "$(date -u) arm skipped: not 2026" >> "$LOG"
    exit 0
fi

# Respect an existing kill switch — never arm if the system is frozen.
if [ -f data/self_improve/AUTONOMY_DISABLED ]; then
    echo "$(date -u) arm ABORTED: AUTONOMY_DISABLED present" >> "$LOG"
    exit 0
fi

touch data/self_improve/AUTONOMY_ARMED
echo "$(date -u) AUTONOMY_ARMED created" >> "$LOG"

# Best-effort Telegram confirmation so Chen sees it armed.
set +e
set -a; . ./.env 2>/dev/null; set +a
if [ -n "${TELEGRAM_ALERT_BOT_TOKEN:-}" ]; then
    curl -s "https://api.telegram.org/bot${TELEGRAM_ALERT_BOT_TOKEN}/sendMessage" \
        --data-urlencode "chat_id=${TELEGRAM_CHAT_ID:--5243679323}" \
        --data-urlencode "text=🟢 Autonomous loop ARMED on schedule ($(date -u +%H:%M) UTC). exp #2 will auto-apply when its paper window closes (~20:18 UTC), then 48h canary → live. Circuit breaker active. Freeze anytime: touch data/self_improve/AUTONOMY_DISABLED" \
        >/dev/null 2>&1
fi
echo "$(date -u) arm script done" >> "$LOG"
exit 0
