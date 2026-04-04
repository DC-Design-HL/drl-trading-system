# MEMORY.md

## User Preferences

### Chen Chen (@Chen4700)
- **Coding workflow (UPDATED 2026-03-26)**: CEO should do code edits directly when it's cheaper than spawning subagents. Only use subagents for large multi-file tasks, deep analysis, or when the task truly benefits from delegation. Direct edits save tokens.
- **Subagent model**: Use Sonnet (not Opus) for all subagent spawns to save costs. Set model override on every `sessions_spawn`.
- **Context compaction**: Remind Chen to compact at every 100K context tokens. Alert at 100K, 200K, 300K thresholds.
- **Claude Code bash script (DEPRECATED)**: The bash script with ANTHROPIC_API_KEY no longer works (key expired). Use `sessions_spawn` with `runtime="subagent"` instead if needed.
- **NO TRAINING ON SERVER (CRITICAL)**: NEVER run model training (walk-forward, championship, etc.) on the VPS. Only 2 CPUs + 3.7GB RAM — training freezes the whole server and kills the live bots. ALL training happens on Chen's Mac M3 Pro only.
- **Testnet Rules (CRITICAL)**:
  1. Internal bot triggers → opens REAL futures testnet position (not simulated)
  2. Every position MUST have TP and SL configured on the exchange
  3. ALL data in testnet UI tab = real data from Binance futures API, zero local calculations
  4. **NEVER open a position without SL and TP** — if TP placement fails, immediately close the position. No unprotected positions allowed. This is the #1 rule.
  - These are non-negotiable. Always validate after any testnet code change.
- **Communication & Task Management**: 
  - ALWAYS proactively update Chen on blockers immediately — don't wait for him to ask
  - He should NEVER have to ask "any updates?" — push updates proactively
  - When Claude Code finishes (success OR failure/timeout/kill), IMMEDIATELY report back
  - If a background task gets killed/times out, report it RIGHT AWAY — never silently fail
  - **TASK LIST**: Maintain active task list in TASKS.md. Every request from Chen becomes a tracked task.
  - **NO AI CRONS**: Do NOT use cron jobs that fire into the AI session — they waste tokens on every invocation (loading full 200K+ context). Trade alerts, hourly checks, etc. should be system-level scripts (bash/Python cron) that only message the group when something actually needs attention. The AI should only be invoked when a human sends a message or a subagent completes.
  - **COMPACTION THRESHOLD**: Alert Chen when context reaches 100K tokens to compact. Compact proactively — don't wait.
  - Always check TASKS.md at the start of each interaction

## Trading Guard System

### Orderbook Guard (Golden Guard) — deployed 2026-03-31
- **What**: Blocks trades when orderbook bias contradicts trade direction (SHORT+bullish OB → block, LONG+bearish OB → block)
- **Location**: `live_trading_htf.py` → `_check_orderbook_guard()`, constant `ORDERBOOK_GUARD_ENABLED`
- **Applies to**: ALL tiers including Tier 1 autonomous (unlike signal gate which only checks Tier 2)
- **Backtest results (Mar 24-31, 67 trades)**: WR 46.3%→51.8%, PnL $53→$170 (+217%), catches 3/3 whipsaw shorts
- **Fail-open**: If market data unavailable, trade proceeds
- **Toggle**: Set `ORDERBOOK_GUARD_ENABLED = False` to disable
- **Future upgrade**: Layer on BTC SMA+RSI combo guard for 56.5% WR (Option B, test after 1 week)
- **Git**: Committed to `dev` branch on 2026-03-31

### Rollout Reminders (set 2026-03-31)
- **Apr 7**: Review orderbook guard performance → deploy OB tiered sizing in shadow mode
- **Apr 14**: Review shadow results → enable actual tiered sizing if good

## Standing Rules (Chen)

### Auto-Update Skills (added 2026-03-28)
- When changing ANY existing logic, automatically update the relevant skill doc (SKILL.md) if one exists
- Do NOT ask Chen for permission — just do it
- Always mention in the reply: "Updated skill: <skill-name>"
- Applies to all code changes, not just trading logic

## Trading Rules (Chen)

### BOS/CHOCH Clean-Line Validation (added 2026-03-28)
- Structure break must start from Candle A's **wick** and end at Candle B's **body**
- The line between A and B must be **clean** — no intermediate candle wicks can intersect it
- If any wick between A and B crosses the line → the BOS/CHOCH is **invalid**
- Only clean, uninterrupted price movement = valid structural break
- Implemented in `src/signals/bos_choch.py` → `_is_clean_break()` method
- Documented in `skills/bos-choch-dynamic-sltp/SKILL.md`

## Technical Notes

### Claude Code on this server
- Server runs as root — Claude Code refuses `--permission-mode bypassPermissions` as root
- **Solution**: Created skill `claude-code-runner` at `skills/claude-code-runner/`
- Uses `claudeuser` (non-root) + copies project + runs + syncs back
- Always use the script: `bash skills/claude-code-runner/scripts/run-claude.sh <project-dir> "<prompt>"`
- ANTHROPIC_API_KEY is saved in /root/.bashrc (⚠️ key returned 401 as of 2026-03-21 — may be expired/exhausted)
- When bash script fails, fallback: use `sessions_spawn` with `runtime="acp"`, `agentId: "claude"` — this uses OpenClaw's own API routing
- Chen's preference: ALWAYS use `sessions_spawn` with `runtime="acp"` for Claude Code tasks, not the bash script

## Whale Behavior Training

### Training Parameters (what works on this server)
- **Server**: 3.7GB RAM total, ~400MB available for training
- **Recommended**: `--batch-size 8 --accum-steps 4` (effective batch 32, ~360MB RAM)
- **DON'T USE**: batch_size ≥ 32 — gets OOM killed
- **Script**: `python3 -u train_whale_behavior.py --batch-size 8 --accum-steps 4`
- **Log**: `logs/whale_behavior_training.log`
- **Model output**: `data/whale_behavior/models/whale_behavior_lstm.pt`
- **Results**: `data/whale_behavior/models/training_results.json`
- **Data**: 128K sequences from 11 wallets
- **Epoch time**: ~2 min on CPU
- **Full run**: ~100 min for 50 epochs
- **Log tip**: Use `python3 -u` (unbuffered) or `stdbuf -oL` — nohup buffers output

### Alert Integration (added 2026-03-29)
- Whale behavior confidence shown in trade alerts as `🧬 Whale Behavior: ...`
- **DISPLAY ONLY** — does NOT affect trade decisions (Chen's explicit instruction)
- Predictor: `src/whale_behavior/models/predictor.py`

## Apr 2 Session — Major Changes Made

### Guards Deployed
1. **Rescue Override disabled** (`RESCUE_ENABLED = False`) — RSI/ADX blocks are final
2. **ADX Guard raised to 20** (`ADX_GUARD_MIN = 20`) — blocks ranging market trades
3. **Trailing activation: +1% → +0.5%** (`TRAILING_BREAKEVEN_PCT = 0.005`)
4. **Trailing distance: 0.5% → 0.3%** (`TRAILING_DISTANCE_PCT = 0.003`)
5. **SQLite local storage** — replaced broken MongoDB Atlas (`STORAGE_TYPE=sqlite`)

### ADX Exhaustion Guard — PENDING (decision Apr 4)
- ADX > 60 block proposed — saves ~$93, loses ~$12 in missed winners
- Sweet spot is ADX 30-50 (71% and 67% WR)
- Both extremes terrible: ADX<20 (29% WR) and ADX>60 (29% WR)
- Doc: `docs/adx-exhaustion-guard-proposal.md`
- Reminder set for Apr 4 15:24 UTC to re-evaluate with 48h more data

### Wyckoff Implementation — ON HOLD (Apr 2)
- Detector built: 14/14 events, 12/12 tests passing, all 4 symbols
- Backtested against 47 trades on 4H/1H/15m — NOT ready as guard
- 15m too noisy (all Phase B/D), 4H too aggressive (blocks 80% trades)
- Conflict trades actually performed better (62% WR vs 39% aligned)
- **Decision**: park it, collect labeled data for 2-3 months, then train LSTM
- Full analysis: `docs/wyckoff-analysis-summary.md`
- Skill: `skills/wyckoff-implementation/SKILL.md`

### Whale Signal — NOT USEFUL YET
- Whale SELL never reached 40% threshold in 71 trades (range: 23-34%)
- No correlation with trade outcomes
- Robinhood wallet dead since Feb 14 (stale data)
- Retraining data pushed to dev, Chen to retrain on Mac M3
- Reminder set for Apr 2 18:50 UTC

### MongoDB — BROKEN
- Atlas unreachable: DNS shard-00 has no A record, shard-01/02 timeout
- Likely free tier paused or cluster migrated
- Bots fall back to SQLite (now default)
- When Chen wants cloud DB back: fix Atlas IP whitelist + set STORAGE_TYPE=mongo

### Trade Analysis Summary (Mar 29 - Apr 2, 53 trades)
- Overall: 53% WR, -$218 PnL
- Best band: ADX 30-50 (67-71% WR, profitable)
- Worst bands: ADX<20 (29% WR) and ADX>60 (29% WR)
- Average MFE on winners: ~1.4%, only captured ~60% → new trailing should improve to ~70-75%
- Stale wallets: Robinhood dead, 3 SOL + 3 XRP wallets have 0 txs

### Skills Created Today
- `skills/local-storage/SKILL.md` — SQLite storage backend rules
- `skills/wyckoff-implementation/SKILL.md` — Wyckoff implementation checkpoint

## Projects

### drl-trading-system
- Location: `~/.openclaw/projects/drl-trading-system/repo`
- Type: DRL (Deep Reinforcement Learning) crypto trading system
- Stack: Python, Binance API (testnet), MongoDB, HuggingFace
- Status: Pre-existing codebase, not built through the SaaS pipeline
- All env vars configured (Binance, MongoDB, HF, proxies, whale tracking APIs, news APIs)
- **Deployment**: All changes push to GitHub `dev` branch (`origin/dev`). NEVER push to main/prod unless Chen explicitly says so.
- **HuggingFace**: No longer used. UI is self-hosted on the server (Streamlit + Caddy).
- **Git workflow**: Work on `dev` branch, push to `origin dev`. The repo stays checked out on `dev`.
- **Architecture**: Server runs locally, HF Space is client/UI connecting via sockets
- **Binance testnet**: https://testnet.binance.vision/ — local server connects directly, no proxy needed
- **HF logs**: Use `curl -N -H "Authorization: Bearer $HF_TOKEN" "https://huggingface.co/api/spaces/Chen4700/drl-trading-bot-dev/logs/run"` to check
- **Workflow skill**: See `skills/drl-trading-workflow/SKILL.md` for full rules
