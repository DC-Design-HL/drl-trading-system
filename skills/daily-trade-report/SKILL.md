---
name: daily-trade-report
description: Generate daily trade reports with signal correlation analysis. Use when asked for trade reports, daily summaries, performance analysis, signal effectiveness, or "how did we do today/yesterday". Also use when Chen asks for trade stats, win rates, or which signals are working.
---

# Daily Trade Report

Generates a comprehensive trade performance report with signal correlation analysis.

## When to Use

- "daily report" / "trade report" / "how did we do today"
- "whale analysis" (for on-demand whale signal snapshot)
- Any request about trade performance, win rates, signal effectiveness
- Automatic daily at end of trading day (if cron configured)

## How to Run

```bash
cd ~/.openclaw/projects/drl-trading-system/repo

# Last 24 hours (default)
python scripts/daily_trade_report.py

# Custom window
python scripts/daily_trade_report.py --hours 48

# Save to file
python scripts/daily_trade_report.py --save
```

## Report Contents

1. **Summary** — total trades, win rate, PnL
2. **Per-symbol breakdown** (BTC + ETH) — each trade with entry/exit/PnL
3. **Signal at entry** — MTF bias, Order Flow, Regime, Orderbook, Signal Gate
4. **Whale behavior** — sell/buy confidence + top distributing wallets at trade time
5. **Signal correlation** — which signals correlated with winners vs losers
6. **Cumulative stats** — all-time performance

## Signal Correlation Analysis

The report tracks which combinations predict winning trades:

| Signal | What it means |
|--------|---------------|
| High conf (≥0.90) | Model was very confident — historically WORST performance on BTC |
| Low conf (<0.60) | Model uncertain — historically BEST performance on BTC |
| OF=bearish | Order flow showing sell pressure — strongest standalone signal |
| Whale SELL ≥ 40% | Whale behavior model sees distribution — tracking correlation |

## Key Findings (as of 2026-03-29, 48 trades)

### BTC Critical Patterns
- **Confidence is INVERTED**: conf ≥ 0.90 → 11% WR (-$19K), conf < 0.60 → 73% WR (+$6K)
- **Order Flow is the best signal**: OF=bearish → 58% WR, OF=bullish → 25% WR
- **Signal Gate AUTONOMOUS trades lose**: -$14,618 on 4 trades
- **LONG + OF=bullish = disaster**: 0% WR, -$16,380
- **SHORT + conf < 0.60 = golden**: 100% WR, +$3,815

### ETH
- Smaller trade sizes ($0-30 range)
- Different dynamics — MTF=BEARISH actually works for longs (small sample)
- More data needed before drawing conclusions

## DO NOT CHANGE TRADING LOGIC

As of 2026-03-29, Chen has instructed: **observe only, do not modify filters or trading logic.** Track the data, report daily, and wait for enough evidence before making changes.

## Whale Shadow Tracking

Every trade logs whale signal to `logs/whale_shadow.jsonl`. After 1-2 weeks:

```bash
python scripts/analyze_whale_shadow.py
```

This shows whether whale SELL predictions correlate with actual trade outcomes.

## Files

| Path | Role |
|------|------|
| `scripts/daily_trade_report.py` | Report generator |
| `scripts/analyze_whale_shadow.py` | Whale shadow analysis (run after 2 weeks) |
| `logs/htf_pending_alerts.jsonl` | Source: all trade alerts with signals |
| `logs/whale_shadow.jsonl` | Source: whale signals at each trade |
| `logs/daily_reports/` | Saved report files |

## Reminder

- **4-day check (2026-04-02)**: Run first whale shadow analysis, review daily reports
- **2-week check (2026-04-12)**: Full analysis, decide whether to integrate whale SELL into trading logic
