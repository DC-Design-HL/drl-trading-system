"""Reviewer — nightly post-mortem over recent closes.

Two outputs each night (UTC 03:00 by cron):

  1. A statistical post-mortem to logs/decisions/post-mortems/YYYY-MM-DD.md
     — symbol/side breakdown, exit-reason distribution, biggest winners/
     losers, current trailing streaks. Always written, even without API.

  2. (Optional) An LLM-authored pattern summary appended to the same
     file. The LLM looks for non-obvious patterns the numbers don't
     surface — e.g. "5 of 7 ETH SHORT losses had same regime+news
     pattern". Only runs if ANTHROPIC_API_KEY is present.

If the LLM pass surfaces a high-confidence pattern, the Reviewer
escalates to Telegram (the trading channel) with a 1-line summary and
seeds the Researcher (M4+) with a hypothesis row in `decisions`.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from textwrap import dedent
from typing import Any, Optional

from .llm_client import (
    MODEL_OPUS,
    CallContext,
    LLMClient,
    LLMResponse,
    default_client,
    load_persona,
)
from .metrics import TradeClose, parse_ts, summarize

UTC = timezone.utc

# Same set used elsewhere (trades.action values that count as a close)
CLOSE_ACTIONS = (
    "CLOSE_LONG",
    "CLOSE_SHORT",
    "REVERSE_CLOSE_LONG",
    "REVERSE_CLOSE_SHORT",
    "SL_HIT",
    "TP_HIT",
)


# ─────────────────────────────────────────────────────────────────────────
# Data assembly
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class CloseRecord:
    id: int
    ts: datetime
    symbol: str
    side: str
    pnl: float
    reason: str
    confidence_at_open: float | None
    open_id: int | None
    open_ts: datetime | None


def load_window(
    conn: sqlite3.Connection,
    *,
    since: datetime,
    until: datetime,
) -> list[CloseRecord]:
    """Load testnet closes in [since, until] with their matched OPEN
    confidence (FIFO pair, same algorithm as the backtest harness)."""
    rows = conn.execute(
        f"""
        SELECT id, timestamp, symbol, action, pnl, confidence, reason
        FROM trades
        WHERE is_testnet=1 AND timestamp <= ?
        ORDER BY timestamp, id
        """,
        (until.isoformat(),),
    ).fetchall()

    open_stack: dict[str, list[tuple]] = {}
    closes: list[CloseRecord] = []
    for r in rows:
        _id, ts_str, symbol, action, pnl, conf, reason = r
        if not action:
            continue
        if action.startswith("OPEN_"):
            open_stack.setdefault(symbol, []).append(r)
        elif action in CLOSE_ACTIONS or action.startswith("CLOSE_"):
            stack = open_stack.get(symbol)
            o_id = o_ts = o_conf = None
            if stack:
                o_row = stack.pop(0)
                o_id, o_ts_str, _, _, _, o_conf, _ = o_row
                o_ts = parse_ts(o_ts_str) if o_ts_str else None
            close_ts = parse_ts(ts_str)
            if close_ts < since or close_ts > until:
                continue
            side = "LONG" if "LONG" in action else "SHORT"
            closes.append(
                CloseRecord(
                    id=_id,
                    ts=close_ts,
                    symbol=symbol,
                    side=side,
                    pnl=float(pnl or 0.0),
                    reason=reason or "",
                    confidence_at_open=float(o_conf) if o_conf is not None else None,
                    open_id=o_id,
                    open_ts=o_ts,
                )
            )
    return closes


# ─────────────────────────────────────────────────────────────────────────
# Statistical breakdown
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class WindowSummary:
    since: datetime
    until: datetime
    closes: list[CloseRecord]

    @property
    def n(self) -> int:
        return len(self.closes)

    def overall(self) -> dict[str, float]:
        return summarize(self._to_metrics_input())

    def by_symbol(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for sym, rows in self._group("symbol").items():
            out[sym] = summarize([self._mk(r) for r in rows])
        return out

    def by_side(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for side, rows in self._group("side").items():
            out[side] = summarize([self._mk(r) for r in rows])
        return out

    def by_exit_reason(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        groups = defaultdict(list)
        for c in self.closes:
            groups[c.reason or "(unset)"].append(c)
        for reason, rows in groups.items():
            out[reason] = summarize([self._mk(r) for r in rows])
        return out

    def biggest_wins(self, n: int = 5) -> list[CloseRecord]:
        return sorted(self.closes, key=lambda c: -c.pnl)[:n]

    def biggest_losses(self, n: int = 5) -> list[CloseRecord]:
        return sorted(self.closes, key=lambda c: c.pnl)[:n]

    def trailing_streaks(self) -> dict[tuple[str, str], int]:
        """For each (symbol, side), trailing run of pnl<=0 closes."""
        out: dict[tuple[str, str], int] = {}
        for c in sorted(self.closes, key=lambda c: c.ts, reverse=True):
            key = (c.symbol, c.side)
            if key not in out:
                out[key] = 0
            # only count contiguous trailing run
        # Recompute properly
        seen: dict[tuple[str, str], bool] = {}
        streaks: dict[tuple[str, str], int] = defaultdict(int)
        by_key: dict[tuple[str, str], list[CloseRecord]] = defaultdict(list)
        for c in self.closes:
            by_key[(c.symbol, c.side)].append(c)
        for key, rows in by_key.items():
            rows.sort(key=lambda c: c.ts, reverse=True)
            run = 0
            for c in rows:
                if c.pnl <= 0:
                    run += 1
                else:
                    break
            if run > 0:
                streaks[key] = run
        return dict(streaks)

    # Internal helpers
    def _group(self, attr: str) -> dict[str, list[CloseRecord]]:
        out: dict[str, list[CloseRecord]] = defaultdict(list)
        for c in self.closes:
            out[getattr(c, attr)].append(c)
        return out

    def _to_metrics_input(self) -> list[TradeClose]:
        return [self._mk(c) for c in self.closes]

    @staticmethod
    def _mk(c: CloseRecord) -> TradeClose:
        return TradeClose(ts=c.ts, symbol=c.symbol, side=c.side, pnl=c.pnl)


# ─────────────────────────────────────────────────────────────────────────
# Markdown rendering
# ─────────────────────────────────────────────────────────────────────────


def render_markdown(summary: WindowSummary, *, llm_pattern_section: str = "") -> str:
    overall = summary.overall()
    by_sym = summary.by_symbol()
    by_side = summary.by_side()
    by_exit = summary.by_exit_reason()
    streaks = summary.trailing_streaks()
    wins = summary.biggest_wins(5)
    losses = summary.biggest_losses(5)

    lines: list[str] = []
    lines.append(
        f"# Post-Mortem · {summary.since.date()} → {summary.until.date()}"
    )
    lines.append("")
    lines.append(
        f"Window: `{summary.since.isoformat()}` → "
        f"`{summary.until.isoformat()}` (UTC). "
        f"Closes: **{summary.n}**."
    )
    lines.append("")

    lines.append("## Overall")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| net pnl | ${overall['net_pnl_usd']:+.2f} |")
    lines.append(f"| closes | {overall['num_closes']} |")
    lines.append(f"| win rate | {overall['win_rate'] * 100:.1f}% |")
    lines.append(f"| profit factor | {_fmt_pf(overall['profit_factor'])} |")
    lines.append(f"| Sharpe | {overall['sharpe']:.2f} |")
    lines.append(f"| Sortino | {overall['sortino']:.2f} |")
    lines.append(f"| max drawdown | {overall['max_drawdown_pct']:.2f}% |")
    lines.append("")

    lines.append("## By symbol")
    lines.append("")
    lines.append("| symbol | n | WR | net pnl | PF | Sharpe |")
    lines.append("|---|---|---|---|---|---|")
    for sym, m in sorted(by_sym.items()):
        lines.append(
            f"| {sym} | {m['num_closes']} | "
            f"{m['win_rate'] * 100:.1f}% | ${m['net_pnl_usd']:+.2f} | "
            f"{_fmt_pf(m['profit_factor'])} | {m['sharpe']:.2f} |"
        )
    lines.append("")

    lines.append("## By side")
    lines.append("")
    lines.append("| side | n | WR | net pnl | PF |")
    lines.append("|---|---|---|---|---|")
    for side, m in sorted(by_side.items()):
        lines.append(
            f"| {side} | {m['num_closes']} | "
            f"{m['win_rate'] * 100:.1f}% | ${m['net_pnl_usd']:+.2f} | "
            f"{_fmt_pf(m['profit_factor'])} |"
        )
    lines.append("")

    lines.append("## By exit reason")
    lines.append("")
    lines.append("| reason | n | net pnl | avg pnl |")
    lines.append("|---|---|---|---|")
    for reason, m in sorted(by_exit.items(), key=lambda kv: kv[1]["net_pnl_usd"]):
        avg = m["net_pnl_usd"] / m["num_closes"] if m["num_closes"] else 0
        lines.append(
            f"| `{reason}` | {m['num_closes']} | "
            f"${m['net_pnl_usd']:+.2f} | ${avg:+.2f} |"
        )
    lines.append("")

    if streaks:
        lines.append("## Trailing loss streaks")
        lines.append("")
        flag = []
        for (sym, side), n in sorted(streaks.items(), key=lambda kv: -kv[1]):
            mark = " ⚠️" if n >= 3 else ""
            flag.append(f"- {sym} {side}: **{n} losses in a row**{mark}")
        lines.extend(flag)
        lines.append("")

    lines.append("## Biggest moves")
    lines.append("")
    if wins:
        lines.append("**Top winners**")
        for c in wins:
            if c.pnl <= 0:
                continue
            lines.append(
                f"- `{c.ts.isoformat()}` {c.symbol} {c.side}: "
                f"${c.pnl:+.2f} ({c.reason})"
            )
        lines.append("")
    if losses:
        lines.append("**Top losers**")
        for c in losses:
            if c.pnl >= 0:
                continue
            lines.append(
                f"- `{c.ts.isoformat()}` {c.symbol} {c.side}: "
                f"${c.pnl:+.2f} ({c.reason})"
            )
        lines.append("")

    if llm_pattern_section:
        lines.append("## Pattern analysis (LLM)")
        lines.append("")
        lines.append(llm_pattern_section.strip())
        lines.append("")

    return "\n".join(lines)


def _fmt_pf(pf: float) -> str:
    import math
    if math.isinf(pf) or pf >= 9999:
        return "∞"
    return f"{pf:.2f}"


# ─────────────────────────────────────────────────────────────────────────
# LLM pattern analysis
# ─────────────────────────────────────────────────────────────────────────


_REVIEWER_SYSTEM = """\
You are the Reviewer for an autonomous self-improving crypto trading
system. Your job: read a structured post-mortem of recent closes and
surface NON-OBVIOUS patterns — patterns the numeric tables do not
already make obvious.

Examples of useful patterns:
  * "All 3 ETH SHORT losses happened within 60 min of an EXT_POS_NEWS
    block being missed (i.e. news arrived between the guard window's
    end and the exit)"
  * "BTCUSDT LONG took 4 trades; 3 of 4 entered while ADX was in
    20-25, suggesting the ranging-regime gate may be too permissive"
  * "All SL_HIT events on XRP fired between 02:00-04:00 UTC during
    the low-liquidity window"

Avoid these (the human reader can see the numbers):
  * "XRP had the worst PF" — already in the table
  * Restating "There were N losses on symbol X"
  * Generic advice ("consider tighter stops")

Reply in MARKDOWN — no JSON envelope. 3-6 bullet points, each starting
with a 1-3 word headline in **bold**, followed by the specifics. End
with one EXPLICIT next-step suggestion the Researcher could test, or
"No high-confidence next step." if nothing stands out.
"""


def _build_user_prompt(summary: WindowSummary) -> str:
    """Render the data the LLM should reason over — already-summarized,
    not a row dump."""
    lines = []
    lines.append(f"# Window: {summary.since.isoformat()} → {summary.until.isoformat()}")
    lines.append(f"Total closes: {summary.n}")
    lines.append("")
    lines.append("## Overall")
    lines.append(json.dumps(_clean(summary.overall()), indent=2))
    lines.append("\n## By symbol")
    lines.append(json.dumps({k: _clean(v) for k, v in summary.by_symbol().items()}, indent=2))
    lines.append("\n## By side")
    lines.append(json.dumps({k: _clean(v) for k, v in summary.by_side().items()}, indent=2))
    lines.append("\n## By exit reason")
    lines.append(json.dumps({k: _clean(v) for k, v in summary.by_exit_reason().items()}, indent=2))
    lines.append("\n## Trailing streaks")
    lines.append(json.dumps({f"{k[0]}/{k[1]}": v for k, v in summary.trailing_streaks().items()}, indent=2))
    lines.append("\n## Biggest losers (id, ts, sym, side, pnl, reason, conf_at_open)")
    for c in summary.biggest_losses(10):
        if c.pnl >= 0:
            continue
        lines.append(
            f"- id={c.id} {c.ts.isoformat()} {c.symbol} {c.side} "
            f"pnl=${c.pnl:+.2f} reason={c.reason} "
            f"conf={c.confidence_at_open}"
        )
    return "\n".join(lines)


def _clean(d: dict[str, float]) -> dict[str, float]:
    """Strip inf/nan for JSON encoding into the LLM prompt."""
    import math
    out = {}
    for k, v in d.items():
        if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
            out[k] = 9999.0 if math.isinf(v) else 0.0
        else:
            out[k] = v
    return out


def run_pattern_analysis(
    summary: WindowSummary,
    *,
    client: LLMClient,
) -> LLMResponse:
    persona = load_persona("professional-trader")
    full_system = (
        _REVIEWER_SYSTEM
        + ("\n\n--- TRADER PERSONA ---\n\n" + persona if persona else "")
    )
    return client.call(
        ctx=CallContext(
            agent="reviewer",
            context_summary=f"window {summary.since.date()}→{summary.until.date()} n={summary.n}",
        ),
        model=MODEL_OPUS,
        system=full_system,
        user=_build_user_prompt(summary),
        max_tokens=1600,
    )


# ─────────────────────────────────────────────────────────────────────────
# Orchestration entry — for the cron script
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class ReviewResult:
    summary: WindowSummary
    markdown_path: Path
    markdown: str
    llm_response: Optional[LLMResponse] = None
    telegram_digest: Optional[str] = None


def run_review(
    *,
    db_path: str | Path = "data/trading.db",
    out_dir: str | Path = "logs/decisions/post-mortems",
    window_hours: float = 24.0,
    now: Optional[datetime] = None,
    client: Optional[LLMClient] = None,
    enable_llm: bool = True,
) -> ReviewResult:
    """Assemble the post-mortem, render Markdown, optionally run LLM pass.

    Returns a ReviewResult — caller decides whether to Telegram-ping or
    not (the cron wrapper does the ping).
    """
    now = now or datetime.now(UTC)
    since = now - timedelta(hours=window_hours)
    db = Path(db_path)
    with sqlite3.connect(str(db)) as conn:
        closes = load_window(conn, since=since, until=now)

    summary = WindowSummary(since=since, until=now, closes=closes)

    llm_response: Optional[LLMResponse] = None
    llm_section = ""
    if enable_llm and summary.n > 0:
        cli = client or default_client(db_path)
        llm_response = run_pattern_analysis(summary, client=cli)
        if llm_response.degraded:
            llm_section = (
                f"_LLM pass skipped — {llm_response.error or 'no API key'}_"
            )
        else:
            llm_section = llm_response.text

    markdown = render_markdown(summary, llm_pattern_section=llm_section)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{now.strftime('%Y-%m-%d')}.md"
    path.write_text(markdown, encoding="utf-8")

    # Telegram digest text — 1-2 lines summarizing this window. Cron
    # wrapper decides whether to send.
    overall = summary.overall()
    digest = (
        f"📋 Post-mortem {since.date()}→{now.date()}: "
        f"{summary.n} closes, pnl ${overall['net_pnl_usd']:+.2f}, "
        f"WR {overall['win_rate'] * 100:.1f}%, PF {_fmt_pf(overall['profit_factor'])}"
    )
    streaks = summary.trailing_streaks()
    flags = [k for k, v in streaks.items() if v >= 3]
    if flags:
        digest += " · ⚠ streaks: " + ", ".join(f"{s}/{d}={streaks[(s,d)]}" for (s, d) in flags)

    return ReviewResult(
        summary=summary,
        markdown_path=path,
        markdown=markdown,
        llm_response=llm_response,
        telegram_digest=digest,
    )
