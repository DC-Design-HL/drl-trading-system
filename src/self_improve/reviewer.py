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


_CLOSE_ACTIONS = (
    "CLOSE_LONG", "CLOSE_SHORT", "REVERSE_CLOSE_LONG",
    "REVERSE_CLOSE_SHORT", "SL_HIT", "TP_HIT",
)


def experiment_attribution_section(
    conn: sqlite3.Connection, *, since: datetime, until: datetime,
) -> str:
    """PROFITABILITY_PLAN.md P4 §4 — per active experiment, the stamped-trade
    PnL and suppression counts over the window, plus the latest canary verdict
    (the evidence-based counterfactual decision). Returns '' if the P4 schema
    isn't present yet (older DB) or nothing was attributable."""
    s, u = since.isoformat(), until.isoformat()
    try:
        # Experiments that touched this window: stamped trades, suppressions,
        # or currently in a live-ish stage.
        stamped = {
            r[0]: (r[1], r[2]) for r in conn.execute(
                f"SELECT experiment_id, COUNT(*), COALESCE(SUM(pnl),0) FROM trades "
                f"WHERE experiment_id IS NOT NULL AND is_testnet=1 AND pnl IS NOT NULL "
                f"AND action IN ({','.join('?' * len(_CLOSE_ACTIONS))}) "
                f"AND COALESCE(created_at,timestamp) BETWEEN ? AND ? "
                f"GROUP BY experiment_id",
                (*_CLOSE_ACTIONS, s, u),
            ).fetchall()
        }
        suppressed = {
            r[0]: r[1] for r in conn.execute(
                "SELECT experiment_id, COUNT(*) FROM suppressed_entries "
                "WHERE experiment_id IS NOT NULL AND ts BETWEEN ? AND ? "
                "GROUP BY experiment_id",
                (s, u),
            ).fetchall()
        }
        live_stages = {
            r[0]: r[1] for r in conn.execute(
                "SELECT id, stage FROM experiments "
                "WHERE stage IN ('canary','live','awaiting_canary_approval')",
            ).fetchall()
        }
    except sqlite3.OperationalError:
        return ""  # P4 schema not migrated yet

    exp_ids = set(stamped) | set(suppressed) | set(live_stages)
    exp_ids.discard(None)
    if not exp_ids:
        return ""

    lines = ["## Active-experiment attribution (P4)", ""]
    lines.append("| exp | stage | stamped closes | stamped PnL | suppressed | latest verdict |")
    lines.append("|---|---|---|---|---|---|")
    for eid in sorted(exp_ids):
        stage = live_stages.get(eid)
        if stage is None:
            row = conn.execute(
                "SELECT stage FROM experiments WHERE id=?", (eid,)).fetchone()
            stage = row[0] if row else "?"
        n_cl, pnl = stamped.get(eid, (0, 0.0))
        n_sup = suppressed.get(eid, 0)
        v = conn.execute(
            "SELECT decision_type, outcome, substr(rationale,1,80) FROM decisions "
            "WHERE experiment_id=? AND decision_type IN "
            "('promote','rollback','canary_eval','reject') ORDER BY id DESC LIMIT 1",
            (eid,),
        ).fetchone()
        verdict = f"{v[0]}/{v[1]}: {v[2]}" if v else "—"
        lines.append(
            f"| #{eid} | {stage} | {n_cl} | ${pnl:+.2f} | {n_sup} | {verdict} |"
        )
    lines.append("")
    return "\n".join(lines)


def entry_signal_patterns_section(
    conn: sqlite3.Connection, *, since: datetime, until: datetime,
) -> str:
    """PROFITABILITY_PLAN.md P5 §4 — mine the entry-signal snapshots for
    loser patterns the numeric tables miss: of this window's losing trades,
    how many had model disagreement / whale opposition AT ENTRY. Returns ''
    if the P5 schema is absent or there are too few losers to bother."""
    open_actions = ("OPEN_LONG", "OPEN_SHORT")
    close_actions = ("CLOSE_LONG", "CLOSE_SHORT", "REVERSE_CLOSE_LONG",
                     "REVERSE_CLOSE_SHORT", "SL_HIT", "TP_HIT")
    try:
        # Snapshots keyed by (symbol, open_ts).
        snaps = {}
        for ts, symbol, side, m_action, sigs in conn.execute(
            "SELECT ts, symbol, side, model_action, signals_json FROM entry_signals "
            "WHERE snapshot_type='entry'"
        ).fetchall():
            try:
                signals = json.loads(sigs) if sigs else {}
            except json.JSONDecodeError:
                signals = {}
            snaps[(symbol, ts)] = {"side": side, "model_action": m_action,
                                   "signals": signals}
    except sqlite3.OperationalError:
        return ""  # P5 schema not migrated yet
    if not snaps:
        return ""

    # FIFO-pair OPEN→CLOSE; keep losers whose close is in the window.
    rows = conn.execute(
        "SELECT id, timestamp, symbol, action, pnl FROM trades "
        "WHERE is_testnet=1 ORDER BY id"
    ).fetchall()
    stack: dict[str, list[str]] = {}
    losers: list[dict] = []
    s_iso, u_iso = since.isoformat(), until.isoformat()
    for _id, ts, symbol, action, pnl in rows:
        if action in open_actions:
            stack.setdefault(symbol, []).append(ts)
        elif action in close_actions and pnl is not None:
            st = stack.get(symbol)
            if not st:
                continue
            open_ts = st.pop(0)
            if float(pnl) < 0 and s_iso <= str(ts) <= u_iso:
                snap = snaps.get((symbol, open_ts))
                if snap:
                    losers.append(snap)

    if len(losers) < 3:
        return ""

    n = len(losers)
    disagreed = sum(1 for s in losers if s["model_action"]
                    and str(s["model_action"]).upper() != str(s["side"]).upper())
    whale_opp = 0
    for s in losers:
        whale = s["signals"].get("whale") or {}
        d = whale.get("direction")
        if d is not None and whale.get("intent") != "unavailable":
            if (float(d) > 0.5) != (str(s["side"]).upper() == "LONG"):
                whale_opp += 1

    return "\n".join([
        "## Entry-signal patterns (P5)", "",
        f"Of the **{n}** losing trades this window with an entry snapshot:",
        f"- **{disagreed}** ({disagreed * 100 // n}%) had the PPO model "
        f"DISAGREEING with the side at entry.",
        f"- **{whale_opp}** ({whale_opp * 100 // n}%) had the whale signal "
        f"OPPOSING the side at entry.",
        "",
    ])


def render_markdown(
    summary: WindowSummary, *, llm_pattern_section: str = "",
    experiment_section: str = "", entry_signal_section: str = "",
) -> str:
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

    if experiment_section:
        lines.append(experiment_section.strip())
        lines.append("")

    if entry_signal_section:
        lines.append(entry_signal_section.strip())
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
        experiment_section = experiment_attribution_section(
            conn, since=since, until=now)
        entry_signal_section = entry_signal_patterns_section(
            conn, since=since, until=now)

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

    markdown = render_markdown(
        summary, llm_pattern_section=llm_section,
        experiment_section=experiment_section,
        entry_signal_section=entry_signal_section)

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
