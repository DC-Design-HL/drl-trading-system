"""BacktestRunner — config in, results out.

Lifecycle:
    1. Validate config + warn on soft errors
    2. Pre-warm kline cache
    3. Build strategy + guard chain + broker
    4. Walk Replay, dispatch to strategy/guards/broker
    5. Write trades.parquet, blocked.parquet, summary.json, equity.csv,
       config.resolved.yaml
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .config import BacktestConfig
from .data import KlineCache
from .guards import build_guard_chain, ReverseCloseLongGuard
from .results import (compute_summary, write_blocked_parquet, write_equity_csv,
                       write_summary_json, write_trades_parquet)
from .sim.broker import Broker
from .sim.context import Ctx
from .sim.replay import Replay
from .strategy.base import Intent, get_strategy
from .strategy.indicators import atr as _atr

logger = logging.getLogger(__name__)


class BacktestRunner:
    """One config in, one results directory out."""

    def __init__(self, config: BacktestConfig,
                 cache: Optional[KlineCache] = None):
        self.config = config
        self.cache = cache or KlineCache()
        self.warnings: List[str] = []
        self.out_dir = config.resolve_output_dir()

    def dry_run(self) -> Dict[str, Any]:
        self.warnings = self.config.validate()
        return {
            "run_id": self.config.run_id,
            "window": [self.config.window.start.isoformat(),
                       self.config.window.end.isoformat()],
            "symbols": self.config.symbols,
            "intervals": {
                "primary": self.config.intervals.primary,
                "htf": list(self.config.intervals.htf),
            },
            "strategy": self.config.strategy,
            "guards_enabled": list(self.config.guards.enabled),
            "sweep_mode": self.config.sweep.mode,
            "sweep_n_axes": len(self.config.sweep.axes),
            "output_dir": str(self.out_dir),
            "warnings": self.warnings,
            "kline_probe": self._probe_klines(),
        }

    def run(self) -> Dict[str, Any]:
        """Full simulation."""
        self.warnings = self.config.validate()
        for w in self.warnings:
            logger.warning("config: %s", w)

        # 1. Pre-load klines
        primary, htf = self._load_klines()

        # 2. Build strategy + guards + broker
        strategy_cls = get_strategy(self.config.strategy)
        strategy = strategy_cls(**self.config.strategy_overrides)
        guard_chain = build_guard_chain(
            self.config.guards.enabled, self.config.guards.params,
        )
        # The reverse-close-long canary is its own check on REVERSE_CLOSE_LONG
        rcl_guard = ReverseCloseLongGuard()

        # TP override config (Phase 1 sweep support)
        atr_floor = self.config.exits.atr_floor or {}
        broker = Broker(
            fees_taker=self.config.fees.taker,
            slippage_pct=self.config.fees.slippage_pct,
            starting_balance=5000.0,
            max_concurrent=self.config.sizing.max_concurrent,
            sl_atr_mult=float(atr_floor.get("sl_mult", 1.5)),
            tp_atr_mult=float(atr_floor.get("tp_mult", 3.0)),
            tp_pct_override=self._extras_get("tp_pct_override"),
            tp_multiplier=float(self._extras_get("tp_multiplier", 1.0)),
            short_only_tp_override=bool(self._extras_get("short_only_tp_override", False)),
            conditional_tp_max_confidence=self._extras_get("conditional_tp_max_confidence"),
        )

        # 3. Replay
        replay = Replay(primary=primary, htf=htf)
        logger.info("Replay schedule: %d bars across %d symbols",
                    len(replay), len(primary))

        trades = []
        blocked = []
        equity_curve = []

        # Pre-build BTC 4h slope helper (cache by ms timestamp into a 1h grid)
        btc_4h_df = htf.get("BTCUSDT", {}).get("4h")
        # Pre-build basket-change cache (USDT.D proxy)
        # For simplicity Phase 1: compute basket_change_pct once per bar from
        # the 4 symbol primary closes' average % change over lookback hours.
        basket_lookback_bars = max(1, int(self.config.exits.stagnant_hours))  # placeholder
        basket_lookback_bars = 8  # 2h on 15m bars

        t0 = time.time()
        for ctx in replay:
            # Compute USDT.D proxy basket_change_pct: avg pct change of
            # the 4-symbol basket over the trailing N bars
            ctx.extras["basket_change_pct"] = self._basket_change_pct(
                ctx.now_ms, primary, lookback_bars=basket_lookback_bars,
            )

            # BTC 4h slope for the canary guard
            ctx.extras["btc_4h_slope_pct"] = self._btc_4h_slope_pct(
                ctx.now_ms, btc_4h_df,
            )

            # 1) Update broker (exits)
            closed = broker.on_bar(ctx)
            trades.extend(closed)

            # 2) Strategy intent
            intent = strategy.on_bar(ctx)

            # 3) REVERSE_CLOSE handling: if a position exists and intent is opposite
            pos = broker.positions.get(ctx.symbol)
            if pos is not None and intent.action in ("OPEN_LONG", "OPEN_SHORT"):
                wanted_side = "LONG" if intent.action == "OPEN_LONG" else "SHORT"
                if pos.side != wanted_side:
                    if pos.side == "LONG":
                        # Apply REVERSE_CLOSE_LONG canary
                        rcl_intent = Intent(action="REVERSE_CLOSE_LONG",
                                            confidence=intent.confidence)
                        rcl_result = rcl_guard(rcl_intent, ctx)
                        if rcl_result.allowed:
                            tr = broker.force_close(ctx, "reverse_close_long")
                            if tr: trades.append(tr)
                            pos = None
                        else:
                            blocked.append({
                                "ts_ms": ctx.now_ms, "symbol": ctx.symbol,
                                "side_intent": "REVERSE_CLOSE_LONG",
                                "blocking_guard": rcl_result.reason.split(":")[0]
                                                     if ":" in rcl_result.reason
                                                     else "reverse_close_long",
                                "reason": rcl_result.reason,
                                "confidence": intent.confidence,
                            })
                    else:  # SHORT → flip to LONG via reverse_close_short
                        tr = broker.force_close(ctx, "reverse_close_short")
                        if tr: trades.append(tr)
                        pos = None

            # 4) Open new position if FLAT and entry intent passes guard chain
            if pos is None and intent.action in ("OPEN_LONG", "OPEN_SHORT"):
                guard_result = guard_chain(intent, ctx)
                if not guard_result.allowed:
                    blocked.append({
                        "ts_ms": ctx.now_ms, "symbol": ctx.symbol,
                        "side_intent": intent.action,
                        "blocking_guard": guard_result.reason.split(":")[0],
                        "reason": guard_result.reason,
                        "confidence": intent.confidence,
                    })
                else:
                    if broker.can_open():
                        broker.open_position(intent, ctx,
                                             sizing_usd=self.config.sizing.usd)

            # Track equity (balance + open-position notional, approximation)
            equity_curve.append({
                "ts_ms": ctx.now_ms,
                "balance": broker.balance,
                "open_positions": len(broker.positions),
            })

        # 5) Force-close any remaining positions at window end
        for sym in list(broker.positions):
            df = primary[sym]
            last_ctx = Ctx(
                symbol=sym, now_ms=int(df["open_time"].iloc[-1]),
                cursor_index=len(df) - 1, primary=df, htf={},
            )
            tr = broker.force_close(last_ctx, "eod")
            if tr: trades.append(tr)

        elapsed = time.time() - t0
        logger.info("Backtest complete: %d trades, %d blocked, %.1fs",
                    len(trades), len(blocked), elapsed)

        # 6) Write outputs
        self.out_dir.mkdir(parents=True, exist_ok=True)
        write_trades_parquet(trades, self.out_dir / "trades.parquet")
        write_blocked_parquet(blocked, self.out_dir / "blocked.parquet")
        write_equity_csv(equity_curve, self.out_dir / "equity.csv")
        summary = compute_summary(
            trades=trades, blocked=blocked, equity_curve=equity_curve,
            starting_balance=broker.starting_balance,
            run_id=self.config.run_id, config_dict=self.config.raw,
        )
        write_summary_json(summary, self.out_dir / "summary.json")
        with open(self.out_dir / "config.resolved.yaml", "w") as f:
            yaml.safe_dump(self.config.raw, f, sort_keys=False)

        return summary

    def _extras_get(self, key, default=None):
        """Read a value from `exits.extras` in the YAML for sweep parameters
        that don't have first-class schema fields yet."""
        extras = self.config.raw.get("exits", {}).get("extras", {}) if self.config.raw else {}
        return extras.get(key, default)

    # ── kline + basket helpers ─────────────────────────────────────
    def _load_klines(self) -> tuple[Dict, Dict]:
        from datetime import datetime as _dt, timezone as _tz, timedelta as _td
        start = _dt.combine(self.config.window.start, _dt.min.time(), tzinfo=_tz.utc)
        end = _dt.combine(self.config.window.end, _dt.min.time(), tzinfo=_tz.utc)

        # Pad start by 30 days for indicator warmup (ATR/RSI/ADX)
        start_padded = start - _td(days=30)

        primary = {}
        htf = {}
        for sym in self.config.symbols:
            primary[sym] = self.cache.get(sym, self.config.intervals.primary,
                                           start_padded, end)
            htf[sym] = {}
            for iv in self.config.intervals.htf:
                htf[sym][iv] = self.cache.get(sym, iv, start_padded, end)
        # Trim primary to actual window (keep HTF padded for HTF indicators)
        start_ms = int(start.timestamp() * 1000)
        for sym in primary:
            primary[sym] = primary[sym][primary[sym]["open_time"] >= start_ms].reset_index(drop=True)
        return primary, htf

    @staticmethod
    def _basket_change_pct(now_ms: int, primary: Dict, lookback_bars: int) -> Optional[float]:
        """Average % change of the 4-symbol basket over the trailing N bars."""
        changes = []
        for sym, df in primary.items():
            mask = df["open_time"] <= now_ms
            sub = df[mask]
            if len(sub) <= lookback_bars: continue
            cur = float(sub["close"].iloc[-1])
            prev = float(sub["close"].iloc[-lookback_bars - 1])
            if prev <= 0: continue
            changes.append((cur - prev) / prev * 100.0)
        if not changes: return None
        return sum(changes) / len(changes)

    @staticmethod
    def _btc_4h_slope_pct(now_ms: int, btc_4h_df) -> Optional[float]:
        """BTC 4h EMA(10) slope over a trailing 20h window, as percent.
        Mirrors the live bot's _compute_btc_4h_ema_slope_pct.
        """
        if btc_4h_df is None or btc_4h_df.empty: return None
        sub = btc_4h_df[btc_4h_df["open_time"] < now_ms]
        if len(sub) < 15: return None
        closes = sub["close"].astype(float).to_numpy()
        # EMA(10) over last 10 vs over 5 bars earlier (20h window)
        def _ema(vals, period):
            k = 2.0 / (period + 1)
            e = float(vals[0])
            for v in vals[1:]:
                e = float(v) * k + e * (1 - k)
            return e
        ema_now = _ema(closes[-10:], 10)
        ema_prev = _ema(closes[-15:-5], 10)
        if ema_prev == 0: return None
        return (ema_now - ema_prev) / ema_prev * 100.0

    # ── connectivity probe ────────────────────────────────────────
    def _probe_klines(self) -> Dict[str, Any]:
        from datetime import timedelta as _td
        end = datetime.now(timezone.utc)
        start = end - _td(hours=2)
        out: Dict[str, Any] = {}
        intervals = [self.config.intervals.primary] + list(self.config.intervals.htf)
        for sym in self.config.symbols:
            sym_out: Dict[str, int] = {}
            for iv in intervals:
                try:
                    df = self.cache.get(sym, iv, start, end)
                    sym_out[iv] = int(len(df))
                except Exception as exc:
                    logger.warning("probe %s %s failed: %s", sym, iv, exc)
                    sym_out[iv] = -1
            out[sym] = sym_out
        return out
