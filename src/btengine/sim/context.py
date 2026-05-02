"""Ctx — the per-bar context object passed to entry rules and guards.

Exposes everything a strategy needs at a single point in time, without
ever revealing future bars (mitigates lookahead bias):

  ctx.symbol          str
  ctx.now_ms          int   (open_time of the current bar, in ms)
  ctx.primary         pd.DataFrame  (bars up to and including current)
  ctx.htf             dict[str, pd.DataFrame]  (each HTF interval)
  ctx.position_state  str  ('LONG' | 'SHORT' | 'FLAT')
  ctx.position_units  float
  ctx.entry_price     float | 0.0
  ctx.balance         float
  ctx.config          BacktestConfig
  ctx.extras          dict (free-form for signal context, etc.)

Backtest-only fields (never on the live ctx):
  ctx.cursor_index    int   (row index in primary)
  ctx.t_now_iso       str   (UTC ISO8601 — derived, for logging)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class Ctx:
    symbol: str
    now_ms: int
    cursor_index: int
    primary: pd.DataFrame
    htf: Dict[str, pd.DataFrame] = field(default_factory=dict)

    # Position state (pre-fill view: what the broker has after last bar's exits)
    position_state: str = "FLAT"     # 'FLAT' | 'LONG' | 'SHORT'
    position_units: float = 0.0
    entry_price: float = 0.0

    # Portfolio-level
    balance: float = 0.0

    # Free-form scratchpad for signal context (e.g., last MarketStructure signal)
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def t_now_iso(self) -> str:
        return datetime.fromtimestamp(self.now_ms / 1000, tz=timezone.utc).isoformat()

    @property
    def current_bar(self) -> pd.Series:
        return self.primary.iloc[self.cursor_index]

    @property
    def current_close(self) -> float:
        return float(self.current_bar["close"])

    def htf_up_to_now(self, interval: str) -> Optional[pd.DataFrame]:
        """Slice an HTF dataframe to bars whose open_time < now_ms.

        Strict inequality: an HTF bar that *starts* at now_ms is the
        currently-forming bar; we don't peek at it. Returns the full HTF
        history before the cursor.
        """
        df = self.htf.get(interval)
        if df is None or df.empty:
            return None
        return df[df["open_time"] < self.now_ms]
