"""Bar-by-bar replay across multi-symbol, multi-interval data.

For each (symbol, primary_bar) the replay loop:
  1. Builds a Ctx with primary bars [:cursor+1] and HTF bars [:cursor_aligned]
  2. Yields the Ctx to the caller (typically Strategy.on_bar)
  3. Caller may set ctx.extras to communicate state across bars

Multi-symbol mode: bars from all symbols are interleaved by open_time so
that order respects wall-clock causality. Two BTC bars with the same
open_time as one ETH bar appear in deterministic alphabetical order
(BTC then ETH then SOL then XRP — matches live's per-symbol thread loop
when iterations are simultaneous).

Lookahead invariant: ctx.htf_up_to_now(iv) returns ONLY HTF bars whose
open_time < ctx.now_ms. The currently-forming HTF bar is invisible.
This is critical for MarketStructure correctness on the boundary case
where a 15m bar's close coincides with the start of a 1h bar.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Iterable, Iterator, List, Tuple

import pandas as pd

from .context import Ctx


class Replay:
    """Iterator over a (symbol → primary_df) collection, producing Ctx
    objects in chronological order.

    HTF dataframes are pre-loaded and sliced lazily per ctx.

    Usage:
        rp = Replay(
            primary={"BTCUSDT": btc_15m, ...},
            htf={"BTCUSDT": {"1h": btc_1h, "4h": btc_4h}, ...},
        )
        for ctx in rp:
            intent = strategy.on_bar(ctx)
    """

    def __init__(self,
                 primary: Dict[str, pd.DataFrame],
                 htf: Dict[str, Dict[str, pd.DataFrame]] | None = None):
        self.primary = {sym: df.reset_index(drop=True) for sym, df in primary.items()}
        self.htf = htf or {}
        # Pre-build a flat schedule of (open_time_ms, symbol, row_idx) tuples
        self._schedule: List[Tuple[int, str, int]] = []
        for sym, df in sorted(self.primary.items()):
            if df.empty:
                continue
            for i, t in enumerate(df["open_time"].astype("int64").to_numpy()):
                self._schedule.append((int(t), sym, i))
        # Stable sort: open_time asc, then symbol asc (alphabetical) for determinism
        self._schedule.sort(key=lambda x: (x[0], x[1]))

    def __len__(self) -> int:
        return len(self._schedule)

    def __iter__(self) -> Iterator[Ctx]:
        for now_ms, sym, idx in self._schedule:
            ctx = Ctx(
                symbol=sym,
                now_ms=now_ms,
                cursor_index=idx,
                primary=self.primary[sym].iloc[: idx + 1],
                htf={iv: df for iv, df in self.htf.get(sym, {}).items()},
            )
            yield ctx
