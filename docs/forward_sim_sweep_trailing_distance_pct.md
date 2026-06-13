# Forward-Sim Sweep — `trailing_distance_pct`

**Window:** 2026-05-30 → 2026-06-13 (2w)  
**Baseline config**, varying only `trailing_distance_pct`.  
**Ranking:** net PnL (PROFITABILITY_PLAN.md §7); max DD gate ≤ 8%.  
**Not deployed — demonstration of the sim's ranking capability.**

| Rank | trailing_distance_pct | Net PnL | Trades | Win rate | Max DD | DD gate |
|---:|---:|---:|---:|---:|---:|:--:|
| 1 | 0.008 | $+437.07 | 165 | 66.7% | 4.1% | ✅ |
| 2 | 0.003 | $+413.21 | 174 | 68.4% | 4.9% | ✅ |
| 3 | 0.005 | $+299.94 | 172 | 67.4% | 6.3% | ✅ |

**Winner: `trailing_distance_pct=0.008`** — net $+437.07, max DD 4.1%.

_Sharpe / Sortino not yet computed (need a trade-level return series — future P5). Sim PnL runs optimistic vs live (it can't replay orderbook/whale/news/USDT.D guards), so treat these as relative rankings, not absolute forecasts._
