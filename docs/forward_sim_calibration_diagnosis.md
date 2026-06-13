# Forward-Sim Calibration Diagnosis

**Window:** 2026-05-30 → 2026-06-13 (2w)  
**Live entries analysed:** 139  
**Match window:** ±30 min  

## Why each LIVE entry did / didn't match the sim

| Verdict | Count | % | Meaning |
|---|---:|---:|---|
| match | 76 | 54.7% | sim entered same side within window ✅ |
| sim_in_position | 32 | 23.0% | sim was holding a different trade at that time |
| no_decision_bar | 23 | 16.5% | no sim decision near that time (cadence/data gap) |
| sim_no_signal | 3 | 2.2% | sim saw no tradable structure (trend skip) |
| blocked:rsi | 3 | 2.2% | blocked by the rsi gate |
| sim_dir_opposite | 1 | 0.7% | sim's structure signal pointed the OTHER way |
| blocked:s5_ob | 1 | 0.7% | blocked by the s5_ob gate |

## Per (symbol, side)

- **BTCUSDT LONG** (n=4): sim_in_position=2, match=2
- **BTCUSDT SHORT** (n=46): match=26, no_decision_bar=9, sim_in_position=8, sim_no_signal=1, blocked:rsi=1, sim_dir_opposite=1
- **ETHUSDT LONG** (n=8): no_decision_bar=3, match=3, sim_in_position=2
- **ETHUSDT SHORT** (n=41): match=23, sim_in_position=11, no_decision_bar=6, blocked:s5_ob=1
- **SOLUSDT SHORT** (n=40): match=22, sim_in_position=9, no_decision_bar=5, sim_no_signal=2, blocked:rsi=2

## Sim-only entries (sim traded, live did not)

Total sim-only: 98

- BTCUSDT LONG: 5
- BTCUSDT SHORT: 19
- ETHUSDT LONG: 12
- ETHUSDT SHORT: 19
- SOLUSDT SHORT: 43

_Note: live HOLD/skip reasons are not in trading.db, so sim-only entries can only be counted, not attributed to a specific live guard (orderbook / whale / news / USDT.D) without parsing live bot logs._
