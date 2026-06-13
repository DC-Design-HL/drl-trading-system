# Forward-Sim Calibration Diagnosis

**Window:** 2026-05-30 → 2026-06-13 (2w)  
**Live entries analysed:** 140  
**Match window:** ±30 min  

## Why each LIVE entry did / didn't match the sim

| Verdict | Count | % | Meaning |
|---|---:|---:|---|
| match | 67 | 47.9% | sim entered same side within window ✅ |
| no_decision_bar | 23 | 16.4% | no sim decision near that time (cadence/data gap) |
| sim_in_position | 20 | 14.3% | sim was holding a different trade at that time |
| blocked:rsi | 17 | 12.1% | blocked by the rsi gate |
| blocked:struct_first_adx | 6 | 4.3% | blocked by the struct_first_adx gate |
| sim_no_signal | 5 | 3.6% | sim saw no tradable structure (trend skip) |
| blocked:cooldown | 1 | 0.7% | blocked by the cooldown gate |
| sim_dir_opposite | 1 | 0.7% | sim's structure signal pointed the OTHER way |

## Per (symbol, side)

- **BTCUSDT LONG** (n=4): match=2, sim_in_position=2
- **BTCUSDT SHORT** (n=46): match=24, no_decision_bar=9, sim_in_position=5, blocked:rsi=3, sim_no_signal=2, blocked:struct_first_adx=2, sim_dir_opposite=1
- **ETHUSDT LONG** (n=8): blocked:rsi=3, no_decision_bar=2, sim_no_signal=1, sim_in_position=1, match=1
- **ETHUSDT SHORT** (n=42): match=21, no_decision_bar=7, sim_in_position=6, blocked:rsi=4, blocked:struct_first_adx=4
- **SOLUSDT SHORT** (n=40): match=19, blocked:rsi=7, sim_in_position=6, no_decision_bar=5, sim_no_signal=2, blocked:cooldown=1

## Sim-only entries (sim traded, live did not)

Total sim-only: 94

- BTCUSDT LONG: 10
- BTCUSDT SHORT: 19
- ETHUSDT LONG: 11
- ETHUSDT SHORT: 14
- SOLUSDT SHORT: 40

_Note: live HOLD/skip reasons are not in trading.db, so sim-only entries can only be counted, not attributed to a specific live guard (orderbook / whale / news / USDT.D) without parsing live bot logs._
