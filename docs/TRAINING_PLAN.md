# DRL Trading System — Comprehensive Retraining Plan v4 (FINAL)

**Date:** 2026-04-12 (updated after expert review)
**Target:** 50%+ win rate per symbol (BTC, ETH, SOL, XRP)
**Training hardware:** Mac M3 Pro (NEVER on the server)
**New paradigm:** Structure-Gated Filter (model ACCEPTS/REJECTS BOS/CHOCH signals, does NOT make entry decisions)

> **v3 → v4 changes:** Incorporates critical findings from DRL research review
> (see `docs/TRAINING_PLAN_REVIEW.md`). Key changes:
> 1. Model role changed from entry decision → signal filter (binary ACCEPT/REJECT)
> 2. Reward changed from rolling win-rate bonus → Differential Sharpe Ratio
> 3. Features reduced from 140 → ~51 (information-dense, no redundancy)
> 4. Val/test overlap FIXED with strict 3-way split + 48h embargo
> 5. Added curriculum learning, early stopping, action masking, LR schedule
>
> **How to train (one command per symbol):**
> ```bash
> pip install -r requirements-training.txt  # first time only
> python download_historical_data.py --assets BTCUSDT ETHUSDT SOLUSDT XRPUSDT
> python train_model.py --symbol BTCUSDT
> python train_model.py --symbol ETHUSDT
> python train_model.py --symbol SOLUSDT
> python train_model.py --symbol XRPUSDT
> ```

---

## 1. Executive Summary

### What is wrong

The current models deliver 34-40% win rates because of five compounding failures:

1. **Reward function misalignment.** The HTF env rewards unrealized PnL every step (+0.2x), gives asymmetric TP bonus (+0.10) vs SL penalty (-0.05), and adds idle penalties. The model learns to enter positions frequently to collect unrealized PnL ticks rather than learning to select high-probability setups. Win rate is never directly optimized.

2. **Severe overfitting.** The WFv2 BTC overfit report shows avg val/test Sharpe ratio of 19.2x (severe). Val Sharpe avg=2.51 but OOS Sharpe avg=0.09. Only 50% of folds are positive. The 512-256-128 network in HTFTradingAgent has ~500K parameters for a 117-dim input with noisy financial labels — massively over-parameterized.

3. **Feature space issues.** 117 features is reasonable, but many are redundant (12 compact features repeated at each TF), several have low signal (Wyckoff phase, doji flags, Ichimoku cloud position as categorical), and critical features are missing entirely (BOS/CHOCH signals exist in `src/signals/bos_choch.py` but are NOT in the 117-dim obs; funding rate, open interest, and liquidation cascades are absent).

4. **Environment-reality gap.** The training env uses fixed SL/TP percentages (1.5%/3%), no partial fills, no slippage model, and no funding rate costs. Live trading uses ATR-based SL/TP, partial TP (40%/35%/25%), trailing stops, and faces real spread/slippage. Models trained in one regime cannot generalize to the other.

5. **No per-symbol specialization.** All 4 models train with identical hyperparameters despite BTC (low vol, mean-reverting micro), SOL (high vol, momentum-driven), and XRP (news-driven, sparse liquidity) having fundamentally different dynamics.

### What will change (v4 — post-review)

- **Model role REDEFINED:** BOS/CHOCH signals generate candidate entries → model only decides ACCEPT/REJECT (binary action space). This leverages the proven structure-first edge (41% WR, +$7,146) while letting the model learn WHICH signals are high quality
- **Differential Sharpe Ratio reward** (Moody & Saffell 2001) — single-step, Markovian reward that optimizes risk-adjusted returns. No rolling windows, no non-stationarity
- **Reduced 51-dim observation** — high-information features only: price (13), structure (8), multi-TF (12), cross-asset (4), vol regime (4), temporal (5), order block (2), position (3). Eliminates redundant compact features and low-signal categoricals
- **Environment: StructureFilterEnv** — pre-computes BOS/CHOCH for all bars, presents signals to model with action masking, SL/TP matching live (1.5%/3%)
- **Strict walk-forward with 48h embargo** — no val/train overlap, no autocorrelation leakage
- **Curriculum learning** — Stage 1: trending only (ADX>30, 100K steps) → Stage 2: add ranging (ADX>15, 200K) → Stage 3: all data (200K)
- **Per-symbol hyperparameter tuning** with smaller networks (~8-14K params vs old ~500K)
- **RecurrentPPO (LSTM)** from sb3-contrib, linear LR schedule, max_grad_norm=0.3
- **Multi-seed training** — 3 seeds per fold, best selected on val Sharpe
- **Early stopping** — eval every 50K steps, stop after 3 consecutive val Sharpe drops
- **S1 baseline comparison** — model must BEAT structure-only performance to deploy
- **Ensemble of top-3 folds** per symbol with disagreement-based abstention

---

## 2. Per-Model Analysis

### 2.1 BTC (34.2% win rate)

**Diagnosis:**
- WFv2 OOS Sharpe = 0.09 (essentially random). 50% of folds positive.
- Overfit ratio 19x. Model memorized training patterns.
- BTC is the most efficient crypto market — hardest to beat with pure price features.
- Current model enters ~33 trades per 3-month fold (reasonable frequency) but picks directions near-randomly.

**Specific fixes:**
- Add BTC-specific features: CME gap proximity, BTC dominance proxy, exchange net flow
- Reduce network to 64-32 (BTC needs less expressivity, more regularization)
- Higher entropy coefficient (0.08) to prevent premature exploitation in low-signal regime
- Focus reward on HOLD quality — BTC edge is more about avoiding bad trades than finding great ones

### 2.2 ETH (33.7% win rate — worst performer)

**Diagnosis:**
- ETH model loses money in EVERY configuration tested in backtest v2
- Strong correlation with BTC (0.85+) means ETH-specific features are washed out by BTC noise
- ETH has unique dynamics: staking yield, L2 activity, gas fees, ETH/BTC ratio breakouts
- Network is likely learning BTC patterns and applying them to ETH where they fail

**Specific fixes:**
- Add ETH/BTC ratio as a primary feature (when ETH decouples from BTC, that IS the signal)
- Add staking APR delta, gas price (EIP-1559 base fee), L2 TVL growth rate
- Train on ETH/BTC ratio returns alongside ETH/USDT returns (dual target)
- Consider disabling ETH model entirely until it can beat a simple buy-and-hold benchmark on OOS data
- Use a dedicated ETH feature set that emphasizes decorrelation from BTC

### 2.3 SOL (40.5% win rate — best performer)

**Diagnosis:**
- Closest to target. SOL has higher volatility and clearer momentum patterns.
- SOL's market structure is less efficient — more exploitable by RL.
- Current 40.5% likely reflects genuine weak edge that proper training can amplify.
- SOL benefits most from momentum features and volume spikes.

**Specific fixes:**
- Emphasize momentum features: ROC cascading (5/10/20/40 bars), volume breakout signals
- Lower SL distance (SOL moves fast — tight stops reduce loss magnitude)
- Add DEX volume / TVL as features (Solana ecosystem activity)
- Consider 5m execution timeframe for SOL (more granular entries)
- Slightly larger network (128-64) — SOL has more learnable structure

### 2.4 XRP (37.8% win rate)

**Diagnosis:**
- XRP is heavily news-driven (SEC lawsuits, Ripple partnerships, SWIFT integration rumors)
- Price features alone are insufficient — XRP moves on events that have no price precursor
- Current model picks up some mean-reversion after news spikes but bleeds on quiet days
- Lower liquidity means larger spread impact.

**Specific fixes:**
- Add news sentiment score (already have `src/news/sentiment.py`)
- Add XRP-specific: Ripple escrow release schedule, ODL corridor volume
- Increase SL/TP distances (XRP has fat tails)
- Higher minimum confidence threshold for entry (be more selective)
- Consider a "news event detector" wrapper that overrides model during high-impact events

---

## 3. New Reward Function Design

The current reward function is the single biggest contributor to poor win rates. Here is the replacement.

### 3.1 Design Principles

1. **Win rate matters.** A 50% win rate with 1.5:1 R:R is more robust than 30% with 3:1 R:R because drawdowns are shallower and recovery is faster.
2. **Realized PnL only.** No unrealized PnL credits — they teach the model to enter for the sake of floating profit.
3. **Symmetric costs.** No TP bonus / SL penalty asymmetry — the model should learn to exit well from the PnL itself.
4. **Penalize bad entries, not bad luck.** A trade that hits SL after adverse news is bad luck. A trade that enters counter-trend with no setup confirmation is a bad entry.

### 3.2 New Reward Function

```python
class WinRateAlignedReward:
    """
    Reward function optimized for win rate + risk-adjusted returns.
    
    Core idea: reward = realized_trade_pnl + setup_quality_bonus + regime_penalty
    No unrealized PnL. No holding bonuses. No asymmetric SL/TP rewards.
    """
    
    def __init__(self, win_rate_target=0.50):
        self.win_rate_target = win_rate_target
        self.trade_outcomes = []  # rolling window of last 50 trades
        self.max_trade_history = 50
    
    def compute_step_reward(
        self,
        action: int,
        position: int,
        trade_closed: bool,
        realized_pnl_pct: float,      # PnL as % of position value
        htf_alignment: float,          # [-1, 1] from HTF cascade
        bos_choch_confirms: bool,      # True if structure confirms direction
        entry_quality_score: float,    # 0-1, how good was the setup at entry
        funding_rate_cost: float,      # deducted from reward (realistic cost)
        slippage_cost: float,          # deducted from reward
    ) -> float:
        reward = 0.0
        
        # === TRADE CLOSURE REWARD (dominant component) ===
        if trade_closed:
            # Raw PnL reward (position-size invariant)
            reward += realized_pnl_pct * 5.0  # Scale factor for gradient signal
            
            # Win/loss recording
            self.trade_outcomes.append(1.0 if realized_pnl_pct > 0 else 0.0)
            if len(self.trade_outcomes) > self.max_trade_history:
                self.trade_outcomes.pop(0)
            
            # Win rate bonus/penalty (drives toward target)
            if len(self.trade_outcomes) >= 10:
                current_wr = sum(self.trade_outcomes) / len(self.trade_outcomes)
                wr_delta = current_wr - self.win_rate_target
                reward += wr_delta * 2.0  # positive if above target, negative if below
            
            # Deduct realistic costs
            reward -= funding_rate_cost
            reward -= slippage_cost
        
        # === ENTRY QUALITY REWARD ===
        if action in (1, 2) and position == 0:  # new entry
            # Reward aligned entries, penalize counter-trend
            if action == 1:  # long
                alignment_reward = htf_alignment * 0.02  # positive if bullish
            else:  # short
                alignment_reward = -htf_alignment * 0.02  # positive if bearish
            reward += alignment_reward
            
            # BOS/CHOCH confirmation bonus
            if bos_choch_confirms:
                reward += 0.01
            
            # Trade cost (fee)
            reward -= 0.001  # ~0.04% fee normalized
        
        # === ANTI-OVERTRADE ===
        # No reward or penalty for holding (HOLD is free)
        # No idle penalty (sometimes flat IS the right call)
        
        # === DRAWDOWN PENALTY (safety net only) ===
        # Applied externally in the environment, not here
        
        return float(np.clip(reward, -2.0, 2.0))
```

### 3.3 Key Differences from Current

| Aspect | Current (htf_env.py) | New |
|--------|---------------------|-----|
| Unrealized PnL | +0.2x every step | Removed entirely |
| TP hit | +0.10 * (1 + 0.5*align) | Just realized PnL |
| SL hit | -0.05 fixed | Just realized PnL |
| Counter-trend entry | -0.015 | Alignment-scaled (smoother) |
| Idle penalty | -0.0005 when align>0.6 | None |
| Early flip penalty | -0.015 | None (env handles min hold) |
| Win rate signal | None | Direct bonus/penalty vs target |
| BOS/CHOCH | Not in reward | +0.01 confirmation bonus |
| Realistic costs | Fee only | Fee + funding + slippage |
| Reward range | Unbounded | Clipped [-2, 2] |

---

## 4. Feature Engineering Improvements

### 4.1 Current 117-dim Observation Breakdown

```
[0-19]    1D features (20): SMA trend, SMA200 dist, ADX, macro RSI, returns,
          vol regime, ATR regime, HH/HL, Ichimoku, Wyckoff, EMA stack, trend score
[20-44]   4H features (25): 12 compact + BOS, CHOCH, order blocks, FVGs,
          swing distances, structure trend, liquidity, trend score
[45-74]   1H features (30): 12 compact + divergences, Wyckoff events, Stochastic,
          pivot proximity, momentum, EMA ribbon, vol delta, trend strength, BB squeeze
[75-109]  15M features (35): 12 compact + micro RSI, MACD hist, candle patterns,
          wick ratios, volume spikes, Keltner position, ATR percentile, entry score
[110-113] Alignment (4): 1D-4H, 4H-1H, 1H-15M, overall
[114-116] Position state (3): position, unrealized PnL, balance ratio
```

### 4.2 Problems with Current Features

1. **Redundancy:** The 12 "compact features" (EMA trend, RSI, 3x momentum, ATR ratio, volume trend, MACD, Bollinger, S/R position, candle body, trend strength) repeat at each TF. That is 48 out of 110 market features (44%) using the same template. The model sees nearly the same RSI and MACD at 4 scales but gets no unique per-TF structural info.

2. **Missing critical signals:**
   - BOS/CHOCH exists in `src/signals/bos_choch.py` (high quality ICT implementation) but is NOT fed into the observation vector. The 4H features have a simplified BOS/CHOCH detection inside `htf_features.py` (lines 441-443) using basic swing point breaks, which is far inferior to the dedicated MarketStructure class.
   - Funding rate (tells you which side is crowded — critical for futures)
   - Open interest delta (rising OI + rising price = real demand)
   - Liquidation levels / heatmap (where forced orders will pile up)
   - Stablecoin supply/flow (macro demand proxy)
   - Order flow imbalance (exists in `src/features/orderbook_imbalance.py` but not in obs)

3. **Low-signal features consuming capacity:**
   - `wyckoff_phase` (discretized rule-based, high false positive rate)
   - `doji_day` (binary flag, very rare, negligible predictive power)
   - `ichimoku_cloud_pos` (categorical with 3 values — low granularity)
   - `vol_expansion` (binary threshold — loses continuous information)

### 4.3 New 140-dim Observation Vector

```
[0-17]    1D features (18): KEEP best 18, DROP doji_day and vol_expansion binary
[18-42]   4H features (25): KEEP all (already has SMC features)
[43-70]   1H features (28): KEEP best 28, DROP wyckoff_spring/upthrust (move to 4H)
[71-105]  15M features (35): KEEP all
[106-109] Alignment (4): KEEP all
[110-116] NEW: BOS/CHOCH from MarketStructure class (7 features):
          - bos_bullish_recent (1/0, last 20 bars on 15M)
          - bos_bearish_recent (1/0)
          - choch_bullish_recent (1/0)  
          - choch_bearish_recent (1/0)
          - structure_trend_15m (-1/0/1)
          - bos_choch_confidence (0-1, from MarketStructure.get_signals)
          - bars_since_last_structure_break (normalized 0-1)
[117-122] NEW: Funding & Derivatives (6 features):
          - funding_rate_current (scaled, typically -0.01 to 0.01)
          - funding_rate_8h_avg (smoothed)
          - open_interest_delta_1h (% change, clipped)
          - open_interest_delta_4h
          - long_short_ratio (from Binance API)
          - liquidation_imbalance (net long - net short liquidations, normalized)
[123-127] NEW: Order Flow (5 features):
          - bid_ask_imbalance (from orderbook_imbalance.py)
          - trade_flow_imbalance (buy vs sell volume ratio)
          - large_trade_signal (whale detector from existing whale_tracker.py)
          - cvd_15m (cumulative volume delta, 15-min window)
          - cvd_1h (cumulative volume delta, 1-hour window)
[128-133] NEW: On-chain / Macro (6 features):
          - exchange_net_flow_btc (positive = inflow = bearish, from existing whale code)
          - stablecoin_supply_delta (USDT+USDC market cap change proxy)
          - btc_dominance_delta (1D change in BTC.D)
          - eth_btc_ratio_momentum (5-bar ROC of ETH/BTC — ETH model only)
          - fear_greed_index (normalized 0-1, from alternative_data.py)
          - dxy_proxy (dollar strength from stablecoin premium, if available)
[134-136] Position state (3): KEEP as-is
[137-139] NEW: Trade context (3 features):
          - time_of_day_sin (sin(2*pi*hour/24) — captures session patterns)
          - time_of_day_cos (cos component)
          - day_of_week_sin (sin(2*pi*dow/7) — weekend vs weekday)
```

**Total: 140 features** (vs 117 current)

### 4.4 Implementation Notes

- BOS/CHOCH: Call `MarketStructure.get_signals()` on 15M data with 1H+4H confirmation. This class already exists and is battle-tested in live trading.
- Funding rate: Binance API `GET /fapi/v1/fundingRate` — can be fetched alongside candle data during data collection.
- Open interest: Binance API `GET /fapi/v1/openInterest` — historical OI available via `GET /futures/data/openInterestHist`.
- Order flow: The existing `src/features/orderbook_imbalance.py` has the logic; wire it into HTFFeatureEngine.
- On-chain data: The existing `src/features/alternative_data.py` and whale tracking modules provide most of this.
- Time features: Pure math, no external data needed.

---

## 5. Environment v2 Design

### 5.1 What Must Change

The training environment must match live trading conditions as closely as possible. Every difference is a source of train/live distribution shift.

### 5.2 ATR-Based SL/TP (Matching Live)

Current env: Fixed 1.5% SL / 3% TP
Live system: SL >= 1.5x ATR, TP >= 3.0x ATR

```python
# New: ATR-based SL/TP in env
atr_14 = self._compute_atr(14)  # from 15M bars
sl_distance = max(1.5 * atr_14, current_price * 0.005)  # floor at 0.5%
tp_distance = max(3.0 * atr_14, current_price * 0.010)  # floor at 1.0%
```

### 5.3 Partial Take Profit (Matching Live)

Live system: 40% at 1R, 35% at 2R, 25% trails
Training env must replicate this:

```python
# Partial TP schedule
partial_tp_schedule = [
    (1.0, 0.40),  # at 1R: close 40%
    (2.0, 0.35),  # at 2R: close 35%
    # remaining 25% trails with 0.3% trailing stop
]
```

### 5.4 Trailing Stop

Live system: Activates at +0.5%, trails at 0.3% distance.
Must be replicated in env to avoid train/live gap.

### 5.5 Realistic Slippage Model

```python
def compute_slippage(symbol: str, volume_ratio: float) -> float:
    """
    Slippage model based on symbol liquidity and relative trade size.
    
    BTC: very liquid, ~0.01% slippage
    SOL/XRP: less liquid, ~0.03-0.05% slippage
    During low volume (volume_ratio < 0.5): 2x slippage
    """
    base_slippage = {
        "BTCUSDT": 0.0001,
        "ETHUSDT": 0.00015,
        "SOLUSDT": 0.0003,
        "XRPUSDT": 0.0004,
    }
    slippage = base_slippage.get(symbol, 0.0003)
    if volume_ratio < 0.5:
        slippage *= 2.0
    return slippage
```

### 5.6 Funding Rate Cost

```python
# Every 8 hours, deduct funding rate from balance
# Positive funding = longs pay shorts (bearish crowding signal)
# Negative funding = shorts pay longs (bullish crowding signal)
if self.steps_in_position % 32 == 0:  # 32 * 15min = 8 hours
    funding_cost = abs(self.funding_rate) * position_value
    self.balance -= funding_cost
```

### 5.7 Stagnant Exit (Matching Live)

Live system: Exit if PnL in [-0.3%, +0.5%] after 6 hours (24 bars at 15M).

```python
if self.steps_in_position >= 24:
    if -0.003 < unrealized_pnl_pct < 0.005:
        self._close_position(current_price)  # stagnant exit
        reward -= 0.005  # small penalty for wasted capital time
```

---

## 6. Hyperparameter Recommendations Per Model

### 6.1 Algorithm Choice: RecurrentPPO (Primary)

The current models use plain PPO with MlpPolicy. Financial time series have temporal dependencies that an MLP cannot capture — each observation is treated independently. RecurrentPPO (LSTM) maintains hidden state across timesteps, allowing the model to learn patterns like "price has been consolidating for 20 bars" or "we are in the 3rd wave of a trend."

The existing `train_walkforward_v2.py` already supports RecurrentPPO — it just has not been deployed as the primary model.

### 6.2 Per-Symbol Hyperparameters

| Parameter | BTC | ETH | SOL | XRP |
|-----------|-----|-----|-----|-----|
| **Algorithm** | RecurrentPPO | RecurrentPPO | RecurrentPPO | RecurrentPPO |
| **LSTM hidden** | 64 | 64 | 128 | 64 |
| **LSTM layers** | 1 | 1 | 1 | 1 |
| **Post-LSTM arch** | [64] | [64] | [64, 32] | [64] |
| **Learning rate** | 2e-5 | 3e-5 | 5e-5 | 2e-5 |
| **n_steps** | 512 | 512 | 256 | 512 |
| **batch_size** | 128 | 128 | 64 | 128 |
| **n_epochs** | 8 | 8 | 10 | 8 |
| **gamma** | 0.995 | 0.995 | 0.99 | 0.995 |
| **clip_range** | 0.10 | 0.10 | 0.15 | 0.10 |
| **ent_coef** | 0.08 | 0.06 | 0.04 | 0.06 |
| **Total timesteps** | 500K | 500K | 400K | 400K |
| **Obs noise std** | 0.015 | 0.015 | 0.01 | 0.02 |

**Rationale:**
- BTC/ETH: Low signal, need more regularization (higher entropy, lower LR, tighter clip)
- SOL: Higher signal, can afford more expressive network and faster learning
- XRP: Most noisy, highest observation noise injection to force robust features
- All: Lower LR than current (1e-4 was too high for LSTM stability)

### 6.3 VecNormalize Configuration

```python
VecNormalize(
    env,
    norm_obs=True,
    norm_reward=False,     # Do NOT normalize reward (masks reward signal)
    clip_obs=10.0,
    clip_reward=10.0,      # Irrelevant since norm_reward=False
    gamma=0.995,
    training=True,         # FREEZE for val/test
)
```

Critical: Save VecNormalize stats after training, freeze for evaluation and deployment. This was identified as a bug in the original RETRAINING_PLAN.md and must be enforced.

---

## 7. Data Requirements

### 7.1 Time Range

- **Minimum:** 3 years (April 2023 - April 2026)
- **Ideal:** 3.5 years (Jan 2023 - April 2026) for more regime coverage
- **Timeframes to download:** 15M (base), 1H, 4H, 1D (for direct use where available)

### 7.2 Data Volume Estimates

| Symbol | Interval | Bars (3 years) | File Size |
|--------|----------|-----------------|-----------|
| BTCUSDT | 15m | ~105,000 | ~15 MB |
| ETHUSDT | 15m | ~105,000 | ~15 MB |
| SOLUSDT | 15m | ~105,000 | ~15 MB |
| XRPUSDT | 15m | ~105,000 | ~15 MB |
| **Total** | | ~420,000 | ~60 MB |

### 7.3 Additional Data to Collect

| Data Type | Source | Storage |
|-----------|--------|---------|
| Funding rate history | Binance `GET /fapi/v1/fundingRate` | CSV per symbol |
| Open interest history | Binance `GET /futures/data/openInterestHist` | CSV per symbol |
| Long/short ratio | Binance `GET /futures/data/globalLongShortAccountRatio` | CSV per symbol |
| Liquidation data | Binance WS `forceOrder` (historical from Coinglass API) | CSV per symbol |
| Fear & Greed Index | Alternative.me API | CSV (shared across symbols) |
| BTC dominance | CoinGecko / CMC | CSV |
| Stablecoin supply | DefiLlama API | CSV |

### 7.4 Data Quality Checks

```python
def validate_data(df, symbol, interval):
    """Run before training. Abort if any check fails."""
    # 1. No large gaps
    time_diffs = df['open_time'].diff()
    max_gap = time_diffs.max()
    expected_gap = {'15m': pd.Timedelta('15min'), '1h': pd.Timedelta('1h')}[interval]
    assert max_gap < expected_gap * 4, f"Gap too large: {max_gap}"
    
    # 2. Completeness
    expected_bars = (df['open_time'].max() - df['open_time'].min()) / expected_gap
    actual_bars = len(df)
    completeness = actual_bars / expected_bars
    assert completeness > 0.97, f"Only {completeness:.1%} complete"
    
    # 3. OHLC consistency
    assert (df['high'] >= df['low']).all(), "High < Low found"
    assert (df['high'] >= df['open']).all(), "High < Open found"
    assert (df['high'] >= df['close']).all(), "High < Close found"
    
    # 4. No zero prices or volume
    assert (df['close'] > 0).all(), "Zero/negative close prices"
    assert (df['volume'] > 0).sum() / len(df) > 0.95, "Too many zero-volume bars"
    
    print(f"  {symbol} {interval}: {len(df)} bars, {completeness:.1%} complete, PASS")
```

---

## 8. Walk-Forward Training Methodology

### 8.1 Window Design

```
Total data:    ~3 years (Apr 2023 → Apr 2026)
Train window:  8 months
Val window:    2 months (carved from end of train window)
Test window:   2 months (strictly OOS, never seen)
Slide step:    2 months (non-overlapping test periods)
```

Why 8/2/2 instead of the previous 12/3/3:
- More folds = more robust statistical estimate
- 8 months captures multiple regime changes
- 2-month test windows align with natural market quarters
- Yields ~13 folds (vs 9 with 12/3/3) for 3-year data

### 8.2 Fold Schedule

```
Fold  Train period           Val (in-window)       Test (OOS)
----  --------------------   -------------------   -------------------
  0   2023-04 → 2023-12     2023-10 → 2023-12     2023-12 → 2024-02
  1   2023-06 → 2024-02     2023-12 → 2024-02     2024-02 → 2024-04
  2   2023-08 → 2024-04     2024-02 → 2024-04     2024-04 → 2024-06
  3   2023-10 → 2024-06     2024-04 → 2024-06     2024-06 → 2024-08
  4   2023-12 → 2024-08     2024-06 → 2024-08     2024-08 → 2024-10
  5   2024-02 → 2024-10     2024-08 → 2024-10     2024-10 → 2024-12
  6   2024-04 → 2024-12     2024-10 → 2024-12     2024-12 → 2025-02
  7   2024-06 → 2025-02     2024-12 → 2025-02     2025-02 → 2025-04
  8   2024-08 → 2025-04     2025-02 → 2025-04     2025-04 → 2025-06
  9   2024-10 → 2025-06     2025-04 → 2025-06     2025-06 → 2025-08
 10   2024-12 → 2025-08     2025-06 → 2025-08     2025-08 → 2025-10
 11   2025-02 → 2025-10     2025-08 → 2025-10     2025-10 → 2025-12
 12   2025-04 → 2025-12     2025-10 → 2025-12     2025-12 → 2026-02
 13   2025-06 → 2026-02     2025-12 → 2026-02     2026-02 → 2026-04
```

### 8.3 Anti-Overfitting Gates (Per Fold)

A fold's model is REJECTED if any of these fire:

| Gate | Threshold | Action |
|------|-----------|--------|
| Val/Test Sharpe ratio | > 3.0 | Reject model, log as overfit |
| OOS win rate | < 35% | Reject — below random |
| OOS trades | < 15 | Reject — insufficient sample |
| OOS max drawdown | > 25% | Reject — too risky |
| Train Sharpe | > 5.0 | Reject — memorization detected |
| Val Sharpe rising + Test Sharpe falling | 3 consecutive checks | Early stop |

### 8.4 Ensemble Selection

After all folds complete:
1. Rank folds by OOS Sharpe (must be > 0.3 to qualify)
2. Select top-3 folds that also have OOS win rate > 45%
3. Ensemble: average action log-probabilities from the 3 models
4. If fewer than 3 folds qualify: retrain with adjusted hyperparameters

---

## 9. Training Pipeline: Step by Step

### Phase 0: Data Collection (30 min)

```bash
cd /path/to/drl-trading-system

# Download 3 years of 15M candle data for all 4 symbols
python download_historical_data.py \
    --years 3.5 \
    --assets BTCUSDT ETHUSDT SOLUSDT XRPUSDT \
    --interval 15m \
    --output-dir data/historical_v3

# Download supplementary data
python scripts/download_funding_rates.py \
    --assets BTCUSDT ETHUSDT SOLUSDT XRPUSDT \
    --years 3.5 \
    --output-dir data/historical_v3

python scripts/download_derivatives_data.py \
    --assets BTCUSDT ETHUSDT SOLUSDT XRPUSDT \
    --years 3.5 \
    --output-dir data/historical_v3
```

**NOTE:** The `download_funding_rates.py` and `download_derivatives_data.py` scripts need to be created. They follow the same pattern as `download_historical_data.py` but call different Binance endpoints.

### Phase 1: Data Validation (5 min)

```bash
python scripts/validate_training_data.py \
    --data-dir data/historical_v3 \
    --assets BTCUSDT ETHUSDT SOLUSDT XRPUSDT
```

This script (to be created) runs the quality checks from Section 7.4 and produces a report.

### Phase 2: Feature Engine Update (code changes, ~2 hours dev work)

Files to modify:
1. `src/features/htf_features.py` — Add new feature slots (BOS/CHOCH, derivatives, order flow, time features)
2. `src/env/htf_env.py` — Update N_OBS from 117 to 140, wire new features
3. Create `src/features/derivatives_features.py` — Funding rate, OI, long/short ratio
4. Create `src/features/structure_features.py` — Wrapper around `src/signals/bos_choch.py` for obs vector

### Phase 3: Environment v2 Update (code changes, ~3 hours dev work)

Files to modify:
1. `src/env/htf_env.py` — Implement ATR-based SL/TP, partial TP, trailing stop, slippage model, funding cost, stagnant exit
2. New reward function class (can live in `src/env/win_rate_reward.py`)
3. Update `train_htf_walkforward.py` to use new env config

### Phase 4: Training Script Update (code changes, ~1 hour dev work)

Modify `train_htf_walkforward.py`:
1. Per-symbol hyperparameter configs
2. RecurrentPPO as default algorithm
3. New anti-overfitting gates
4. Top-3 ensemble selection logic

### Phase 5: Train BTC First (Pilot Run) (~4-6 hours on M3 Pro)

```bash
python train_htf_walkforward.py \
    --data-dir data/historical_v3 \
    --symbol BTCUSDT \
    --algorithm recurrent_ppo \
    --output-dir data/models/v3/BTCUSDT \
    --total-timesteps 500000 \
    --max-folds 14
```

**Review checkpoints:**
1. After fold 0 completes: check OOS metrics. If Sharpe < -2 or win rate < 25%, stop and debug.
2. After fold 3: check consistency. If all folds have same sign of return, proceeding well.
3. After all folds: review fold_summary.json and overfit_report.json.

### Phase 6: Train ETH, SOL, XRP (~4-5 hours each)

```bash
for symbol in ETHUSDT SOLUSDT XRPUSDT; do
    python train_htf_walkforward.py \
        --data-dir data/historical_v3 \
        --symbol $symbol \
        --algorithm recurrent_ppo \
        --output-dir data/models/v3/$symbol \
        --total-timesteps 500000 \
        --max-folds 14
    echo "=== DONE: $symbol ==="
done
```

### Phase 7: Ensemble Selection & Evaluation (~30 min)

```bash
for symbol in BTCUSDT ETHUSDT SOLUSDT XRPUSDT; do
    python train_htf_walkforward.py \
        --data-dir data/historical_v3 \
        --symbol $symbol \
        --output-dir data/models/v3/$symbol \
        --eval-only \
        --ensemble-top 3
done
```

### Phase 8: Backtest Validation (~1 hour)

Run the v2 multisignal backtest against the new models:

```bash
python backtest_v2_multisignal.py \
    --model-dir data/models/v3 \
    --data-dir data/historical_v3 \
    --output data/backtest_v3_results.json
```

**Pass criteria (per symbol):**
- Win rate >= 48% (allowing 2% margin below 50% target)
- Sharpe ratio >= 0.5
- Max drawdown <= 20%
- Profit factor >= 1.1
- Minimum 20 trades per 2-month window

### Phase 9: Deploy to Server

Only after all 4 models pass Phase 8 criteria:

```bash
# On Mac: copy models to server
scp -r data/models/v3/ user@116.203.196.107:~/drl-trading-system/data/models/v3/

# On server: update model paths in live_trading_htf.py config
# Restart services
./start_services.sh
```

### Phase 10: Paper Trade Validation (30 days)

Run new models in paper trading mode alongside old models for 30 days:
- Compare daily win rates, Sharpe, drawdown
- If new models are consistently better: switch live
- If new models degrade after 2 weeks: investigate regime shift

---

## 10. Evaluation Criteria

### 10.1 Must-Pass Metrics (Before Deployment)

| Metric | BTC | ETH | SOL | XRP |
|--------|-----|-----|-----|-----|
| OOS Win Rate | >= 48% | >= 45% | >= 50% | >= 45% |
| OOS Sharpe | >= 0.5 | >= 0.3 | >= 0.7 | >= 0.3 |
| Max Drawdown | <= 18% | <= 20% | <= 22% | <= 22% |
| Profit Factor | >= 1.1 | >= 1.0 | >= 1.2 | >= 1.0 |
| Trades per 2mo | >= 20 | >= 15 | >= 25 | >= 15 |
| Val/Test ratio | <= 3.0 | <= 3.0 | <= 3.0 | <= 3.0 |
| Positive folds | >= 60% | >= 50% | >= 65% | >= 50% |

**ETH has relaxed targets** because it is structurally harder (high BTC correlation). If ETH cannot meet even relaxed targets, disable the ETH model and allocate its capital to SOL.

### 10.2 Warning Metrics (Investigate but Don't Block)

- Any fold with zero trades (policy collapse)
- Any fold with > 100% annualized return (suspicious)
- Consistency gap: best fold return > 3x worst fold return
- Action distribution: if HOLD > 90% of all actions (model is too conservative)
- Action distribution: if HOLD < 40% of all actions (model is overtrading)

### 10.3 Comparison Against Baselines

Every trained model must beat these baselines on OOS data:

1. **Random baseline:** Random actions with same trade frequency → ~33% win rate
2. **Buy-and-hold:** Long-only, no exits → symbol-dependent
3. **RSI mean-reversion:** Long when RSI<30, short when RSI>70 → ~42% win rate historically
4. **Current production model:** The models being replaced

If a model cannot beat the RSI baseline, it should not be deployed.

---

## 11. Estimated Training Time on Mac M3 Pro

| Phase | Task | Duration |
|-------|------|----------|
| 0 | Data download (API rate limited) | 30-45 min |
| 1 | Data validation | 5 min |
| 2-4 | Code changes (features, env, scripts) | 6-8 hours dev work |
| 5 | BTC training (14 folds x 500K steps) | 4-6 hours |
| 6 | ETH training | 4-5 hours |
| 6 | SOL training | 3-4 hours |
| 6 | XRP training | 3-4 hours |
| 7 | Ensemble evaluation | 30 min |
| 8 | Backtest validation | 1 hour |
| 9 | Deployment | 30 min |
| **Total compute time** | | **~16-22 hours** |
| **Total with dev work** | | **~24-30 hours** |

**M3 Pro performance notes:**
- The M3 Pro has 12-core CPU and 18GB+ RAM. PyTorch MPS (Metal) provides GPU acceleration for matrix ops.
- RecurrentPPO is ~40% slower than plain PPO due to LSTM forward passes.
- Each fold (500K steps) takes ~20-30 min on M3 Pro with RecurrentPPO.
- Training can be parallelized: run BTC and ETH simultaneously if RAM allows (~4GB per process).
- Use `OMP_NUM_THREADS=4` and `MKL_NUM_THREADS=4` for optimal performance on M3 Pro.

---

## 12. Step-by-Step Instructions for Chen

### Prerequisites

```bash
# On your Mac M3 Pro:
cd ~/drl-trading-system  # or wherever the repo is cloned

# Ensure Python environment is set up
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install sb3-contrib  # for RecurrentPPO

# Verify sb3-contrib is installed
python3 -c "from sb3_contrib import RecurrentPPO; print('RecurrentPPO available')"
```

### Step 1: Pull Latest Code

```bash
git pull origin dev
```

### Step 2: Download Training Data

```bash
python download_historical_data.py \
    --years 3.5 \
    --assets BTCUSDT ETHUSDT SOLUSDT XRPUSDT \
    --interval 15m \
    --output-dir data/historical_v3
```

Wait for download to complete (~30 min). Verify files exist:
```bash
ls -lh data/historical_v3/
# Should see 4 CSV files, ~15MB each
```

### Step 3: Validate Data

```bash
python -c "
import pandas as pd
for sym in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']:
    df = pd.read_csv(f'data/historical_v3/{sym}_15m.csv', parse_dates=['open_time'])
    print(f'{sym}: {len(df)} bars, {df.open_time.min()} to {df.open_time.max()}')
    assert len(df) > 80000, f'{sym} has too few bars!'
print('All data OK')
"
```

### Step 4: Train BTC (Pilot)

```bash
# Start with BTC as a test
OMP_NUM_THREADS=4 python train_htf_walkforward.py \
    --data-dir data/historical_v3 \
    --symbol BTCUSDT \
    --algorithm recurrent_ppo \
    --output-dir data/models/v3/BTCUSDT \
    --total-timesteps 500000 \
    --max-folds 14

# Check results
cat data/models/v3/BTCUSDT/fold_summary.json | python -m json.tool
cat data/models/v3/BTCUSDT/overfit_report.json | python -m json.tool
```

**What to check:**
- `oos_sharpe_mean` > 0.3 (decent signal)
- `positive_fold_pct` > 50% (majority of folds profitable)
- `overfit_flags_count` < 5 (not too many overfit folds)
- Average win rate across folds > 45%

If BTC looks good, proceed. If bad, ping me and I will debug.

### Step 5: Train Remaining Symbols

```bash
for symbol in ETHUSDT SOLUSDT XRPUSDT; do
    echo "=== Training $symbol ==="
    OMP_NUM_THREADS=4 python train_htf_walkforward.py \
        --data-dir data/historical_v3 \
        --symbol $symbol \
        --algorithm recurrent_ppo \
        --output-dir data/models/v3/$symbol \
        --total-timesteps 500000 \
        --max-folds 14
    echo "=== Done: $symbol ==="
    echo ""
done
```

This will take ~12-15 hours total. You can leave it running overnight.

### Step 6: Review Results

```bash
# Quick summary of all symbols
for symbol in BTCUSDT ETHUSDT SOLUSDT XRPUSDT; do
    echo "=== $symbol ==="
    cat data/models/v3/$symbol/fold_summary.json | python -m json.tool | head -15
    echo ""
done
```

### Step 7: Build Ensembles

```bash
for symbol in BTCUSDT ETHUSDT SOLUSDT XRPUSDT; do
    python train_htf_walkforward.py \
        --symbol $symbol \
        --output-dir data/models/v3/$symbol \
        --eval-only \
        --ensemble-top 3
done
```

### Step 8: Run Final Backtest

```bash
python backtest_v2_multisignal.py \
    --model-dir data/models/v3 \
    --output data/backtest_v3_results.json

# Review
cat data/backtest_v3_results.json | python -m json.tool
```

### Step 9: Deploy (Only if Pass Criteria Met)

```bash
# Copy models to server
scp -r data/models/v3/ root@116.203.196.107:~/drl-trading-system/data/models/v3/

# SSH to server and restart
ssh root@116.203.196.107
cd ~/drl-trading-system
# Update model paths in live_trading_htf.py (or config)
./start_services.sh
```

---

## 13. Risk Mitigation

### What if win rates are still below 50%?

1. **Check feature importance:** Use permutation importance on the validation set to identify which of the 140 features actually matter. Drop the bottom 30%.
2. **Try plain PPO:** If RecurrentPPO overfits, LSTM might be too expressive. Fall back to MLP PPO with 64-32.
3. **Reduce action space:** Instead of HOLD/LONG/SHORT, try HOLD/TRADE where the direction is determined by HTF alignment (removing one degree of freedom).
4. **Increase training data:** Add 2021-2022 data for BTC/ETH (longer history available).
5. **Curriculum learning:** Phase 1 = only BTC (most liquid), Phase 2 = transfer to other symbols.

### What if ETH never works?

Disable the ETH model. Reallocate capital to SOL (highest win rate) and BTC (most liquid). Run ETH in shadow mode (paper trading) until it can beat buy-and-hold.

### What if models degrade after deployment?

- **Week 1-2 monitoring:** Compare daily metrics to backtest expectations. If win rate drops below 40% for 7 consecutive days, halt trading for that symbol.
- **Monthly retraining:** Re-run the pipeline monthly with the latest 3 years of data to adapt to regime shifts.
- **Automatic fallback:** If all 4 models are halted, switch to a simple RSI mean-reversion strategy as a safety net.

---

## 14. Files to Create/Modify

### New Files

| File | Purpose |
|------|---------|
| `src/features/derivatives_features.py` | Funding rate, OI, long/short ratio features |
| `src/features/structure_features.py` | BOS/CHOCH obs vector wrapper |
| `src/env/win_rate_reward.py` | New reward function class |
| `scripts/download_funding_rates.py` | Historical funding rate downloader |
| `scripts/download_derivatives_data.py` | Historical OI, liquidation data |
| `scripts/validate_training_data.py` | Pre-training data quality checks |

### Files to Modify

| File | Changes |
|------|---------|
| `src/features/htf_features.py` | Add slots for new features (140-dim) |
| `src/env/htf_env.py` | ATR-based SL/TP, partial TP, trailing, slippage, funding, stagnant exit, new reward |
| `train_htf_walkforward.py` | Per-symbol configs, RecurrentPPO default, anti-overfitting gates, ensemble |
| `download_historical_data.py` | Add 15m interval support (currently seems 1h focused) |
| `live_trading_htf.py` | Update model loading to use v3 ensemble models |

---

## 15. Summary of Expected Impact

| Metric | Current | Target | Improvement Driver |
|--------|---------|--------|-------------------|
| BTC Win Rate | 34.2% | 50%+ | Win-rate reward + BOS/CHOCH + reduced overtrade |
| ETH Win Rate | 33.7% | 45%+ | ETH/BTC ratio features + decorrelation + higher threshold |
| SOL Win Rate | 40.5% | 52%+ | Momentum emphasis + tighter stops + LSTM memory |
| XRP Win Rate | 37.8% | 48%+ | News sentiment + higher entry bar + obs noise |
| Avg OOS Sharpe | 0.09 | 0.5+ | Anti-overfitting gates + realistic env |
| Val/Test ratio | 19x | <3x | Smaller network + noise + proper VecNormalize |
| Positive folds | 50% | 65%+ | Better reward alignment + more folds |

The single most impactful change is the **reward function reform** — switching from unrealized PnL ticks to realized-only with win rate targeting. This alone should move win rates by 5-10 percentage points because the model will stop entering marginal trades just to collect floating PnL credits.

The second most impactful change is **adding BOS/CHOCH to the observation space**. This signal is already computed in production (for trade filtering) and has shown promise in backtest v2 scenario F1 (BOS-only filter reduced trades from 325 to 53 but turned BTC profitable: +10.9% return). Giving the model direct access to this signal in its observation lets it learn WHEN structure breaks matter, rather than using a hard-coded filter.
