# Break of Structure (BOS), Change of Character (CHOCH), Fake Variants & Dynamic SL/TP Management

## Deep Research Document

**Date:** 2026-03-23
**Purpose:** Comprehensive analysis of BOS, CHOCH, Fake BOS & Fake CHOCH signals for dynamic stop-loss and take-profit adjustment on open positions in crypto/futures markets.
**Context:** BTC/ETH trading on Binance Futures (testnet), 15-minute candles with multi-timeframe analysis (15m, 1H, 4H, 1D).

---

## Table of Contents

1. [Theoretical Foundation](#1-theoretical-foundation)
2. [Break of Structure (BOS)](#2-break-of-structure-bos)
3. [Change of Character (CHOCH)](#3-change-of-character-choch)
4. [Fake BOS (False Breakout)](#4-fake-bos-false-breakout)
5. [Fake CHOCH](#5-fake-choch)
6. [Swing Point Detection Algorithms](#6-swing-point-detection-algorithms)
7. [Dynamic SL/TP Adjustment Strategy](#7-dynamic-sltp-adjustment-strategy)
8. [Implementation Pseudocode](#8-implementation-pseudocode)
9. [Decision Trees](#9-decision-trees)
10. [Risk Management Rules](#10-risk-management-rules)
11. [Backtesting Considerations](#11-backtesting-considerations)
12. [References & Sources](#12-references--sources)

---

## 1. Theoretical Foundation

### 1.1 Origins: Dow Theory and Market Structure

The concepts of Break of Structure and Change of Character originate from **Dow Theory** (Charles Dow, 1890s), which defines trends through the sequence of swing highs and swing lows:

- **Uptrend:** Series of Higher Highs (HH) and Higher Lows (HL)
- **Downtrend:** Series of Lower Lows (LL) and Lower Highs (LH)

### 1.2 Smart Money Concepts (SMC) Framework

Modern BOS/CHOCH concepts were popularized post-2017 through **ICT (Inner Circle Trader)** methodology and the broader **Smart Money Concepts (SMC)** framework. Key principles:

- **Markets are driven by institutional order flow**, not retail sentiment
- **Liquidity pools** form at swing highs and lows (where stop-losses cluster)
- **Order blocks** are zones where institutions placed large orders
- **Fair Value Gaps (FVG)** represent institutional imbalance
- BOS and CHOCH are the structural signals that reveal institutional intent

### 1.3 Market Structure Hierarchy

```
MACRO STRUCTURE (1D/4H)
│
├── External BOS — Major swing high/low break on HTF
│   (Confirms the dominant trend direction)
│
├── Internal Structure (1H/15m)
│   ├── Internal BOS — Minor continuation within macro trend
│   └── CHOCH — First break against the prevailing internal trend
│
└── Micro Structure (5m/1m)
    └── Used for precision entries (not for SL/TP management)
```

---

## 2. Break of Structure (BOS)

### 2.1 Definition

**Break of Structure (BOS)** occurs when price breaks beyond a previous swing high or swing low **in the direction of the existing trend**, confirming trend continuation.

- **Bullish BOS:** Price closes above the most recent swing high in an uptrend
- **Bearish BOS:** Price closes below the most recent swing low in a downtrend

BOS signals that the "smart money" (institutional traders) is continuing to drive price in the current direction.

### 2.2 Visual Pattern — Bullish BOS

```
                         HH ← New BOS
                        /
                   HH  /
                  / \ /
                 /   HL
           HH  /
          / \ /
         /   HL
    HH  /
   / \ /
  /   HL
 /
HL (starting point)

Each time price makes a new HH → that's a Bullish BOS
The HL after each BOS is the new "swing low to protect"
```

### 2.3 Visual Pattern — Bearish BOS

```
LH (starting point)
 \
  \   LH
   \ / \
    LL   \   LH
          \ / \
           LL   \   LH
                 \ / \
                  LL   \
                        LL ← New BOS

Each time price makes a new LL → that's a Bearish BOS
The LH after each BOS is the new "swing high to protect"
```

### 2.4 BOS Confirmation Criteria

Based on ICT/SMC methodology and the `smartmoneyconcepts` Python library (by joshyattridge), a valid BOS requires:

| Criterion | Description | Weight |
|-----------|------------|--------|
| **Candle Close** | Candle body must close beyond the swing level (not just wick) | Critical |
| **Momentum** | The breaking candle should be impulsive (large body, small wicks) | High |
| **Sustained Hold** | Price should not immediately reverse back inside the structure | High |
| **Volume Confirmation** | Higher volume on the break validates institutional participation | Medium |
| **Higher-TF Alignment** | BOS on LTF aligned with HTF trend has higher reliability | High |

**CRITICAL: A wick break does NOT count as BOS.** As noted in research from strike.money: "A valid BOS happens when a strong and meaningful candle closes beyond the key structure level."

### 2.5 Types of BOS

#### 2.5.1 External BOS
- Break of a **major** swing high/low visible on higher timeframes (4H, 1D)
- Confirms strong trend continuation
- Higher reliability, suitable for swing trading positions
- Used for our **primary SL/TP adjustment signals**

#### 2.5.2 Internal BOS
- Break of a **minor** swing high/low within a larger structure
- Occurs on lower timeframes (15m, 1H)
- Represents smaller momentum shifts within the broader trend
- Used for **fine-tuning SL trailing** within an existing position

#### 2.5.3 Liquidity BOS
- Price briefly breaks a high/low to grab liquidity, then moves in the intended direction
- Traps breakout traders and sweeps stop-losses before the real move
- Essentially a **fake BOS that resolves** in the original trend direction
- This is the most deceptive variant — see Section 4

### 2.6 Multi-Timeframe BOS

| Timeframe | Swing Lookback | BOS Significance | Use in Our System |
|-----------|---------------|-------------------|-------------------|
| **1D** | 10-20 candles | Major trend direction, rarely changes | Overall bias, TP extension targets |
| **4H** | 10-15 candles | Medium-term trend, key SL levels | Primary SL/TP adjustment trigger |
| **1H** | 8-12 candles | Short-term structure, transition signals | Secondary confirmation |
| **15m** | 5-10 candles | Intraday structure, entry-level signals | Trigger for initial position entry, fast SL trailing |

**Key principle:** BOS on a higher timeframe overrides signals from a lower timeframe. A bullish BOS on 4H holds more weight than a bearish BOS on 15m.

### 2.7 Volume Confirmation for BOS

Volume confirms institutional participation:

```
STRONG BOS (High Confidence):
- Volume on breakout candle > 1.5x average volume (20-period)
- Multiple consecutive candles with above-average volume
- Breakout candle has large body (>70% of total range)

WEAK BOS (Low Confidence):
- Volume on breakout candle < average volume
- Breakout candle has long wicks relative to body
- Price barely clears the swing level
```

**Metric:** `volume_ratio = breakout_candle_volume / SMA(volume, 20)`
- `volume_ratio > 1.5` → Strong BOS (high confidence for SL/TP adjustment)
- `volume_ratio 1.0-1.5` → Normal BOS (proceed with standard adjustment)
- `volume_ratio < 1.0` → Weak/suspect BOS (wait for confirmation, do NOT adjust TP)

---

## 3. Change of Character (CHOCH)

### 3.1 Definition

**Change of Character (CHOCH)** is the **first break of structure against the prevailing trend**. It signals a potential trend reversal.

- In an uptrend: CHOCH is the **first break below a swing low** (price makes a Lower Low for the first time)
- In a downtrend: CHOCH is the **first break above a swing high** (price makes a Higher High for the first time)

**Key distinction from BOS:** BOS confirms trend continuation; CHOCH signals potential trend reversal. CHOCH is always the FIRST break against the trend — subsequent breaks in the new direction become BOS.

### 3.2 Visual Pattern — Bearish CHOCH (Reversal from Uptrend)

```
                HH (last high)
               / \
              /   \
         HH /     \
        / \/       \
       /  HL        \
  HH /               \
 / \/                  \
/  HL                   \
                         \
                    HL ---X--- ← CHOCH! First break below HL
                               \
                                \  LH
                                 \/
                                  LL → Now bearish, future breaks are BOS
```

### 3.3 Visual Pattern — Bullish CHOCH (Reversal from Downtrend)

```
LL → Now bullish, future breaks are BOS
 /
/   HH ← CHOCH! First break above LH
   /  \
  /    \
LH ----X
  \   /
   \ /
    LH
     \
      \   LH
       \ /
        LL
         \
          LL (last low)
```

### 3.4 CHOCH vs BOS — The Critical Difference

| Aspect | BOS | CHOCH |
|--------|-----|-------|
| **Direction** | With the trend | Against the trend |
| **Meaning** | Trend continuation | Potential trend reversal |
| **Frequency** | Multiple per trend | Only ONE per trend transition |
| **Reliability** | Higher (follows momentum) | Needs confirmation (may be fake) |
| **SL/TP Action** | Trail SL, extend TP | Tighten SL aggressively |
| **Order Block Association** | Forms after BOS (continuation OB) | Creates reversal order block |
| **Risk** | Risk of trend exhaustion | Risk of false reversal (fake CHOCH) |

### 3.5 CHOCH and Order Blocks

When CHOCH occurs, it creates a **reversal order block** — the last opposing candle before the break:

```
Bearish CHOCH creating a bearish Order Block:

     ┌─────┐ ← Bearish Order Block (last bullish candle before break)
     │ OB  │    This is where institutions placed their sell orders
     └──┬──┘
        │
        ▼  Price breaks below swing low (CHOCH)
     ┌─────┐
     │     │
     │     │ Large bearish candle
     └─────┘
```

This order block becomes a zone where price may return to (pullback) before continuing in the new direction. For dynamic SL/TP, this is critical — if holding a LONG and a bearish CHOCH forms, the order block above becomes resistance and a potential TP target.

### 3.6 Programmatic CHOCH Detection

The algorithm from the `smartmoneyconcepts` library (`smc.bos_choch()`):

```
1. Track the current trend state (bullish or bearish)
2. On each new candle, check swing highs and lows
3. IF in uptrend:
   - Break above swing high → BOS (bullish continuation)
   - Break below swing low → CHOCH (bearish reversal signal)
4. IF in downtrend:
   - Break below swing low → BOS (bearish continuation)
   - Break above swing high → CHOCH (bullish reversal signal)
5. After CHOCH, update trend state to the new direction
6. Subsequent breaks in the new direction become BOS
```

### 3.7 When CHOCH Matters Most

CHOCH is **most significant** when:

1. **It occurs on a higher timeframe** (4H or 1D) — this indicates genuine institutional repositioning
2. **Volume confirms the break** — high volume on the CHOCH candle
3. **It aligns with an order block or demand/supply zone** — confluence increases reliability
4. **Divergence is present** — RSI/MACD divergence before CHOCH adds confirmation
5. **It breaks a strong swing point** — not just a minor internal structure point

CHOCH is **less significant** when:
1. It occurs only on a low timeframe (5m/15m) without HTF confirmation
2. Volume is low on the break
3. Price is in a ranging/consolidating market (many false CHOCHs)
4. It occurs during low-liquidity sessions (Asian session for crypto)

---

## 4. Fake BOS (False Breakout)

### 4.1 Definition

A **Fake BOS** (also called **Liquidity BOS**, **Liquidity Grab**, or **Stop Hunt**) occurs when price breaks beyond a swing high or low but **fails to sustain** the break. Price quickly reverses back inside the previous structure.

### 4.2 Why Fake BOS Happens — Institutional Mechanics

Institutions need **liquidity** to fill large orders. Retail stop-losses cluster at obvious swing points:

```
                    Liquidity Pool (Stop-Losses)
                    ▼▼▼▼▼▼▼
     ─────────── Swing High ───────────
    |                                   |
    |   Retail Shorts: "I'll place     |
    |   my stop above this high"        |
    |                                   |
    └───────────────────────────────────┘

1. Institutions push price above the swing high
2. Retail stop-losses get triggered (buying)
3. Institutions SELL into this buying (they get filled at good prices)
4. Price reverses sharply back down
5. Retail traders who went long on the "breakout" get trapped
```

### 4.3 Anatomy of a Fake BOS (Stop Hunt)

```
  Fake Bullish BOS (Bearish Trap):

  Price action:
       ╭──╮ ← Wick above swing high (liquidity grab)
       │  │
  ─────┼──┼───── Swing High Level
       │  │
       │  ╰──── Body closes BELOW or barely above
       │
       │ ← Sharp reversal follows
       │
       ╰───── Price moves decisively in opposite direction


  KEY IDENTIFIERS:
  - Long wick above level, body closes below/near level
  - Rapid reversal within 1-3 candles
  - Volume spike on the wick (stop hunts create volume)
  - Often occurs at times of institutional activity
```

### 4.4 Detection Methods for Fake BOS

#### 4.4.1 Wick Rejection Analysis

```python
def is_wick_rejection(candle, swing_level, direction):
    """
    Detect wick rejection at swing level indicating fake BOS.
    
    For bullish fake BOS (price spikes above then reverses):
    - High extends above swing_level
    - Close is below swing_level (or barely above)
    - Upper wick is >60% of total candle range
    
    For bearish fake BOS (price spikes below then reverses):
    - Low extends below swing_level  
    - Close is above swing_level (or barely below)
    - Lower wick is >60% of total candle range
    """
    total_range = candle.high - candle.low
    if total_range == 0:
        return False
    
    if direction == 'bullish_fake':  # price went above but rejected
        upper_wick = candle.high - max(candle.open, candle.close)
        wick_ratio = upper_wick / total_range
        return (candle.high > swing_level and 
                candle.close < swing_level and 
                wick_ratio > 0.6)
    
    elif direction == 'bearish_fake':  # price went below but rejected
        lower_wick = min(candle.open, candle.close) - candle.low
        wick_ratio = lower_wick / total_range
        return (candle.low < swing_level and 
                candle.close > swing_level and 
                wick_ratio > 0.6)
```

#### 4.4.2 Volume Divergence

```
REAL BOS:
- High volume ON the breakout candle
- Continued volume in the breakout direction
- Volume sustains above average for 2-3 candles

FAKE BOS (Liquidity Grab):
- Volume spike on the breakout candle (stops being triggered)
- REVERSAL candle has EVEN HIGHER volume (institutional orders filling)
- Volume drops quickly after the grab
- Volume divergence: price makes new high/low but volume pattern reverses
```

**Detection formula:**

```python
def detect_volume_divergence(candles, breakout_idx):
    """
    Check if volume pattern suggests fake BOS.
    
    Returns True if the reversal candle(s) show stronger volume
    than the breakout candle, suggesting institutional counter-move.
    """
    breakout_vol = candles[breakout_idx].volume
    avg_vol = mean([c.volume for c in candles[breakout_idx-20:breakout_idx]])
    
    # Check next 1-3 candles for reversal with higher volume
    for i in range(1, min(4, len(candles) - breakout_idx)):
        reversal_candle = candles[breakout_idx + i]
        if (reversal_candle.volume > breakout_vol * 1.2 and
            is_reversal_candle(candles[breakout_idx], reversal_candle)):
            return True  # Likely fake BOS
    
    return False
```

#### 4.4.3 Time-Based Validation

A real BOS should sustain beyond the level for a meaningful duration:

```
VALIDATION WINDOWS (by timeframe):
- 15m candles: Price must hold beyond level for 2-3 candles (30-45 min)
- 1H candles:  Price must hold beyond level for 2-3 candles (2-3 hours)
- 4H candles:  Price must hold beyond level for 1-2 candles (4-8 hours)
- 1D candles:  Price must hold beyond level for 1 candle (daily close)

If price reverses within the validation window → likely Fake BOS
```

#### 4.4.4 Correlation with Liquidity Zones

```python
def check_liquidity_zone(swing_level, order_book_data):
    """
    If there's a known cluster of stop-losses near the swing level,
    a break that reverses quickly is likely a liquidity grab (fake BOS).
    
    Proxy: If price has tested this level multiple times (creating
    visible equal highs/lows), more stops cluster there.
    """
    touches = count_level_touches(swing_level, tolerance=0.001)
    if touches >= 3:
        return "HIGH_LIQUIDITY"  # High probability of stop hunt
    elif touches >= 2:
        return "MEDIUM_LIQUIDITY"
    return "LOW_LIQUIDITY"
```

### 4.5 Fake BOS Signal Score

Combine all detection methods into a composite score:

```
FAKE_BOS_SCORE = (
    0.30 * wick_rejection_score +      # 0 or 1
    0.25 * volume_divergence_score +    # 0 or 1
    0.20 * time_failure_score +         # 0 or 1 (failed to hold)
    0.15 * liquidity_zone_score +       # 0 to 1 (based on touches)
    0.10 * htf_contradiction_score      # 0 or 1 (HTF trend opposes break)
)

if FAKE_BOS_SCORE > 0.6 → HIGH probability of fake BOS
if FAKE_BOS_SCORE 0.3-0.6 → MEDIUM probability, wait for confirmation
if FAKE_BOS_SCORE < 0.3 → LOW probability, treat as real BOS
```

---

## 5. Fake CHOCH

### 5.1 Definition

A **Fake CHOCH** is a break against the prevailing trend that appears to signal a reversal but is actually noise, a temporary retracement, or a manipulation. The trend resumes in its original direction afterward.

### 5.2 Why Fake CHOCH Occurs

1. **Ranging/Consolidating Markets:** In sideways markets, price oscillates between support and resistance. Each break can look like CHOCH but the market lacks true directional intent.

2. **News Spikes:** Sudden volatility from news/events can temporarily break structure but doesn't represent genuine institutional repositioning.

3. **Low-Timeframe Noise:** On 5m/15m, CHOCH signals are much less reliable due to noise and micro-structure fluctuations.

4. **Liquidity Grabs Disguised as CHOCH:** Institutions may briefly break a swing low in an uptrend to grab liquidity before continuing up.

### 5.3 Distinguishing Real CHOCH from Fake CHOCH

```
                    REAL CHOCH                      FAKE CHOCH
                    ──────────                      ──────────
Volume:             High on break candle            Low/average volume
Follow-through:     2-3 candles sustain             Quick reversal (1-2 candles)
                    in new direction                back to original trend
HTF alignment:      Aligns with HTF structure       Contradicts HTF trend
Order block:        Forms clear reversal OB         No clear OB forms
Body vs wick:       Strong body close beyond        Mostly wick, weak close
Momentum:           RSI/MACD confirm shift          No momentum divergence
Market context:     After extended trend            During consolidation/range
                    (exhaustion point)              
```

### 5.4 Filters to Validate CHOCH

#### 5.4.1 Volume Filter

```python
def validate_choch_volume(candles, choch_idx, lookback=20):
    """
    CHOCH is more likely real if:
    1. Break candle volume > 1.5x average
    2. Next 2 candles maintain elevated volume in new direction
    """
    avg_vol = mean([c.volume for c in candles[choch_idx-lookback:choch_idx]])
    break_vol = candles[choch_idx].volume
    
    if break_vol < avg_vol * 1.3:
        return "WEAK"  # Likely fake CHOCH
    
    # Check follow-through
    follow_vols = [candles[choch_idx+i].volume for i in range(1, 3) 
                   if choch_idx+i < len(candles)]
    if all(v > avg_vol for v in follow_vols):
        return "STRONG"  # Likely real CHOCH
    
    return "MEDIUM"
```

#### 5.4.2 Momentum Filter (RSI Divergence)

```
REAL CHOCH often preceded by divergence:

For Bearish CHOCH (end of uptrend):
  Price: Higher High → Higher High
  RSI:   Higher High → Lower High  ← BEARISH DIVERGENCE
  + CHOCH break = HIGH confidence reversal

For Bullish CHOCH (end of downtrend):
  Price: Lower Low → Lower Low
  RSI:   Lower Low → Higher Low   ← BULLISH DIVERGENCE
  + CHOCH break = HIGH confidence reversal

NO divergence + CHOCH = SUSPECT — might be fake
```

#### 5.4.3 Multi-Timeframe Confirmation

**This is the most important filter for avoiding fake CHOCH:**

```
CHOCH VALIDATION MATRIX:

                    15m CHOCH    1H CHOCH     4H CHOCH    1D CHOCH
1D trend supports:  WEAK         MEDIUM       STRONG      VERY STRONG
4H trend supports:  MEDIUM       STRONG       STRONG      N/A
1H trend supports:  STRONG       STRONG       N/A         N/A
No HTF support:     LIKELY FAKE  SUSPECT      WEAK        POSSIBLE

RULE: Never act on CHOCH that contradicts ALL higher timeframes.
```

### 5.5 Fake CHOCH in Ranging Markets

```
     ┌─────────── Resistance ───────────┐
     │                                   │
     │    HH  ← Looks like bullish CHOCH │
     │   / \     but it's just ranging   │
     │  /   \  /\                        │
     │ /     \/  \                       │
     │/   Noise   \                      │
     │              \                    │
     │               LL ← Looks like     │
     │                    bearish CHOCH  │
     │                    but it's noise │
     └─────────── Support ──────────────┘

DETECTION: Check if price has been ranging:
- ATR is declining
- Bollinger Bands are squeezing
- No clear HH/HL or LH/LL sequence
- Multiple CHOCHs in both directions = RANGING MARKET
  → Don't trust any CHOCH signals until range breaks
```

---

## 6. Swing Point Detection Algorithms

### 6.1 Basic Swing Detection (Lookback Method)

This is the method used by the `smartmoneyconcepts` library:

```python
def detect_swing_highs_lows(ohlc_df, swing_length=10):
    """
    Detect swing highs and lows using a lookback/lookforward window.
    
    A swing high: candle whose HIGH is the highest in 
                  [swing_length] candles before AND after.
    A swing low:  candle whose LOW is the lowest in 
                  [swing_length] candles before AND after.
    
    Parameters:
        ohlc_df: DataFrame with columns ['open', 'high', 'low', 'close']
        swing_length: number of candles to look before and after
    
    Returns:
        DataFrame with columns:
        - HighLow: 1 for swing high, -1 for swing low, 0 otherwise
        - Level: price level of the swing point
    
    NOTE: This method has a lag of swing_length candles (needs future data).
    For real-time: use confirmation-based approach (Section 6.2).
    """
    result = pd.DataFrame(index=ohlc_df.index)
    result['HighLow'] = 0
    result['Level'] = np.nan
    
    highs = ohlc_df['high'].values
    lows = ohlc_df['low'].values
    
    for i in range(swing_length, len(ohlc_df) - swing_length):
        # Check swing high
        window_highs = highs[i - swing_length : i + swing_length + 1]
        if highs[i] == max(window_highs):
            result.iloc[i, result.columns.get_loc('HighLow')] = 1
            result.iloc[i, result.columns.get_loc('Level')] = highs[i]
        
        # Check swing low
        window_lows = lows[i - swing_length : i + swing_length + 1]
        if lows[i] == min(window_lows):
            result.iloc[i, result.columns.get_loc('HighLow')] = -1
            result.iloc[i, result.columns.get_loc('Level')] = lows[i]
    
    return result
```

### 6.2 Real-Time Swing Detection (Confirmation-Based)

For live trading with our WebSocket feed, we can't look forward. Use **confirmation-based** detection:

```python
def detect_swing_realtime(ohlc_df, confirmation_candles=3):
    """
    Real-time swing detection that doesn't require future data.
    
    A swing high is confirmed after [confirmation_candles] lower highs.
    A swing low is confirmed after [confirmation_candles] higher lows.
    
    Parameters:
        ohlc_df: DataFrame with OHLCV data
        confirmation_candles: number of candles needed to confirm
    
    Returns:
        List of (index, type, level) tuples
    """
    swings = []
    n = len(ohlc_df)
    
    for i in range(confirmation_candles, n):
        # Check if candle [i - confirmation_candles] is a swing high
        candidate_high = ohlc_df['high'].iloc[i - confirmation_candles]
        is_swing_high = True
        
        # All subsequent candles must have lower highs
        for j in range(1, confirmation_candles + 1):
            if ohlc_df['high'].iloc[i - confirmation_candles + j] >= candidate_high:
                is_swing_high = False
                break
        
        # Also check candles before the candidate
        for j in range(1, confirmation_candles + 1):
            idx = i - confirmation_candles - j
            if idx < 0:
                break
            if ohlc_df['high'].iloc[idx] >= candidate_high:
                is_swing_high = False
                break
        
        if is_swing_high:
            swings.append((i - confirmation_candles, 'SH', candidate_high))
        
        # Similarly for swing low
        candidate_low = ohlc_df['low'].iloc[i - confirmation_candles]
        is_swing_low = True
        
        for j in range(1, confirmation_candles + 1):
            if ohlc_df['low'].iloc[i - confirmation_candles + j] <= candidate_low:
                is_swing_low = False
                break
        
        for j in range(1, confirmation_candles + 1):
            idx = i - confirmation_candles - j
            if idx < 0:
                break
            if ohlc_df['low'].iloc[idx] <= candidate_low:
                is_swing_low = False
                break
        
        if is_swing_low:
            swings.append((i - confirmation_candles, 'SL', candidate_low))
    
    return swings
```

### 6.3 Recommended Lookback Periods for Our System

| Timeframe | Swing Length (candles) | Confirmation Candles | Effective Window |
|-----------|----------------------|---------------------|------------------|
| **15m** | 5-8 | 3 | 1.25-2 hours lookback |
| **1H** | 8-12 | 3-5 | 8-12 hours lookback |
| **4H** | 10-15 | 3-5 | 1.7-2.5 days lookback |
| **1D** | 10-20 | 3-5 | 10-20 days lookback |

**For our 15m primary timeframe with multi-TF features:**
- Use `swing_length=5` on 15m for responsive swing detection
- Use `swing_length=10` on 1H for medium-term structure
- Use `swing_length=12` on 4H for trend structure
- Use `swing_length=15` on 1D for major structure

### 6.4 BOS/CHOCH Detection Algorithm

```python
def detect_bos_choch(ohlc_df, swing_highs_lows, close_break=True):
    """
    Detect BOS and CHOCH from swing highs/lows.
    
    Based on the smartmoneyconcepts library approach:
    - Track trend state (bullish/bearish)
    - Break in trend direction → BOS
    - Break against trend direction → CHOCH (resets trend state)
    
    Parameters:
        ohlc_df: OHLCV DataFrame
        swing_highs_lows: output from swing detection
        close_break: if True, use close price; else use high/low
    
    Returns:
        DataFrame with BOS, CHOCH, Level, BrokenIndex columns
    """
    result = pd.DataFrame(index=ohlc_df.index)
    result['BOS'] = 0
    result['CHOCH'] = 0
    result['Level'] = np.nan
    result['BrokenIndex'] = np.nan
    
    # Get swing points
    swing_points = swing_highs_lows[swing_highs_lows['HighLow'] != 0].copy()
    
    if len(swing_points) < 2:
        return result
    
    # Determine initial trend
    first_two = swing_points.head(2)
    if first_two['HighLow'].iloc[0] == -1 and first_two['HighLow'].iloc[1] == 1:
        trend = 'bullish'
    else:
        trend = 'bearish'
    
    # Track last significant swing high and low
    last_swing_high = None
    last_swing_low = None
    last_swing_high_idx = None
    last_swing_low_idx = None
    
    for idx, row in swing_points.iterrows():
        if row['HighLow'] == 1:  # Swing high
            last_swing_high = row['Level']
            last_swing_high_idx = idx
        elif row['HighLow'] == -1:  # Swing low
            last_swing_low = row['Level']
            last_swing_low_idx = idx
    
    # Now scan candles for breaks
    for i in range(len(ohlc_df)):
        price = ohlc_df['close'].iloc[i] if close_break else None
        
        if trend == 'bullish' and last_swing_high is not None:
            check_price = ohlc_df['close'].iloc[i] if close_break else ohlc_df['high'].iloc[i]
            if check_price > last_swing_high:
                result.iloc[i, result.columns.get_loc('BOS')] = 1  # Bullish BOS
                result.iloc[i, result.columns.get_loc('Level')] = last_swing_high
        
        if trend == 'bullish' and last_swing_low is not None:
            check_price = ohlc_df['close'].iloc[i] if close_break else ohlc_df['low'].iloc[i]
            if check_price < last_swing_low:
                result.iloc[i, result.columns.get_loc('CHOCH')] = -1  # Bearish CHOCH
                result.iloc[i, result.columns.get_loc('Level')] = last_swing_low
                trend = 'bearish'  # Trend changes
        
        if trend == 'bearish' and last_swing_low is not None:
            check_price = ohlc_df['close'].iloc[i] if close_break else ohlc_df['low'].iloc[i]
            if check_price < last_swing_low:
                result.iloc[i, result.columns.get_loc('BOS')] = -1  # Bearish BOS
                result.iloc[i, result.columns.get_loc('Level')] = last_swing_low
        
        if trend == 'bearish' and last_swing_high is not None:
            check_price = ohlc_df['close'].iloc[i] if close_break else ohlc_df['high'].iloc[i]
            if check_price > last_swing_high:
                result.iloc[i, result.columns.get_loc('CHOCH')] = 1  # Bullish CHOCH
                result.iloc[i, result.columns.get_loc('Level')] = last_swing_high
                trend = 'bullish'  # Trend changes
        
        # Update swing points as they're confirmed
        # (simplified - in practice, update as new swings are detected)
    
    return result
```

---

## 7. Dynamic SL/TP Adjustment Strategy

### 7.1 Core Philosophy

> "The goal is not to predict the future. The goal is to respond to what the market shows us. BOS says 'the trend is alive' — trail your stop and let it ride. CHOCH says 'something changed' — protect your gains."

### 7.2 Pre-Conditions for Dynamic Adjustment

Before ANY SL/TP adjustment, these conditions MUST be met:

```
PRE-CONDITIONS:
1. Position is currently IN PROFIT (unrealized PnL > 0)
2. Unrealized profit exceeds minimum threshold:
   - BTC: > 0.3% of position value
   - ETH: > 0.4% of position value
   (These account for typical spread + fees on Binance Futures)
3. Position has been open for at least 2 candles (30 min on 15m TF)
4. SL can only be moved TOWARD profit (never worse than current)
5. WebSocket price feed is active and healthy (no stale data)
```

### 7.3 Signal-Based Adjustment Rules

#### 7.3.1 LONG Position Adjustments

```
┌─────────────────────────────────────────────────────────────────┐
│                    LONG POSITION SL/TP RULES                     │
├──────────────┬──────────────────────────────────────────────────┤
│ Signal       │ Action                                           │
├──────────────┼──────────────────────────────────────────────────┤
│ Bullish BOS  │ TRAIL SL:                                        │
│ (15m/1H)     │   Move SL to last confirmed swing low            │
│              │   Minimum: SL at breakeven + 0.1%                │
│              │   Buffer: Leave 0.2% below swing low             │
│              │                                                  │
│              │ EXTEND TP:                                        │
│              │   If volume confirms BOS (>1.5x avg):            │
│              │   → Move TP to next resistance / order block      │
│              │   → Or set TP at 1.5x the new SL distance        │
│              │   If weak volume: keep current TP                 │
├──────────────┼──────────────────────────────────────────────────┤
│ Bullish BOS  │ STRONG TRAIL SL:                                  │
│ (4H/1D)      │   Move SL to 4H swing low + 0.3% buffer         │
│              │                                                  │
│              │ AGGRESSIVE TP EXTENSION:                           │
│              │   Move TP to next 4H/1D resistance level          │
│              │   or 2x the new SL distance                       │
├──────────────┼──────────────────────────────────────────────────┤
│ Bearish      │ TIGHTEN SL:                                       │
│ CHOCH        │   Move SL to 50% of current profit                │
│ (15m/1H)     │   (e.g., if entry at 60000 and price at 62000,   │
│              │    move SL to 61000)                              │
│              │                                                  │
│              │ TP OPTIONS:                                        │
│              │   Option A: Set TP at current price (take profit) │
│              │   Option B: Set TP slightly above current price   │
│              │   (allow one more push before exit)               │
│              │   Decision: based on CHOCH validation score       │
├──────────────┼──────────────────────────────────────────────────┤
│ Bearish      │ EMERGENCY TIGHTEN:                                │
│ CHOCH (4H)   │   Move SL to 75% of current profit               │
│              │   Set TP at nearest resistance                    │
│              │   Consider closing 50% of position at market     │
├──────────────┼──────────────────────────────────────────────────┤
│ Fake BOS     │ HOLD:                                             │
│ (bearish)    │   If identified as fake → do NOT panic exit       │
│              │   Keep current SL/TP unchanged                    │
│              │   Wait for price to reclaim structure              │
│              │   If price reclaims within 3 candles → confirmed  │
│              │   fake, continue position                         │
├──────────────┼──────────────────────────────────────────────────┤
│ Fake CHOCH   │ HOLD:                                             │
│ (bearish)    │   If identified as fake → don't over-tighten     │
│              │   Keep current SL/TP unchanged                    │
│              │   If initially tightened SL on CHOCH signal       │
│              │   and then determined it's fake:                  │
│              │   → Do NOT loosen SL (never move SL away)         │
│              │   → But don't tighten further on noise            │
└──────────────┴──────────────────────────────────────────────────┘
```

#### 7.3.2 SHORT Position Adjustments

```
┌─────────────────────────────────────────────────────────────────┐
│                   SHORT POSITION SL/TP RULES                     │
├──────────────┬──────────────────────────────────────────────────┤
│ Signal       │ Action                                           │
├──────────────┼──────────────────────────────────────────────────┤
│ Bearish BOS  │ TRAIL SL:                                        │
│ (15m/1H)     │   Move SL to last confirmed swing high           │
│              │   Buffer: Leave 0.2% above swing high            │
│              │                                                  │
│              │ EXTEND TP:                                        │
│              │   If volume confirms: TP to next support/OB      │
│              │   If weak volume: keep current TP                 │
├──────────────┼──────────────────────────────────────────────────┤
│ Bearish BOS  │ STRONG TRAIL SL:                                  │
│ (4H/1D)      │   Move SL to 4H swing high + 0.3% buffer        │
│              │                                                  │
│              │ AGGRESSIVE TP EXTENSION:                           │
│              │   Move TP to next 4H/1D support level             │
├──────────────┼──────────────────────────────────────────────────┤
│ Bullish      │ TIGHTEN SL:                                       │
│ CHOCH        │   Move SL to 50% of current profit                │
│ (15m/1H)     │                                                  │
│              │ TP OPTIONS:                                        │
│              │   Based on CHOCH validation score                 │
├──────────────┼──────────────────────────────────────────────────┤
│ Bullish      │ EMERGENCY TIGHTEN:                                │
│ CHOCH (4H)   │   Move SL to 75% of current profit               │
│              │   Consider closing 50% of position at market     │
├──────────────┼──────────────────────────────────────────────────┤
│ Fake BOS     │ HOLD: Keep current SL/TP                          │
│ (bullish)    │                                                  │
├──────────────┼──────────────────────────────────────────────────┤
│ Fake CHOCH   │ HOLD: Don't over-tighten                          │
│ (bullish)    │                                                  │
└──────────────┴──────────────────────────────────────────────────┘
```

### 7.4 SL Calculation Details

#### 7.4.1 Trailing SL to Swing Low (LONG position, after Bullish BOS)

```python
def calculate_new_sl_long(current_sl, entry_price, current_price, 
                           last_swing_low, buffer_pct=0.002):
    """
    Calculate new SL for a LONG position after Bullish BOS.
    
    Rules:
    1. New SL = last_swing_low - (buffer_pct * current_price)
    2. New SL must be higher than current SL (only trail up)
    3. New SL must be below current price (leave room)
    4. Minimum: SL at breakeven + small buffer
    """
    buffer = buffer_pct * current_price
    proposed_sl = last_swing_low - buffer
    
    # Ensure SL only moves up
    if proposed_sl <= current_sl:
        return current_sl  # No change
    
    # Ensure SL is below current price
    if proposed_sl >= current_price:
        return current_sl  # Would close position immediately
    
    # Ensure minimum breakeven + buffer
    breakeven_sl = entry_price + (entry_price * 0.001)  # 0.1% above entry
    proposed_sl = max(proposed_sl, breakeven_sl)
    
    return proposed_sl
```

#### 7.4.2 Tightening SL (LONG position, after Bearish CHOCH)

```python
def calculate_choch_sl_long(entry_price, current_price, 
                             current_sl, tighten_pct=0.50):
    """
    Calculate tightened SL for LONG position after CHOCH.
    
    Move SL to capture [tighten_pct] of current profit.
    E.g., if 50% and profit is $2000, move SL to lock in $1000.
    """
    current_profit = current_price - entry_price
    
    if current_profit <= 0:
        return current_sl  # Not in profit, don't adjust
    
    # New SL locks in tighten_pct of profit
    proposed_sl = entry_price + (current_profit * tighten_pct)
    
    # Only tighten (move up), never loosen
    return max(proposed_sl, current_sl)
```

### 7.5 TP Calculation Details

#### 7.5.1 TP Extension (after BOS)

```python
def calculate_new_tp_long(current_price, new_sl, last_bos_level,
                           next_resistance=None, rr_multiplier=1.5):
    """
    Calculate new TP for LONG position after BOS.
    
    Strategy:
    1. If next resistance/order block is known → use it
    2. Otherwise → use R:R multiplier from new SL distance
    """
    if next_resistance is not None:
        # Use the identified resistance level
        return next_resistance
    
    # Calculate based on R:R ratio
    sl_distance = current_price - new_sl
    tp_distance = sl_distance * rr_multiplier
    proposed_tp = current_price + tp_distance
    
    return proposed_tp
```

#### 7.5.2 Finding Next Resistance/Support

```python
def find_next_level(current_price, direction, 
                     swing_points, order_blocks, fvgs):
    """
    Find the next significant level for TP extension.
    
    Checks (in priority order):
    1. Order blocks on higher timeframe
    2. Unfilled Fair Value Gaps
    3. Previous swing highs/lows
    4. Psychological levels (round numbers)
    """
    candidates = []
    
    # Order blocks
    for ob in order_blocks:
        if direction == 'long' and ob['top'] > current_price:
            candidates.append(('OB', ob['bottom'], ob['strength']))
        elif direction == 'short' and ob['bottom'] < current_price:
            candidates.append(('OB', ob['top'], ob['strength']))
    
    # Fair Value Gaps
    for fvg in fvgs:
        if direction == 'long' and fvg['bottom'] > current_price:
            candidates.append(('FVG', fvg['bottom'], 0.5))
        elif direction == 'short' and fvg['top'] < current_price:
            candidates.append(('FVG', fvg['top'], 0.5))
    
    # Swing points
    for sp in swing_points:
        if direction == 'long' and sp['level'] > current_price:
            candidates.append(('SWING', sp['level'], 0.3))
        elif direction == 'short' and sp['level'] < current_price:
            candidates.append(('SWING', sp['level'], 0.3))
    
    # Sort by proximity to current price
    if direction == 'long':
        candidates.sort(key=lambda x: x[1])
    else:
        candidates.sort(key=lambda x: -x[1])
    
    return candidates[0] if candidates else None
```

### 7.6 Adjustment Frequency and Throttling

```
THROTTLING RULES:
- Minimum interval between SL adjustments: 15 minutes (1 candle)
- Minimum interval between TP adjustments: 30 minutes (2 candles)
- Maximum adjustments per hour: 4
- After a CHOCH tighten, cooldown: 30 minutes before any TP extension

WHY: Binance Algo Order API has rate limits:
  - 10 orders per second per IP
  - 1200 orders per minute
  - Each SL/TP adjustment = 1 cancel + 1 new order = 2 API calls
  
  Our throttling is well within limits, but prevents whipsawing.
```

---

## 8. Implementation Pseudocode

### 8.1 Main Signal Processing Loop

```python
class DynamicSLTPManager:
    """
    Manages SL/TP adjustments for open positions based on 
    BOS/CHOCH signals across multiple timeframes.
    """
    
    def __init__(self, config):
        self.config = config
        self.last_adjustment_time = None
        self.adjustment_count = 0
        self.trend_state = {}  # per timeframe
        self.swing_points = {}  # per timeframe
        
    def process_new_candle(self, timeframe, candle, position):
        """
        Called when a new candle closes on any timeframe.
        """
        # Step 1: Update swing points
        self.update_swing_points(timeframe, candle)
        
        # Step 2: Check for BOS/CHOCH
        signal = self.detect_signal(timeframe, candle)
        
        if signal is None:
            return None
        
        # Step 3: Validate signal (check for fake)
        validation = self.validate_signal(signal, timeframe, candle)
        
        # Step 4: If position exists and in profit, calculate adjustment
        if position and position.unrealized_pnl > 0:
            adjustment = self.calculate_adjustment(
                signal, validation, timeframe, position
            )
            return adjustment
        
        return None
    
    def detect_signal(self, timeframe, candle):
        """Detect BOS or CHOCH from latest candle."""
        trend = self.trend_state.get(timeframe, 'unknown')
        swings = self.swing_points.get(timeframe, [])
        
        if not swings:
            return None
        
        last_swing_high = self._get_last_swing('high', swings)
        last_swing_low = self._get_last_swing('low', swings)
        
        if trend == 'bullish':
            if candle.close > last_swing_high['level']:
                return Signal('BOS', 'bullish', last_swing_high['level'], 
                            timeframe)
            elif candle.close < last_swing_low['level']:
                self.trend_state[timeframe] = 'bearish'
                return Signal('CHOCH', 'bearish', last_swing_low['level'], 
                            timeframe)
        
        elif trend == 'bearish':
            if candle.close < last_swing_low['level']:
                return Signal('BOS', 'bearish', last_swing_low['level'], 
                            timeframe)
            elif candle.close > last_swing_high['level']:
                self.trend_state[timeframe] = 'bullish'
                return Signal('CHOCH', 'bullish', last_swing_high['level'], 
                            timeframe)
        
        return None
    
    def validate_signal(self, signal, timeframe, candle):
        """
        Validate signal to detect fakes.
        Returns validation score 0-1 (1 = very likely real).
        """
        scores = {}
        
        # 1. Wick rejection check
        scores['wick'] = self._check_wick_rejection(signal, candle)
        
        # 2. Volume confirmation
        scores['volume'] = self._check_volume(signal, candle, timeframe)
        
        # 3. Higher timeframe alignment
        scores['htf'] = self._check_htf_alignment(signal, timeframe)
        
        # 4. Time-based (for BOS: has it held?)
        scores['time'] = 1.0  # Will be updated on subsequent candles
        
        # 5. Momentum (RSI divergence)
        scores['momentum'] = self._check_momentum(signal, timeframe)
        
        # Composite score
        composite = (
            0.25 * scores['wick'] +
            0.25 * scores['volume'] +
            0.25 * scores['htf'] +
            0.15 * scores['time'] +
            0.10 * scores['momentum']
        )
        
        return ValidationResult(
            score=composite,
            is_likely_fake=(composite < 0.4),
            details=scores
        )
    
    def calculate_adjustment(self, signal, validation, timeframe, position):
        """
        Calculate the SL/TP adjustment based on signal and validation.
        """
        # Don't adjust on fake signals
        if validation.is_likely_fake:
            return Adjustment(action='HOLD', reason='Fake signal detected',
                            fake_score=1 - validation.score)
        
        direction = position.side  # 'LONG' or 'SHORT'
        
        # BOS in position direction → TRAIL
        if (signal.type == 'BOS' and 
            self._signal_supports_position(signal, direction)):
            return self._calculate_trail(signal, timeframe, position)
        
        # CHOCH against position → TIGHTEN
        if (signal.type == 'CHOCH' and 
            not self._signal_supports_position(signal, direction)):
            return self._calculate_tighten(signal, timeframe, position)
        
        return None
    
    def _calculate_trail(self, signal, timeframe, position):
        """Trail SL toward profit and optionally extend TP."""
        if position.side == 'LONG':
            last_swing_low = self._get_last_swing('low', 
                                self.swing_points[timeframe])
            buffer = position.current_price * self.config.sl_buffer_pct
            new_sl = last_swing_low['level'] - buffer
            
            # Enforce: only trail up
            new_sl = max(new_sl, position.current_sl)
            
            # Enforce: minimum breakeven
            breakeven = position.entry_price * (1 + self.config.min_profit_pct)
            new_sl = max(new_sl, breakeven)
            
            # Calculate new TP
            new_tp = self._calculate_new_tp(position, new_sl, timeframe)
            
            return Adjustment(
                action='TRAIL',
                new_sl=new_sl,
                new_tp=new_tp,
                reason=f'Bullish BOS on {timeframe}',
                confidence=signal.confidence
            )
        
        else:  # SHORT
            last_swing_high = self._get_last_swing('high',
                                self.swing_points[timeframe])
            buffer = position.current_price * self.config.sl_buffer_pct
            new_sl = last_swing_high['level'] + buffer
            
            # Enforce: only trail down for short
            new_sl = min(new_sl, position.current_sl)
            
            new_tp = self._calculate_new_tp(position, new_sl, timeframe)
            
            return Adjustment(
                action='TRAIL',
                new_sl=new_sl,
                new_tp=new_tp,
                reason=f'Bearish BOS on {timeframe}',
                confidence=signal.confidence
            )
    
    def _calculate_tighten(self, signal, timeframe, position):
        """Tighten SL to protect profit on CHOCH."""
        tighten_pct = self.config.choch_tighten_pct  # 0.50 for 15m/1H
        
        if timeframe in ['4H', '1D']:
            tighten_pct = self.config.choch_emergency_pct  # 0.75 for 4H/1D
        
        current_profit = abs(position.current_price - position.entry_price)
        
        if position.side == 'LONG':
            proposed_sl = position.entry_price + (current_profit * tighten_pct)
            new_sl = max(proposed_sl, position.current_sl)
        else:
            proposed_sl = position.entry_price - (current_profit * tighten_pct)
            new_sl = min(proposed_sl, position.current_sl)
        
        return Adjustment(
            action='TIGHTEN',
            new_sl=new_sl,
            new_tp=position.current_tp,  # Keep TP or tighten slightly
            reason=f'CHOCH on {timeframe} - protecting profit',
            confidence=signal.confidence
        )
```

### 8.2 Binance API Integration

```python
async def update_sl_tp_binance(symbol, position_side, new_sl, new_tp, 
                                 current_sl_order_id, current_tp_order_id):
    """
    Update SL/TP on Binance Futures.
    
    Process:
    1. Cancel existing SL order
    2. Cancel existing TP order
    3. Place new SL order (STOP_MARKET)
    4. Place new TP order (TAKE_PROFIT_MARKET)
    
    API endpoint: https://fapi.binance.com (or demo-fapi.binance.com for testnet)
    """
    base_url = "https://demo-fapi.binance.com"
    
    # Step 1: Cancel existing SL
    if current_sl_order_id:
        await cancel_order(base_url, symbol, current_sl_order_id)
    
    # Step 2: Cancel existing TP
    if current_tp_order_id:
        await cancel_order(base_url, symbol, current_tp_order_id)
    
    # Step 3: Place new SL (STOP_MARKET)
    sl_side = 'SELL' if position_side == 'LONG' else 'BUY'
    sl_params = {
        'symbol': symbol,
        'side': sl_side,
        'type': 'STOP_MARKET',
        'stopPrice': str(round(new_sl, 2)),
        'closePosition': 'true',  # Close entire position
        'workingType': 'MARK_PRICE',
        'timeInForce': 'GTC',
        'timestamp': int(time.time() * 1000),
    }
    sl_order = await place_order(base_url, sl_params)
    
    # Step 4: Place new TP (TAKE_PROFIT_MARKET)
    tp_params = {
        'symbol': symbol,
        'side': sl_side,  # Same side as SL (opposite of position)
        'type': 'TAKE_PROFIT_MARKET',
        'stopPrice': str(round(new_tp, 2)),
        'closePosition': 'true',
        'workingType': 'MARK_PRICE',
        'timeInForce': 'GTC',
        'timestamp': int(time.time() * 1000),
    }
    tp_order = await place_order(base_url, tp_params)
    
    return {
        'sl_order_id': sl_order['orderId'],
        'tp_order_id': tp_order['orderId'],
        'new_sl': new_sl,
        'new_tp': new_tp,
    }


async def cancel_order(base_url, symbol, order_id):
    """
    Cancel an existing order.
    
    DELETE /fapi/v1/order
    Parameters: symbol, orderId, timestamp, signature
    """
    endpoint = f"{base_url}/fapi/v1/order"
    params = {
        'symbol': symbol,
        'orderId': order_id,
        'timestamp': int(time.time() * 1000),
    }
    params['signature'] = generate_signature(params)
    
    async with aiohttp.ClientSession() as session:
        async with session.delete(endpoint, params=params, 
                                    headers={'X-MBX-APIKEY': API_KEY}) as resp:
            return await resp.json()


async def place_order(base_url, params):
    """
    Place a new order.
    
    POST /fapi/v1/order
    """
    endpoint = f"{base_url}/fapi/v1/order"
    params['signature'] = generate_signature(params)
    
    async with aiohttp.ClientSession() as session:
        async with session.post(endpoint, params=params,
                                  headers={'X-MBX-APIKEY': API_KEY}) as resp:
            return await resp.json()
```

---

## 9. Decision Trees

### 9.1 Master Decision Tree — New Signal Received

```
NEW SIGNAL DETECTED
│
├── Is position open?
│   ├── NO → Ignore (SL/TP management only applies to open positions)
│   └── YES → Continue
│       │
│       ├── Is position in profit?
│       │   ├── NO → Do nothing (only adjust when in profit)
│       │   └── YES → Continue
│       │       │
│       │       ├── Minimum profit threshold met?
│       │       │   ├── NO → Do nothing
│       │       │   └── YES → Continue
│       │       │       │
│       │       │       ├── What type of signal?
│       │       │       │
│       │       │       ├── BOS (with trend) ──────────────────┐
│       │       │       │                                       │
│       │       │       │   ├── Validate: is it fake?           │
│       │       │       │   │   ├── Fake score > 0.6            │
│       │       │       │   │   │   └── HOLD (don't adjust)    │
│       │       │       │   │   └── Fake score <= 0.6           │
│       │       │       │   │       │                           │
│       │       │       │   │       ├── Which timeframe?        │
│       │       │       │   │       ├── 15m/1H: TRAIL SL       │
│       │       │       │   │       │   + optional TP extend    │
│       │       │       │   │       └── 4H/1D: STRONG TRAIL    │
│       │       │       │   │           + aggressive TP extend  │
│       │       │       │                                       │
│       │       │       ├── CHOCH (against trend) ─────────────┐
│       │       │       │                                       │
│       │       │       │   ├── Validate: is it fake?           │
│       │       │       │   │   ├── Fake score > 0.6            │
│       │       │       │   │   │   └── HOLD (noise)           │
│       │       │       │   │   └── Fake score <= 0.6           │
│       │       │       │   │       │                           │
│       │       │       │   │       ├── Which timeframe?        │
│       │       │       │   │       ├── 15m/1H: TIGHTEN SL     │
│       │       │       │   │       │   (50% of profit)         │
│       │       │       │   │       └── 4H/1D: EMERGENCY       │
│       │       │       │   │           (75% of profit)         │
│       │       │       │   │           + consider partial close │
│       │       │       │                                       │
│       │       │       └── Conflicting signals? ──────────────┐
│       │       │           (e.g., BOS on 15m but CHOCH on 4H)  │
│       │       │           └── HIGHER TIMEFRAME WINS           │
│       │       │               Use 4H/1D signal over 15m/1H   │
```

### 9.2 Fake Signal Detection Decision Tree

```
SIGNAL DETECTED — IS IT FAKE?
│
├── Check 1: Wick Rejection
│   ├── Break is wick only (body inside) → +0.30 fake score
│   └── Body closes beyond level → +0.00 fake score
│
├── Check 2: Volume
│   ├── Volume < average on break → +0.20 fake score
│   ├── Reversal candle has higher volume → +0.25 fake score
│   └── Volume confirms break direction → +0.00 fake score
│
├── Check 3: HTF Alignment
│   ├── Signal contradicts all HTFs → +0.25 fake score
│   ├── Mixed alignment → +0.10 fake score
│   └── Aligned with HTF → +0.00 fake score
│
├── Check 4: Liquidity Zone
│   ├── Multiple touches at level (3+) → +0.15 fake score
│   ├── 2 touches → +0.08 fake score
│   └── First test of level → +0.00 fake score
│
├── Check 5: Time Validation
│   ├── Reversed within 1 candle → +0.20 fake score
│   ├── Reversed within 2-3 candles → +0.10 fake score
│   └── Held for 3+ candles → +0.00 fake score
│
└── TOTAL FAKE SCORE
    ├── > 0.6: HIGH probability fake → HOLD position
    ├── 0.4-0.6: MEDIUM → wait 1-2 candles for confirmation
    └── < 0.4: LOW → treat as real signal
```

---

## 10. Risk Management Rules

### 10.1 Inviolable Rules

```
RULE 1: NEVER MOVE SL AWAY FROM PROFIT
  - For LONG: SL can only go UP or stay same
  - For SHORT: SL can only go DOWN or stay same
  - This rule has ZERO exceptions

RULE 2: MINIMUM PROFIT BEFORE ACTIVATION
  - Dynamic SL/TP only activates when position is in profit
  - Minimum profit threshold: 0.3% for BTC, 0.4% for ETH
  - Below threshold: original SL/TP from entry signal remains

RULE 3: MAXIMUM DRAWDOWN FROM PEAK
  - Track the highest unrealized profit reached
  - If drawdown from peak > 40% → emergency tighten SL to lock 60% of peak
  - This is independent of BOS/CHOCH signals

RULE 4: POSITION AGE MINIMUM
  - No SL/TP adjustments within first 30 minutes (2 candles on 15m)
  - Allows position to "breathe" initially

RULE 5: CONFLICTING TIMEFRAMES
  - When signals conflict across timeframes, higher TF always wins
  - 1D > 4H > 1H > 15m
  - Exception: if ALL lower TFs show CHOCH and only 1D shows BOS,
    the preponderance of evidence suggests caution → tighten

RULE 6: API FAILURE HANDLING
  - If SL update fails → retry 3 times with exponential backoff
  - If all retries fail → close position at market immediately
  - Never be in a position without a SL

RULE 7: MAXIMUM POSITION DURATION
  - After 24 hours without meaningful BOS → begin tightening SL
  - After 48 hours → aggressive tighten (lock 70% of peak profit)
  - Prevents "stale" positions from giving back all gains
```

### 10.2 Risk Parameters

```python
RISK_CONFIG = {
    # Profit thresholds
    'min_profit_pct_btc': 0.003,     # 0.3% minimum profit before adjusting
    'min_profit_pct_eth': 0.004,     # 0.4% minimum profit before adjusting
    
    # SL buffer (distance below/above swing point)
    'sl_buffer_pct': 0.002,          # 0.2% buffer from swing level
    'sl_buffer_pct_htf': 0.003,      # 0.3% buffer for 4H/1D signals
    
    # CHOCH tightening
    'choch_tighten_pct_ltf': 0.50,   # Lock 50% of profit on 15m/1H CHOCH
    'choch_tighten_pct_htf': 0.75,   # Lock 75% of profit on 4H/1D CHOCH
    
    # TP extension
    'tp_rr_multiplier': 1.5,         # Risk:Reward for TP when no level found
    'tp_rr_multiplier_htf': 2.0,     # R:R for HTF BOS TP extension
    
    # Throttling
    'min_sl_interval_seconds': 900,  # 15 min between SL adjustments
    'min_tp_interval_seconds': 1800, # 30 min between TP adjustments
    'max_adjustments_per_hour': 4,
    
    # Swing detection
    'swing_length_15m': 5,
    'swing_length_1h': 10,
    'swing_length_4h': 12,
    'swing_length_1d': 15,
    
    # Fake detection thresholds
    'fake_score_threshold': 0.6,     # Above this = fake signal
    'fake_score_medium': 0.4,        # Between medium and threshold = wait
    
    # Volume thresholds
    'volume_strong_multiplier': 1.5, # Volume > 1.5x avg = strong
    'volume_weak_threshold': 1.0,    # Volume < avg = weak
    
    # Safety
    'max_drawdown_from_peak_pct': 0.40,  # Emergency tighten at 40% drawdown
    'stale_position_hours': 24,           # Start tightening after 24h
    'critical_position_hours': 48,        # Aggressive tighten after 48h
}
```

---

## 11. Backtesting Considerations

### 11.1 Data Requirements

```
For meaningful backtesting of BOS/CHOCH dynamic SL/TP:

DATA NEEDED:
- 15m OHLCV for BTC and ETH: minimum 6 months (26,000+ candles)
- 1H OHLCV: 6 months
- 4H OHLCV: 1 year
- 1D OHLCV: 2 years
- Volume data MUST be included (critical for fake detection)

DATA SOURCE:
- Binance public kline data via REST API
- Or download from Binance Data Vision (data.binance.vision)

PITFALLS:
- Avoid look-ahead bias: swing points need future confirmation
- Account for slippage on SL execution (use 0.05% slippage assumption)
- Account for funding rates (every 8 hours on Binance)
- Include trading fees (0.02% maker, 0.04% taker for Binance)
```

### 11.2 Backtesting Framework

```python
def backtest_dynamic_sltp(ohlcv_data, entry_signals, config):
    """
    Backtesting framework for BOS/CHOCH dynamic SL/TP.
    
    1. Use entry_signals from your DRL model or other entry system
    2. On each candle, check for BOS/CHOCH signals
    3. Apply SL/TP adjustment rules
    4. Track results
    """
    results = []
    
    for signal in entry_signals:
        position = open_position(signal)
        
        for candle in ohlcv_data[signal.entry_idx:]:
            # Check if SL or TP hit
            if position.side == 'LONG':
                if candle.low <= position.current_sl:
                    close_at = position.current_sl * (1 - SLIPPAGE)
                    results.append(close_position(position, close_at, 'SL'))
                    break
                if candle.high >= position.current_tp:
                    close_at = position.current_tp * (1 - SLIPPAGE)
                    results.append(close_position(position, close_at, 'TP'))
                    break
            
            # Update market structure and check for signals
            manager = DynamicSLTPManager(config)
            adjustment = manager.process_new_candle('15m', candle, position)
            
            if adjustment:
                position.apply_adjustment(adjustment)
    
    return analyze_results(results)
```

### 11.3 Key Metrics to Track

```
BACKTESTING METRICS:
1. Win Rate (with vs without dynamic SL/TP)
2. Average Win Size (expecting larger with TP extension)
3. Average Loss Size (expecting smaller with SL trailing)
4. Profit Factor = Gross Profit / Gross Loss
5. Maximum Drawdown
6. Sharpe Ratio
7. Number of SL/TP adjustments per trade
8. Average adjustment frequency
9. False signal rate (how often fake detection was correct)
10. "Give-back" ratio: how much profit was returned from peak
    before exit (target: <30%)
```

### 11.4 Expected Behavior

```
WITHOUT Dynamic SL/TP:
- Fixed SL/TP from entry
- Some winners hit SL just before continuing in profit direction
- Large wins get TP'd too early; market continues without us
- During reversals, gives back most unrealized profit before SL

WITH Dynamic SL/TP:
- BOS trailing captures more of trending moves
- CHOCH tightening protects profit during reversals
- Fake detection prevents premature exits on stop hunts
- Net effect: larger average win, similar or fewer losses
- Trade-off: some positions closed earlier on valid tighten
  that turns out to be temporary
```

---

## 12. References & Sources

### 12.1 ICT / Smart Money Concepts

- **ICT (Inner Circle Trader)** — Michael J. Huddleston's YouTube mentorship. Core concepts of market structure, BOS, CHOCH, order blocks, and fair value gaps.
  - BOS defined as continuation of HH/HL (bullish) or LH/LL (bearish) sequence
  - CHOCH defined as the first break against the prevailing structure
  - Order blocks as zones of institutional order placement
  - Liquidity pools at swing highs/lows where retail stops cluster

- **Smart Money Concepts (SMC)** — Broader community that has formalized ICT concepts into trading frameworks. Post-2017 popularization of BOS/CHOCH terminology.

### 12.2 Dow Theory (Foundation)

- Charles Dow, Wall Street Journal editorials (1899-1902)
- William Hamilton, "The Stock Market Barometer" (1922)
- Robert Rhea, "Dow Theory" (1932)
- Core principle: Trends defined by sequence of highs and lows. Trend continues until structure breaks.

### 12.3 Technical References Used

- **strike.money/technical-analysis/break-of-structure** — Comprehensive BOS guide covering 5 types of BOS (Bullish, Bearish, Internal, External, Liquidity), confirmation criteria (candle close required, not wick), multi-timeframe usage table, and win rate data. Notes that a valid BOS requires: correct structural level, candle close beyond point, strong momentum, and sustained hold.

- **joshyattridge/smart-money-concepts** (GitHub) — Open-source Python library implementing ICT concepts. Key functions:
  - `smc.swing_highs_lows(ohlc, swing_length=50)` — Swing detection using lookback/lookforward window
  - `smc.bos_choch(ohlc, swing_highs_lows, close_break=True)` — BOS/CHOCH detection with close_break parameter
  - `smc.ob(ohlc, swing_highs_lows)` — Order block detection
  - `smc.liquidity(ohlc, swing_highs_lows)` — Liquidity zone identification
  - `smc.fvg(ohlc)` — Fair value gap detection
  - Library available via `pip install smartmoneyconcepts`

- **Binance Futures API** (developers.binance.com/docs/derivatives):
  - `POST /fapi/v1/order` — New order (STOP_MARKET, TAKE_PROFIT_MARKET)
  - `DELETE /fapi/v1/order` — Cancel order
  - `DELETE /fapi/v1/allOpenOrders` — Cancel all open orders
  - Rate limits: 10 orders/sec, 1200 orders/min per IP

### 12.4 Academic / Quantitative References

- **Market Microstructure Theory** — Observes that large institutional orders create temporary supply/demand imbalances visible as order blocks and FVGs. O'Hara (1995) "Market Microstructure Theory."

- **Stop Hunting / Liquidity Provision** — Academic literature on how market makers and institutional traders actively sweep known stop-loss clusters. Danielsson et al. (2012) "Endogenous and Systemic Risk."

- **Swing Point Detection** — Quantitative approaches include:
  - Fractal-based detection (Bill Williams' fractals): 5-bar patterns
  - Zigzag indicator: percentage-based swing detection
  - Rolling window max/min: the approach used by `smartmoneyconcepts`
  - Adaptive swing detection: using ATR-scaled thresholds

### 12.5 Volume Analysis

- **Volume Spread Analysis (VSA)** — Tom Williams' methodology for reading volume in context of price spread. Key principles applied to BOS/CHOCH validation:
  - High volume + wide spread + close near high = institutional buying
  - High volume + wide spread + close near low = institutional selling
  - High volume + narrow spread = absorption (potential reversal)
  - Low volume breakout = likely false (no institutional participation)

---

## Appendix A: Configuration Quick Reference

```python
# Copy this into your trading bot configuration

BOS_CHOCH_CONFIG = {
    # Swing Detection
    'swing_lengths': {
        '15m': 5,   # ~1.25 hours each direction
        '1H': 10,   # ~10 hours each direction
        '4H': 12,   # ~2 days each direction
        '1D': 15,   # ~15 days each direction
    },
    'confirmation_candles': 3,  # For real-time detection
    
    # Signal Validation
    'close_break': True,  # Require candle close (not just wick)
    'volume_strong_mult': 1.5,
    'fake_threshold': 0.6,
    
    # SL/TP Adjustment
    'min_profit_pct': {'BTCUSDT': 0.003, 'ETHUSDT': 0.004},
    'sl_buffer_pct': {'ltf': 0.002, 'htf': 0.003},
    'choch_tighten': {'ltf': 0.50, 'htf': 0.75},
    'tp_rr_multiplier': {'ltf': 1.5, 'htf': 2.0},
    
    # Throttling
    'min_sl_interval_s': 900,
    'min_tp_interval_s': 1800,
    'max_adjustments_per_hour': 4,
    
    # Safety
    'max_drawdown_from_peak': 0.40,
    'stale_position_hours': 24,
}
```

## Appendix B: Integration Points with Our DRL System

```
CURRENT SYSTEM ARCHITECTURE:
├── src/features/htf_features.py    ← 117-dim feature vector (15m, 1H, 4H, 1D)
├── logs/htf_trading_state.json     ← BTC position state
├── logs/htf_trading_state_ETHUSDT.json ← ETH position state
├── WebSocket price monitor         ← Real-time tick data
└── Binance Algo Order API          ← SL/TP placement

WHERE BOS/CHOCH INTEGRATES:
├── NEW: src/signals/bos_choch_detector.py
│   ├── SwingPointDetector class
│   ├── BOSCHOCHDetector class
│   └── FakeSignalValidator class
│
├── NEW: src/management/dynamic_sltp.py
│   ├── DynamicSLTPManager class
│   ├── SLCalculator class
│   ├── TPCalculator class
│   └── BinanceSLTPUpdater class
│
├── MODIFY: WebSocket handler
│   └── On each 15m candle close → trigger BOS/CHOCH check
│   └── On each 1H/4H/1D close → trigger HTF signal check
│
├── MODIFY: Feature engine (htf_features.py)
│   └── Add BOS/CHOCH signal features to the feature vector:
│       - bos_bullish_15m, bos_bearish_15m
│       - choch_bullish_15m, choch_bearish_15m
│       - bos_count_last_N (trend strength proxy)
│       - choch_count_last_N (reversal frequency proxy)
│       - fake_score (signal reliability)
│
└── MODIFY: State files
    └── Add fields:
        - last_sl_adjustment_time
        - last_tp_adjustment_time
        - peak_unrealized_pnl
        - sl_adjustment_count
        - current_trend_state per TF
        - last_swing_points per TF
```

---

*This research document was compiled on 2026-03-23 for the BTC/ETH Binance Futures trading system. All pseudocode is Python 3.10+ compatible. The strategies described are for educational and testnet purposes. Never risk capital you cannot afford to lose.*
