# Risk Management Analysis: Proposed Fixed-Dollar-Risk Approach

**Date**: 2026-03-26  
**Status**: Analysis Complete  
**Author**: Trading Systems Architect  

---

## 0. Summary

The proposed approach replaces the current fixed-percentage position sizing (50% of balance per trade) with a **fixed-dollar-risk model** using isolated margin and dynamic leverage. This analysis concludes that the proposed approach is **fundamentally superior** in risk management principles but needs **significant modifications** before implementation. The recommended path is a hybrid version that preserves the core risk-per-trade concept while capping leverage, simplifying the scaling rule, and accounting for testnet limitations.

---

## 1. Math Breakdown

### 1.1 Core Calculations (Starting Balance: $5,000)

```
Risk Pool       = 10% × $5,000 = $500
Fixed Risk/Trade = $500 ÷ 20   = $25
```

Each trade risks exactly **$25** — if the stop loss is hit, the maximum loss is $25.

### 1.2 Position Sizing with 1.5% Stop Loss

The key formula:

```
Position Size (Notional) = Fixed Risk / SL Percentage
Notional = $25 / 0.015 = $1,666.67
```

The **leverage required** depends on how much margin you allocate:

```
Margin = Notional / Leverage
```

In isolated margin mode, the margin is the collateral for this specific trade. To open a $1,666.67 notional position:

| Leverage | Margin Required | % of Balance |
|----------|----------------|--------------|
| 1x       | $1,666.67      | 33.3%        |
| 2x       | $833.33        | 16.7%        |
| 5x       | $333.33        | 6.7%         |
| 10x      | $166.67        | 3.3%         |
| 20x      | $83.33         | 1.7%         |
| 50x      | $33.33         | 0.7%         |

**The dollar risk is the same ($25) regardless of leverage.** Leverage only determines how much margin is locked up.

### 1.3 Concrete Examples

#### BTC at $70,000 (BTCUSDT)

```
Fixed Risk         = $25
SL %               = 1.5%
SL Distance        = $70,000 × 0.015 = $1,050

Notional           = $25 / 0.015 = $1,666.67
Position in BTC    = $1,666.67 / $70,000 = 0.02381 BTC

At 10x leverage:
  Margin Required  = $1,666.67 / 10 = $166.67
  % of Balance     = $166.67 / $5,000 = 3.3%

Entry Price        = $70,000
SL Price (LONG)    = $70,000 × 0.985 = $68,950
TP Price (LONG)    = $70,000 × 1.030 = $72,100

Loss if SL hit     = 0.02381 × ($70,000 - $68,950) = 0.02381 × $1,050 = $25.00 ✓
Profit if TP hit   = 0.02381 × ($72,100 - $70,000) = 0.02381 × $2,100 = $50.00

Risk:Reward        = 1:2 ($25 risk → $50 reward)
```

**Liquidation price at 10x leverage (LONG, isolated):**

```
Liquidation Price ≈ Entry × (1 - 1/Leverage + maintenance margin rate)
                  ≈ $70,000 × (1 - 1/10 + 0.004)
                  ≈ $70,000 × 0.904
                  ≈ $63,280

SL Price           = $68,950
Liq Price          = $63,280
Buffer (SL - Liq)  = $5,670 (8.1% of entry)    ✅ SAFE
```

#### ETH at $2,000 (ETHUSDT)

```
Fixed Risk         = $25
SL %               = 1.5%
SL Distance        = $2,000 × 0.015 = $30

Notional           = $25 / 0.015 = $1,666.67
Position in ETH    = $1,666.67 / $2,000 = 0.83333 ETH

At 10x leverage:
  Margin Required  = $1,666.67 / 10 = $166.67
  % of Balance     = 3.3%

Entry Price        = $2,000
SL Price (LONG)    = $2,000 × 0.985 = $1,970
TP Price (LONG)    = $2,000 × 1.030 = $2,060

Loss if SL hit     = 0.83333 × ($2,000 - $1,970) = 0.83333 × $30 = $25.00 ✓
Profit if TP hit   = 0.83333 × ($2,060 - $2,000) = 0.83333 × $60 = $50.00

Risk:Reward        = 1:2
```

**Liquidation price at 10x leverage (LONG, isolated):**

```
Liquidation Price ≈ $2,000 × 0.904 ≈ $1,808
SL Price          = $1,970
Buffer            = $162 (8.1%)    ✅ SAFE
```

### 1.4 Comparison with Current System

| Metric | Current System | Proposed System |
|--------|---------------|-----------------|
| **Position Size** | 50% of balance = $2,500 | $1,666.67 (fixed by risk) |
| **Notional (capped)** | min($2,500, $3,000) = $2,500 | $1,666.67 |
| **Max Loss per Trade** | $2,500 × 1.5% = $37.50 | $25.00 |
| **Max Loss % of Balance** | 0.75% | 0.50% |
| **BTC Units** | 0.03571 BTC | 0.02381 BTC |
| **ETH Units** | 1.25 ETH | 0.83333 ETH |
| **Margin Mode** | Cross | Isolated |
| **Leverage** | 1x (implicit) | Dynamic (up to 10-20x) |
| **Consecutive Losses to -10%** | 13 trades | 20 trades |
| **Consecutive Losses to Ruin** | 133 trades (~100%) | 200 trades (100%) |

**Key insight**: The proposed system risks **33% less per trade** ($25 vs $37.50) while keeping the same R:R ratio.

### 1.5 The "Maximize Leverage" Question

The proposal says to "increase leverage as much as possible while staying within the risk limit." Since the dollar risk is fixed at $25 regardless of leverage, the only effect of higher leverage is **less margin locked up**:

| Leverage | Margin Locked | Free Balance | Liq Distance from Entry |
|----------|--------------|-------------|------------------------|
| 5x      | $333.33      | $4,666.67   | ~18% (safe)           |
| 10x     | $166.67      | $4,833.33   | ~9% (safe)            |
| 20x     | $83.33       | $4,916.67   | ~4.5% (tight)         |
| 50x     | $33.33       | $4,966.67   | ~1.8% (**DANGEROUS**)  |
| 100x    | $16.67       | $4,983.33   | ~0.9% (**SL INSIDE LIQ**) |
| 125x    | $13.33       | $4,986.67   | ~0.7% (**SL INSIDE LIQ**) |

⚠️ **At 50x+ leverage, the liquidation price can fall INSIDE the SL zone**, meaning the position gets liquidated before the SL order triggers.

At 100x+ with a 1.5% SL, the liquidation distance (~1%) is LESS than the SL distance (1.5%), so **the position will be liquidated before the SL fires**. This defeats the entire purpose of the system.

---

## 2. Pros vs Current System

### 2.1 Advantages

| Advantage | Explanation |
|-----------|-------------|
| **Defined dollar risk** | Every trade risks exactly $25 — no ambiguity, no surprise losses |
| **Drawdown resilience** | 20 consecutive losses = $500 = 10% of balance (vs ~15% in current system) |
| **Anti-martingale by default** | Position size doesn't grow with losses (unlike fixed % of shrinking balance) |
| **Isolated margin protection** | Losing trade can only lose its allocated margin, not the entire account |
| **Capital efficiency** | Only 3-7% of balance locked per trade vs 50% currently |
| **Multiple positions possible** | Free balance allows opening BTC and ETH simultaneously |
| **Emotional discipline** | "$25 risk" is psychologically clear; "50% position" is not |
| **Compatible with partial TP** | Partial exits still work; the $25 risk is on the initial position |

### 2.2 Drawdown Handling

**Current system (50% position, cross margin, 1.5% SL):**
```
Trade 1 loss: -$37.50  → Balance: $4,962.50
Trade 2 loss: -$37.19  → Balance: $4,925.31  (size shrinks with balance)
Trade 3 loss: -$36.94  → Balance: $4,888.37
...
After 10 losses: ~$4,632  (-7.4%)
After 20 losses: ~$4,290  (-14.2%)
```

**Proposed system ($25 fixed risk per trade):**
```
Trade 1 loss:  -$25.00  → Balance: $4,975.00
Trade 2 loss:  -$25.00  → Balance: $4,950.00
Trade 3 loss:  -$25.00  → Balance: $4,925.00
...
After 10 losses: $4,750  (-5.0%)
After 20 losses: $4,500  (-10.0%)  ← All 20 "parts" exhausted, must recalculate
```

The proposed system loses **less per trade** and the losses are perfectly linear and predictable.

### 2.3 Risk Scaling at Different Balance Levels

| Balance | Baseline | Doublings | Risk/Trade | Risk Pool | 20-Trade Max DD |
|---------|----------|-----------|-----------|-----------|-----------------|
| $5,000  | $5,000   | 0         | $25       | $500      | $500 (10%)      |
| $10,000 | $5,000   | 1         | $50       | $1,000    | $1,000 (10%)    |
| $20,000 | $5,000   | 2         | $100      | $2,000    | $2,000 (10%)    |
| $40,000 | $5,000   | 3         | $200      | $4,000    | $4,000 (10%)    |
| $80,000 | $5,000   | 4         | $400      | $8,000    | $8,000 (10%)    |

At each level, the risk/trade doubles, but it stays proportional to the account size at approximately 0.5% risk per trade.

**Position sizes at each level (BTC @ $70,000):**

| Balance | Risk/Trade | Notional | BTC Qty | Leverage 10x Margin |
|---------|-----------|----------|---------|-------------------|
| $5,000  | $25       | $1,667   | 0.0238  | $167              |
| $10,000 | $50       | $3,333   | 0.0476  | $333              |
| $20,000 | $100      | $6,667   | 0.0952  | $667              |
| $40,000 | $200      | $13,333  | 0.1905  | $1,333            |

---

## 3. Cons and Risks

### 3.1 Dangers of Maximizing Leverage

**This is the most dangerous part of the proposal.**

| Risk | Severity | Explanation |
|------|----------|-------------|
| **Liquidation before SL** | 🔴 CRITICAL | At high leverage (>20x with 1.5% SL), the liquidation price is closer to entry than the SL. The exchange liquidates the position before the SL order can trigger. |
| **Funding rate drain** | 🟡 MEDIUM | High-leverage positions accrue larger funding rate charges (every 8h on Binance). At 50x on a $1,667 notional, 0.01% funding = $0.17/8h = $0.50/day. Small but adds up. |
| **Flash crash slippage** | 🔴 CRITICAL | Market orders (including SL STOP_MARKET) fill at the **next available price**, not the trigger price. A 3% flash crash could fill a 1.5% SL at 3%, turning a $25 loss into a $50 loss. With high leverage, the position is liquidated before the SL even has a chance to fill. |
| **Exchange maintenance** | 🟡 MEDIUM | During exchange maintenance or outages, high-leverage positions face liquidation risk from normal price movement that wouldn't affect low-leverage positions. |
| **Auto-deleveraging** | 🟠 HIGH | Binance has an Auto-Deleverage (ADL) system that can forcibly close profitable positions when counter-party liquidations occur. Higher leverage = higher ADL priority. |

### 3.2 Binance Futures Testnet Limitations

| Limitation | Impact |
|------------|--------|
| **STOP_MARKET / TAKE_PROFIT_MARKET** | Not always supported (error -4509). The current bot already works around this with bot-side WS monitoring. |
| **Isolated margin mode** | ✅ Supported on testnet. The `marginType` parameter can be set via `POST /fapi/v1/marginType` with `marginType=ISOLATED`. |
| **Dynamic leverage** | ✅ Supported. `POST /fapi/v1/leverage` accepts up to 125x (symbol-dependent). BTC: max 125x, ETH: max 100x on testnet. |
| **Liquidation behavior** | ⚠️ Testnet liquidation engine is less reliable than production. Liquidation orders may execute with different slippage than mainnet. |
| **Order book depth** | ⚠️ Testnet has thin order books. Large orders ($3,000+) may experience significant slippage compared to mainnet. |
| **changeMarginType errors** | ⚠️ If a position is already open in cross margin and you try to switch to isolated, it fails. Must be set BEFORE opening a position. |

### 3.3 Flash Crash / Slippage Analysis

**Scenario: BTC at $70,000, LONG position, SL at $68,950 (1.5%)**

| Event | Impact at 10x | Impact at 50x | Impact at 100x |
|-------|--------------|--------------|----------------|
| SL fills at exact price | -$25 (planned) | -$25 (planned) | -$25 (planned) |
| SL slips 0.5% ($350) | -$33.33 | -$33.33 | **LIQUIDATED** |
| SL slips 1.0% ($700) | -$41.67 | **LIQUIDATED** | **LIQUIDATED** |
| Flash crash 3% ($2,100) | -$75.00 | **LIQUIDATED** | **LIQUIDATED** |
| Flash crash 5% ($3,500) | -$108.33 (liq) | **LIQUIDATED** | **LIQUIDATED** |

At 10x leverage, the system survives moderate slippage (0.5-1%) with controlled extra loss. At 50x+, even minor slippage causes liquidation.

### 3.4 The 20 "Parts" Question

**Critical ambiguity: Are the 20 parts for 20 simultaneous positions or 20 sequential trades?**

**Interpretation A: 20 Sequential Trades (Most Likely Intent)**
- You have a $500 risk budget for this "cycle"
- Each trade risks $25, and after 20 losses, you recalculate from the new balance
- At any time, you have 1-2 positions open (BTC + ETH)
- **Verdict**: Reasonable. Risk budget is a mental accounting tool.

**Interpretation B: 20 Simultaneous Positions**
- You open up to 20 positions at once, each risking $25
- Total capital at risk: 20 × $25 = $500 (10% of balance)
- **Verdict**: Dangerous due to correlation. If BTC drops 5%, all crypto positions drop together. 20 correlated losses = instant 10% drawdown.

**Recommendation**: Interpret as sequential, cap simultaneous positions at 2 (one per asset).

### 3.5 Risk Scaling Pitfalls

The "double risk when balance doubles" rule has a dangerous edge case:

```
Starting:  $5,000 → risk $25/trade
Growth:    $10,000 → risk $50/trade (doubled ✓)
Drawdown:  $7,500 → still risking $50/trade (balance dropped below threshold!)
```

**Problem**: After scaling up, a drawdown doesn't scale the risk back down. The rule says "each time balance doubles from baseline, double risk" — but it doesn't say what happens on the way back down.

**At $7,500 with $50 risk/trade:**
- Risk/trade as % of balance: $50 / $7,500 = 0.67% (was 0.50% at $10K)
- 20 consecutive losses: $1,000 = 13.3% drawdown (was 10% at $10K)

This creates an **asymmetric ratchet** where risk goes up but never comes back down.

---

## 4. Implementation Complexity

### 4.1 Changes to `live_trading_htf.py`

| Change | Difficulty | Description |
|--------|-----------|-------------|
| Replace `POSITION_SIZE = 0.50` | 🟢 Easy | Replace with risk calculation function |
| Add risk pool tracking | 🟡 Medium | Track risk_pool, fixed_risk, risk_level in state file |
| Add balance doubling logic | 🟡 Medium | Track baseline, compute doublings, update fixed_risk |
| Remove `FIXED_MAX_NOTIONAL` | 🟢 Easy | Notional is now computed from risk/SL |
| Update `_open_position()` | 🟡 Medium | Compute notional from risk, pass leverage to executor |
| Update `session_balance` logic | 🟡 Medium | session_balance now feeds risk pool, not position size |
| Add risk scaling state | 🟡 Medium | Persist baseline, current_risk_level, risk_per_trade |

**Estimated new `_open_position` logic:**

```python
def _compute_position_params(self, price: float, sl_pct: float) -> dict:
    """Compute position size, leverage, and margin from fixed-dollar-risk model."""
    # Risk per trade from pool
    risk_pool = self.balance * 0.10
    fixed_risk = risk_pool / 20
    
    # Apply doubling rule
    doublings = int(math.log2(self.balance / self.baseline_balance)) if self.balance >= self.baseline_balance else 0
    fixed_risk *= (2 ** doublings)
    
    # Notional = risk / SL%
    notional = fixed_risk / sl_pct
    
    # Determine leverage (cap at 20x for safety)
    target_leverage = min(20, int(notional / (self.balance * 0.05)))  # Max 5% margin
    target_leverage = max(1, target_leverage)
    
    # Margin required
    margin = notional / target_leverage
    
    # Validate liquidation vs SL
    # For LONG: liq ≈ entry × (1 - 1/leverage)
    # SL = entry × (1 - sl_pct)
    # Need: liq < SL - delta
    liq_distance_pct = 1.0 / target_leverage
    if liq_distance_pct <= sl_pct * 1.5:  # Need 50% buffer
        # Reduce leverage until safe
        target_leverage = int(1.0 / (sl_pct * 1.5))
        target_leverage = max(1, target_leverage)
    
    return {
        "notional": notional,
        "leverage": target_leverage,
        "margin": notional / target_leverage,
        "fixed_risk": fixed_risk,
        "units": notional / price,
    }
```

### 4.2 Changes to `futures_executor.py`

| Change | Difficulty | Description |
|--------|-----------|-------------|
| Switch to isolated margin | 🟡 Medium | Call `POST /fapi/v1/marginType` with `ISOLATED` before opening |
| Accept dynamic leverage | 🟢 Easy | `open_long`/`open_short` already accept `leverage` param |
| Add margin adjustment | 🟠 Hard | `POST /fapi/v1/positionMargin` to add margin if liq too close |
| Pre-trade margin validation | 🟡 Medium | Before opening, verify margin type + compute liq price |
| Margin type state tracking | 🟡 Medium | Track per-symbol whether isolated mode is set |

**New method needed:**

```python
def set_margin_type(self, symbol: str, margin_type: str = "ISOLATED") -> bool:
    """Set margin type for a symbol. Must be called before opening a position."""
    try:
        self.connector.client.change_margin_type(symbol=symbol, marginType=margin_type)
        return True
    except Exception as exc:
        if "No need to change margin type" in str(exc):
            return True  # Already set
        logger.error("Failed to set margin type %s for %s: %s", margin_type, symbol, exc)
        return False

def add_position_margin(self, symbol: str, amount: float) -> bool:
    """Add margin to an isolated position to push liquidation price further."""
    try:
        self.connector.client.change_position_margin(
            symbol=symbol, amount=amount, type=1  # 1 = add margin
        )
        return True
    except Exception as exc:
        logger.error("Failed to add margin for %s: %s", symbol, exc)
        return False
```

### 4.3 Interaction with Partial TP System

The partial TP system (40% at 1R, 35% at 2R, remaining 25% trails) **is fully compatible**:

```
Position: 0.02381 BTC ($1,667 notional) with $25 risk

Partial TP1 (1R = +1.5%):
  Close 40% = 0.00952 BTC
  Profit = 0.00952 × $1,050 = $10.00

Partial TP2 (2R = +3.0%):  
  Close 35% = 0.00833 BTC
  Profit = 0.00833 × $2,100 = $17.50

Remaining 25% = 0.00595 BTC (trails with SL at breakeven+)
  If exits at 4R: 0.00595 × $4,200 = $25.00

Total potential: $10 + $17.50 + $25 = $52.50 (2.1R)
```

The partial TP percentages (40/35/25) work the same regardless of position sizing method. No changes needed to partial TP logic.

### 4.4 Risk Doubling + Drawdown Interaction

**The "ratchet problem" in detail:**

```
Phase 1: Balance $5,000 → $10,000 (100 trades, 60% win rate)
  Risk/trade: $25
  Status: First doubling achieved ✓

Phase 2: Risk doubles to $50/trade
  Next 20 trades: 8 wins, 12 losses (bad streak)
  Net PnL: (8 × $100) - (12 × $50) = $800 - $600 = +$200
  Balance: $10,200

Phase 3: Another bad streak
  Next 20 trades: 5 wins, 15 losses
  Net PnL: (5 × $100) - (15 × $50) = $500 - $750 = -$250
  Balance: $9,950

  ⚠️ Balance is below $10,000 but risk/trade is still $50
  ⚠️ At $9,950, $50 risk = 0.50% — barely acceptable
  
Phase 4: Continued drawdown
  Next 20 trades: 4 wins, 16 losses
  Net PnL: (4 × $100) - (16 × $50) = $400 - $800 = -$400
  Balance: $9,550

  ⚠️ $50 risk on $9,550 = 0.52% — creeping up
  
Phase 5: If it falls to $7,500:
  $50 risk on $7,500 = 0.67% per trade
  20 losses = $1,000 = 13.3% drawdown
  
  This is WORSE than the original $25-risk level at $5,000 (10% max DD)
```

**The rule needs a downward adjustment mechanism.** See Section 5 for recommendation.

---

## 5. Recommendation

### 5.1 Verdict: Implement with Modifications

The proposed approach is **better than the current system** for these reasons:

1. **Defined risk per trade** ($25 vs ambiguous 50% × 1.5% = $37.50)
2. **Lower risk per trade** (0.50% vs 0.75% of balance)
3. **Isolated margin** prevents account-level liquidation
4. **Capital efficiency** — only 3-7% margin locked vs 50% of balance

However, it needs these **critical modifications**:

### 5.2 Required Modifications

#### Modification 1: Cap Leverage at 10x

**Never exceed 10x leverage.** This ensures:
- Liquidation distance ≈ 9% (vs 1.5% SL) — massive buffer
- Flash crash protection up to ~7% before liquidation
- SL always triggers well before liquidation

```python
MAX_LEVERAGE = 10  # Hard cap — non-negotiable
```

#### Modification 2: Bidirectional Risk Scaling (Replace "Double on Double")

Instead of only scaling up, use a **smooth formula** that scales both ways:

```python
def compute_risk_per_trade(balance: float, initial_balance: float = 5000.0) -> float:
    """
    Risk per trade = 0.5% of current balance, quantized to $5 increments.
    
    This naturally scales with balance in both directions:
    - $5,000 → $25/trade
    - $10,000 → $50/trade  
    - $7,500 → $37.50 → rounded to $35/trade
    - $3,000 → $15/trade (scales DOWN on drawdown)
    """
    raw_risk = balance * 0.005  # 0.5% of balance
    # Floor to nearest $5 for clean numbers
    risk = max(5.0, math.floor(raw_risk / 5) * 5)
    return risk
```

This is simpler, safer, and achieves the same goal: risk grows with success, shrinks with drawdowns.

#### Modification 3: Liquidation Buffer Validation

Before every trade, validate that the liquidation price has at least a **2x buffer** beyond the SL:

```python
def validate_liq_buffer(entry, sl_pct, leverage, direction):
    """Ensure liquidation price is 2x further than SL from entry."""
    liq_distance = 1.0 / leverage  # Approximate
    sl_distance = sl_pct
    
    if liq_distance < sl_distance * 2.0:
        # Reduce leverage until safe
        safe_leverage = int(1.0 / (sl_distance * 2.0))
        return max(1, safe_leverage)
    return leverage
```

With a 1.5% SL and 2x buffer requirement:
```
Max leverage = 1 / (0.015 × 2.0) = 33x
Capped at 10x by rule 1 → 10x (safe)
```

#### Modification 4: Set Margin Type Per-Symbol on Startup

Add an initialization step to set isolated margin for all traded symbols:

```python
# In futures_executor.py __init__ or in live_trading_htf.py startup
for symbol in ["BTCUSDT", "ETHUSDT"]:
    executor.set_margin_type(symbol, "ISOLATED")
```

⚠️ This MUST be done when no positions are open. If a cross-margin position exists, the call will fail.

#### Modification 5: Maximum 2 Simultaneous Positions

Cap at one position per asset (BTC + ETH = 2 max). The "20 parts" should be interpreted as a sequential budget, not simultaneous capacity.

### 5.3 Final Comparison: Current vs Modified Proposed

| Metric | Current | Proposed (Modified) |
|--------|---------|-------------------|
| Risk/trade ($5K bal) | $37.50 (0.75%) | $25.00 (0.50%) |
| Risk/trade ($10K bal) | $75.00 (0.75%) | $50.00 (0.50%) |
| Max drawdown (20 losses) | ~14% | 10% |
| Margin mode | Cross | Isolated |
| Leverage | 1x | 10x (capped) |
| Margin per trade | $2,500 (50%) | $167 (3.3%) |
| Simultaneous positions | 1 | 2 (BTC + ETH) |
| Scales with drawdowns | Yes (% of shrinking bal) | Yes (0.5% of current) |
| Partial TP compatible | ✅ | ✅ |
| Trailing SL compatible | ✅ | ✅ |
| Regime multipliers | ✅ | ✅ |
| Liquidation risk | Low (1x, cross) | Very Low (10x, isolated, buffered) |

### 5.4 Implementation Priority

| Priority | Task | Effort | Files Changed |
|----------|------|--------|--------------|
| **P0** | Set isolated margin on startup | 2h | `futures_executor.py`, `binance_futures.py` |
| **P1** | Replace POSITION_SIZE with risk calculator | 3h | `live_trading_htf.py` |
| **P2** | Dynamic leverage (capped at 10x) | 2h | `futures_executor.py`, `live_trading_htf.py` |
| **P3** | Liquidation buffer validation | 2h | `futures_executor.py` |
| **P4** | Add margin adjustment (push liq away) | 3h | `futures_executor.py`, `binance_futures.py` |
| **P5** | State persistence (risk level, baseline) | 1h | `live_trading_htf.py` |
| **P6** | Telegram alerts for risk-level changes | 1h | `live_trading_htf.py` |

**Total estimated effort: 14 hours** (2 working days)

### 5.5 Testnet Rollout Plan

1. **Day 1**: Implement P0-P2 (isolated margin + risk calculator + leverage)
2. **Day 1**: Close any existing cross-margin positions first
3. **Day 2**: Implement P3-P4 (liquidation validation + margin adjustment)
4. **Day 2**: Deploy to testnet with both BTC and ETH
5. **Day 3-7**: Monitor for 5 days with Telegram alerts
6. **Day 7**: Review first 20 trades — compare actual loss per trade to $25 target

### 5.6 What NOT to Implement

| Feature from Proposal | Recommendation | Reason |
|----------------------|----------------|--------|
| "Maximize leverage" | ❌ SKIP | Liquidation risk is unacceptable above 10x |
| "Double on double" scaling | ❌ REPLACE | Use smooth 0.5% formula instead |
| 20 simultaneous positions | ❌ REINTERPRET | Cap at 2 positions, 20 is a sequential budget |
| "Add margin to push liq away" | ✅ IMPLEMENT | But as a safety net, not a primary strategy |

---

## 6. Appendix: Key Formulas

### Isolated Margin Liquidation Price

**LONG:**
```
Liquidation Price = Entry Price × (1 - Initial Margin Rate + Maintenance Margin Rate)
                  = Entry Price × (1 - 1/Leverage + MMR)

BTC MMR on Binance Futures: 0.40% for notional < $50K
```

**SHORT:**
```
Liquidation Price = Entry Price × (1 + Initial Margin Rate - Maintenance Margin Rate)
                  = Entry Price × (1 + 1/Leverage - MMR)
```

### Dynamic Leverage Calculator

```python
def compute_safe_leverage(sl_pct: float, buffer_multiplier: float = 2.0, max_leverage: int = 10) -> int:
    """
    Compute the maximum safe leverage given SL% and buffer requirement.
    
    The liquidation distance (1/leverage) must be at least buffer_multiplier × SL%.
    """
    theoretical_max = int(1.0 / (sl_pct * buffer_multiplier))
    return min(theoretical_max, max_leverage)

# Examples:
# SL=1.5%, buffer=2x → max 33x, capped to 10x
# SL=2.0%, buffer=2x → max 25x, capped to 10x  
# SL=3.0%, buffer=2x → max 16x, capped to 10x
```

### Risk Per Trade Formula

```python
def compute_trade_params(balance: float, sl_pct: float, entry_price: float):
    risk_per_trade = max(5.0, math.floor(balance * 0.005 / 5) * 5)  # 0.5% of balance, $5 floor
    notional = risk_per_trade / sl_pct
    leverage = compute_safe_leverage(sl_pct)
    margin = notional / leverage
    units = notional / entry_price
    
    return {
        "risk_per_trade": risk_per_trade,
        "notional": notional,
        "leverage": leverage,
        "margin": margin,
        "units": units,
        "risk_pct": risk_per_trade / balance * 100,
    }

# BTC @ $70K, balance $5K:
# risk=$25, notional=$1,667, leverage=10, margin=$167, units=0.0238
```

---

## 7. Conclusion

The proposed fixed-dollar-risk approach is a **significant improvement** over the current 50% position sizing. It introduces proper risk management principles:

1. **Know your max loss before entering** ($25, not "roughly 1.5% of 50% of balance")
2. **Isolated margin prevents cascading liquidation**
3. **Position size is determined by risk tolerance, not arbitrary percentage**
4. **Capital efficiency allows multi-asset trading**

The main danger is the "maximize leverage" directive, which must be hard-capped at 10x. The "double on double" scaling rule should be replaced with a smooth 0.5%-of-balance formula.

**Recommendation: Implement the modified version. It's strictly better than the current system.**
