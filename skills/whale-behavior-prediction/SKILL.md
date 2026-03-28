---
name: whale-behavior-prediction
description: Per-wallet behavioral pattern prediction for ETH whale wallets. Use when working on whale wallet data collection, action classification, price labeling, sequence model training, or real-time whale intent prediction. Also use when adding new wallets, debugging collection pipelines, improving action classification, tuning the LSTM/Transformer model, or integrating whale behavior signals into the trading bot. This module is SEPARATE from the existing whale tracking system (src/features/whale_tracker.py) — it runs alongside it as an additional signal source.
---

# Whale Behavior Prediction

Predicts the likely next action of individual ETH whale wallets by learning their historical behavioral patterns before major price moves.

## Key Principle

This is NOT aggregate whale tracking. It learns **per-wallet behavioral fingerprints** — each wallet has its own pattern (e.g., Wallet X always does EXCHANGE_WITHDRAWAL → DORMANT → DEX_SWAP_BUY before ETH pumps).

## Architecture

```
Phase 1: Data Collection (Etherscan API → JSONL timelines)
Phase 2: Price Labeling (Binance OHLCV → action + outcome pairs)
Phase 3: Sequence Model (LSTM/Transformer → per-wallet intent prediction)
Phase 4: Real-time Signal (live prediction → trading bot integration)
```

**Does NOT replace** `src/features/whale_tracker.py`. Produces an additional signal that feeds alongside existing signals.

## Module Structure

```
src/whale_behavior/
├── __init__.py
├── data/
│   ├── __init__.py
│   ├── eth_collector.py     # Etherscan historical data collection
│   └── price_labeler.py     # Label actions with price outcomes
├── models/
│   ├── __init__.py
│   ├── sequence_model.py    # LSTM/Transformer training (Phase 3)
│   └── predictor.py         # Real-time prediction (Phase 4)
data/whale_behavior/
├── eth/                     # Raw action timelines per wallet (JSONL)
├── labeled/                 # Price-labeled timelines (JSONL)
├── price_cache/             # ETH hourly OHLCV cache (parquet)
└── models/                  # Trained model checkpoints
```

## Phase 1: Data Collection

### Collector: `src/whale_behavior/data/eth_collector.py`

Fetches complete transaction history per wallet from Etherscan:
- **Normal transactions** — ETH transfers, contract calls
- **Internal transactions** — contract-to-contract ETH movements
- **ERC-20 transfers** — USDT, USDC, token movements

Supports incremental updates (tracks last collected block per wallet).

### Action Classification

Each transaction is classified into one of these action types:

| Action | Direction | Meaning |
|--------|-----------|---------|
| `EXCHANGE_DEPOSIT` | out | Sent ETH to a known exchange (preparing to sell?) |
| `EXCHANGE_WITHDRAWAL` | in | Pulled ETH from exchange (accumulating?) |
| `DEX_SWAP` | out | Interacted with DEX router (Uniswap, 1inch, etc.) |
| `DEX_INTERACTION` | out | Called a DEX contract |
| `DEX_RECEIVED` | in | Received from DEX (swap output) |
| `STAKING_DEPOSIT` | out | Deposited to ETH2, Lido, RocketPool (long-term bullish) |
| `STAKING_WITHDRAWAL` | in | Withdrew from staking (preparing to sell?) |
| `LARGE_TRANSFER_OUT` | out | Sent ETH to unknown address (distribution?) |
| `LARGE_TRANSFER_IN` | in | Received ETH from unknown address (accumulation?) |
| `CONTRACT_CALL` | out | Called a contract (0 ETH value) |
| `CONTRACT_RECEIVED` | in | Received from contract |
| `TOKEN_TO_EXCHANGE` | out | Sent ERC-20 tokens to exchange |
| `TOKEN_FROM_EXCHANGE` | in | Received ERC-20 from exchange |
| `TOKEN_TRANSFER_OUT` | out | Sent ERC-20 to unknown |
| `TOKEN_TRANSFER_IN` | in | Received ERC-20 from unknown |

### Known Address Registry

The collector maintains labels for known addresses to classify counterparties:
- **Exchanges**: Binance, Coinbase, Kraken, Robinhood hot/cold wallets
- **DEX routers**: Uniswap V2/V3, SushiSwap, 1inch, 0x
- **Staking**: ETH 2.0 Deposit, Lido stETH, RocketPool

To add new known addresses: edit `KNOWN_EXCHANGES`, `KNOWN_STAKING`, or `KNOWN_DEX_ROUTERS` dicts in `eth_collector.py`.

### Running Collection

```bash
# Collect + label all wallets
python collect_whale_behavior.py

# Collect only (no labeling)
python collect_whale_behavior.py --collect-only

# Single wallet
python collect_whale_behavior.py --wallet "Galaxy Digital"

# Force re-collect from scratch
python collect_whale_behavior.py --force
```

### Data Format

Each wallet's timeline is stored as JSONL at `data/whale_behavior/eth/<wallet_label>.jsonl`:

```json
{
  "timestamp": 1711234567,
  "action": "EXCHANGE_DEPOSIT",
  "value_eth": 150.5,
  "to_type": "binance",
  "tx_hash": "0x...",
  "gas_used": 21000,
  "block": 19500000,
  "direction": "out"
}
```

## Phase 2: Price Labeling

### Labeler: `src/whale_behavior/data/price_labeler.py`

For each action, labels what ETH price did afterward:

| Field | Description |
|-------|-------------|
| `price_at_action` | ETH/USDT price when the action occurred |
| `price_change_1h` | % change after 1 hour |
| `price_change_4h` | % change after 4 hours |
| `price_change_12h` | % change after 12 hours |
| `price_change_24h` | % change after 24 hours |
| `label_1h/4h/12h/24h` | `BUY_SIGNAL` (>1.5%), `SELL_SIGNAL` (<-1.5%), `NEUTRAL` |

Uses Binance hourly OHLCV, cached as parquet at `data/whale_behavior/price_cache/eth_hourly.parquet`.

Labeled data saved to `data/whale_behavior/labeled/<wallet>_labeled.jsonl`.

## Phase 3: Sequence Model (TODO)

### Concept

Each wallet has its own behavioral fingerprint. The model learns temporal patterns:

```
Wallet X typical pre-pump sequence:
  DORMANT(48h) → EXCHANGE_WITHDRAWAL → DORMANT(6h) → DEX_SWAP_BUY → DEX_SWAP_BUY
  → ETH pumps 5-8% in 24h

Wallet Y typical pre-dump sequence:
  STAKING_WITHDRAWAL → EXCHANGE_DEPOSIT → EXCHANGE_DEPOSIT → EXCHANGE_DEPOSIT
  → ETH dumps 3-5% in 12h
```

### Model Architecture

```
Input: Last N actions of wallet W
  ↓
Action embedding (action_type + value_bucket + time_gap → vector)
  ↓
Wallet ID embedding (learned per-wallet bias)
  ↓
LSTM or Transformer encoder (learns temporal patterns)
  ↓
Attention over sequence (which actions matter most?)
  ↓
Output heads:
  - Intent classification: [P(accumulating), P(distributing), P(neutral)]
  - Direction: P(bullish) vs P(bearish)
  - Confidence: 0-1
  - Expected magnitude: regression (expected % move)
```

### Feature Engineering for Sequences

Each action in the sequence becomes a feature vector:

| Feature | Description |
|---------|-------------|
| `action_type` | One-hot or learned embedding (15 types) |
| `value_bucket` | Log-scaled value bucket (tiny/small/medium/large/whale) |
| `time_since_prev` | Hours since previous action (captures dormancy) |
| `direction` | In/Out binary |
| `to_type` | Counterparty type embedding |
| `gas_ratio` | Gas used relative to average (urgency indicator) |
| `token_type` | If ERC-20: USDT/USDC/ETH/other |

### Training Strategy

- **Per-wallet models** or a **single model with wallet ID embedding** (more data-efficient)
- Train on sequences where the last action's `label_4h` is BUY_SIGNAL or SELL_SIGNAL (skip NEUTRAL for training, use for evaluation)
- Validation: time-based split (train on older data, test on recent)
- Loss: Cross-entropy for classification + MSE for magnitude

## Phase 4: Real-time Signal (TODO)

### Integration with Trading Bot

```python
# In the trading loop (alongside existing whale_tracker.get_whale_signals())
from src.whale_behavior.models.predictor import WhaleIntentPredictor

predictor = WhaleIntentPredictor()
intent_signal = predictor.get_signal("ETHUSDT")
# Returns:
# {
#     "intent": "accumulating",  # or "distributing", "neutral"
#     "direction": 0.72,          # bullish probability
#     "confidence": 0.85,
#     "active_wallets": 3,        # how many wallets show clear intent
#     "wallet_details": {...}      # per-wallet breakdown
# }
```

The signal feeds into the DRL agent's feature vector as additional features, NOT replacing existing whale signals.

### Update Frequency

- **Collection**: Every 1 hour (incremental, only new blocks)
- **Prediction**: Every 15 minutes (aligned with trading loop)
- **Model retraining**: Weekly (or when accuracy drops below threshold)

## Adding New Wallets

1. Add wallet to `src/features/whale_wallet_registry.py` (existing registry)
2. Run `python collect_whale_behavior.py --wallet "New Wallet Label"`
3. Labeled data auto-generated
4. Model picks up new wallet on next training cycle

## Tracked Wallets (ETH)

Current registry (15 wallets): see `src/features/whale_wallet_registry.py` → `get_wallets_by_chain("ETH")`

Includes: Binance (3 wallets), Smart Money Whale 1, Galaxy Digital, Coinbase Institutional, Robinhood, ETH 2.0 Deposit, Wintermute, Jump Trading, Justin Sun, FTX Estate, Arbitrum Bridge, Kraken.

## Files

| Path | Role |
|------|------|
| `src/whale_behavior/__init__.py` | Module init |
| `src/whale_behavior/data/eth_collector.py` | Etherscan historical collection + action classification |
| `src/whale_behavior/data/price_labeler.py` | Price outcome labeling (Binance OHLCV) |
| `src/whale_behavior/models/sequence_model.py` | LSTM/Transformer model (Phase 3 — TODO) |
| `src/whale_behavior/models/predictor.py` | Real-time predictor (Phase 4 — TODO) |
| `collect_whale_behavior.py` | Runner script (collection + labeling) |
| `data/whale_behavior/eth/*.jsonl` | Raw action timelines |
| `data/whale_behavior/labeled/*_labeled.jsonl` | Price-labeled data |
| `data/whale_behavior/price_cache/` | Binance OHLCV cache |
| `src/features/whale_wallet_registry.py` | Shared wallet registry (existing) |
