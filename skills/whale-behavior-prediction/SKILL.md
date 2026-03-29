---
name: whale-behavior-prediction
description: Per-wallet behavioral pattern prediction for ETH whale wallets. Use when working on whale wallet data collection, action classification, price labeling, sequence model training, or real-time whale intent prediction. Also use when adding new wallets, debugging collection pipelines, improving action classification, tuning the LSTM model, or integrating whale behavior signals into the trading bot. This module is SEPARATE from the existing whale tracking system (src/features/whale_tracker.py) — it runs alongside it as an additional signal source.
---

# Whale Behavior Prediction

Predicts the likely next action of individual ETH whale wallets by learning their historical behavioral patterns before major price moves.

## Key Principle

This is NOT aggregate whale tracking. It learns **per-wallet behavioral fingerprints** — each wallet has its own pattern (e.g., Wallet X always does EXCHANGE_WITHDRAWAL → DORMANT → DEX_SWAP_BUY before ETH pumps).

## Architecture

```
Phase 1: Data Collection (Etherscan API → JSONL timelines)       ✅ COMPLETE
Phase 2: Price Labeling (Binance OHLCV → action + outcome pairs) ✅ COMPLETE
Phase 3: Sequence Model (LSTM → per-wallet intent prediction)    🔄 IN PROGRESS
Phase 4: Real-time Signal (live prediction → alert display)      🔄 IN PROGRESS
```

**Does NOT replace** `src/features/whale_tracker.py`. Produces an additional signal displayed in trade alerts.

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
│   ├── sequence_model.py    # LSTM training (Phase 3)
│   └── predictor.py         # Real-time prediction (Phase 4)
data/whale_behavior/
├── eth/                     # Raw action timelines per wallet (JSONL)
├── labeled/                 # Price-labeled timelines (JSONL)
├── price_cache/             # ETH hourly OHLCV cache (parquet)
└── models/                  # Trained model checkpoints
    ├── whale_behavior_lstm.pt     # Best model checkpoint
    └── training_results.json      # Latest training results
```

## Phase 1: Data Collection (COMPLETE)

### Collector: `src/whale_behavior/data/eth_collector.py`

Fetches complete transaction history per wallet from Etherscan:
- **Normal transactions** — ETH transfers, contract calls
- **Internal transactions** — contract-to-contract ETH movements
- **ERC-20 transfers** — USDT, USDC, token movements

Supports incremental updates (tracks last collected block per wallet).

### Action Classification (16 types)

| Action | Direction | Meaning |
|--------|-----------|---------|
| `EXCHANGE_DEPOSIT` | out | Sent ETH to a known exchange |
| `EXCHANGE_WITHDRAWAL` | in | Pulled ETH from exchange |
| `DEX_SWAP` | out | Interacted with DEX router |
| `DEX_INTERACTION` | out | Called a DEX contract |
| `DEX_RECEIVED` | in | Received from DEX |
| `STAKING_DEPOSIT` | out | Deposited to ETH2/Lido/RocketPool |
| `STAKING_WITHDRAWAL` | in | Withdrew from staking |
| `LARGE_TRANSFER_OUT` | out | Sent ETH to unknown address |
| `LARGE_TRANSFER_IN` | in | Received ETH from unknown |
| `CONTRACT_CALL` | out | Called a contract (0 ETH) |
| `CONTRACT_RECEIVED` | in | Received from contract |
| `TOKEN_TO_EXCHANGE` | out | Sent ERC-20 to exchange |
| `TOKEN_FROM_EXCHANGE` | in | Received ERC-20 from exchange |
| `TOKEN_TRANSFER_OUT` | out | Sent ERC-20 to unknown |
| `TOKEN_TRANSFER_IN` | in | Received ERC-20 from unknown |
| `UNKNOWN` | — | Unclassified |

### Running Collection

```bash
# Collect + label all wallets
python collect_whale_behavior.py

# Collect only (no labeling)
python collect_whale_behavior.py --collect-only

# Single wallet
python collect_whale_behavior.py --wallet "Galaxy Digital"
```

### Current Data (11 wallets, ~128K labeled actions)

| Wallet | Actions |
|--------|---------|
| Binance Hot Wallet | 20,049 |
| Binance Cold Wallet | 19,947 |
| Binance Reserve | 20,001 |
| Smart Money Whale 1 | 20,051 |
| ETH 2.0 Deposit | 11,447 |
| Coinbase Institutional | 10,602 |
| Binance Cold 2 | 10,415 |
| Kraken Deposit | 10,730 |
| Robinhood | 3,605 |
| Jump Trading | 1,310 |
| Galaxy Digital | 393 |

## Phase 2: Price Labeling (COMPLETE)

### Labeler: `src/whale_behavior/data/price_labeler.py`

Labels each action with what ETH price did afterward:

| Field | Description |
|-------|-------------|
| `label_1h/4h/12h/24h` | `BUY_SIGNAL` (>1.5%), `SELL_SIGNAL` (<-1.5%), `NEUTRAL` |
| `price_change_1h/4h/12h/24h` | Actual % change |

Label distribution (4h window): BUY=30,909 SELL=21,412 NEUTRAL=76,009

## Phase 3: Training

### Model Architecture

```
Input: Last 20 actions of wallet W (sliding window)
  ↓
Embeddings: action_type(8d) + value_bucket(4d) + counterparty(4d) + wallet_id(16d)
  ↓
Continuous features: time_gap_norm, direction, gas_ratio
  ↓
BiLSTM (2 layers, hidden=64, dropout=0.3)
  ↓
Self-Attention pooling
  ↓
FC(128) + dropout
  ↓
3 output heads:
  - Intent: [P(BUY), P(SELL), P(NEUTRAL)]  — CrossEntropy with class weights
  - Direction: P(bullish)                    — BCE
  - Magnitude: expected % move              — MSE
```

### Training Script

```bash
# Default training (4h window, 50 epochs)
python train_whale_behavior.py

# Custom params
python train_whale_behavior.py --window 4h --epochs 50 --batch-size 8 --lr 0.001 --patience 10

# With gradient accumulation (memory-efficient, recommended for this server)
python train_whale_behavior.py --batch-size 8 --accum-steps 4
```

### CRITICAL: Memory Constraints

**Server has only 3.7GB RAM.** Training 128K sequences OOMs with batch_size ≥ 32.

| batch_size | RAM usage | Status |
|------------|-----------|--------|
| 64 | ~600MB+ | ❌ OOM killed at epoch 1 |
| 32 | ~500MB+ | ❌ OOM killed at epoch 6 |
| 16 | ~420MB | ⚠️ Got to epoch 15, then OOM |
| 8 + accum=4 | ~300MB | ✅ Recommended — simulates batch 32 |

**Always use `--batch-size 8 --accum-steps 4`** on this server.

### Training Results (Latest)

Trained with batch_size=16, got to epoch 15 before OOM:
- **Best val_loss: 0.983**
- **SELL precision: 78.4%** 🔥 (usable as sell-only signal)
- BUY precision: 11.7% (poor)
- NEUTRAL precision: 9.9% (poor)
- Direction accuracy: 56.4%
- Overall intent accuracy: 23.5%

**Interpretation:** Model is heavily SELL-biased but accurate when it predicts SELL. Currently integrated as a display-only confidence signal in trade alerts (not used for decision-making).

### Training Output Files

| File | Description |
|------|-------------|
| `data/whale_behavior/models/whale_behavior_lstm.pt` | Best model checkpoint (saves on val_loss improvement) |
| `data/whale_behavior/models/training_results.json` | Final test metrics |
| `logs/whale_behavior_training.log` | Training log (check for epoch progress) |

### Monitoring Training

```bash
# Check if training is running
ps aux | grep train_whale | grep -v grep

# Check training log (may be buffered — use stdbuf for real-time)
tail -f logs/whale_behavior_training.log

# Check model checkpoint age
ls -la data/whale_behavior/models/whale_behavior_lstm.pt

# Run with unbuffered output for real-time monitoring
stdbuf -oL python3 -u train_whale_behavior.py --batch-size 8 --accum-steps 4 2>&1 | tee logs/whale_behavior_training.log
```

### Known Issues

1. **Log buffering**: Python buffers nohup output. Use `python3 -u` (unbuffered) or `stdbuf -oL` to see real-time epochs.
2. **OOM**: Always use batch_size=8 with gradient accumulation on this server.
3. **Class imbalance**: 59% NEUTRAL, 24% BUY, 17% SELL — model tends to predict the majority class. Class weights help but don't fully solve it.

## Phase 4: Real-time Signal (Display Only)

### Integration: Trade Alerts

Whale behavior model confidence is shown in trade alert messages as an informational signal:
```
🐋 Whale Behavior: SELL 78% confidence
```

**NOT used for trade decisions.** Display only — helps human traders see the whale context.

### Predictor: `src/whale_behavior/models/predictor.py`

Loads the trained model and runs inference on recent wallet actions:

```python
from src.whale_behavior.models.predictor import WhaleIntentPredictor

predictor = WhaleIntentPredictor()
signal = predictor.get_signal()
# Returns:
# {
#     "intent": "distributing",     # accumulating/distributing/neutral
#     "confidence": 0.78,           # model confidence
#     "direction": 0.28,            # P(bullish) — low = bearish
#     "active_wallets": 3,
# }
```

## Adding New Wallets

1. Add wallet address to `src/whale_behavior/data/eth_collector.py` wallet registry
2. Run `python collect_whale_behavior.py --wallet "New Wallet Label"`
3. Retrain: `python train_whale_behavior.py --batch-size 8 --accum-steps 4`

## Files

| Path | Role |
|------|------|
| `src/whale_behavior/__init__.py` | Module init |
| `src/whale_behavior/data/eth_collector.py` | Etherscan collection + action classification |
| `src/whale_behavior/data/price_labeler.py` | Price outcome labeling |
| `src/whale_behavior/models/sequence_model.py` | LSTM model + training pipeline |
| `src/whale_behavior/models/predictor.py` | Real-time predictor |
| `collect_whale_behavior.py` | Runner script (collection + labeling) |
| `train_whale_behavior.py` | Training runner script |
| `trade_alerter.py` | Alert service (displays whale confidence) |
| `data/whale_behavior/eth/*.jsonl` | Raw action timelines |
| `data/whale_behavior/labeled/*_labeled.jsonl` | Price-labeled data |
| `data/whale_behavior/models/` | Model checkpoints + results |
