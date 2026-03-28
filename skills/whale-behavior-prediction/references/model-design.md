# Whale Behavior Model — Detailed Design

## Problem Statement

Given the last N on-chain actions of a specific ETH whale wallet, predict whether the wallet is accumulating (bullish), distributing (bearish), or neutral — and use that as a trading signal.

## Why Per-Wallet Matters

Aggregate whale signals (what our existing system does) treat all wallets the same. But:

- **Exchange hot wallets** move tokens constantly for operational reasons — not signals
- **Institutional wallets** (Galaxy Digital, Jump Trading) have deliberate strategies
- **Accumulators** (Justin Sun) show clear pre-move patterns
- **Smart money wallets** have proven track records

A per-wallet model learns: "When *this specific wallet* does X→Y→Z, price tends to move in direction D."

## Input Representation

### Action Sequence

Each wallet's recent activity is represented as a sequence of action vectors:

```
Sequence length: N = 20 (last 20 actions)
Each action = [action_type, value_bucket, time_gap, direction, counterparty_type, gas_ratio]
```

### Feature Details

**action_type** (categorical, 15 classes → embedding dim 8):
```
EXCHANGE_DEPOSIT, EXCHANGE_WITHDRAWAL, DEX_SWAP, DEX_INTERACTION,
DEX_RECEIVED, STAKING_DEPOSIT, STAKING_WITHDRAWAL, LARGE_TRANSFER_OUT,
LARGE_TRANSFER_IN, CONTRACT_CALL, CONTRACT_RECEIVED, TOKEN_TO_EXCHANGE,
TOKEN_FROM_EXCHANGE, TOKEN_TRANSFER_OUT, TOKEN_TRANSFER_IN
```

**value_bucket** (categorical, 6 classes → embedding dim 4):
```
DUST:      < 0.01 ETH
SMALL:     0.01 - 1 ETH
MEDIUM:    1 - 100 ETH
LARGE:     100 - 1,000 ETH
WHALE:     1,000 - 10,000 ETH
MEGA:      > 10,000 ETH
```

**time_gap** (continuous, normalized):
```
Hours since previous action. Log-scaled: log(1 + hours) / log(1 + 720)
Captures dormancy patterns (whale going quiet for days = significant)
```

**direction** (binary): 0 = incoming, 1 = outgoing

**counterparty_type** (categorical, 8 classes → embedding dim 4):
```
exchange, dex, staking, unknown, weth_contract, bridge, eth2_deposit, self
```

**gas_ratio** (continuous, normalized):
```
gas_used / median_gas_for_action_type
> 1.5 = urgency (paying premium), < 0.5 = standard
```

### Wallet ID Embedding

Each wallet gets a learned embedding (dim 16). This allows the model to learn wallet-specific biases:
- "When wallet X does EXCHANGE_DEPOSIT, it's usually followed by a dump"
- "When wallet Y does EXCHANGE_DEPOSIT, it's usually rebalancing (neutral)"

## Model Architecture Options

### Option A: LSTM with Attention (simpler, less data needed)

```
action_embedding(dim=8) + value_embed(4) + counterparty_embed(4) + continuous_features(3)
  = 19-dim per action
  ↓
wallet_id_embedding(16) concatenated to each action → 35-dim
  ↓
Bidirectional LSTM(hidden=64, layers=2, dropout=0.3)
  ↓
Self-attention over LSTM outputs (learn which actions in sequence matter most)
  ↓
FC(128) → ReLU → Dropout(0.3)
  ↓
Output heads:
  - intent_head: FC(3) → softmax [accumulating, distributing, neutral]
  - direction_head: FC(1) → sigmoid [P(bullish)]
  - magnitude_head: FC(1) → tanh × 0.1 [expected % move, clamped ±10%]
```

**Parameters**: ~200K (small, trains on CPU in minutes)

### Option B: Transformer Encoder (more powerful, needs more data)

```
action_embedding + positional_encoding
  ↓
TransformerEncoder(d_model=64, nhead=4, num_layers=3, dropout=0.2)
  ↓
CLS token output → wallet_id_embedding concat
  ↓
FC(128) → ReLU → Dropout
  ↓
Same output heads as Option A
```

**Parameters**: ~500K (still small, but needs more training data)

### Recommendation: Start with Option A (LSTM)

LSTM is better for our data size. We can upgrade to Transformer if we collect more data.

## Training Pipeline

### Data Preparation

1. Load labeled timelines for all wallets
2. Create overlapping windows: `[action_{t-19}, ..., action_t]` → `label_4h at action_t`
3. Skip sequences where `label_4h` is None (insufficient future data)
4. Balance classes: undersample NEUTRAL to match BUY_SIGNAL + SELL_SIGNAL count

### Split Strategy

**Time-based split** (NOT random — prevents data leakage):
- Train: oldest 70% of each wallet's timeline
- Validation: next 15%
- Test: newest 15%

### Training Hyperparameters

```python
{
    "sequence_length": 20,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 50,
    "patience": 10,       # early stopping
    "lstm_hidden": 64,
    "lstm_layers": 2,
    "dropout": 0.3,
    "wallet_embed_dim": 16,
    "label_window": "4h",  # primary prediction window
}
```

### Loss Function

Multi-task loss:
```
L = α × CrossEntropy(intent) + β × BCE(direction) + γ × MSE(magnitude)
where α=1.0, β=0.5, γ=0.3
```

### Evaluation Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Intent accuracy | >55% | 3-class (accumulating/distributing/neutral) |
| Direction accuracy | >55% | Binary (bullish/bearish) on non-neutral samples |
| Precision (BUY) | >60% | When model says "accumulating", how often is it right? |
| Precision (SELL) | >60% | When model says "distributing", how often is it right? |
| Profit factor | >1.2 | Simulated PnL if trading on model signals |

## Real-time Inference

### Signal Generation

Every 15 minutes (aligned with trading loop):

1. Fetch latest transactions for all tracked wallets (incremental)
2. For each wallet: take last 20 actions → run through model
3. Get per-wallet prediction: `{intent, direction, confidence}`
4. Aggregate across wallets:
   - Weight by wallet tier (Elite 3x, Strong 1.5x, Experimental 0.5x)
   - Weight by model confidence
   - Require minimum 2 wallets agreeing for high-confidence signal

### Output Signal

```python
{
    "signal": 0.65,           # -1 to +1 (aggregate across wallets)
    "confidence": 0.78,       # 0 to 1
    "intent": "accumulating", # majority intent
    "active_wallets": 4,      # wallets with clear (non-neutral) intent
    "wallet_details": {
        "Galaxy Digital": {"intent": "accumulating", "direction": 0.82, "confidence": 0.91},
        "Jump Trading": {"intent": "accumulating", "direction": 0.75, "confidence": 0.84},
        "Justin Sun": {"intent": "distributing", "direction": 0.35, "confidence": 0.72},
        "Wintermute": {"intent": "neutral", "direction": 0.52, "confidence": 0.45},
    }
}
```

## Future Expansion

### More Chains (after ETH is proven)
- SOL: Solscan API for wallet history
- XRP: XRPL API for wallet history
- Same model architecture, different collector

### More Features
- **Token-specific flows**: Track USDT/USDC separately (stablecoin accumulation = dry powder)
- **Cross-wallet correlation**: When Wallet A and Wallet B both do X simultaneously
- **Network graph**: Who is Wallet X sending to? Build a transaction graph
- **Gas price context**: Was the action during high gas (urgent) or low gas (patient)?

### Model Improvements
- **Temporal Fusion Transformer**: Better at multi-horizon prediction
- **Contrastive learning**: Learn wallet embeddings by "which wallets behave similarly"
- **Online learning**: Update model weights as new data arrives (no full retrain)
