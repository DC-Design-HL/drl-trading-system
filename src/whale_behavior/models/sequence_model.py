"""
Whale Behavior Sequence Model — LSTM with Attention

Learns per-wallet behavioral patterns from historical action sequences.
Predicts intent (accumulating/distributing/neutral) and direction (bullish/bearish).

Architecture:
  Action embeddings + wallet ID embedding → BiLSTM → Self-Attention → Output heads
"""

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

MODEL_DIR = Path("data/whale_behavior/models")
LABELED_DIR = Path("data/whale_behavior/labeled")

# ── Action type vocabulary ────────────────────────────────────────────
ACTION_TYPES = [
    "EXCHANGE_DEPOSIT", "EXCHANGE_WITHDRAWAL", "DEX_SWAP", "DEX_INTERACTION",
    "DEX_RECEIVED", "STAKING_DEPOSIT", "STAKING_WITHDRAWAL",
    "LARGE_TRANSFER_OUT", "LARGE_TRANSFER_IN", "CONTRACT_CALL",
    "CONTRACT_RECEIVED", "TOKEN_TO_EXCHANGE", "TOKEN_FROM_EXCHANGE",
    "TOKEN_TRANSFER_OUT", "TOKEN_TRANSFER_IN", "UNKNOWN",
]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_TYPES)}
NUM_ACTION_TYPES = len(ACTION_TYPES)

# Counterparty type vocabulary
COUNTERPARTY_TYPES = [
    "binance", "coinbase", "kraken", "robinhood", "dex", "staking",
    "eth2_deposit", "weth_contract", "bridge", "unknown", "",
]
COUNTER_TO_IDX = {c: i for i, c in enumerate(COUNTERPARTY_TYPES)}
NUM_COUNTER_TYPES = len(COUNTERPARTY_TYPES)

# Value buckets
def value_to_bucket(value_eth: float) -> int:
    """Convert ETH value to bucket index."""
    if value_eth < 0.01:
        return 0   # DUST
    elif value_eth < 1:
        return 1   # SMALL
    elif value_eth < 100:
        return 2   # MEDIUM
    elif value_eth < 1000:
        return 3   # LARGE
    elif value_eth < 10000:
        return 4   # WHALE
    else:
        return 5   # MEGA

NUM_VALUE_BUCKETS = 6

# Label mapping
INTENT_LABELS = {"BUY_SIGNAL": 0, "SELL_SIGNAL": 1, "NEUTRAL": 2}
NUM_INTENTS = 3

# Hyperparameters
SEQ_LENGTH = 20
LSTM_HIDDEN = 64
LSTM_LAYERS = 2
DROPOUT = 0.3
WALLET_EMBED_DIM = 16
ACTION_EMBED_DIM = 8
VALUE_EMBED_DIM = 4
COUNTER_EMBED_DIM = 4
CONTINUOUS_FEATURES = 3  # time_gap, direction, gas_ratio


# ── Dataset ───────────────────────────────────────────────────────────

class WhaleSequenceDataset(Dataset):
    """
    Creates sliding-window sequences from labeled wallet timelines.
    Each sample: (action_sequence[SEQ_LENGTH], wallet_id, label)
    """

    def __init__(
        self,
        sequences: np.ndarray,      # (N, seq_len, features) float32
        wallet_ids: np.ndarray,      # (N,) int32
        labels_intent: np.ndarray,   # (N,) int32
        labels_direction: np.ndarray,  # (N,) float32
        labels_magnitude: np.ndarray,  # (N,) float32
    ):
        self.sequences = sequences
        self.wallet_ids = wallet_ids
        self.labels_intent = labels_intent
        self.labels_direction = labels_direction
        self.labels_magnitude = labels_magnitude

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.sequences[idx]),
            torch.LongTensor([self.wallet_ids[idx]]),
            torch.LongTensor([self.labels_intent[idx]]),
            torch.FloatTensor([self.labels_direction[idx]]),
            torch.FloatTensor([self.labels_magnitude[idx]]),
        )


# ── Model ─────────────────────────────────────────────────────────────

class SelfAttention(nn.Module):
    """Simple self-attention over LSTM outputs."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.scale = math.sqrt(hidden_dim)

    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        q = self.query(x)
        k = self.key(x)
        attn_weights = torch.bmm(q, k.transpose(1, 2)) / self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attended = torch.bmm(attn_weights, x)
        # Pool: take mean of attended outputs
        return attended.mean(dim=1)


class WhaleBehaviorLSTM(nn.Module):
    """
    Per-wallet behavioral pattern model.

    Input per timestep: [action_type, value_bucket, counterparty_type, time_gap, direction, gas_ratio]
    Additional: wallet_id embedding

    Output:
      - intent: P(accumulating), P(distributing), P(neutral)
      - direction: P(bullish)
      - magnitude: expected % price move
    """

    def __init__(self, num_wallets: int = 20):
        super().__init__()

        # Embeddings
        self.action_embed = nn.Embedding(NUM_ACTION_TYPES, ACTION_EMBED_DIM)
        self.value_embed = nn.Embedding(NUM_VALUE_BUCKETS, VALUE_EMBED_DIM)
        self.counter_embed = nn.Embedding(NUM_COUNTER_TYPES, COUNTER_EMBED_DIM)
        self.wallet_embed = nn.Embedding(num_wallets, WALLET_EMBED_DIM)

        # Input dim per timestep
        input_dim = ACTION_EMBED_DIM + VALUE_EMBED_DIM + COUNTER_EMBED_DIM + CONTINUOUS_FEATURES
        # + wallet embed is concatenated after LSTM

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=LSTM_HIDDEN,
            num_layers=LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT if LSTM_LAYERS > 1 else 0,
        )

        lstm_out_dim = LSTM_HIDDEN * 2  # bidirectional

        # Attention
        self.attention = SelfAttention(lstm_out_dim)

        # Classification head
        fc_input_dim = lstm_out_dim + WALLET_EMBED_DIM
        self.fc1 = nn.Linear(fc_input_dim, 128)
        self.dropout = nn.Dropout(DROPOUT)

        # Output heads
        self.intent_head = nn.Linear(128, NUM_INTENTS)
        self.direction_head = nn.Linear(128, 1)
        self.magnitude_head = nn.Linear(128, 1)

    def forward(self, x, wallet_ids):
        """
        x: (batch, seq_len, features)
           features = [action_type_idx, value_bucket_idx, counter_type_idx,
                       time_gap_norm, direction, gas_ratio_norm]
        wallet_ids: (batch, 1)
        """
        batch_size = x.size(0)

        # Split features
        action_idx = x[:, :, 0].long()
        value_idx = x[:, :, 1].long()
        counter_idx = x[:, :, 2].long()
        continuous = x[:, :, 3:]  # time_gap, direction, gas_ratio

        # Embeddings
        action_emb = self.action_embed(action_idx)
        value_emb = self.value_embed(value_idx)
        counter_emb = self.counter_embed(counter_idx)

        # Concatenate per-timestep features
        lstm_input = torch.cat([action_emb, value_emb, counter_emb, continuous], dim=-1)

        # LSTM
        lstm_out, _ = self.lstm(lstm_input)

        # Attention pooling
        attended = self.attention(lstm_out)  # (batch, lstm_out_dim)

        # Wallet embedding
        wallet_emb = self.wallet_embed(wallet_ids.squeeze(-1))  # (batch, wallet_embed_dim)

        # Combine
        combined = torch.cat([attended, wallet_emb], dim=-1)

        # FC layers
        h = F.relu(self.fc1(combined))
        h = self.dropout(h)

        # Output heads
        intent_logits = self.intent_head(h)
        direction = torch.sigmoid(self.direction_head(h))
        magnitude = torch.tanh(self.magnitude_head(h)) * 0.1  # clamp to ±10%

        return intent_logits, direction, magnitude


# ── Data Preparation ──────────────────────────────────────────────────

def load_labeled_data(label_window: str = "4h") -> Tuple[List, List]:
    """
    Load all labeled wallet timelines and build training data.

    Returns (wallet_names, wallet_data) where wallet_data[i] is a list of
    action dicts for wallet i.
    """
    wallet_names = []
    wallet_data = []

    for f in sorted(LABELED_DIR.glob("*_labeled.jsonl")):
        actions = []
        with open(f) as fh:
            for line in fh:
                try:
                    rec = json.loads(line.strip())
                    # Skip if no label for this window
                    label_key = f"label_{label_window}"
                    if rec.get(label_key) is None:
                        continue
                    actions.append(rec)
                except json.JSONDecodeError:
                    continue

        if len(actions) >= SEQ_LENGTH + 1:
            name = f.stem.replace("_labeled", "")
            wallet_names.append(name)
            wallet_data.append(actions)
            logger.info("Loaded %s: %d labeled actions", name, len(actions))

    return wallet_names, wallet_data


def action_to_features(action: Dict) -> np.ndarray:
    """Convert a single action dict to a feature vector."""
    action_idx = ACTION_TO_IDX.get(action.get("action", "UNKNOWN"), ACTION_TO_IDX["UNKNOWN"])
    value_bucket = value_to_bucket(abs(action.get("value_eth", 0)))
    counter_idx = COUNTER_TO_IDX.get(action.get("to_type", "unknown"), COUNTER_TO_IDX["unknown"])

    # Continuous features
    time_gap = action.get("_time_gap_hours", 0)
    time_gap_norm = min(math.log(1 + time_gap) / math.log(1 + 720), 1.0)

    direction = 1.0 if action.get("direction") == "out" else 0.0

    gas_ratio = min(action.get("_gas_ratio", 1.0), 5.0) / 5.0  # Normalize to 0-1

    return np.array([action_idx, value_bucket, counter_idx,
                     time_gap_norm, direction, gas_ratio], dtype=np.float32)


def build_sequences(
    wallet_names: List[str],
    wallet_data: List[List[Dict]],
    label_window: str = "4h",
    seq_length: int = SEQ_LENGTH,
) -> Tuple:
    """
    Build sliding-window sequences from wallet timelines.

    Returns (sequences, wallet_ids, labels_intent, labels_direction, labels_magnitude)
    """
    all_sequences = []
    all_wallet_ids = []
    all_labels_intent = []
    all_labels_direction = []
    all_labels_magnitude = []

    label_key = f"label_{label_window}"
    change_key = f"price_change_{label_window}"

    for wallet_idx, (name, actions) in enumerate(zip(wallet_names, wallet_data)):
        # Pre-compute time gaps and gas ratios
        for i, action in enumerate(actions):
            if i > 0:
                time_gap = (action["timestamp"] - actions[i - 1]["timestamp"]) / 3600.0
                action["_time_gap_hours"] = max(time_gap, 0)
            else:
                action["_time_gap_hours"] = 0

            # Gas ratio (relative to median — approximate)
            gas = action.get("gas_used", 21000) or 21000
            action["_gas_ratio"] = gas / 50000.0  # rough normalization

        # Sliding window
        for i in range(seq_length, len(actions)):
            window = actions[i - seq_length: i]
            target = actions[i]

            # Label
            label_str = target.get(label_key)
            if label_str not in INTENT_LABELS:
                continue

            intent = INTENT_LABELS[label_str]
            price_change = target.get(change_key, 0) or 0

            # Direction: 1.0 = bullish (BUY_SIGNAL), 0.0 = bearish (SELL_SIGNAL)
            if label_str == "BUY_SIGNAL":
                direction = 1.0
            elif label_str == "SELL_SIGNAL":
                direction = 0.0
            else:
                direction = 0.5  # neutral

            # Build feature sequence
            seq = np.array([action_to_features(a) for a in window], dtype=np.float32)

            all_sequences.append(seq)
            all_wallet_ids.append(wallet_idx)
            all_labels_intent.append(intent)
            all_labels_direction.append(direction)
            all_labels_magnitude.append(float(price_change))

    # Convert to compact numpy arrays to reduce memory
    return (
        np.array(all_sequences, dtype=np.float32),
        np.array(all_wallet_ids, dtype=np.int32),
        np.array(all_labels_intent, dtype=np.int32),
        np.array(all_labels_direction, dtype=np.float32),
        np.array(all_labels_magnitude, dtype=np.float32),
    )


def time_split(
    sequences, wallet_ids, labels_intent, labels_direction, labels_magnitude,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[WhaleSequenceDataset, WhaleSequenceDataset, WhaleSequenceDataset]:
    """
    Time-based split (no shuffle — preserves temporal order).
    """
    n = len(sequences)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    def make_dataset(start, end):
        return WhaleSequenceDataset(
            sequences[start:end],
            wallet_ids[start:end],
            labels_intent[start:end],
            labels_direction[start:end],
            labels_magnitude[start:end],
        )

    return make_dataset(0, train_end), make_dataset(train_end, val_end), make_dataset(val_end, n)


# ── Training ──────────────────────────────────────────────────────────

def train_model(
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    label_window: str = "4h",
    patience: int = 10,
) -> Dict:
    """
    Full training pipeline: load data → build sequences → train LSTM → save model.

    Returns training results dict.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    # Load data
    wallet_names, wallet_data = load_labeled_data(label_window)
    if not wallet_names:
        logger.error("No labeled data found!")
        return {"error": "No data"}

    logger.info("Loaded %d wallets", len(wallet_names))

    # Build sequences
    # Free raw data before building sequences
    sequences, wallet_ids, labels_intent, labels_direction, labels_magnitude = \
        build_sequences(wallet_names, wallet_data, label_window)
    del wallet_data  # Free memory — no longer needed
    import gc; gc.collect()

    logger.info("Built %d sequences", len(sequences))

    if len(sequences) < 100:
        logger.error("Too few sequences (%d) for training", len(sequences))
        return {"error": "Insufficient sequences"}

    # Class balance info
    from collections import Counter
    intent_dist = Counter(labels_intent)
    logger.info("Intent distribution: BUY=%d SELL=%d NEUTRAL=%d",
                intent_dist.get(0, 0), intent_dist.get(1, 0), intent_dist.get(2, 0))

    # Compute class weights for imbalanced data
    total = len(labels_intent)
    class_counts = [intent_dist.get(i, 1) for i in range(NUM_INTENTS)]
    class_weights = torch.FloatTensor([total / (NUM_INTENTS * c) for c in class_counts]).to(device)
    logger.info("Class weights: %s", class_weights.tolist())

    # Split
    train_ds, val_ds, test_ds = time_split(
        sequences, wallet_ids, labels_intent, labels_direction, labels_magnitude
    )
    logger.info("Split: train=%d val=%d test=%d", len(train_ds), len(val_ds), len(test_ds))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Model
    num_wallets = len(wallet_names)
    model = WhaleBehaviorLSTM(num_wallets=num_wallets).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Loss functions
    intent_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    direction_loss_fn = nn.BCELoss()
    magnitude_loss_fn = nn.MSELoss()

    # Loss weights
    alpha, beta, gamma = 1.0, 0.5, 0.3

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_intent_acc": [], "val_direction_acc": []}

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        train_loss_sum = 0
        train_batches = 0

        for seq, wid, intent, direction, magnitude in train_loader:
            seq, wid = seq.to(device), wid.to(device)
            intent, direction, magnitude = intent.to(device), direction.to(device), magnitude.to(device)

            intent_logits, dir_pred, mag_pred = model(seq, wid)

            loss_intent = intent_loss_fn(intent_logits, intent.squeeze(-1))
            loss_dir = direction_loss_fn(dir_pred.squeeze(-1), direction.squeeze(-1))
            loss_mag = magnitude_loss_fn(mag_pred.squeeze(-1), magnitude.squeeze(-1))

            loss = alpha * loss_intent + beta * loss_dir + gamma * loss_mag

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

        avg_train_loss = train_loss_sum / max(train_batches, 1)

        # ── Validate ──
        model.eval()
        val_loss_sum = 0
        val_batches = 0
        val_intent_correct = 0
        val_dir_correct = 0
        val_total = 0

        with torch.no_grad():
            for seq, wid, intent, direction, magnitude in val_loader:
                seq, wid = seq.to(device), wid.to(device)
                intent, direction, magnitude = intent.to(device), direction.to(device), magnitude.to(device)

                intent_logits, dir_pred, mag_pred = model(seq, wid)

                loss_intent = intent_loss_fn(intent_logits, intent.squeeze(-1))
                loss_dir = direction_loss_fn(dir_pred.squeeze(-1), direction.squeeze(-1))
                loss_mag = magnitude_loss_fn(mag_pred.squeeze(-1), magnitude.squeeze(-1))

                loss = alpha * loss_intent + beta * loss_dir + gamma * loss_mag
                val_loss_sum += loss.item()
                val_batches += 1

                # Accuracy
                pred_intent = intent_logits.argmax(dim=1)
                val_intent_correct += (pred_intent == intent.squeeze(-1)).sum().item()

                pred_dir = (dir_pred.squeeze(-1) > 0.5).float()
                actual_dir = (direction.squeeze(-1) > 0.5).float()
                val_dir_correct += (pred_dir == actual_dir).sum().item()

                val_total += intent.size(0)

        avg_val_loss = val_loss_sum / max(val_batches, 1)
        val_intent_acc = val_intent_correct / max(val_total, 1)
        val_dir_acc = val_dir_correct / max(val_total, 1)

        scheduler.step(avg_val_loss)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_intent_acc"].append(val_intent_acc)
        history["val_direction_acc"].append(val_dir_acc)

        if epoch % 5 == 0 or epoch == epochs - 1:
            logger.info(
                "Epoch %d/%d: train_loss=%.4f val_loss=%.4f intent_acc=%.1f%% dir_acc=%.1f%%",
                epoch + 1, epochs, avg_train_loss, avg_val_loss,
                val_intent_acc * 100, val_dir_acc * 100,
            )

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                "model_state": model.state_dict(),
                "wallet_names": wallet_names,
                "num_wallets": num_wallets,
                "label_window": label_window,
                "epoch": epoch + 1,
                "val_loss": avg_val_loss,
                "val_intent_acc": val_intent_acc,
                "val_dir_acc": val_dir_acc,
            }, MODEL_DIR / "whale_behavior_lstm.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    # ── Test ──
    model.load_state_dict(torch.load(MODEL_DIR / "whale_behavior_lstm.pt", weights_only=False)["model_state"])
    model.eval()

    test_intent_correct = 0
    test_dir_correct = 0
    test_total = 0
    test_preds = []
    test_actuals = []

    with torch.no_grad():
        for seq, wid, intent, direction, magnitude in test_loader:
            seq, wid = seq.to(device), wid.to(device)
            intent = intent.to(device)
            direction = direction.to(device)

            intent_logits, dir_pred, mag_pred = model(seq, wid)

            pred_intent = intent_logits.argmax(dim=1)
            test_intent_correct += (pred_intent == intent.squeeze(-1)).sum().item()

            pred_dir = (dir_pred.squeeze(-1) > 0.5).float()
            actual_dir = (direction.squeeze(-1) > 0.5).float()
            test_dir_correct += (pred_dir == actual_dir).sum().item()

            test_total += intent.size(0)
            test_preds.extend(pred_intent.cpu().tolist())
            test_actuals.extend(intent.squeeze(-1).cpu().tolist())

    test_intent_acc = test_intent_correct / max(test_total, 1)
    test_dir_acc = test_dir_correct / max(test_total, 1)

    # Per-class accuracy
    from collections import defaultdict
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    for pred, actual in zip(test_preds, test_actuals):
        class_total[actual] += 1
        if pred == actual:
            class_correct[actual] += 1

    results = {
        "test_intent_acc": round(test_intent_acc * 100, 1),
        "test_dir_acc": round(test_dir_acc * 100, 1),
        "test_total": test_total,
        "best_val_loss": round(best_val_loss, 4),
        "epochs_trained": len(history["train_loss"]),
        "per_class": {
            "BUY_acc": round(class_correct[0] / max(class_total[0], 1) * 100, 1),
            "SELL_acc": round(class_correct[1] / max(class_total[1], 1) * 100, 1),
            "NEUTRAL_acc": round(class_correct[2] / max(class_total[2], 1) * 100, 1),
        },
        "wallet_names": wallet_names,
        "num_sequences": len(sequences),
    }

    logger.info("=" * 60)
    logger.info("TEST RESULTS:")
    logger.info("  Intent accuracy: %.1f%%", results["test_intent_acc"])
    logger.info("  Direction accuracy: %.1f%%", results["test_dir_acc"])
    logger.info("  Per-class: BUY=%.1f%% SELL=%.1f%% NEUTRAL=%.1f%%",
                results["per_class"]["BUY_acc"],
                results["per_class"]["SELL_acc"],
                results["per_class"]["NEUTRAL_acc"])
    logger.info("  Sequences: %d, Epochs: %d", results["num_sequences"], results["epochs_trained"])
    logger.info("=" * 60)

    # Save results
    with open(MODEL_DIR / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results
