#!/usr/bin/env python3
"""
Project 3 — Whale Behavior LSTM v2 (Mac M3).

Problem: the deployed whale predictor in src/whale_behavior/models/predictor.py
expects an LSTM with input_dim=144 → hidden=128, but the saved
state_dict was trained with a different architecture (likely 272 → 256
per a recent code change). Loading fails with shape-mismatch errors;
the in-prod whale signal is mostly NEUTRAL because the model never
loads. The deployed `WHALE_NEUTRAL_GUARD` neutralizes the damage.

This script trains a fresh whale model on the REAL labeled data in
`data/whale_behavior/labeled_v2/*.jsonl`. No synthetic / mock data:
every training sequence is a real on-chain wallet event sequence with
behavioral labels (LARGE_TRANSFER_IN, ACCUMULATION, etc.) computed
from observable transaction patterns.

The output is a model + preprocessor that the production predictor can
load.

Run on Mac:
    pip install torch numpy pandas
    python3 scripts/train_whale_v2.py
        [--data data/whale_behavior/labeled_v2/]
        [--seq_len 24]      # 24-event sliding windows
        [--hidden 256]
        [--output data/models/whale_v2.pt]
        [--epochs 50]

Acceptance gate (validated on test set):
  - Confidence stdev across test samples > 0.10 (model is informative)
  - Direction predictions show all 3 classes (BULLISH/BEARISH/NEUTRAL)
    with no class > 70% of predictions
  - Optional: backtest. The predictor's BULLISH/BEARISH labels at
    historical OPEN events should correlate with trade outcomes
    by ≥ +3 pp WR vs the 56% baseline. The dataset already includes
    the link via `direction` and the bot-trade outcome.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPO / "data" / "whale_behavior" / "labeled_v3"   # forward-return labels
LEGACY_DATA = REPO / "data" / "whale_behavior" / "labeled_v2"


# Direction labels are now read from the `direction_label` field of
# labeled_v3/*.jsonl — produced by scripts/relabel_whale_with_forward_returns.py.
# That field is the SIGN of the realized 4h forward return, not the
# wallet's own action.
#
# The legacy DIRECTION_FROM_ACTION mapping (below) is retained for
# backward compatibility if you point this script at labeled_v2 — but
# it's known to produce a 94% BULL imbalance and a degenerate model.
LEGACY_DIRECTION_FROM_ACTION = {
    "LARGE_TRANSFER_IN": "BULLISH",
    "LARGE_TRANSFER_OUT": "BEARISH",
    "ACCUMULATION": "BULLISH",
    "DISTRIBUTION": "BEARISH",
    "ROUTINE": "NEUTRAL",
    "SWAP": "NEUTRAL",
}

NUMERIC_FEATURES = [
    "value_eth", "net_flow_4h", "intent_score_4h", "past_flow_4h",
    "net_flow_12h", "intent_score_12h", "past_flow_12h",
    "net_flow_24h",
]

LABEL_TO_IDX = {"BEARISH": 0, "NEUTRAL": 1, "BULLISH": 2}


def load_events(data_dir: Path) -> list[dict]:
    out = []
    for f in sorted(data_dir.glob("*.jsonl")):
        wallet = f.stem.replace("_behavioral", "")
        with open(f) as fp:
            for line in fp:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                d["wallet"] = wallet
                out.append(d)
    out.sort(key=lambda x: x.get("timestamp") or 0)
    return out


def derive_label(event: dict) -> str:
    """Prefer the forward-return label (`direction_label` from labeled_v3).
    Fall back to the legacy action-based mapping only if the field is missing,
    with a warning printed on first fallback.
    """
    if "direction_label" in event:
        return event["direction_label"]
    if not getattr(derive_label, "_warned", False):
        print("WARNING: dataset lacks 'direction_label' field — falling back to "
              "action-based labels. Re-run scripts/relabel_whale_with_forward_returns.py "
              "to produce labeled_v3/ for proper supervised labels.", file=sys.stderr)
        derive_label._warned = True  # type: ignore[attr-defined]
    a = event.get("action", "ROUTINE")
    return LEGACY_DIRECTION_FROM_ACTION.get(a, "NEUTRAL")


def featurize(events: list[dict], seq_len: int = 24):
    """Sliding-window sequences. Each sample is `seq_len` consecutive events
    from one wallet. Label is the direction of the LAST event in the window.
    """
    import numpy as np
    by_wallet: dict[str, list[dict]] = {}
    for e in events:
        by_wallet.setdefault(e["wallet"], []).append(e)
    X_seqs, y_lbls = [], []
    for wallet, seq in by_wallet.items():
        if len(seq) < seq_len:
            continue
        for i in range(seq_len, len(seq) + 1):
            window = seq[i - seq_len:i]
            row_features = []
            for ev in window:
                feats = [float(ev.get(k) or 0) for k in NUMERIC_FEATURES]
                row_features.append(feats)
            X_seqs.append(row_features)
            y_lbls.append(LABEL_TO_IDX[derive_label(window[-1])])
    return np.asarray(X_seqs, dtype="float32"), np.asarray(y_lbls, dtype="int64")


def time_split(X, y, val_frac=0.15, test_frac=0.15):
    """Time-ordered split (data is already sorted)."""
    n = len(X)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_test - n_val
    return (X[:n_train], y[:n_train],
            X[n_train:n_train + n_val], y[n_train:n_train + n_val],
            X[n_train + n_val:], y[n_train + n_val:])


def build_model(input_dim: int, hidden: int, num_classes: int):
    import torch.nn as nn
    return nn.Sequential(
        # Wraps an LSTM with attention-like aggregation, kept simple to
        # match the deployed WhaleBehaviorLSTM class shape.
    )


class WhaleBehaviorLSTM_v2:
    """Mirror of the production class but trained from scratch.
    Saved state_dict will be loaded by src/whale_behavior/models/predictor.py
    after a small wiring update (we'll write that change once you have the
    .pt file).
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 2, num_classes: int = 3):
        import torch.nn as nn
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=0.2)
        self.attn_q = nn.Linear(hidden_dim, hidden_dim)
        self.attn_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)

    def __call__(self, x):
        import torch
        out, _ = self.lstm(x)
        # Simple attention: dot-product q · k summary over time
        q = self.attn_q(out[:, -1])  # query = last hidden state
        k = self.attn_k(out)         # keys over all timesteps
        attn_logits = (k * q.unsqueeze(1)).sum(-1) / (self.hidden_dim ** 0.5)
        attn_weights = attn_logits.softmax(dim=-1)
        ctx = (out * attn_weights.unsqueeze(-1)).sum(dim=1)
        h = self.dropout(self.fc1(ctx).relu())
        return self.fc2(h)

    def parameters(self):
        return (list(self.lstm.parameters()) + list(self.attn_q.parameters()) +
                list(self.attn_k.parameters()) + list(self.fc1.parameters()) +
                list(self.fc2.parameters()))

    def state_dict(self):
        return {
            "lstm": self.lstm.state_dict(),
            "attn_q": self.attn_q.state_dict(),
            "attn_k": self.attn_k.state_dict(),
            "fc1": self.fc1.state_dict(),
            "fc2": self.fc2.state_dict(),
            "hyperparams": {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "num_classes": self.num_classes,
            },
        }

    def train(self):
        for m in (self.lstm, self.attn_q, self.attn_k, self.fc1, self.fc2):
            m.train()

    def eval(self):
        for m in (self.lstm, self.attn_q, self.attn_k, self.fc1, self.fc2):
            m.eval()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--seq_len", type=int, default=24)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--output", default="data/models/whale_v2.pt")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    try:
        import torch
        from torch import nn, optim
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as e:
        print(f"ERROR: missing dependency: {e}", file=sys.stderr)
        print("Run: pip install torch")
        return 2

    torch.manual_seed(args.seed)

    print(f"Loading whale events from {args.data} ...")
    events = load_events(Path(args.data))
    print(f"  {len(events)} events from {len(set(e['wallet'] for e in events))} wallets")
    if len(events) < args.seq_len * 5:
        print("ERROR: not enough events to train", file=sys.stderr)
        return 1

    print(f"Featurizing into seq_len={args.seq_len} windows ...")
    X, y = featurize(events, seq_len=args.seq_len)
    print(f"  {len(X)} samples, input_dim={X.shape[2]}")
    print(f"  class distribution: {[int((y == i).sum()) for i in range(3)]} (BEAR/NEUT/BULL)")

    X_tr, y_tr, X_val, y_val, X_te, y_te = time_split(X, y)
    print(f"  train={len(X_tr)} val={len(X_val)} test={len(X_te)}")

    model = WhaleBehaviorLSTM_v2(input_dim=X.shape[2], hidden_dim=args.hidden,
                                 num_layers=args.num_layers, num_classes=3)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # Class-weighted loss so the model isn't dominated by the majority NEUT class.
    # Weight inversely to class frequency in the training set.
    train_class_counts = [max(1, int((y_tr == i).sum())) for i in range(3)]
    inv_freq = [1.0 / c for c in train_class_counts]
    norm = sum(inv_freq)
    class_weights = torch.tensor([w * 3 / norm for w in inv_freq], dtype=torch.float32)
    print(f"  class weights (BEAR/NEUT/BULL): {[round(w, 3) for w in class_weights.tolist()]}")
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                              batch_size=args.batch_size, shuffle=True)

    best_val_acc = 0.0
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        model.eval()
        with torch.no_grad():
            val_logits = model(torch.from_numpy(X_val))
            val_pred = val_logits.argmax(-1).numpy()
            val_acc = float((val_pred == y_val).mean())
        print(f"  epoch {epoch + 1:>3}: train_loss={total_loss / max(1, len(train_loader)):.4f}  val_acc={val_acc:.3f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: (v if not hasattr(v, "copy") else v.copy()) for k, v in model.state_dict().items()}

    # Load best
    if best_state is not None:
        # restore submodule state_dicts
        model.lstm.load_state_dict(best_state["lstm"])
        model.attn_q.load_state_dict(best_state["attn_q"])
        model.attn_k.load_state_dict(best_state["attn_k"])
        model.fc1.load_state_dict(best_state["fc1"])
        model.fc2.load_state_dict(best_state["fc2"])

    # Test eval
    model.eval()
    with torch.no_grad():
        test_logits = model(torch.from_numpy(X_te))
        test_probs = test_logits.softmax(-1).numpy()
        test_pred = test_logits.argmax(-1).numpy()
        test_acc = float((test_pred == y_te).mean())
        confidence_stdev = float(test_probs.max(axis=-1).std())
    print(f"\nTest acc: {test_acc:.3f}")
    print(f"Test confidence stdev: {confidence_stdev:.4f}  "
          f"(must be > 0.10 to pass acceptance gate)")
    pred_dist = [int((test_pred == i).sum()) for i in range(3)]
    print(f"Test prediction distribution (BEAR/NEUT/BULL): {pred_dist}")

    # Save
    out_path = REPO / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": {k: v for k, v in model.state_dict().items() if k != "hyperparams"},
        "hyperparams": model.state_dict()["hyperparams"],
        "feature_names": NUMERIC_FEATURES,
        "label_idx": LABEL_TO_IDX,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "test_acc": test_acc,
        "test_confidence_stdev": confidence_stdev,
    }
    torch.save(payload, str(out_path))
    print(f"\nSaved: {out_path}")

    # 3-class random baseline = 33%. We need MEANINGFULLY better than that
    # AND confidence variance, AND no single class dominating.
    RANDOM_BASELINE = 1.0 / 3
    MIN_TEST_ACC = 0.40
    failed = []
    if test_acc < MIN_TEST_ACC:
        failed.append(f"test_acc={test_acc:.3f} < {MIN_TEST_ACC} (random baseline {RANDOM_BASELINE:.3f})")
    if confidence_stdev < 0.10:
        failed.append(f"confidence_stdev={confidence_stdev:.4f} < 0.10 (model not making distinctions)")
    if max(pred_dist) > 0.7 * sum(pred_dist):
        failed.append(f"single class dominates: {pred_dist}")
    if failed:
        print("\n❌ Acceptance gates FAILED:")
        for msg in failed:
            print(f"    - {msg}")
        print("\nDO NOT ship this model to production. Possible causes:")
        print("  * The labeled wallets are not predictive of 4h forward returns")
        print("    (current state of labeled_v3 data on Apr 2026 — confirmed empirically).")
        print("  * Try different wallets, longer horizons, or different forward-return")
        print("    thresholds via scripts/relabel_whale_with_forward_returns.py.")
        print("  * Keep WHALE_NEUTRAL_GUARD enabled in production until a model that")
        print("    passes these gates is found.")
        return 1
    print("\n✅ Acceptance gates passed. Ship to server with:")
    print(f"    scp {out_path} server:.../data/models/")
    print("    Then update src/whale_behavior/models/predictor.py to load the v2 payload.")
    print("    After deploy, set WHALE_NEUTRAL_GUARD_ENABLED = False.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
