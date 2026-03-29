"""
Whale Behavior Intent Predictor — Real-time inference.

Loads the trained LSTM model and runs prediction on the latest
wallet actions to produce a whale intent signal.

This is DISPLAY ONLY — not used for trade decisions.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from .sequence_model import (
    WhaleBehaviorLSTM,
    MODEL_DIR,
    LABELED_DIR,
    SEQ_LENGTH,
    INTENT_LABELS,
    action_to_features,
)

# Live data directory (unlabeled, continuously updated by collector)
LIVE_DIR = Path("data/whale_behavior/eth")

logger = logging.getLogger(__name__)

# Reverse intent map
IDX_TO_INTENT = {v: k for k, v in INTENT_LABELS.items()}
INTENT_DISPLAY = {
    "BUY_SIGNAL": "accumulating",
    "SELL_SIGNAL": "distributing",
    "NEUTRAL": "neutral",
}


class WhaleIntentPredictor:
    """
    Loads the trained whale behavior model and predicts current intent
    from the most recent wallet actions.
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or (MODEL_DIR / "whale_behavior_lstm.pt")
        self.model = None
        self.wallet_names = []
        # Predictor always runs on CPU for minimal memory footprint on server
        self.device = torch.device("cpu")
        self._loaded = False

    def _load(self) -> bool:
        """Load the trained model. Returns False if model doesn't exist."""
        if self._loaded:
            return True

        if not self.model_path.exists():
            logger.warning("Whale behavior model not found at %s", self.model_path)
            return False

        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            self.wallet_names = checkpoint.get("wallet_names", [])
            num_wallets = checkpoint.get("num_wallets", len(self.wallet_names))

            self.model = WhaleBehaviorLSTM(num_wallets=num_wallets).to(self.device)
            self.model.load_state_dict(checkpoint["model_state"])
            self.model.eval()
            self._loaded = True

            logger.info("Loaded whale behavior model (epoch %d, val_loss=%.4f, %d wallets)",
                        checkpoint.get("epoch", 0),
                        checkpoint.get("val_loss", 0),
                        num_wallets)
            return True
        except Exception as e:
            logger.error("Failed to load whale behavior model: %s", e)
            return False

    def _load_recent_actions(self, wallet_name: str, n: int = SEQ_LENGTH) -> list:
        """
        Load the N most recent actions for a wallet.
        
        Prefers LIVE data (data/whale_behavior/eth/) which is continuously
        updated by the collector service. Falls back to labeled data if
        live data doesn't exist.
        """
        # Try live data first (continuously updated by collector)
        live_file = LIVE_DIR / f"{wallet_name}.jsonl"
        if live_file.exists():
            actions = self._read_last_n_lines(live_file, n)
            if len(actions) >= n:
                return actions

        # Fallback to labeled data (static, from training)
        labeled_file = LABELED_DIR / f"{wallet_name}_labeled.jsonl"
        if labeled_file.exists():
            actions = self._read_last_n_lines(labeled_file, n)
            if len(actions) >= n:
                return actions

        return []

    @staticmethod
    def _read_last_n_lines(filepath: Path, n: int) -> list:
        """Efficiently read the last N JSON lines from a file."""
        actions = []
        try:
            # Read from end for efficiency on large files
            with open(filepath, 'rb') as f:
                f.seek(0, 2)  # End of file
                file_size = f.tell()
                
                # Read last chunk (estimate ~300 bytes per line)
                chunk_size = min(file_size, n * 500)
                f.seek(max(0, file_size - chunk_size))
                data = f.read().decode('utf-8', errors='replace')
            
            lines = data.strip().split('\n')
            # Take last N lines
            for line in lines[-n:]:
                line = line.strip()
                if not line:
                    continue
                try:
                    actions.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except Exception:
            pass
        return actions

    def predict_wallet(self, wallet_name: str) -> Optional[Dict]:
        """
        Predict the current intent for a specific wallet.

        Returns dict with intent, confidence, direction, or None if unavailable.
        """
        if not self._load():
            return None

        if wallet_name not in self.wallet_names:
            return None

        wallet_idx = self.wallet_names.index(wallet_name)
        actions = self._load_recent_actions(wallet_name)

        if len(actions) < SEQ_LENGTH:
            return None

        # Compute time gaps
        for i, action in enumerate(actions):
            if i > 0:
                time_gap = (action["timestamp"] - actions[i - 1]["timestamp"]) / 3600.0
                action["_time_gap_hours"] = max(time_gap, 0)
            else:
                action["_time_gap_hours"] = 0
            gas = action.get("gas_used", 21000) or 21000
            action["_gas_ratio"] = gas / 50000.0

        # Build feature sequence
        seq = np.array([action_to_features(a) for a in actions[-SEQ_LENGTH:]], dtype=np.float32)
        seq_tensor = torch.from_numpy(seq).unsqueeze(0).to(self.device)
        wid_tensor = torch.LongTensor([[wallet_idx]]).to(self.device)

        with torch.no_grad():
            intent_logits, dir_pred, mag_pred = self.model(seq_tensor, wid_tensor)

        # Softmax for probabilities
        probs = torch.softmax(intent_logits, dim=1).squeeze().cpu().numpy()
        pred_idx = int(probs.argmax())
        intent_label = IDX_TO_INTENT.get(pred_idx, "NEUTRAL")

        return {
            "wallet": wallet_name,
            "intent": INTENT_DISPLAY.get(intent_label, "neutral"),
            "intent_raw": intent_label,
            "confidence": float(probs[pred_idx]),
            "probs": {
                "BUY": float(probs[0]),
                "SELL": float(probs[1]),
                "NEUTRAL": float(probs[2]),
            },
            "direction": float(dir_pred.squeeze().cpu()),
            "magnitude": float(mag_pred.squeeze().cpu()),
        }

    def get_signal(self) -> Dict:
        """
        Get aggregate whale behavior signal across all tracked wallets.

        Returns:
            {
                "intent": "accumulating"|"distributing"|"neutral",
                "confidence": 0.0-1.0,
                "direction": 0.0-1.0 (P(bullish)),
                "active_wallets": int,
                "sell_confidence": 0.0-1.0,
                "buy_confidence": 0.0-1.0,
                "wallet_details": {name: prediction, ...},
            }
        """
        if not self._load():
            return {
                "intent": "unavailable",
                "confidence": 0.0,
                "direction": 0.5,
                "active_wallets": 0,
                "sell_confidence": 0.0,
                "buy_confidence": 0.0,
                "wallet_details": {},
            }

        details = {}
        for name in self.wallet_names:
            pred = self.predict_wallet(name)
            if pred is not None:
                details[name] = pred

        if not details:
            return {
                "intent": "no_data",
                "confidence": 0.0,
                "direction": 0.5,
                "active_wallets": 0,
                "sell_confidence": 0.0,
                "buy_confidence": 0.0,
                "wallet_details": {},
            }

        # Aggregate: weighted average by confidence
        total_conf = sum(d["confidence"] for d in details.values())
        if total_conf == 0:
            total_conf = 1.0

        avg_direction = sum(d["direction"] * d["confidence"] for d in details.values()) / total_conf
        avg_sell = sum(d["probs"]["SELL"] * d["confidence"] for d in details.values()) / total_conf
        avg_buy = sum(d["probs"]["BUY"] * d["confidence"] for d in details.values()) / total_conf

        # Determine overall intent
        if avg_sell > avg_buy and avg_sell > 0.4:
            intent = "distributing"
            confidence = avg_sell
        elif avg_buy > avg_sell and avg_buy > 0.4:
            intent = "accumulating"
            confidence = avg_buy
        else:
            intent = "neutral"
            confidence = max(avg_sell, avg_buy)

        return {
            "intent": intent,
            "confidence": round(confidence, 3),
            "direction": round(avg_direction, 3),
            "active_wallets": len(details),
            "sell_confidence": round(avg_sell, 3),
            "buy_confidence": round(avg_buy, 3),
            "wallet_details": details,
        }
