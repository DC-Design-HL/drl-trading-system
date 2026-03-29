"""
Confidence Calibrator v2 — Post-hoc calibration for HTF agent confidence.

Problem: PPO's raw action probabilities (softmax output) are NOT calibrated.
  - 95% confidence ≠ 95% chance of winning
  - High confidence often means overfitting to training patterns

Solution: Temperature scaling + historical win-rate binning.
  1. Temperature scaling: softens the softmax so extreme probabilities are dampened
  2. Win-rate calibration: maps raw confidence → observed win rate from shadow data
  3. Regime adjustment: scales confidence down in ranging/low-ADX markets

Usage:
    calibrator = ConfidenceCalibrator()
    calibrator.load("data/models/htf/calibration.json")
    
    raw_conf = agent.predict(obs)  # 0.95
    calibrated = calibrator.calibrate(raw_conf, regime="ranging", adx=15)  # 0.52

Training the calibrator:
    python scripts/train_confidence_calibrator.py --trades-file logs/htf_pending_alerts.jsonl
"""

import json
import logging
import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

CALIBRATION_DIR = Path("data/models/htf")


class ConfidenceCalibrator:
    """
    Post-hoc confidence calibration using temperature scaling
    and empirical win-rate mapping.
    """

    def __init__(self):
        self.temperature: float = 1.0  # >1 softens, <1 sharpens
        self.regime_discounts: Dict[str, float] = {
            "RANGING": 0.7,
            "LOW_VOLATILITY": 0.8,
            "TRENDING_UP": 1.0,
            "TRENDING_DOWN": 1.0,
            "HIGH_VOLATILITY": 0.85,
            "unknown": 0.9,
        }
        self.win_rate_bins: Dict[str, float] = {}  # "0.90-1.00" → 0.35 (actual WR)
        self.adx_threshold: float = 20.0  # Below this, discount confidence
        self.adx_discount: float = 0.75
        self._loaded = False

    def calibrate(
        self,
        raw_confidence: float,
        action_probs: Optional[np.ndarray] = None,
        regime: str = "unknown",
        adx: float = 0.0,
    ) -> float:
        """
        Calibrate raw model confidence into a meaningful probability.

        Args:
            raw_confidence: Max action prob from PPO (0-1)
            action_probs: Full action probability vector (if available)
            regime: Current market regime string
            adx: Current ADX value

        Returns:
            Calibrated confidence (0-1) that better reflects actual win probability
        """
        # Step 1: Temperature scaling on raw probs
        if action_probs is not None and self.temperature != 1.0:
            # Apply temperature to logits
            log_probs = np.log(np.clip(action_probs, 1e-8, 1.0))
            scaled = log_probs / self.temperature
            scaled -= np.max(scaled)  # numerical stability
            exp_scaled = np.exp(scaled)
            calibrated_probs = exp_scaled / exp_scaled.sum()
            conf = float(np.max(calibrated_probs))
        else:
            conf = raw_confidence

        # Step 2: Win-rate bin mapping (if trained)
        if self.win_rate_bins:
            bin_key = self._get_bin_key(raw_confidence)
            if bin_key in self.win_rate_bins:
                empirical_wr = self.win_rate_bins[bin_key]
                # Blend: 70% empirical, 30% temperature-scaled
                conf = 0.7 * empirical_wr + 0.3 * conf

        # Step 3: Regime discount
        discount = self.regime_discounts.get(regime, 0.9)
        conf *= discount

        # Step 4: ADX discount (low ADX = choppy, discount confidence)
        if adx > 0 and adx < self.adx_threshold:
            adx_factor = self.adx_discount + (1 - self.adx_discount) * (adx / self.adx_threshold)
            conf *= adx_factor

        return max(0.01, min(0.99, conf))

    def _get_bin_key(self, confidence: float) -> str:
        """Map confidence to a bin key."""
        if confidence >= 0.90:
            return "0.90-1.00"
        elif confidence >= 0.80:
            return "0.80-0.90"
        elif confidence >= 0.70:
            return "0.70-0.80"
        elif confidence >= 0.60:
            return "0.60-0.70"
        elif confidence >= 0.50:
            return "0.50-0.60"
        else:
            return "0.00-0.50"

    def train_from_trades(self, trades_file: str = "logs/htf_pending_alerts.jsonl") -> Dict:
        """
        Train the calibrator from historical trade data.
        
        Computes:
        1. Optimal temperature via grid search
        2. Win-rate per confidence bin
        3. Win-rate per regime

        Returns training results dict.
        """
        # Load completed trades
        records = []
        with open(trades_file) as f:
            for line in f:
                try:
                    r = json.loads(line.strip())
                    if r.get("strategy") != "htf":
                        continue
                    records.append(r)
                except:
                    continue

        # Match OPEN→CLOSE pairs
        open_trades = {}
        pairs = []
        for r in records:
            trade = r.get("trade", {})
            action = trade.get("action", "")
            symbol = trade.get("symbol", "?")
            if "OPEN" in action and "PARTIAL" not in action:
                open_trades[symbol] = r
            elif "CLOSE" in action and "PARTIAL" not in action and symbol in open_trades:
                open_r = open_trades.pop(symbol)
                open_trade = open_r.get("trade", {})
                pnl = trade.get("pnl", 0) or 0
                conf = open_trade.get("confidence", 0)
                signals = open_r.get("signals", {})
                regime = signals.get("regime", {}).get("type",
                         signals.get("regime", {}).get("state", "unknown"))
                adx = signals.get("regime", {}).get("adx", 0) or 0

                pairs.append({
                    "confidence": conf,
                    "pnl": pnl,
                    "won": pnl > 0,
                    "regime": regime,
                    "adx": adx,
                    "symbol": symbol,
                })

        if len(pairs) < 10:
            logger.warning("Not enough trades (%d) to train calibrator", len(pairs))
            return {"error": "insufficient_data", "trades": len(pairs)}

        # Compute win rate per confidence bin
        from collections import defaultdict
        bin_wins = defaultdict(int)
        bin_total = defaultdict(int)

        for p in pairs:
            bk = self._get_bin_key(p["confidence"])
            bin_total[bk] += 1
            if p["won"]:
                bin_wins[bk] += 1

        self.win_rate_bins = {}
        for bk in bin_total:
            wr = bin_wins[bk] / bin_total[bk]
            self.win_rate_bins[bk] = round(wr, 3)

        # Compute win rate per regime
        regime_wins = defaultdict(int)
        regime_total = defaultdict(int)
        for p in pairs:
            regime_total[p["regime"]] += 1
            if p["won"]:
                regime_wins[p["regime"]] += 1

        for regime in regime_total:
            wr = regime_wins[regime] / regime_total[regime]
            # Use empirical WR as discount (relative to best regime)
            self.regime_discounts[regime] = round(wr / max(max(
                regime_wins[r] / regime_total[r] for r in regime_total), 0.01), 3)

        # Grid search for optimal temperature
        best_temp = 1.0
        best_score = float("inf")  # minimize calibration error

        for temp in [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
            self.temperature = temp
            # Expected calibration error
            ece = 0
            for p in pairs:
                calibrated = self.calibrate(p["confidence"], regime=p["regime"], adx=p["adx"])
                actual = 1.0 if p["won"] else 0.0
                ece += abs(calibrated - actual)
            ece /= len(pairs)
            if ece < best_score:
                best_score = ece
                best_temp = temp

        self.temperature = best_temp

        results = {
            "trades_analyzed": len(pairs),
            "temperature": self.temperature,
            "win_rate_bins": self.win_rate_bins,
            "regime_discounts": self.regime_discounts,
            "calibration_error": round(best_score, 4),
        }

        logger.info("Calibrator trained: temp=%.1f, bins=%s", self.temperature, self.win_rate_bins)
        return results

    def save(self, path: Optional[str] = None):
        """Save calibration parameters."""
        path = path or str(CALIBRATION_DIR / "calibration_v2.json")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "temperature": self.temperature,
            "win_rate_bins": self.win_rate_bins,
            "regime_discounts": self.regime_discounts,
            "adx_threshold": self.adx_threshold,
            "adx_discount": self.adx_discount,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved calibration to %s", path)

    def load(self, path: Optional[str] = None) -> bool:
        """Load calibration parameters. Returns False if not found."""
        path = path or str(CALIBRATION_DIR / "calibration_v2.json")
        if not Path(path).exists():
            return False
        try:
            with open(path) as f:
                data = json.load(f)
            self.temperature = data.get("temperature", 1.0)
            self.win_rate_bins = data.get("win_rate_bins", {})
            self.regime_discounts = data.get("regime_discounts", self.regime_discounts)
            self.adx_threshold = data.get("adx_threshold", 20.0)
            self.adx_discount = data.get("adx_discount", 0.75)
            self._loaded = True
            logger.info("Loaded calibration: temp=%.1f, bins=%d", self.temperature, len(self.win_rate_bins))
            return True
        except Exception as e:
            logger.error("Failed to load calibration: %s", e)
            return False
