"""
Cross-Chain Whale Flow Analysis

Tracks capital rotation and whale activity across ETH/SOL/XRP chains.
Detects when whales are rotating between chains, accumulating stablecoins,
or showing unified directional bias.

Features:
1. ETH→SOL flow ratio (capital rotation detection)
2. Total stablecoin flow (USDT/USDC across all chains)
3. Cross-chain whale consensus (% of chains showing same signal)
4. Chain dominance shift (which chain is attracting capital)

Usage:
    analyzer = CrossChainWhaleFlowAnalyzer()
    features = analyzer.compute_cross_chain_features()
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime, timedelta

from src.features.whale_pattern_predictor import WhalePatternPredictor

logger = logging.getLogger(__name__)


class CrossChainWhaleFlowAnalyzer:
    """
    Analyzes whale flow patterns across multiple blockchain networks.

    Cross-chain patterns reveal capital rotation, risk-on/risk-off behavior,
    and institutional positioning that single-chain analysis misses.
    """

    def __init__(self):
        self.whale_predictor = WhalePatternPredictor()
        logger.info("🌐 CrossChainWhaleFlowAnalyzer initialized")

    def compute_cross_chain_features(self) -> Dict[str, float]:
        """
        Compute cross-chain whale flow features.

        Returns dict with:
        - eth_to_sol_flow_ratio: float (-1 to +1, negative = ETH→SOL rotation)
        - total_stablecoin_flow: float (aggregate USD flow across chains)
        - cross_chain_consensus: float (0 to 1, % of chains agreeing)
        - chain_dominance_eth: float (0 to 1, ETH's share of whale activity)
        - chain_dominance_sol: float (0 to 1, SOL's share of whale activity)
        - chain_dominance_xrp: float (0 to 1, XRP's share of whale activity)
        - unified_signal: float (-1 to +1, consensus-weighted signal)
        """
        # Get whale signals for each chain
        eth_signal = self.whale_predictor.get_signal("ETHUSDT")
        sol_signal = self.whale_predictor.get_signal("SOLUSDT")
        xrp_signal = self.whale_predictor.get_signal("XRPUSDT")

        # Extract signal values
        eth_sig = eth_signal.get('signal', 0.0)
        eth_conf = eth_signal.get('confidence', 0.0)

        sol_sig = sol_signal.get('signal', 0.0)
        sol_conf = sol_signal.get('confidence', 0.0)

        xrp_sig = xrp_signal.get('signal', 0.0)
        xrp_conf = xrp_signal.get('confidence', 0.0)

        # 1. ETH→SOL Flow Ratio
        # Negative = capital flowing from ETH to SOL
        # Positive = capital flowing from SOL to ETH
        eth_to_sol_flow = eth_sig - sol_sig  # If ETH selling and SOL buying → negative
        eth_to_sol_flow_ratio = float(np.clip(eth_to_sol_flow, -1.0, 1.0))

        # 2. Total Stablecoin Flow (proxy using overall directional bias)
        # Positive = whales accumulating stables (risk-off)
        # Negative = whales deploying stables (risk-on)
        # Use inverse of weighted signal: if whales buying crypto → stables flowing out
        weighted_crypto_signal = (
            eth_sig * eth_conf +
            sol_sig * sol_conf +
            xrp_sig * xrp_conf
        ) / max(eth_conf + sol_conf + xrp_conf, 0.01)

        stablecoin_flow_proxy = -weighted_crypto_signal  # Inverse relationship
        total_stablecoin_flow = float(np.clip(stablecoin_flow_proxy, -1.0, 1.0))

        # 3. Cross-Chain Consensus
        # How many chains agree on direction?
        signals = [eth_sig, sol_sig, xrp_sig]
        confidences = [eth_conf, sol_conf, xrp_conf]

        # Remove low-confidence signals
        valid_signals = [s for s, c in zip(signals, confidences) if c > 0.1]

        if len(valid_signals) >= 2:
            # Check if majority agree on direction
            positive_count = sum(1 for s in valid_signals if s > 0.1)
            negative_count = sum(1 for s in valid_signals if s < -0.1)
            neutral_count = len(valid_signals) - positive_count - negative_count

            max_agreement = max(positive_count, negative_count, neutral_count)
            cross_chain_consensus = max_agreement / len(valid_signals)
        else:
            cross_chain_consensus = 0.0

        # 4. Chain Dominance (based on whale activity confidence)
        total_conf = eth_conf + sol_conf + xrp_conf
        if total_conf > 0:
            chain_dominance_eth = eth_conf / total_conf
            chain_dominance_sol = sol_conf / total_conf
            chain_dominance_xrp = xrp_conf / total_conf
        else:
            # Default equal weighting
            chain_dominance_eth = 0.33
            chain_dominance_sol = 0.33
            chain_dominance_xrp = 0.34

        # 5. Unified Signal (consensus-weighted)
        # If chains agree strongly, amplify signal
        # If chains disagree, dampen signal
        unified_signal = weighted_crypto_signal * cross_chain_consensus
        unified_signal = float(np.clip(unified_signal, -1.0, 1.0))

        features = {
            'eth_to_sol_flow_ratio': eth_to_sol_flow_ratio,
            'total_stablecoin_flow': total_stablecoin_flow,
            'cross_chain_consensus': cross_chain_consensus,
            'chain_dominance_eth': chain_dominance_eth,
            'chain_dominance_sol': chain_dominance_sol,
            'chain_dominance_xrp': chain_dominance_xrp,
            'unified_signal': unified_signal,

            # Debug info
            '_eth_signal': eth_sig,
            '_sol_signal': sol_sig,
            '_xrp_signal': xrp_sig,
            '_eth_conf': eth_conf,
            '_sol_conf': sol_conf,
            '_xrp_conf': xrp_conf,
        }

        logger.debug(
            f"🌐 Cross-chain features: "
            f"Consensus={cross_chain_consensus:.2f}, "
            f"Unified={unified_signal:+.2f}, "
            f"ETH/SOL Ratio={eth_to_sol_flow_ratio:+.2f}"
        )

        return features

    def get_rotation_signal(self) -> Dict[str, any]:
        """
        Detect capital rotation patterns.

        Returns:
            direction: str (eth_to_sol, sol_to_eth, eth_to_xrp, etc., or neutral)
            strength: float (0 to 1)
            recommendation: str (human-readable)
        """
        features = self.compute_cross_chain_features()

        eth_sol_ratio = features['eth_to_sol_flow_ratio']
        consensus = features['cross_chain_consensus']

        # Determine rotation direction
        if abs(eth_sol_ratio) < 0.2:
            direction = "neutral"
            strength = 0.0
            recommendation = "No clear cross-chain rotation detected"
        elif eth_sol_ratio < -0.3 and consensus > 0.5:
            direction = "eth_to_sol"
            strength = min(abs(eth_sol_ratio) * consensus, 1.0)
            recommendation = "Capital rotating FROM Ethereum TO Solana - Consider SOL long"
        elif eth_sol_ratio > 0.3 and consensus > 0.5:
            direction = "sol_to_eth"
            strength = min(abs(eth_sol_ratio) * consensus, 1.0)
            recommendation = "Capital rotating FROM Solana TO Ethereum - Consider ETH long"
        else:
            direction = "mixed"
            strength = 0.0
            recommendation = "Mixed signals - Low confidence rotation"

        return {
            'direction': direction,
            'strength': strength,
            'recommendation': recommendation,
            'consensus': consensus,
            'eth_sol_ratio': eth_sol_ratio,
        }

    def get_risk_sentiment(self) -> Dict[str, any]:
        """
        Determine risk-on/risk-off sentiment from stablecoin flows.

        Returns:
            sentiment: str (risk_on, risk_off, neutral)
            strength: float (0 to 1)
            recommendation: str
        """
        features = self.compute_cross_chain_features()

        stablecoin_flow = features['total_stablecoin_flow']
        consensus = features['cross_chain_consensus']

        if stablecoin_flow > 0.3 and consensus > 0.5:
            sentiment = "risk_off"
            strength = min(stablecoin_flow * consensus, 1.0)
            recommendation = "Whales accumulating stablecoins - RISK OFF - Consider reducing exposure"
        elif stablecoin_flow < -0.3 and consensus > 0.5:
            sentiment = "risk_on"
            strength = min(abs(stablecoin_flow) * consensus, 1.0)
            recommendation = "Whales deploying stablecoins - RISK ON - Consider increasing exposure"
        else:
            sentiment = "neutral"
            strength = 0.0
            recommendation = "Neutral stablecoin flow - No clear risk bias"

        return {
            'sentiment': sentiment,
            'strength': strength,
            'recommendation': recommendation,
            'stablecoin_flow': stablecoin_flow,
            'consensus': consensus,
        }


# Singleton instance
_analyzer_instance: Optional[CrossChainWhaleFlowAnalyzer] = None


def get_cross_chain_analyzer() -> CrossChainWhaleFlowAnalyzer:
    """Get singleton instance of cross-chain analyzer."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = CrossChainWhaleFlowAnalyzer()
    return _analyzer_instance
