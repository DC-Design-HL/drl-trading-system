"""
Behavioral Intent Labeler

Labels whale actions based on TRANSACTION PATTERNS (what the whale is doing)
rather than price outcomes (what the market does after).

This fixes the fundamental backward-causality problem in the price labeler:
- Old: "price went up 2% after tx → BUY_SIGNAL" (correlation, not causation)
- New: "whale accumulated net +500 ETH over 24h → ACCUMULATING" (actual behavior)

Labels:
  ACCUMULATING: Net inflow of ETH (whale is building a position)
  DISTRIBUTING: Net outflow of ETH (whale is reducing exposure)
  NEUTRAL: No significant directional flow

The model learns to recognize behavioral patterns, not predict price.
"""

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

LABELED_DIR = Path("data/whale_behavior/labeled_v2")

# ── Wallet type classification ────────────────────────────────────────

# Exchange wallets are EXCLUDED from training — their transfers are
# operational (user deposits/withdrawals), not investment intent.
EXCHANGE_WALLETS = {
    "binance_hot_wallet", "binance_cold_wallet", "binance_reserve",
    "binance_cold_2", "coinbase_institutional", "robinhood",
    "kraken_deposit",
}

# These are real whales / smart money — keep for training
WHALE_WALLETS = {
    "smart_money_whale_1", "galaxy_digital", "jump_trading",
    "eth_2.0_deposit_contract", "wintermute_fast_hot",
    "justin_sun_(accumulator)", "ftx_bankruptcy_estate",
    "arbitrum_bridge",
}


# ── Behavioral labeling ──────────────────────────────────────────────

# Flow windows for computing net flow
FLOW_WINDOWS_HOURS = [4, 12, 24]

# Thresholds for labeling (in ETH)
# These vary by wallet type — large institutional wallets need higher thresholds
FLOW_THRESHOLDS = {
    "default": {
        "accumulating": 10.0,    # Net inflow > 10 ETH in window
        "distributing": -10.0,   # Net outflow > 10 ETH in window
    },
    "large_institution": {
        "accumulating": 100.0,   # Big wallets need bigger moves
        "distributing": -100.0,
    },
}

# Wallets that need larger thresholds
LARGE_INSTITUTION_WALLETS = {
    "eth_2.0_deposit_contract",  # Massive staking flows
}

# Action weights — some actions signal intent more strongly
ACTION_INTENT_WEIGHTS = {
    # Strong accumulation signals
    "EXCHANGE_WITHDRAWAL": 1.5,     # Pulling from exchange = accumulating
    "DEX_SWAP": 1.2,                # Active trading
    "STAKING_DEPOSIT": 1.3,         # Long-term bullish
    "TOKEN_FROM_EXCHANGE": 1.3,     # Pulling tokens from exchange
    
    # Strong distribution signals  
    "EXCHANGE_DEPOSIT": -1.5,       # Sending to exchange = preparing to sell
    "TOKEN_TO_EXCHANGE": -1.3,      # Sending tokens to exchange
    "STAKING_WITHDRAWAL": -1.2,     # Pulling from staking = reducing commitment
    
    # Moderate signals
    "LARGE_TRANSFER_OUT": -0.8,     # Could be moving to cold storage or selling
    "LARGE_TRANSFER_IN": 0.8,       # Receiving large amount
    
    # Weak/neutral signals
    "CONTRACT_CALL": 0.0,           # Could be anything
    "CONTRACT_RECEIVED": 0.0,
    "TOKEN_TRANSFER_OUT": -0.3,
    "TOKEN_TRANSFER_IN": 0.3,
    "DEX_INTERACTION": 0.1,
    "DEX_RECEIVED": 0.2,
    "UNKNOWN": 0.0,
}


def compute_net_flow(actions: List[Dict], window_hours: int, 
                     end_idx: int) -> Tuple[float, float]:
    """
    Compute net ETH flow and weighted intent score for a time window
    ending at end_idx.
    
    Returns (net_flow_eth, intent_score).
    """
    if end_idx < 0 or end_idx >= len(actions):
        return 0.0, 0.0
    
    end_ts = actions[end_idx]["timestamp"]
    start_ts = end_ts - window_hours * 3600
    
    net_flow = 0.0
    intent_score = 0.0
    tx_count = 0
    
    for i in range(end_idx, -1, -1):
        if actions[i]["timestamp"] < start_ts:
            break
        
        action = actions[i]
        value_eth = abs(action.get("value_eth", 0))
        direction = action.get("direction", "")
        action_type = action.get("action", "UNKNOWN")
        
        # Net flow (positive = inflow, negative = outflow)
        if direction == "in":
            net_flow += value_eth
        elif direction == "out":
            net_flow -= value_eth
        
        # Weighted intent score
        weight = ACTION_INTENT_WEIGHTS.get(action_type, 0.0)
        # Scale by value — larger transactions count more
        value_weight = math.log(1 + value_eth)
        intent_score += weight * value_weight
        tx_count += 1
    
    # Normalize intent score by number of transactions
    if tx_count > 0:
        intent_score /= math.sqrt(tx_count)  # sqrt to not over-penalize active wallets
    
    return net_flow, intent_score


def label_action(actions: List[Dict], idx: int, 
                 wallet_name: str) -> Dict:
    """
    Label a single action based on surrounding behavioral context.
    
    Uses forward-looking flow (what the whale does in the NEXT N hours)
    to determine if they're accumulating or distributing.
    
    This is still "future" information, but it's the whale's OWN future
    actions — not market price. The model learns to recognize the BEGINNING
    of accumulation/distribution patterns.
    """
    action = dict(actions[idx])
    
    # Determine threshold set
    if wallet_name in LARGE_INSTITUTION_WALLETS:
        thresholds = FLOW_THRESHOLDS["large_institution"]
    else:
        thresholds = FLOW_THRESHOLDS["default"]
    
    # Compute behavioral features for multiple windows
    best_label = "NEUTRAL"
    best_confidence = 0.0
    
    for window_h in FLOW_WINDOWS_HOURS:
        # Look at PAST flow (what whale has been doing)
        past_flow, past_intent = compute_net_flow(actions, window_h, idx)
        
        # Look at FUTURE flow (what whale will do — this is the label target)
        # Find the action index window_h hours ahead
        future_end = idx
        target_ts = actions[idx]["timestamp"] + window_h * 3600
        for j in range(idx + 1, len(actions)):
            if actions[j]["timestamp"] > target_ts:
                break
            future_end = j
        
        future_flow, future_intent = compute_net_flow(
            actions, window_h, future_end
        ) if future_end > idx else (0.0, 0.0)
        
        # Subtract current position to get only future actions
        if future_end > idx:
            # Recompute for only future window
            future_flow_clean = 0.0
            future_intent_clean = 0.0
            count = 0
            for j in range(idx + 1, future_end + 1):
                a = actions[j]
                v = abs(a.get("value_eth", 0))
                d = a.get("direction", "")
                at = a.get("action", "UNKNOWN")
                
                if d == "in":
                    future_flow_clean += v
                elif d == "out":
                    future_flow_clean -= v
                
                w = ACTION_INTENT_WEIGHTS.get(at, 0.0)
                future_intent_clean += w * math.log(1 + v)
                count += 1
            
            if count > 0:
                future_intent_clean /= math.sqrt(count)
        else:
            future_flow_clean = 0.0
            future_intent_clean = 0.0
        
        # Combined score (past behavior + future behavior)
        # Past confirms the pattern, future is what we're predicting
        combined_intent = 0.3 * past_intent + 0.7 * future_intent_clean
        
        # Label based on thresholds
        if future_flow_clean > thresholds["accumulating"] or combined_intent > 1.0:
            confidence = min(abs(future_flow_clean) / (thresholds["accumulating"] * 3), 1.0)
            confidence = max(confidence, min(abs(combined_intent) / 3.0, 1.0))
            if confidence > best_confidence:
                best_label = "ACCUMULATING"
                best_confidence = confidence
                
        elif future_flow_clean < thresholds["distributing"] or combined_intent < -1.0:
            confidence = min(abs(future_flow_clean) / abs(thresholds["distributing"] * 3), 1.0)
            confidence = max(confidence, min(abs(combined_intent) / 3.0, 1.0))
            if confidence > best_confidence:
                best_label = "DISTRIBUTING"
                best_confidence = confidence
        
        # Store per-window features
        action[f"net_flow_{window_h}h"] = round(future_flow_clean, 4)
        action[f"intent_score_{window_h}h"] = round(combined_intent, 4)
        action[f"past_flow_{window_h}h"] = round(past_flow, 4)
    
    action["behavioral_label"] = best_label
    action["behavioral_confidence"] = round(best_confidence, 4)
    
    return action


def label_wallet_behavioral(wallet_name: str, actions: List[Dict]) -> List[Dict]:
    """
    Label all actions in a wallet timeline with behavioral intent.
    
    Returns labeled actions list.
    """
    if not actions:
        return []
    
    # Sort by timestamp
    actions = sorted(actions, key=lambda a: a["timestamp"])
    
    labeled = []
    for i in range(len(actions)):
        labeled_action = label_action(actions, i, wallet_name)
        labeled.append(labeled_action)
    
    return labeled


def is_exchange_wallet(wallet_name: str) -> bool:
    """Check if a wallet is an exchange wallet (should be excluded)."""
    return wallet_name.lower().replace(" ", "_") in EXCHANGE_WALLETS


def label_and_save_wallet(wallet_name: str) -> int:
    """
    Load raw wallet data, apply behavioral labels, save to labeled_v2/.
    
    Returns number of labeled actions, or -1 on error.
    """
    from .eth_collector import EthWhaleHistoryCollector
    
    LABELED_DIR.mkdir(parents=True, exist_ok=True)
    
    actions = EthWhaleHistoryCollector.load_wallet_timeline(wallet_name)
    if not actions:
        logger.warning("No data for wallet: %s", wallet_name)
        return 0
    
    logger.info("Behavioral labeling %d actions for %s...", len(actions), wallet_name)
    labeled = label_wallet_behavioral(wallet_name, actions)
    
    # Save
    safe_name = wallet_name.lower().replace(" ", "_").replace("/", "_")
    out_file = LABELED_DIR / f"{safe_name}_behavioral.jsonl"
    with open(out_file, "w") as f:
        for record in labeled:
            f.write(json.dumps(record, default=str) + "\n")
    
    # Stats
    total = len(labeled)
    acc = sum(1 for r in labeled if r.get("behavioral_label") == "ACCUMULATING")
    dist = sum(1 for r in labeled if r.get("behavioral_label") == "DISTRIBUTING")
    neut = sum(1 for r in labeled if r.get("behavioral_label") == "NEUTRAL")
    
    logger.info(
        "  %s: ACCUM=%d (%.1f%%) DISTRIB=%d (%.1f%%) NEUTRAL=%d (%.1f%%)",
        wallet_name, 
        acc, acc/total*100 if total else 0,
        dist, dist/total*100 if total else 0,
        neut, neut/total*100 if total else 0,
    )
    
    return total


def label_all_whale_wallets() -> Dict[str, int]:
    """
    Label only non-exchange whale wallets with behavioral intent.
    
    Returns {wallet_name: count}.
    """
    from .eth_collector import EthWhaleHistoryCollector
    
    results = {}
    for label, count in EthWhaleHistoryCollector.list_collected_wallets():
        if count == 0:
            continue
        
        safe_label = label.lower().replace(" ", "_")
        if safe_label in EXCHANGE_WALLETS:
            logger.info("Skipping exchange wallet: %s", label)
            continue
        
        try:
            n = label_and_save_wallet(label)
            results[label] = n
        except Exception as exc:
            logger.error("Failed to label %s: %s", label, exc)
            results[label] = -1
    
    return results
