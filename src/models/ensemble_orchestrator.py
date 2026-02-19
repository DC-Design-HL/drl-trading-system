"""
Ensemble Orchestrator

Inference-time coordinator for the Regime-Specialized Agent Pool.
It uses the HMM Regime Classifier to weight the actions of 4 specialist
PPO agents based on the current regime and transition probabilities.

Usage in live trading:
    orchestrator = EnsembleOrchestrator('BTCUSDT')
    action, confidence = orchestrator.predict(obs, df_for_hmm)
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.regime_classifier import RegimeClassifier, REGIME_NAMES

logger = logging.getLogger(__name__)


class EnsembleOrchestrator:
    """
    Manages 4 specialized PPO agents and weights their outputs
    using the HMM regime transition probabilities.
    """
    
    SPECIALISTS_DIR = Path('./data/models/specialists')
    
    def __init__(self, symbol: str, device: str = "auto"):
        self.symbol = symbol
        self.clean_symbol = symbol.replace('/', '')
        self.device = device
        
        self.classifier = RegimeClassifier()
        self.agents: Dict[str, PPO] = {}
        
        # Will be populated by load()
        self.is_ready = False
        
    def load(self) -> bool:
        """Load the regime classifier and all 4 specialist agents."""
        
        # 1. Load HMM
        if not self.classifier.load(self.clean_symbol):
            logger.error(f"Failed to load regime classifier for {self.clean_symbol}")
            return False
            
        # 2. Load 4 Specialists
        expected_regimes = list(REGIME_NAMES.values())
        loaded_count = 0
        
        for regime in expected_regimes:
            model_path = self.SPECIALISTS_DIR / f"best_{self.clean_symbol.lower()}_{regime.lower()}/best_model.zip"
            
            if not model_path.exists():
                # Fallback to final model if best model doesn't exist
                model_path = self.SPECIALISTS_DIR / f"ppo_{self.clean_symbol.lower()}_{regime.lower()}.zip"
                
            if model_path.exists():
                try:
                    self.agents[regime] = PPO.load(str(model_path), device=self.device)
                    loaded_count += 1
                    logger.info(f"🧬 Loaded {regime} specialist")
                except Exception as e:
                    logger.error(f"Failed to load {regime} specialist: {e}")
            else:
                logger.warning(f"⚠️ Missing {regime} specialist for {self.clean_symbol} at {model_path}")
                
        # We need at least the current regime's model to do anything useful,
        # but preferably we want all 4. Let's say we're ready if we have at least 1,
        # but we'll log a warning if we don't have all 4.
        if loaded_count == 0:
            logger.error(f"No specialist agents loaded for {self.clean_symbol}")
            return False
            
        if loaded_count < 4:
            logger.warning(f"Only loaded {loaded_count}/4 specialists. Missing agents will vote HOLD (0).")
            
        self.is_ready = True
        return True
        
    def predict(
        self, 
        observation: np.ndarray, 
        df_for_regime: pd.DataFrame,
    ) -> Tuple[int, float]:
        """
        Predict the best action by ensemble voting.
        
        Args:
            observation: The MTF environment observation vector
            df_for_regime: The raw OHLCV dataframe used to compute the current regime
            
        Returns:
            action (int): 0=HOLD, 1=BUY, 2=SELL
            confidence (float): 0.0 to 1.0 score of how aligned the agents are
        """
        if not self.is_ready:
            logger.warning("Orchestrator not ready, returning HOLD")
            return 0, 0.0
            
        # 1. Get current regime and transition probabilities
        try:
            regime_info = self.classifier.predict(df_for_regime)
            current_regime = regime_info['current_regime']
            trans_probs = regime_info['transition_probs']
            
            logger.info(f"HMM Analysis: Current={current_regime}, Probs={trans_probs}")
        except Exception as e:
            logger.error(f"HMM prediction failed: {e}. Falling back to unweighted average.")
            current_regime = 'UNKNOWN'
            trans_probs = {r: 1.0/len(self.agents) for r in self.agents.keys()}
            
        # 2. Get predictions from all available specialists
        # PPO predict returns (action, state)
        action_votes = {0: 0.0, 1: 0.0, 2: 0.0}
        agent_actions = {}
        
        # We process probabilities. If a regime isn't in trans_probs, it gets 0 weight.
        # If an agent is missing, its weight is essentially lost (equivalent to voting HOLD with 0 weight).
        
        for regime_name, prob in trans_probs.items():
            if regime_name in self.agents:
                agent = self.agents[regime_name]
                try:
                    action, _ = agent.predict(observation, deterministic=True)
                    action = int(action)
                    
                    # Add weighted vote
                    action_votes[action] += prob
                    agent_actions[regime_name] = action
                    
                except Exception as e:
                    logger.error(f"Prediction error for {regime_name} agent: {e}")
            else:
                # Missing agent's probability weight defaults to HOLD
                action_votes[0] += prob
                
        # 3. Aggregate votes
        # Determine the action with the highest total probability weight
        winning_action = max(action_votes.items(), key=lambda x: x[1])[0]
        confidence = action_votes[winning_action]
        
        # Format a nice log message showing individual votes
        vote_str = ", ".join([f"{r[:4]}:{a}" for r, a in agent_actions.items()])
        action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
        logger.info(
            f"Ensemble Vote -> {action_names[winning_action]} "
            f"(Conf: {confidence:.2f}) | Details: {vote_str}"
        )
        
        return int(winning_action), float(confidence)
