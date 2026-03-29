"""
Ensemble Agent for HTF Trading

Combines QRDQN and PPO agents with market regime conditioning for robust trading.
Uses regime detector to weight model predictions and provides uncertainty-aware
position sizing recommendations.

Key features:
- Regime-conditional ensemble weighting
- Calibrated confidence from QRDQN quantile spread
- Position sizing based on uncertainty and regime
- Compatible with existing HTFTradingEnv
- Supports both training and inference modes
"""

import logging
from typing import Dict, Any, Optional, Tuple, Union
import json
import os
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym

from .htf_agent import HTFTradingAgent
from .qrdqn_agent import HTFQRDQNAgent
from .regime_detector import RegimeDetector, RegimeType

logger = logging.getLogger(__name__)


class EnsembleAgent:
    """
    Ensemble agent combining QRDQN and PPO with regime conditioning.
    
    The ensemble dynamically weights predictions from QRDQN and PPO agents
    based on the detected market regime and provides calibrated confidence
    estimates for position sizing.
    """
    
    def __init__(
        self,
        env: gym.Env,
        qrdqn_agent: Optional[HTFQRDQNAgent] = None,
        ppo_agent: Optional[HTFTradingAgent] = None,
        regime_detector: Optional[RegimeDetector] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the ensemble agent.
        
        Args:
            env: HTFTradingEnv instance
            qrdqn_agent: Optional QRDQN agent (created if None)
            ppo_agent: Optional PPO agent (created if None)
            regime_detector: Optional regime detector (created if None)
            config: Optional configuration overrides
        """
        self.env = env
        self.config = {**self._default_config(), **(config or {})}
        
        # Initialize agents
        self.qrdqn_agent = qrdqn_agent or HTFQRDQNAgent(env)
        self.ppo_agent = ppo_agent or HTFTradingAgent(env)
        self.regime_detector = regime_detector or RegimeDetector()
        
        # Ensemble state
        self.last_regime = RegimeType.RANGING
        self.last_confidence = 0.5
        self.last_position_size_multiplier = 1.0
        
        # Confidence calibration parameters
        self.confidence_calibrator = ConfidenceCalibrator()
        
        # Training tracking
        self.training_steps = 0
        self.ensemble_metrics = {
            'regime_accuracy': [],
            'confidence_calibration': [],
            'position_sizing_performance': []
        }
    
    @staticmethod
    def _default_config() -> Dict[str, Any]:
        """Default ensemble configuration."""
        return {
            # Regime-based weighting
            'regime_weights': {
                'trending_up': {'qrdqn': 0.7, 'ppo': 0.3},
                'trending_down': {'qrdqn': 0.7, 'ppo': 0.3},
                'ranging': {'qrdqn': 0.3, 'ppo': 0.7},
                'volatile': {'qrdqn': 0.6, 'ppo': 0.4},
            },
            # Position sizing parameters
            'base_position_size': 0.02,  # 2% base position size
            'max_position_size': 0.05,   # 5% maximum position size
            'min_position_size': 0.005,  # 0.5% minimum position size
            'confidence_scaling': 0.5,   # How much confidence affects sizing
            'regime_scaling': {
                'trending_up': 1.2,
                'trending_down': 1.2,
                'ranging': 0.6,
                'volatile': 0.8,
            },
            # Confidence calibration
            'calibration_window': 1000,  # Recent predictions for calibration
            'confidence_temperature': 1.5,  # Temperature scaling parameter
            # Ensemble decision thresholds
            'min_confidence_threshold': 0.2,  # Minimum confidence for trade
            'disagreement_threshold': 0.5,    # Max allowed agent disagreement
        }
    
    def predict(
        self,
        observation: np.ndarray,
        price_data: Optional[np.ndarray] = None,
        deterministic: bool = True,
    ) -> Tuple[int, float, float]:
        """
        Predict action with confidence and position size multiplier.
        
        Args:
            observation: Environment observation
            price_data: Optional OHLCV data for regime detection
            deterministic: Use deterministic predictions
            
        Returns:
            Tuple of (action, confidence, position_size_multiplier)
        """
        # Detect current regime
        if price_data is not None:
            regime, regime_confidence = self._detect_regime(price_data)
        else:
            # Use last known regime or default
            regime = self.last_regime
            regime_confidence = 0.5
        
        self.last_regime = regime
        
        # Get predictions from both agents
        qrdqn_action, _, qrdqn_confidence = self.qrdqn_agent.predict(
            observation, deterministic=deterministic
        )
        ppo_action, _, ppo_confidence = self.ppo_agent.predict(
            observation, deterministic=deterministic
        )
        
        # Get regime-based weights
        regime_key = regime.value
        weights = self.config['regime_weights'].get(regime_key, {'qrdqn': 0.5, 'ppo': 0.5})
        
        # Combine predictions
        ensemble_action, ensemble_confidence = self._combine_predictions(
            qrdqn_action, qrdqn_confidence,
            ppo_action, ppo_confidence,
            weights, regime_confidence
        )
        
        # Calculate position size multiplier
        position_size_multiplier = self._calculate_position_size(
            ensemble_confidence, regime, regime_confidence
        )
        
        # Store for tracking
        self.last_confidence = ensemble_confidence
        self.last_position_size_multiplier = position_size_multiplier
        
        return ensemble_action, ensemble_confidence, position_size_multiplier
    
    def _detect_regime(self, price_data: np.ndarray) -> Tuple[RegimeType, float]:
        """Detect market regime from price data."""
        if len(price_data.shape) == 1:
            # Assume it's just close prices, create minimal DataFrame
            import pandas as pd
            df = pd.DataFrame({
                'open': price_data,
                'high': price_data,
                'low': price_data,
                'close': price_data,
                'volume': np.ones_like(price_data)
            })
        else:
            # Assume OHLCV format
            import pandas as pd
            df = pd.DataFrame(price_data, columns=['open', 'high', 'low', 'close', 'volume'])
        
        return self.regime_detector.detect_regime(df)
    
    def _combine_predictions(
        self,
        qrdqn_action: int,
        qrdqn_confidence: float,
        ppo_action: int,
        ppo_confidence: float,
        weights: Dict[str, float],
        regime_confidence: float,
    ) -> Tuple[int, float]:
        """Combine predictions from both agents."""
        
        # Check for agreement
        agents_agree = (qrdqn_action == ppo_action)
        
        if agents_agree:
            # Agents agree - high confidence
            action = qrdqn_action
            confidence = (
                qrdqn_confidence * weights['qrdqn'] +
                ppo_confidence * weights['ppo']
            )
            # Boost confidence when agents agree
            confidence = min(0.95, confidence * 1.2)
        else:
            # Agents disagree - weight by regime and confidence
            qrdqn_score = qrdqn_confidence * weights['qrdqn']
            ppo_score = ppo_confidence * weights['ppo']
            
            # Choose action with higher weighted score
            if qrdqn_score > ppo_score:
                action = qrdqn_action
                confidence = qrdqn_confidence * regime_confidence
            else:
                action = ppo_action
                confidence = ppo_confidence * regime_confidence
            
            # Reduce confidence when agents disagree
            disagreement_penalty = abs(qrdqn_score - ppo_score)
            if disagreement_penalty > self.config['disagreement_threshold']:
                confidence *= 0.7
        
        # Apply confidence calibration
        calibrated_confidence = self.confidence_calibrator.calibrate(confidence)
        
        # Ensure minimum confidence threshold
        if calibrated_confidence < self.config['min_confidence_threshold']:
            action = 0  # Hold when confidence is too low
            calibrated_confidence = self.config['min_confidence_threshold']
        
        return action, calibrated_confidence
    
    def _calculate_position_size(
        self,
        confidence: float,
        regime: RegimeType,
        regime_confidence: float,
    ) -> float:
        """Calculate position size multiplier based on confidence and regime."""
        
        base_size = self.config['base_position_size']
        
        # Confidence-based scaling
        confidence_multiplier = (
            1.0 + (confidence - 0.5) * self.config['confidence_scaling']
        )
        
        # Regime-based scaling
        regime_multiplier = self.config['regime_scaling'].get(regime.value, 1.0)
        
        # Regime confidence adjustment
        regime_adj = 0.5 + regime_confidence * 0.5
        
        # Combine factors
        position_multiplier = (
            confidence_multiplier * 
            regime_multiplier * 
            regime_adj
        )
        
        # Apply bounds
        position_multiplier = np.clip(
            position_multiplier,
            self.config['min_position_size'] / base_size,
            self.config['max_position_size'] / base_size
        )
        
        return float(position_multiplier)
    
    def train_ensemble(
        self,
        qrdqn_steps: int = 150_000,
        ppo_steps: int = 100_000,
        ensemble_steps: int = 50_000,
        eval_env: Optional[gym.Env] = None,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train the ensemble in phases.
        
        Phase 1: Train QRDQN (uncertainty estimation)
        Phase 2: Train PPO (policy optimization)
        Phase 3: Ensemble fine-tuning (calibration)
        
        Args:
            qrdqn_steps: Steps for QRDQN training
            ppo_steps: Steps for PPO training  
            ensemble_steps: Steps for ensemble calibration
            eval_env: Optional evaluation environment
            save_path: Optional save directory
            
        Returns:
            Training metrics dictionary
        """
        logger.info("Starting ensemble training...")
        
        metrics = {}
        
        # Phase 1: Train QRDQN
        logger.info("Phase 1: Training QRDQN (%d steps)", qrdqn_steps)
        qrdqn_metrics = self.qrdqn_agent.train(
            total_timesteps=qrdqn_steps,
            eval_env=eval_env,
            save_path=save_path + "/qrdqn" if save_path else None,
        )
        metrics['qrdqn'] = qrdqn_metrics
        
        # Free memory after QRDQN training
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        
        # Phase 2: Train PPO
        logger.info("Phase 2: Training PPO (%d steps)", ppo_steps)
        ppo_metrics = self.ppo_agent.train(
            total_timesteps=ppo_steps,
            eval_env=eval_env,
            save_path=save_path + "/ppo" if save_path else None,
        )
        metrics['ppo'] = ppo_metrics
        
        # Phase 3: Ensemble calibration
        logger.info("Phase 3: Ensemble calibration (%d steps)", ensemble_steps)
        calibration_metrics = self._calibrate_ensemble(
            steps=ensemble_steps,
            eval_env=eval_env
        )
        metrics['ensemble'] = calibration_metrics
        
        self.training_steps += qrdqn_steps + ppo_steps + ensemble_steps
        
        # Save ensemble
        if save_path:
            self.save(save_path + "/ensemble")
        
        logger.info("Ensemble training complete")
        return metrics
    
    def _calibrate_ensemble(self, steps: int, eval_env: Optional[gym.Env] = None) -> Dict[str, Any]:
        """Calibrate confidence estimates using recent predictions."""
        logger.info("Calibrating ensemble confidence...")
        
        calibration_data = []
        
        # Collect predictions and outcomes for calibration
        if eval_env:
            for episode in range(min(10, steps // 1000)):
                obs, _ = eval_env.reset()
                episode_data = []
                
                done = False
                while not done and len(episode_data) < 100:
                    action, confidence, _ = self.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = eval_env.step(action)
                    
                    # Record prediction and outcome
                    outcome = 1.0 if reward > 0 else 0.0
                    episode_data.append((confidence, outcome))
                    
                    if done or truncated:
                        break
                
                calibration_data.extend(episode_data)
        
        # Update confidence calibrator
        if calibration_data:
            confidences = [x[0] for x in calibration_data]
            outcomes = [x[1] for x in calibration_data]
            self.confidence_calibrator.fit(confidences, outcomes)
        
        return {
            'calibration_samples': len(calibration_data),
            'mean_confidence': np.mean([x[0] for x in calibration_data]) if calibration_data else 0.0,
            'mean_outcome': np.mean([x[1] for x in calibration_data]) if calibration_data else 0.0,
        }
    
    def get_ensemble_state(self) -> Dict[str, Any]:
        """Get current ensemble state for monitoring."""
        return {
            'last_regime': self.last_regime.value,
            'last_confidence': self.last_confidence,
            'last_position_multiplier': self.last_position_size_multiplier,
            'training_steps': self.training_steps,
            'qrdqn_steps': self.qrdqn_agent.training_steps,
            'ppo_steps': self.ppo_agent.training_steps,
        }
    
    def save(self, path: str) -> None:
        """Save the entire ensemble."""
        ensemble_dir = Path(path)
        ensemble_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual agents
        self.qrdqn_agent.save(str(ensemble_dir / "qrdqn_model.zip"))
        self.ppo_agent.save(str(ensemble_dir / "ppo_model.zip"))
        
        # Save ensemble state and config
        ensemble_state = {
            'config': self.config,
            'training_steps': self.training_steps,
            'last_regime': self.last_regime.value,
            'confidence_calibrator': self.confidence_calibrator.to_dict(),
        }
        
        with open(ensemble_dir / "ensemble_state.json", 'w') as f:
            json.dump(ensemble_state, f, indent=2)
        
        logger.info("Saved ensemble to %s", path)
    
    def load(self, path: str) -> None:
        """Load the entire ensemble."""
        ensemble_dir = Path(path)
        
        # Load individual agents
        self.qrdqn_agent.load(str(ensemble_dir / "qrdqn_model.zip"))
        self.ppo_agent.load(str(ensemble_dir / "ppo_model.zip"))
        
        # Load ensemble state
        state_file = ensemble_dir / "ensemble_state.json"
        if state_file.exists():
            with open(state_file, 'r') as f:
                ensemble_state = json.load(f)
            
            self.config.update(ensemble_state.get('config', {}))
            self.training_steps = ensemble_state.get('training_steps', 0)
            self.last_regime = RegimeType(ensemble_state.get('last_regime', 'ranging'))
            
            calibrator_state = ensemble_state.get('confidence_calibrator', {})
            self.confidence_calibrator.from_dict(calibrator_state)
        
        logger.info("Loaded ensemble from %s", path)
    
    def __repr__(self) -> str:
        return (
            f"EnsembleAgent("
            f"qrdqn_steps={self.qrdqn_agent.training_steps}, "
            f"ppo_steps={self.ppo_agent.training_steps}, "
            f"total_steps={self.training_steps})"
        )


class ConfidenceCalibrator:
    """
    Calibrates confidence estimates using temperature scaling and Platt scaling.
    """
    
    def __init__(self):
        self.temperature = 1.5
        self.platt_a = 1.0
        self.platt_b = 0.0
        self.is_fitted = False
    
    def fit(self, confidences: list, outcomes: list) -> None:
        """Fit calibration parameters to confidence-outcome pairs."""
        if len(confidences) < 10:
            return  # Need minimum samples
        
        confidences = np.array(confidences)
        outcomes = np.array(outcomes)
        
        # Simple temperature scaling
        best_temp = 1.0
        best_loss = float('inf')
        
        for temp in np.linspace(0.5, 3.0, 26):
            calibrated = self._apply_temperature(confidences, temp)
            loss = self._calibration_loss(calibrated, outcomes)
            
            if loss < best_loss:
                best_loss = loss
                best_temp = temp
        
        self.temperature = best_temp
        self.is_fitted = True
        
        logger.debug("Calibrated temperature: %.2f", self.temperature)
    
    def calibrate(self, confidence: float) -> float:
        """Apply calibration to a confidence score."""
        if not self.is_fitted:
            return confidence
        
        return self._apply_temperature([confidence], self.temperature)[0]
    
    def _apply_temperature(self, confidences: np.ndarray, temperature: float) -> np.ndarray:
        """Apply temperature scaling."""
        # Convert confidence to logits (approximately)
        logits = np.log(confidences / (1 - confidences + 1e-8) + 1e-8)
        # Scale by temperature
        scaled_logits = logits / temperature
        # Convert back to probabilities
        return 1.0 / (1.0 + np.exp(-scaled_logits))
    
    def _calibration_loss(self, confidences: np.ndarray, outcomes: np.ndarray) -> float:
        """Calculate calibration loss (Brier score)."""
        return np.mean((confidences - outcomes) ** 2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'temperature': self.temperature,
            'platt_a': self.platt_a,
            'platt_b': self.platt_b,
            'is_fitted': self.is_fitted,
        }
    
    def from_dict(self, state: Dict[str, Any]) -> None:
        """Load from dictionary."""
        self.temperature = state.get('temperature', 1.5)
        self.platt_a = state.get('platt_a', 1.0)
        self.platt_b = state.get('platt_b', 0.0)
        self.is_fitted = state.get('is_fitted', False)


def create_ensemble_agent(
    env: gym.Env,
    config_path: Optional[str] = None,
    model_path: Optional[str] = None,
) -> EnsembleAgent:
    """
    Factory function to create an EnsembleAgent.
    
    Args:
        env: HTFTradingEnv instance
        config_path: Optional path to YAML config file
        model_path: Optional path to saved ensemble
        
    Returns:
        EnsembleAgent instance
    """
    config = None
    if config_path:
        import yaml
        with open(config_path, 'r') as f:
            raw = yaml.safe_load(f)
        config = raw.get('ensemble', {}) if isinstance(raw, dict) else {}
        logger.info("Loaded ensemble config from %s", config_path)
    
    # Create ensemble
    ensemble = EnsembleAgent(env=env, config=config)
    
    # Load saved model if provided
    if model_path and os.path.exists(model_path):
        ensemble.load(model_path)
    
    return ensemble