"""
PPO-LSTM Trading Agent
Implements the deep reinforcement learning brain for trading decisions.
"""

import os
import numpy as np
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym


class ConfidenceCallback(BaseCallback):
    """
    Callback to track action confidence during training.
    """
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.confidences = []
        
    def _on_step(self) -> bool:
        return True
        
    def get_average_confidence(self) -> float:
        if not self.confidences:
            return 0.0
        return np.mean(self.confidences)


class TradingAgent:
    """
    PPO-LSTM agent for cryptocurrency trading.
    
    Uses Stable-Baselines3's RecurrentPPO with LSTM policy to capture
    temporal dependencies in market data.
    """
    
    def __init__(
        self,
        env: gym.Env,
        config: Optional[Dict] = None,
        model_path: Optional[str] = None,
    ):
        """
        Initialize the trading agent.
        
        Args:
            env: Gymnasium trading environment
            config: Model hyperparameters
            model_path: Path to load pretrained model from
        """
        self.config = config or self._default_config()
        self.env = env
        
        # Wrap in DummyVecEnv for SB3 compatibility
        self.vec_env = DummyVecEnv([lambda: env])
        
        # Optional: Add observation normalization
        if self.config.get('normalize_observations', True):
            self.vec_env = VecNormalize(
                self.vec_env,
                norm_obs=True,
                norm_reward=True,
                clip_obs=10.0,
                clip_reward=10.0,
            )
        
        # Initialize or load model
        if model_path and os.path.exists(model_path):
            self.model = self._load_model(model_path)
        else:
            self.model = self._create_model()
            
        # Tracking
        self.training_steps = 0
        self.last_action_probs = None
        
    def _default_config(self) -> Dict:
        """Default hyperparameters for PPO."""
        return {
            'policy': 'MlpPolicy',  # Use standard MlpPolicy (MlpLstmPolicy requires sb3-contrib)
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'normalize_observations': True,
            'verbose': 1,
        }
        
    def _create_model(self) -> PPO:
        """Create a new PPO model with LSTM policy."""
        return PPO(
            policy=self.config.get('policy', 'MlpPolicy'),
            env=self.vec_env,
            learning_rate=self.config.get('learning_rate', 3e-4),
            n_steps=self.config.get('n_steps', 2048),
            batch_size=self.config.get('batch_size', 64),
            n_epochs=self.config.get('n_epochs', 10),
            gamma=self.config.get('gamma', 0.99),
            gae_lambda=self.config.get('gae_lambda', 0.95),
            clip_range=self.config.get('clip_range', 0.2),
            ent_coef=self.config.get('ent_coef', 0.01),
            vf_coef=self.config.get('vf_coef', 0.5),
            max_grad_norm=self.config.get('max_grad_norm', 0.5),
            verbose=self.config.get('verbose', 1),
            tensorboard_log="./logs/tensorboard/",
        )
        
    def _load_model(self, path: str) -> PPO:
        """Load a pretrained model."""
        print(f"Loading model from {path}")
        model = PPO.load(path, env=self.vec_env)
        return model
        
    def train(
        self,
        total_timesteps: int = 100000,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = 10000,
        save_path: Optional[str] = None,
        callbacks: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Train the agent.
        
        Args:
            total_timesteps: Total training steps
            eval_env: Optional evaluation environment
            eval_freq: Evaluation frequency
            save_path: Path to save best model
            callbacks: Additional callbacks
            
        Returns:
            Training metrics dictionary
        """
        callback_list = callbacks or []
        
        # Add evaluation callback if eval_env provided
        if eval_env:
            eval_vec_env = DummyVecEnv([lambda: eval_env])
            eval_callback = EvalCallback(
                eval_vec_env,
                best_model_save_path=save_path or "./data/models/",
                log_path="./logs/eval/",
                eval_freq=eval_freq,
                deterministic=True,
                render=False,
            )
            callback_list.append(eval_callback)
            
        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            progress_bar=True,
        )
        
        self.training_steps += total_timesteps
        
        return {
            'total_timesteps': self.training_steps,
        }
        
    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        deterministic: bool = True,
    ) -> Tuple[int, Optional[np.ndarray], float]:
        """
        Predict action for given observation.
        
        Args:
            observation: Current observation
            state: LSTM hidden state (for recurrent policy)
            deterministic: Whether to use deterministic action
            
        Returns:
            Tuple of (action, new_state, confidence)
        """
        # Get action and state
        action, state = self.model.predict(
            observation,
            state=state,
            deterministic=deterministic,
        )
        
        # Calculate confidence from action probabilities
        confidence = self._get_action_confidence(observation)
        
        return int(action), state, confidence
        
    def _get_action_confidence(self, observation: np.ndarray) -> float:
        """
        Calculate confidence score for the predicted action.
        Higher confidence = agent is more certain about its decision.
        """
        import torch
        
        obs = observation.reshape(1, -1)
        
        # Get action distribution
        with torch.no_grad():
            obs_tensor = self.model.policy.obs_to_tensor(obs)[0]
            dist = self.model.policy.get_distribution(obs_tensor)
            
            # Get probabilities
            probs = dist.distribution.probs.detach().cpu().numpy()[0]
            
        self.last_action_probs = probs
        
        # Confidence is max probability
        confidence = float(np.max(probs))
        
        return confidence
        
    def get_action_probabilities(self) -> Optional[np.ndarray]:
        """Get the last computed action probabilities."""
        return self.last_action_probs
        
    def save(self, path: str):
        """Save the model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        
        # Also save VecNormalize statistics if applicable
        if isinstance(self.vec_env, VecNormalize):
            vec_norm_path = path.replace('.zip', '_vecnorm.pkl')
            self.vec_env.save(vec_norm_path)
            
        print(f"Model saved to {path}")
        
    def load(self, path: str):
        """Load model from disk."""
        self.model = PPO.load(path, env=self.vec_env)
        
        # Load VecNormalize if exists
        vec_norm_path = path.replace('.zip', '_vecnorm.pkl')
        if os.path.exists(vec_norm_path) and isinstance(self.vec_env, VecNormalize):
            self.vec_env = VecNormalize.load(vec_norm_path, self.vec_env.venv)
            
        print(f"Model loaded from {path}")


def create_agent(
    env: gym.Env,
    config_path: Optional[str] = None,
    model_path: Optional[str] = None,
) -> TradingAgent:
    """
    Factory function to create a trading agent.
    
    Args:
        env: Trading environment
        config_path: Path to config YAML
        model_path: Path to pretrained model
        
    Returns:
        TradingAgent instance
    """
    config = None
    if config_path:
        import yaml
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)
            config = full_config.get('model', {})
            
    return TradingAgent(env=env, config=config, model_path=model_path)
