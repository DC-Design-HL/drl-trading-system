"""
Quantile Regression DQN Agent for HTF Trading

Implements a Quantile Regression DQN (QRDQN) agent with built-in uncertainty estimation
for the HTF trading system. Uses 31 quantiles to model the full Q-value distribution,
providing natural confidence estimates from quantile spread.

Key features:
- 31 quantiles for fine-grained uncertainty quantification
- Deep network architecture [512, 512, 256, 128] with LayerNorm and Dropout
- Built-in confidence calibration via temperature scaling
- Compatible with existing HTFTradingEnv
- Efficient CPU inference for deployment
"""

import logging
from typing import Optional, Dict, Any, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib import QRDQN
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import TensorDict
import gymnasium as gym

logger = logging.getLogger(__name__)


class QRDQNNetwork(nn.Module):
    """
    Deep Quantile Regression network for DQN.
    
    Architecture: [512, 512, 256, 128] with LayerNorm, Dropout, and quantile heads.
    Returns quantile values for each action across the full distribution.
    """
    
    def __init__(
        self,
        input_dim: int,
        n_actions: int,
        n_quantiles: int = 31,
        hidden_dims: Tuple[int, ...] = (512, 512, 256, 128),
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.n_quantiles = n_quantiles
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # Feature extraction backbone
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate if i < len(hidden_dims) - 1 else dropout_rate * 0.5)
            ])
            prev_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        
        # Quantile regression heads - one per quantile
        self.quantile_heads = nn.ModuleList([
            nn.Linear(hidden_dims[-1], n_actions) 
            for _ in range(n_quantiles)
        ])
        
        # Temperature parameter for confidence calibration
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights using Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Quantile values tensor of shape (batch_size, n_quantiles, n_actions)
        """
        # Extract features
        features = self.backbone(x)
        
        # Compute quantile values for each head
        quantiles = []
        for head in self.quantile_heads:
            q_values = head(features)
            quantiles.append(q_values)
        
        # Stack quantiles: (batch_size, n_quantiles, n_actions)
        quantiles = torch.stack(quantiles, dim=1)
        
        # Apply temperature scaling for calibration
        quantiles = quantiles / self.temperature.clamp(min=0.1)
        
        return quantiles
    
    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        """Get mean Q-values across quantiles for action selection."""
        quantiles = self.forward(x)  # (batch_size, n_quantiles, n_actions)
        return quantiles.mean(dim=1)  # (batch_size, n_actions)
    
    def get_confidence(self, x: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get confidence estimates from quantile spread.
        
        Args:
            x: Input tensor
            action: Optional action tensor. If None, returns confidence for all actions.
            
        Returns:
            Confidence scores between 0 and 1 (narrow spread = high confidence)
        """
        quantiles = self.forward(x)  # (batch_size, n_quantiles, n_actions)
        
        # Use 10th and 90th percentiles for confidence estimation
        q10_idx = int(0.1 * self.n_quantiles)  # ~3rd quantile
        q90_idx = int(0.9 * self.n_quantiles)  # ~28th quantile
        
        q10 = quantiles[:, q10_idx, :]  # (batch_size, n_actions)
        q90 = quantiles[:, q90_idx, :]  # (batch_size, n_actions)
        
        # Quantile spread (uncertainty)
        spread = torch.abs(q90 - q10)
        
        # Convert to confidence (inverse of uncertainty)
        confidence = 1.0 / (1.0 + spread)
        
        if action is not None:
            # Extract confidence for specific actions
            batch_size = confidence.size(0)
            confidence = confidence[torch.arange(batch_size), action]
        
        return confidence


class HTFQRDQNPolicy(BasePolicy):
    """
    Custom QRDQN policy for HTF trading with quantile regression.
    """
    
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: callable,
        n_quantiles: int = 31,
        net_arch: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs
        )
        
        # Default architecture
        if net_arch is None:
            net_arch = {"hidden_dims": [512, 512, 256, 128], "dropout_rate": 0.1}
        
        self.n_quantiles = n_quantiles
        self.net_arch = net_arch
        
        # Build network
        obs_dim = observation_space.shape[0]
        n_actions = action_space.n
        
        self.quantile_net = QRDQNNetwork(
            input_dim=obs_dim,
            n_actions=n_actions,
            n_quantiles=n_quantiles,
            hidden_dims=net_arch.get("hidden_dims", [512, 512, 256, 128]),
            dropout_rate=net_arch.get("dropout_rate", 0.1)
        )
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.quantile_net.parameters(),
            lr=lr_schedule(1),
            weight_decay=1e-5
        )
    
    def forward(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Forward pass for action selection."""
        q_values = self.quantile_net.get_q_values(obs)
        
        if deterministic:
            return q_values.argmax(dim=1, keepdim=True)
        else:
            # Add noise for exploration
            noise = torch.randn_like(q_values) * 0.01
            return (q_values + noise).argmax(dim=1, keepdim=True)
    
    def _predict(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Predict action from observation."""
        return self.forward(observation, deterministic)


class HTFQRDQNAgent:
    """
    Quantile Regression DQN agent for HTF trading.
    
    Provides uncertainty-aware Q-learning with built-in confidence estimation
    via quantile regression. Compatible with the existing HTFTradingEnv.
    """
    
    def __init__(
        self,
        env: gym.Env,
        config: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None,
    ):
        """
        Initialize the QRDQN agent.
        
        Args:
            env: HTFTradingEnv instance
            config: Optional configuration overrides
            model_path: Optional path to pretrained model
        """
        self.env = env
        self.config = {**self._default_config(), **(config or {})}
        
        # Create model
        if model_path:
            self.model = self._load_model(model_path)
        else:
            self.model = self._create_model()
        
        self.training_steps = 0
        self.last_confidence = None
    
    @staticmethod
    def _default_config() -> Dict[str, Any]:
        """Default configuration for QRDQN."""
        return {
            "learning_rate": 3e-4,
            "buffer_size": 500_000,
            "learning_starts": 50_000,
            "batch_size": 256,
            "tau": 1.0,  # Hard update for target network
            "gamma": 0.995,
            "train_freq": 4,
            "gradient_steps": 1,
            "target_update_interval": 1000,
            "exploration_fraction": 0.3,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.01,
            "max_grad_norm": 0.5,
            "n_quantiles": 31,
            "net_arch": {
                "hidden_dims": [512, 512, 256, 128],
                "dropout_rate": 0.1
            },
            "verbose": 1,
            "tensorboard_log": "./logs/tensorboard/qrdqn/",
        }
    
    def _create_model(self) -> QRDQN:
        """Create a new QRDQN model."""
        # QRDQN expects net_arch as a flat list [512, 256, 128]
        net_arch_config = self.config["net_arch"]
        if isinstance(net_arch_config, dict):
            net_arch_list = net_arch_config.get("hidden_dims", [512, 512, 256, 128])
        else:
            net_arch_list = net_arch_config
        
        policy_kwargs = {
            "n_quantiles": self.config["n_quantiles"],
            "net_arch": net_arch_list
        }
        
        model = QRDQN(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=self.config["learning_rate"],
            buffer_size=self.config["buffer_size"],
            learning_starts=self.config["learning_starts"],
            batch_size=self.config["batch_size"],
            tau=self.config["tau"],
            gamma=self.config["gamma"],
            train_freq=self.config["train_freq"],
            gradient_steps=self.config["gradient_steps"],
            target_update_interval=self.config["target_update_interval"],
            exploration_fraction=self.config["exploration_fraction"],
            exploration_initial_eps=self.config["exploration_initial_eps"],
            exploration_final_eps=self.config["exploration_final_eps"],
            max_grad_norm=self.config["max_grad_norm"],
            policy_kwargs=policy_kwargs,
            verbose=self.config["verbose"],
            tensorboard_log=self.config["tensorboard_log"],
        )
        
        logger.info("Created QRDQN model with %d quantiles", self.config["n_quantiles"])
        return model
    
    def _load_model(self, path: str) -> QRDQN:
        """Load a pretrained model."""
        logger.info("Loading QRDQN model from %s", path)
        model = QRDQN.load(path, env=self.env)
        return model
    
    def train(
        self,
        total_timesteps: int,
        callback=None,
        eval_env=None,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train the QRDQN agent.
        
        Args:
            total_timesteps: Number of training steps
            callback: Optional training callback
            eval_env: Optional evaluation environment
            save_path: Optional path to save best model
            
        Returns:
            Training metrics dictionary
        """
        logger.info("Training QRDQN for %d timesteps", total_timesteps)
        
        # Setup evaluation callback if needed
        callbacks = callback if isinstance(callback, list) else ([callback] if callback else [])
        
        if eval_env and save_path:
            from stable_baselines3.common.callbacks import EvalCallback
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=save_path,
                log_path=save_path + "/logs/",
                eval_freq=max(10000, total_timesteps // 10),
                n_eval_episodes=5,
                deterministic=True,
            )
            callbacks.append(eval_callback)
        
        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=False,
            reset_num_timesteps=False,
        )
        
        self.training_steps += total_timesteps
        
        return {
            "total_timesteps": total_timesteps,
            "cumulative_timesteps": self.training_steps,
            "algorithm": "QRDQN",
        }
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
        return_confidence: bool = True,
    ) -> Tuple[int, Optional[Any], float]:
        """
        Predict action with confidence estimate.
        
        Args:
            observation: Environment observation
            deterministic: Use deterministic action selection
            return_confidence: Whether to compute confidence
            
        Returns:
            Tuple of (action, state, confidence)
        """
        action, state = self.model.predict(observation, deterministic=deterministic)
        
        confidence = 1.0  # Default confidence
        if return_confidence:
            try:
                confidence = self._compute_confidence(observation, action)
            except Exception as e:
                logger.debug("Failed to compute confidence: %s", e)
        
        self.last_confidence = confidence
        return int(action), state, float(confidence)
    
    def _compute_confidence(self, observation: np.ndarray, action: int) -> float:
        """Compute confidence from quantile spread."""
        if not hasattr(self.model.policy, 'quantile_net'):
            return 1.0  # Fallback if custom policy not available
        
        # Convert observation to tensor
        obs_tensor = torch.FloatTensor(observation.reshape(1, -1))
        action_tensor = torch.LongTensor([action])
        
        with torch.no_grad():
            confidence = self.model.policy.quantile_net.get_confidence(obs_tensor, action_tensor)
            return float(confidence.item())
    
    def get_last_confidence(self) -> Optional[float]:
        """Get confidence from last prediction."""
        return self.last_confidence
    
    def save(self, path: str) -> None:
        """Save the model."""
        self.model.save(path)
        logger.info("Saved QRDQN model to %s", path)
    
    def load(self, path: str) -> None:
        """Load the model."""
        self.model = QRDQN.load(path, env=self.env)
        logger.info("Loaded QRDQN model from %s", path)
    
    def __repr__(self) -> str:
        return (
            f"HTFQRDQNAgent("
            f"quantiles={self.config['n_quantiles']}, "
            f"lr={self.config['learning_rate']}, "
            f"steps_trained={self.training_steps})"
        )


def create_qrdqn_agent(
    env: gym.Env,
    config_path: Optional[str] = None,
    model_path: Optional[str] = None,
) -> HTFQRDQNAgent:
    """
    Factory function to create an HTFQRDQNAgent.
    
    Args:
        env: HTFTradingEnv instance
        config_path: Optional path to YAML config file
        model_path: Optional path to pretrained model
        
    Returns:
        HTFQRDQNAgent instance
    """
    config = None
    if config_path:
        import yaml
        with open(config_path, 'r') as f:
            raw = yaml.safe_load(f)
        config = raw.get('qrdqn', {}) if isinstance(raw, dict) else {}
        logger.info("Loaded QRDQN config from %s", config_path)
    
    return HTFQRDQNAgent(env=env, config=config, model_path=model_path)