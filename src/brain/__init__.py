"""
Brain Layer Package
PPO-LSTM agent with self-improvement capabilities.
"""

from .agent import TradingAgent
from .replay_buffer import HighRewardBuffer
from .trainer import SelfImprovementTrainer

__all__ = ['TradingAgent', 'HighRewardBuffer', 'SelfImprovementTrainer']
