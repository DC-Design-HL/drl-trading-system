"""
High-Reward Replay Buffer
Stores successful trade sequences for self-improvement fine-tuning.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from collections import deque
from dataclasses import dataclass, field
import threading
import time


@dataclass
class TradeSequence:
    """Represents a high-reward trade sequence."""
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    total_reward: float
    sharpe_ratio: float
    trade_pnl: float
    timestamp: float = field(default_factory=time.time)
    
    def __len__(self) -> int:
        return len(self.observations)


class HighRewardBuffer:
    """
    Buffer that stores high-reward trade sequences for fine-tuning.
    
    Only sequences that exceed the reward threshold are stored.
    Implements a priority queue based on total reward.
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        reward_threshold: float = 0.5,
        min_sequence_length: int = 10,
        save_path: Optional[str] = None,
    ):
        """
        Initialize the replay buffer.
        
        Args:
            max_size: Maximum number of sequences to store
            reward_threshold: Minimum total reward to store a sequence
            min_sequence_length: Minimum sequence length to consider
            save_path: Path to save/load buffer
        """
        self.max_size = max_size
        self.reward_threshold = reward_threshold
        self.min_sequence_length = min_sequence_length
        self.save_path = save_path
        
        self.sequences: List[TradeSequence] = []
        self.lock = threading.Lock()
        
        # Statistics
        self.total_sequences_seen = 0
        self.total_sequences_stored = 0
        
        # Load existing buffer if available
        if save_path and Path(save_path).exists():
            self.load()
            
    def add_sequence(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        episode_metrics: Dict[str, Any],
    ) -> bool:
        """
        Add a trade sequence to the buffer if it exceeds threshold.
        
        Args:
            observations: Array of observations
            actions: Array of actions taken
            rewards: Array of rewards received
            episode_metrics: Dictionary with sharpe_ratio, total_pnl, etc.
            
        Returns:
            True if sequence was added, False otherwise
        """
        self.total_sequences_seen += 1
        
        total_reward = np.sum(rewards)
        sequence_length = len(observations)
        
        # Check if sequence meets criteria
        if sequence_length < self.min_sequence_length:
            return False
            
        if total_reward < self.reward_threshold:
            return False
            
        # Create sequence object
        sequence = TradeSequence(
            observations=observations,
            actions=actions,
            rewards=rewards,
            total_reward=total_reward,
            sharpe_ratio=episode_metrics.get('sharpe_ratio', 0.0),
            trade_pnl=episode_metrics.get('total_pnl', 0.0),
        )
        
        with self.lock:
            # Add to buffer
            self.sequences.append(sequence)
            self.total_sequences_stored += 1
            
            # Keep only top-k sequences by reward
            if len(self.sequences) > self.max_size:
                self.sequences.sort(key=lambda x: x.total_reward, reverse=True)
                self.sequences = self.sequences[:self.max_size]
                
        return True
        
    def sample_batch(
        self,
        batch_size: int = 32,
        prioritized: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch of transitions from the buffer.
        
        Args:
            batch_size: Number of transitions to sample
            prioritized: Whether to use priority sampling
            
        Returns:
            Tuple of (observations, actions, rewards)
        """
        if len(self.sequences) == 0:
            raise ValueError("Buffer is empty")
            
        with self.lock:
            if prioritized:
                # Priority sampling based on total reward
                rewards = np.array([s.total_reward for s in self.sequences])
                probs = rewards / rewards.sum()
                indices = np.random.choice(
                    len(self.sequences),
                    size=min(batch_size, len(self.sequences)),
                    replace=False,
                    p=probs,
                )
            else:
                indices = np.random.choice(
                    len(self.sequences),
                    size=min(batch_size, len(self.sequences)),
                    replace=False,
                )
                
            # Collect transitions from sampled sequences
            obs_list = []
            action_list = []
            reward_list = []
            
            for idx in indices:
                seq = self.sequences[idx]
                # Sample a random window from the sequence
                if len(seq) > batch_size:
                    start = np.random.randint(0, len(seq) - batch_size)
                    obs_list.extend(seq.observations[start:start+batch_size])
                    action_list.extend(seq.actions[start:start+batch_size])
                    reward_list.extend(seq.rewards[start:start+batch_size])
                else:
                    obs_list.extend(seq.observations)
                    action_list.extend(seq.actions)
                    reward_list.extend(seq.rewards)
                    
        return (
            np.array(obs_list),
            np.array(action_list),
            np.array(reward_list),
        )
        
    def get_all_transitions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get all transitions in the buffer.
        
        Returns:
            Tuple of (all_observations, all_actions, all_rewards)
        """
        if len(self.sequences) == 0:
            return np.array([]), np.array([]), np.array([])
            
        with self.lock:
            obs_list = []
            action_list = []
            reward_list = []
            
            for seq in self.sequences:
                obs_list.extend(seq.observations)
                action_list.extend(seq.actions)
                reward_list.extend(seq.rewards)
                
        return (
            np.array(obs_list),
            np.array(action_list),
            np.array(reward_list),
        )
        
    def save(self, path: Optional[str] = None):
        """Save buffer to disk."""
        save_path = path or self.save_path
        if not save_path:
            raise ValueError("No save path specified")
            
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        with self.lock:
            data = {
                'sequences': self.sequences,
                'total_sequences_seen': self.total_sequences_seen,
                'total_sequences_stored': self.total_sequences_stored,
            }
            with open(save_path, 'wb') as f:
                pickle.dump(data, f)
                
        print(f"Buffer saved to {save_path} ({len(self.sequences)} sequences)")
        
    def load(self, path: Optional[str] = None):
        """Load buffer from disk."""
        load_path = path or self.save_path
        if not load_path or not Path(load_path).exists():
            return
            
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
            
        with self.lock:
            self.sequences = data['sequences']
            self.total_sequences_seen = data['total_sequences_seen']
            self.total_sequences_stored = data['total_sequences_stored']
            
        print(f"Buffer loaded from {load_path} ({len(self.sequences)} sequences)")
        
    def clear(self):
        """Clear the buffer."""
        with self.lock:
            self.sequences = []
            
    def __len__(self) -> int:
        return len(self.sequences)
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        if len(self.sequences) == 0:
            return {
                'size': 0,
                'total_seen': self.total_sequences_seen,
                'total_stored': self.total_sequences_stored,
                'storage_rate': 0.0,
            }
            
        with self.lock:
            rewards = [s.total_reward for s in self.sequences]
            sharpes = [s.sharpe_ratio for s in self.sequences]
            
        return {
            'size': len(self.sequences),
            'total_seen': self.total_sequences_seen,
            'total_stored': self.total_sequences_stored,
            'storage_rate': self.total_sequences_stored / max(1, self.total_sequences_seen),
            'avg_reward': np.mean(rewards),
            'max_reward': np.max(rewards),
            'avg_sharpe': np.mean(sharpes),
            'total_transitions': sum(len(s) for s in self.sequences),
        }
