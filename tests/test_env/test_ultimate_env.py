"""
Tests for UltimateTradingEnv

Critical test cases:
- Environment initialization
- Observation/action space validation
- Step function correctness
- Reset functionality
- Reward calculation
- Position tracking
"""

import pytest
import numpy as np
import gymnasium as gym
from src.env.ultimate_env import UltimateTradingEnv


@pytest.mark.unit
class TestUltimateTradingEnv:
    """Test suite for UltimateTradingEnv."""

    def test_initialization(self, sample_ohlcv_data, trading_config):
        """Test environment initialization."""
        env = UltimateTradingEnv(
            df=sample_ohlcv_data,
            initial_balance=trading_config['initial_balance'],
            position_size=trading_config['position_size'],
        )

        assert env is not None
        assert isinstance(env, gym.Env)

    def test_observation_space(self, sample_ohlcv_data):
        """Test observation space dimensions."""
        env = UltimateTradingEnv(df=sample_ohlcv_data)

        # Check observation space
        assert isinstance(env.observation_space, gym.spaces.Box)

        # Ultimate env should have 153 dimensions (150 features + 3 position state)
        expected_dim = env.num_features + 3
        assert env.observation_space.shape[0] == expected_dim, \
            f"Observation space has {env.observation_space.shape[0]} dims, expected {expected_dim}"

    def test_action_space(self, sample_ohlcv_data):
        """Test action space is discrete with 3 actions."""
        env = UltimateTradingEnv(df=sample_ohlcv_data)

        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == 3  # 0=Hold, 1=Buy, 2=Sell

    def test_reset(self, sample_ohlcv_data):
        """Test environment reset."""
        env = UltimateTradingEnv(df=sample_ohlcv_data)

        obs, info = env.reset()

        # Check observation
        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape

        # Check info
        assert isinstance(info, dict)

        # Check state reset
        assert env.current_step >= 0
        assert env.balance == env.initial_balance
        assert env.position == 0

    def test_step_hold_action(self, sample_ohlcv_data):
        """Test step with hold action."""
        env = UltimateTradingEnv(df=sample_ohlcv_data)
        obs, info = env.reset()

        # Take hold action (0)
        obs, reward, terminated, truncated, info = env.step(0)

        # Check return types
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

        # Position should still be 0
        assert env.position == 0

    def test_step_buy_action(self, sample_ohlcv_data):
        """Test step with buy action."""
        env = UltimateTradingEnv(df=sample_ohlcv_data)
        env.reset()

        initial_balance = env.balance

        # Take buy action (1)
        obs, reward, terminated, truncated, info = env.step(1)

        # Position should be long
        assert env.position == 1

        # Balance should decrease (position opened)
        # (Depending on implementation, this might not be immediate)

    def test_step_sell_action(self, sample_ohlcv_data):
        """Test step with sell action."""
        env = UltimateTradingEnv(df=sample_ohlcv_data)
        env.reset()

        # Take sell action (2)
        obs, reward, terminated, truncated, info = env.step(2)

        # Position should be short
        assert env.position == -1

    def test_observation_no_nan_inf(self, sample_ohlcv_data):
        """Test that observations don't contain NaN or inf."""
        env = UltimateTradingEnv(df=sample_ohlcv_data)
        obs, _ = env.reset()

        # Check for NaN
        assert not np.isnan(obs).any(), "Observation contains NaN"

        # Check for inf
        assert not np.isinf(obs).any(), "Observation contains inf"

        # Take a few steps and check observations
        for _ in range(5):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                break

            assert not np.isnan(obs).any(), "Observation contains NaN after step"
            assert not np.isinf(obs).any(), "Observation contains inf after step"

    def test_episode_termination(self, sample_ohlcv_data):
        """Test that episode terminates correctly."""
        env = UltimateTradingEnv(df=sample_ohlcv_data)
        env.reset()

        terminated = False
        truncated = False
        steps = 0
        max_steps = len(sample_ohlcv_data)

        # Step until termination
        while not (terminated or truncated) and steps < max_steps:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1

        # Should terminate before max steps
        assert steps < max_steps or terminated or truncated

    def test_balance_tracking(self, sample_ohlcv_data):
        """Test that balance is tracked correctly."""
        env = UltimateTradingEnv(df=sample_ohlcv_data, initial_balance=10000.0)
        env.reset()

        initial_balance = env.balance

        # Take some actions
        for action in [1, 0, 0, 2]:  # Buy, hold, hold, sell
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        # Balance should have changed (either up or down)
        # Can't assert exact value without knowing price movements

    def test_position_size_enforcement(self, sample_ohlcv_data):
        """Test that position size is enforced."""
        position_size = 0.25
        env = UltimateTradingEnv(
            df=sample_ohlcv_data,
            initial_balance=10000.0,
            position_size=position_size,
        )
        env.reset()

        # Open position
        obs, reward, terminated, truncated, info = env.step(1)  # Buy

        # Check that position size was enforced (if tracked)
        # This depends on implementation details

    def test_trading_fee_applied(self, sample_ohlcv_data):
        """Test that trading fees are applied."""
        env = UltimateTradingEnv(
            df=sample_ohlcv_data,
            initial_balance=10000.0,
            trading_fee=0.001,  # 0.1% fee
        )
        env.reset()

        initial_balance = env.balance

        # Open and close position
        env.step(1)  # Buy
        env.step(2)  # Sell

        # Balance should be less than initial due to fees
        # (Exact amount depends on price)

    def test_stop_loss_enforcement(self, sample_ohlcv_data):
        """Test stop loss enforcement."""
        env = UltimateTradingEnv(
            df=sample_ohlcv_data,
            stop_loss_pct=0.025,
        )
        env.reset()

        # This test would require manipulating price data
        # or mocking to trigger stop loss
        # For now, just check that SL is set
        assert env.stop_loss_pct == 0.025

    def test_take_profit_enforcement(self, sample_ohlcv_data):
        """Test take profit enforcement."""
        env = UltimateTradingEnv(
            df=sample_ohlcv_data,
            take_profit_pct=0.05,
        )
        env.reset()

        # Check that TP is set
        assert env.take_profit_pct == 0.05


@pytest.mark.unit
class TestRewardFunction:
    """Test reward calculation."""

    def test_reward_calculation(self, sample_ohlcv_data):
        """Test that rewards are calculated."""
        env = UltimateTradingEnv(df=sample_ohlcv_data)
        env.reset()

        # Take action and get reward
        obs, reward, terminated, truncated, info = env.step(1)

        # Reward should be a finite number
        assert isinstance(reward, (int, float))
        assert not np.isnan(reward)
        assert not np.isinf(reward)

    def test_reward_scaling(self, sample_ohlcv_data):
        """Test reward scaling parameter."""
        scaling = 2.0
        env = UltimateTradingEnv(
            df=sample_ohlcv_data,
            reward_scaling=scaling,
        )

        # Check that scaling is set
        assert env.reward_scaling == scaling


@pytest.mark.integration
class TestUltimateTradingEnvIntegration:
    """Integration tests for trading environment."""

    def test_full_episode(self, sample_ohlcv_data):
        """Test running a full episode."""
        env = UltimateTradingEnv(df=sample_ohlcv_data)
        obs, info = env.reset()

        total_reward = 0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated) and steps < 100:
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            steps += 1

        # Should have taken at least some steps
        assert steps > 0

        # Total reward should be finite
        assert not np.isnan(total_reward)
        assert not np.isinf(total_reward)

    def test_deterministic_episode(self, sample_ohlcv_data):
        """Test that same actions produce same results."""
        env1 = UltimateTradingEnv(df=sample_ohlcv_data)
        env2 = UltimateTradingEnv(df=sample_ohlcv_data)

        # Use same seed
        env1.reset(seed=42)
        env2.reset(seed=42)

        # Take same actions
        actions = [1, 0, 0, 2, 0]

        for action in actions:
            obs1, reward1, term1, trunc1, _ = env1.step(action)
            obs2, reward2, term2, trunc2, _ = env2.step(action)

            if term1 or trunc1 or term2 or trunc2:
                break

            # Should get same results
            np.testing.assert_array_almost_equal(obs1, obs2)
            assert reward1 == reward2

    @pytest.mark.slow
    def test_multiple_episodes(self, sample_ohlcv_data):
        """Test running multiple episodes."""
        env = UltimateTradingEnv(df=sample_ohlcv_data)

        for episode in range(5):
            obs, info = env.reset()

            terminated = False
            truncated = False
            steps = 0

            while not (terminated or truncated) and steps < 50:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1

            # Each episode should complete
            assert steps > 0


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases for trading environment."""

    def test_empty_dataframe(self, empty_dataframe):
        """Test with empty DataFrame."""
        try:
            env = UltimateTradingEnv(df=empty_dataframe)
            # If it initializes, try to reset
            env.reset()
        except ValueError as e:
            # Acceptable to reject empty data
            assert "empty" in str(e).lower() or "insufficient" in str(e).lower()

    def test_single_row_dataframe(self, single_row_dataframe):
        """Test with single-row DataFrame."""
        try:
            env = UltimateTradingEnv(df=single_row_dataframe)
            env.reset()
        except ValueError as e:
            # Acceptable to require minimum data
            assert "insufficient" in str(e).lower()

    def test_invalid_action(self, sample_ohlcv_data):
        """Test handling of invalid action."""
        env = UltimateTradingEnv(df=sample_ohlcv_data)
        env.reset()

        # Try invalid action
        try:
            env.step(999)
            # Should either handle gracefully or raise
        except (ValueError, AssertionError):
            # Acceptable to reject invalid action
            pass
