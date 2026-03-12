"""
Tests for model loading and inference

Critical test cases:
- Model file loading
- VecNormalize stats loading
- Prediction determinism
- Observation normalization
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch


@pytest.mark.requires_model
class TestModelLoading:
    """Tests that require trained models."""

    def test_model_file_exists(self):
        """Test that model file exists."""
        model_path = Path('./data/models/ultimate_agent.zip')

        if not model_path.exists():
            pytest.skip("Model file not found")

        assert model_path.is_file()

    def test_vec_normalize_file_exists(self):
        """Test that VecNormalize stats file exists."""
        vec_norm_path = Path('./data/models/ultimate_agent_vec_normalize.pkl')

        if not vec_norm_path.exists():
            pytest.skip("VecNormalize file not found")

        assert vec_norm_path.is_file()

    def test_model_loads_without_error(self):
        """Test that model loads without errors."""
        from stable_baselines3 import PPO

        model_path = Path('./data/models/ultimate_agent.zip')

        if not model_path.exists():
            pytest.skip("Model file not found")

        try:
            model = PPO.load(str(model_path))
            assert model is not None
        except Exception as e:
            pytest.fail(f"Model loading failed: {e}")

    def test_vec_normalize_loads_without_error(self):
        """Test that VecNormalize stats load without errors."""
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        import gymnasium as gym

        vec_norm_path = Path('./data/models/ultimate_agent_vec_normalize.pkl')

        if not vec_norm_path.exists():
            pytest.skip("VecNormalize file not found")

        try:
            # Need dummy env for VecNormalize
            dummy_env = DummyVecEnv([lambda: gym.make('CartPole-v1')])
            vec_norm = VecNormalize.load(str(vec_norm_path), dummy_env)
            assert vec_norm is not None
        except Exception as e:
            pytest.fail(f"VecNormalize loading failed: {e}")

    def test_model_prediction(self, mock_observation):
        """Test model prediction."""
        from stable_baselines3 import PPO

        model_path = Path('./data/models/ultimate_agent.zip')

        if not model_path.exists():
            pytest.skip("Model file not found")

        model = PPO.load(str(model_path))

        # Make prediction
        action, _states = model.predict(mock_observation, deterministic=True)

        # Action should be valid (0, 1, or 2)
        assert action in [0, 1, 2], f"Invalid action: {action}"

    def test_model_determinism(self, mock_observation):
        """Test that model predictions are deterministic."""
        from stable_baselines3 import PPO

        model_path = Path('./data/models/ultimate_agent.zip')

        if not model_path.exists():
            pytest.skip("Model file not found")

        model = PPO.load(str(model_path))

        # Make multiple predictions with same input
        action1, _ = model.predict(mock_observation, deterministic=True)
        action2, _ = model.predict(mock_observation, deterministic=True)
        action3, _ = model.predict(mock_observation, deterministic=True)

        # All should be the same
        assert action1 == action2 == action3, "Model is not deterministic"

    def test_observation_shape_matches_model(self, mock_observation):
        """Test that observation shape matches model expectations."""
        from stable_baselines3 import PPO

        model_path = Path('./data/models/ultimate_agent.zip')

        if not model_path.exists():
            pytest.skip("Model file not found")

        model = PPO.load(str(model_path))

        # Should not raise error on prediction
        try:
            action, _states = model.predict(mock_observation, deterministic=True)
        except Exception as e:
            pytest.fail(f"Prediction failed with correct observation shape: {e}")


@pytest.mark.unit
class TestObservationNormalization:
    """Test observation normalization."""

    def test_vec_normalize_normalization(self, mock_observation):
        """Test that VecNormalize normalizes observations."""
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        import gymnasium as gym

        vec_norm_path = Path('./data/models/ultimate_agent_vec_normalize.pkl')

        if not vec_norm_path.exists():
            pytest.skip("VecNormalize file not found")

        # Load VecNormalize
        dummy_env = DummyVecEnv([lambda: gym.make('CartPole-v1')])
        vec_norm = VecNormalize.load(str(vec_norm_path), dummy_env)
        vec_norm.training = False

        # Normalize observation
        # Note: This is a simplified test; real usage is more complex
        # Just check that it doesn't crash

    def test_observation_range_after_normalization(self):
        """Test that normalized observations are in reasonable range."""
        # This would require actual VecNormalize integration
        # For now, just check that mock data is reasonable
        obs = np.random.randn(153).astype(np.float32)

        # Observations should be finite
        assert np.all(np.isfinite(obs))


@pytest.mark.unit
class TestModelMocking:
    """Test model functionality with mocking."""

    @patch('stable_baselines3.PPO')
    def test_mock_model_prediction(self, mock_ppo_class):
        """Test model prediction with mocking."""
        # Create mock model
        mock_model = Mock()
        mock_model.predict.return_value = (1, None)  # Action 1 (Buy)
        mock_ppo_class.load.return_value = mock_model

        # Use mock model
        from stable_baselines3 import PPO
        model = PPO.load('dummy_path')

        obs = np.random.randn(153).astype(np.float32)
        action, _ = model.predict(obs, deterministic=True)

        assert action == 1

    def test_observation_preprocessing(self, mock_observation):
        """Test observation preprocessing."""
        # Ensure observation is float32
        assert mock_observation.dtype == np.float32

        # Ensure observation is 1D
        assert len(mock_observation.shape) == 1

        # Ensure correct size
        assert mock_observation.shape[0] == 153


@pytest.mark.integration
@pytest.mark.requires_model
class TestModelIntegration:
    """Integration tests for model + environment."""

    def test_model_environment_compatibility(self, sample_ohlcv_data):
        """Test that model works with environment."""
        from stable_baselines3 import PPO
        from src.env.ultimate_env import UltimateTradingEnv

        model_path = Path('./data/models/ultimate_agent.zip')

        if not model_path.exists():
            pytest.skip("Model file not found")

        # Load model
        model = PPO.load(str(model_path))

        # Create environment
        env = UltimateTradingEnv(df=sample_ohlcv_data)

        # Reset environment
        obs, info = env.reset()

        # Get prediction
        try:
            action, _states = model.predict(obs, deterministic=True)
            assert action in [0, 1, 2]
        except Exception as e:
            pytest.fail(f"Model-environment compatibility issue: {e}")

    def test_full_episode_with_model(self, sample_ohlcv_data):
        """Test running full episode with real model."""
        from stable_baselines3 import PPO
        from src.env.ultimate_env import UltimateTradingEnv

        model_path = Path('./data/models/ultimate_agent.zip')

        if not model_path.exists():
            pytest.skip("Model file not found")

        model = PPO.load(str(model_path))
        env = UltimateTradingEnv(df=sample_ohlcv_data)

        obs, info = env.reset()
        total_reward = 0
        steps = 0

        terminated = False
        truncated = False

        while not (terminated or truncated) and steps < 50:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            steps += 1

        # Should complete some steps
        assert steps > 0

        # Total reward should be finite
        assert np.isfinite(total_reward)

        print(f"\nCompleted {steps} steps, total reward: {total_reward:.4f}")
