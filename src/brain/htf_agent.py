"""
HTF (Hierarchical Multi-Timeframe) Trading Agent

Wraps PPO from stable-baselines3 with enhancements for a 4-timeframe cascade
system: 1D (macro) -> 4H (structure) -> 1H (momentum) -> 15M (execution).

Key enhancements over the base TradingAgent:
- Wider network architecture ([512, 256, 128]) suited for 117-dim HTF observations
- VecNormalize for stable training across large feature spaces
- Curriculum training: Phase 1 (HTF alignment focus), Phase 2 (full-cascade execution)
- Entropy annealing schedule: exploration -> exploitation across curriculum phases
- Best model checkpointing via EvalCallback
"""

import os
import logging
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class EntropyAnnealCallback(BaseCallback):
    """
    Linearly anneal the entropy coefficient during a training phase.

    This drives a smooth exploration -> exploitation transition without
    requiring a full restart of the model.
    """

    def __init__(
        self,
        start_ent: float,
        end_ent: float,
        total_steps: int,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.start_ent = start_ent
        self.end_ent = end_ent
        self.total_steps = total_steps
        self._phase_start_step = 0

    def _on_training_start(self) -> None:
        self._phase_start_step = self.num_timesteps

    def _on_step(self) -> bool:
        elapsed = self.num_timesteps - self._phase_start_step
        frac = min(elapsed / max(self.total_steps, 1), 1.0)
        new_ent = self.start_ent + frac * (self.end_ent - self.start_ent)
        self.model.ent_coef = new_ent
        return True


class HTFMetricsCallback(BaseCallback):
    """Collect episode-level metrics exposed by HTFTradingEnv info dicts."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: list = []
        self.episode_lengths: list = []
        self.htf_alignment_rates: list = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
            if "htf_alignment_rate" in info:
                self.htf_alignment_rates.append(info["htf_alignment_rate"])
        return True

    def get_summary(self) -> Dict[str, float]:
        summary: Dict[str, float] = {}
        if self.episode_rewards:
            summary["mean_episode_reward"] = float(np.mean(self.episode_rewards))
            summary["std_episode_reward"] = float(np.std(self.episode_rewards))
            summary["num_episodes"] = len(self.episode_rewards)
        if self.htf_alignment_rates:
            summary["mean_htf_alignment_rate"] = float(np.mean(self.htf_alignment_rates))
        return summary


# ---------------------------------------------------------------------------
# Main agent class
# ---------------------------------------------------------------------------

class HTFTradingAgent:
    """
    Hierarchical Multi-Timeframe Trading Agent.

    Wraps PPO with:
    - Wider network architecture (policy_kwargs net_arch [512, 256, 128]) for
      117-dim obs produced by the 4-timeframe cascade feature set.
    - VecNormalize for stable training (clip_obs=10, clip_reward=10).
    - Curriculum training: Phase 1 (1H+4H alignment focus with high entropy),
      Phase 2 (full 4-TF execution with tighter clip_range and lower entropy).
    - Entropy annealing schedule for smooth exploration -> exploitation.
    - Best model checkpointing via EvalCallback.
    """

    # Human-readable action labels
    _ACTION_LABELS = {0: "HOLD", 1: "LONG", 2: "SHORT"}

    def __init__(
        self,
        env: gym.Env,
        config: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None,
    ):
        """
        Initialise the HTF trading agent.

        Args:
            env: A HTFTradingEnv (or any Gymnasium env with a compatible
                 observation space).
            config: Optional hyperparameter overrides. Unset keys fall back to
                    the defaults defined in _default_config().
            model_path: If given and the path exists, the model (and its
                        paired VecNormalize stats) will be loaded from disk
                        rather than created fresh.
        """
        self.base_env = env
        self.config = {**self._default_config(), **(config or {})}

        # ------------------------------------------------------------------
        # Wrap env: DummyVecEnv -> VecNormalize
        # ------------------------------------------------------------------
        self.vec_env: VecNormalize = VecNormalize(
            DummyVecEnv([lambda: env]),
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=self.config["gamma"],
        )

        # ------------------------------------------------------------------
        # Build or load model
        # ------------------------------------------------------------------
        if model_path and os.path.exists(model_path):
            self.model = self._load_model(model_path)
        else:
            self.model = self._create_model()

        # Bookkeeping
        self.training_steps: int = 0
        self.last_action_probs: Optional[np.ndarray] = None
        self._phase_metrics: list = []

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        return {
            "policy": "MlpPolicy",
            "learning_rate": 1e-4,
            "n_steps": 4096,
            "batch_size": 256,
            "n_epochs": 10,
            "gamma": 0.995,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.02,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "verbose": 1,
            # Network architecture: three shared hidden layers for both policy
            # and value function heads.
            "net_arch": [dict(pi=[512, 256, 128], vf=[512, 256, 128])],
            # Tensorboard log directory
            "tensorboard_log": "./logs/tensorboard/htf/",
        }

    # ------------------------------------------------------------------
    # Model creation / loading
    # ------------------------------------------------------------------

    def _create_model(self) -> PPO:
        """Instantiate a fresh PPO model with the HTF network architecture."""
        policy_kwargs = {"net_arch": self.config["net_arch"]}

        model = PPO(
            policy=self.config["policy"],
            env=self.vec_env,
            learning_rate=self.config["learning_rate"],
            n_steps=self.config["n_steps"],
            batch_size=self.config["batch_size"],
            n_epochs=self.config["n_epochs"],
            gamma=self.config["gamma"],
            gae_lambda=self.config["gae_lambda"],
            clip_range=self.config["clip_range"],
            ent_coef=self.config["ent_coef"],
            vf_coef=self.config["vf_coef"],
            max_grad_norm=self.config["max_grad_norm"],
            policy_kwargs=policy_kwargs,
            verbose=self.config["verbose"],
            tensorboard_log=self.config["tensorboard_log"],
        )
        logger.info("Created new HTF PPO model (net_arch=%s)", self.config["net_arch"])
        return model

    def _load_model(self, path: str) -> PPO:
        """Load a previously saved PPO model and pair it with vec_env."""
        logger.info("Loading HTF model from %s", path)
        model = PPO.load(path, env=self.vec_env)

        # Restore VecNormalize statistics if a companion file exists
        vecnorm_path = self._vecnorm_path(path)
        if os.path.exists(vecnorm_path):
            self.vec_env = VecNormalize.load(vecnorm_path, self.vec_env.venv)
            logger.info("Loaded VecNormalize stats from %s", vecnorm_path)

        return model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _vecnorm_path(model_path: str) -> str:
        """Derive the VecNormalize companion path from a model path."""
        base = model_path.removesuffix(".zip") if model_path.endswith(".zip") else model_path
        return base + "_vecnorm.pkl"

    def _build_eval_callback(
        self,
        eval_env: gym.Env,
        save_path: Optional[str],
        eval_freq: int = 20_000,
        phase_tag: str = "",
    ) -> EvalCallback:
        """Wrap eval_env in a normalised VecEnv and return an EvalCallback."""
        eval_vec = VecNormalize(
            DummyVecEnv([lambda: eval_env]),
            norm_obs=True,
            norm_reward=False,   # Don't normalise rewards during evaluation
            clip_obs=10.0,
            training=False,      # Keep stats frozen for fair evaluation
        )

        best_path = save_path or "./data/models/htf/"
        log_path = "./logs/eval/htf/" + (phase_tag + "/" if phase_tag else "")

        return EvalCallback(
            eval_vec,
            best_model_save_path=best_path,
            log_path=log_path,
            eval_freq=max(eval_freq, self.config["n_steps"]),
            n_eval_episodes=5,
            deterministic=True,
            render=False,
        )

    # ------------------------------------------------------------------
    # Curriculum training phases
    # ------------------------------------------------------------------

    def train_phase1(
        self,
        timesteps: int = 500_000,
        eval_env: Optional[gym.Env] = None,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Curriculum Phase 1 — HTF alignment focus.

        Higher entropy (0.05) encourages wide exploration of the state space
        so the agent can discover which 1H/4H alignment signals are actionable.
        The entropy coefficient is annealed down to 0.02 by end of phase.

        Args:
            timesteps: Number of environment steps for this phase.
            eval_env:  Optional evaluation environment for checkpointing.
            save_path: Directory to save the best model checkpoint.

        Returns:
            Metrics dict with phase tag and training summary.
        """
        logger.info("=== Phase 1: HTF Alignment Focus (%d steps) ===", timesteps)

        # Temporarily raise entropy for exploration
        self.model.ent_coef = 0.05

        callbacks: list = [
            EntropyAnnealCallback(start_ent=0.05, end_ent=0.02, total_steps=timesteps),
            HTFMetricsCallback(),
        ]
        metrics_cb: HTFMetricsCallback = callbacks[1]  # type: ignore[assignment]

        if eval_env is not None:
            callbacks.append(
                self._build_eval_callback(eval_env, save_path, phase_tag="phase1")
            )

        self.model.learn(
            total_timesteps=timesteps,
            callback=CallbackList(callbacks),
            progress_bar=True,
            reset_num_timesteps=False,
        )

        self.training_steps += timesteps

        metrics = {
            "phase": "phase1",
            "timesteps": timesteps,
            "cumulative_timesteps": self.training_steps,
            **metrics_cb.get_summary(),
        }
        self._phase_metrics.append(metrics)
        logger.info("Phase 1 complete. Summary: %s", metrics)
        return metrics

    def train_phase2(
        self,
        timesteps: int = 1_000_000,
        eval_env: Optional[gym.Env] = None,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Curriculum Phase 2 — Full 4-TF cascade execution.

        Lower entropy (0.01) and a tighter clip_range (0.15) guide the agent
        toward precise 15M-timeframe entries that align with the full HTF
        cascade (1D -> 4H -> 1H -> 15M).

        Args:
            timesteps: Number of environment steps for this phase.
            eval_env:  Optional evaluation environment for checkpointing.
            save_path: Directory to save the best model checkpoint.

        Returns:
            Metrics dict with phase tag and training summary.
        """
        logger.info("=== Phase 2: Full 4-TF Cascade Execution (%d steps) ===", timesteps)

        # Tighter clip_range and lower entropy for exploitation
        self.model.clip_range = lambda _: 0.15
        self.model.ent_coef = 0.01

        callbacks: list = [
            EntropyAnnealCallback(start_ent=0.01, end_ent=0.005, total_steps=timesteps),
            HTFMetricsCallback(),
        ]
        metrics_cb: HTFMetricsCallback = callbacks[1]  # type: ignore[assignment]

        if eval_env is not None:
            callbacks.append(
                self._build_eval_callback(eval_env, save_path, phase_tag="phase2")
            )

        self.model.learn(
            total_timesteps=timesteps,
            callback=CallbackList(callbacks),
            progress_bar=True,
            reset_num_timesteps=False,
        )

        self.training_steps += timesteps

        metrics = {
            "phase": "phase2",
            "timesteps": timesteps,
            "cumulative_timesteps": self.training_steps,
            **metrics_cb.get_summary(),
        }
        self._phase_metrics.append(metrics)
        logger.info("Phase 2 complete. Summary: %s", metrics)
        return metrics

    def train(
        self,
        total_timesteps: int = 1_500_000,
        eval_env: Optional[gym.Env] = None,
        save_path: Optional[str] = None,
        use_curriculum: bool = True,
    ) -> Dict[str, Any]:
        """
        Full training run, optionally using the two-phase curriculum.

        When use_curriculum=True the timestep budget is split ~1:2 between
        phase1 (exploration) and phase2 (exploitation), matching the
        500k / 1M defaults. If a custom total_timesteps is given, the split
        is preserved proportionally.

        Args:
            total_timesteps: Total env steps across all phases.
            eval_env:        Optional evaluation environment.
            save_path:       Directory for model checkpoints.
            use_curriculum:  If False, run a single flat training phase.

        Returns:
            Combined metrics dict with keys from both phases (if curriculum)
            or a single-phase summary.
        """
        if use_curriculum:
            # Proportional split: 1/3 phase1, 2/3 phase2 (mirrors 500k/1M defaults)
            p1_steps = max(1, round(total_timesteps / 3))
            p2_steps = total_timesteps - p1_steps

            m1 = self.train_phase1(
                timesteps=p1_steps, eval_env=eval_env, save_path=save_path
            )
            m2 = self.train_phase2(
                timesteps=p2_steps, eval_env=eval_env, save_path=save_path
            )

            combined: Dict[str, Any] = {
                "curriculum": True,
                "total_timesteps": total_timesteps,
                "phase1": m1,
                "phase2": m2,
            }
            # Surface the phase2 episode quality metrics at the top level
            combined.update(
                {f"final_{k}": v for k, v in m2.items() if k.startswith("mean_")}
            )
            return combined

        else:
            logger.info("=== Single-phase training (%d steps) ===", total_timesteps)
            metrics_cb = HTFMetricsCallback()
            callbacks: list = [metrics_cb]
            if eval_env is not None:
                callbacks.append(
                    self._build_eval_callback(eval_env, save_path, phase_tag="single")
                )

            self.model.learn(
                total_timesteps=total_timesteps,
                callback=CallbackList(callbacks),
                progress_bar=True,
                reset_num_timesteps=False,
            )
            self.training_steps += total_timesteps

            metrics = {
                "curriculum": False,
                "phase": "single",
                "total_timesteps": total_timesteps,
                "cumulative_timesteps": self.training_steps,
                **metrics_cb.get_summary(),
            }
            self._phase_metrics.append(metrics)
            return metrics

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[int, Optional[np.ndarray], float]:
        """
        Predict the next action for a given observation.

        Args:
            observation:   Raw (un-normalised) observation from the env.
            deterministic: Use greedy action selection when True.

        Returns:
            Tuple of (action: int, state: None, confidence: float).
            confidence is the maximum action probability (0–1).
        """
        action, state = self.model.predict(observation, deterministic=deterministic)
        confidence = self._get_action_confidence(observation)
        return int(action), state, confidence

    def _get_action_confidence(self, observation: np.ndarray) -> float:
        """
        Compute the max action probability for the given observation.

        Falls back to 1/n_actions if the policy distribution is unavailable.
        """
        import torch

        try:
            obs = np.array(observation).reshape(1, -1)
            with torch.no_grad():
                obs_tensor = self.model.policy.obs_to_tensor(obs)[0]
                dist = self.model.policy.get_distribution(obs_tensor)
                probs = dist.distribution.probs.detach().cpu().numpy()[0]

            self.last_action_probs = probs
            return float(np.max(probs))

        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not compute action confidence: %s", exc)
            n_actions = self.base_env.action_space.n
            return 1.0 / n_actions

    def get_action_probabilities(self) -> Optional[np.ndarray]:
        """Return the last computed per-action probability vector."""
        return self.last_action_probs

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save the model and its VecNormalize statistics.

        Writes two files:
          - <path>        : the PPO model weights (model.zip convention)
          - <path>_vecnorm.pkl : serialised VecNormalize running stats

        Args:
            path: Full path for the model file (e.g. data/models/htf/model.zip).
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)

        vecnorm_path = self._vecnorm_path(path)
        self.vec_env.save(vecnorm_path)

        logger.info("Saved model -> %s", path)
        logger.info("Saved VecNormalize stats -> %s", vecnorm_path)

    def load(self, path: str) -> None:
        """
        Load a model (and companion VecNormalize stats) from disk.

        Args:
            path: Path to the model file previously saved with save().
        """
        self.model = PPO.load(path, env=self.vec_env)

        vecnorm_path = self._vecnorm_path(path)
        if os.path.exists(vecnorm_path):
            self.vec_env = VecNormalize.load(vecnorm_path, self.vec_env.venv)
            logger.info("Restored VecNormalize stats from %s", vecnorm_path)

        logger.info("Loaded model from %s", path)

    # ------------------------------------------------------------------
    # Interpretability helpers
    # ------------------------------------------------------------------

    def get_htf_action_interpretation(
        self,
        action: int,
        htf_alignment: int,
    ) -> str:
        """
        Return a human-readable description of the action in the context of
        the current HTF cascade alignment.

        Args:
            action:        The action chosen by the agent (0=Hold, 1=Long, 2=Short).
            htf_alignment: The macro HTF alignment signal:
                             +1 = bullish cascade (1D/4H/1H all bullish)
                             -1 = bearish cascade
                              0 = mixed / no clear alignment

        Returns:
            A descriptive string, e.g. "LONG (aligned)" or "SHORT (counter-trend!)".
        """
        label = self._ACTION_LABELS.get(action, "UNKNOWN")

        if action == 0:  # HOLD
            if htf_alignment == 0:
                return "HOLD (waiting for HTF alignment)"
            return "HOLD (in position / no signal)"

        if action == 1:  # LONG
            if htf_alignment == 1:
                return "LONG (aligned)"
            if htf_alignment == -1:
                return "LONG (counter-trend!)"
            return "LONG (mixed HTF)"

        if action == 2:  # SHORT
            if htf_alignment == -1:
                return "SHORT (aligned)"
            if htf_alignment == 1:
                return "SHORT (counter-trend!)"
            return "SHORT (mixed HTF)"

        return f"{label} (unknown alignment)"

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"HTFTradingAgent("
            f"steps_trained={self.training_steps}, "
            f"lr={self.config['learning_rate']}, "
            f"net_arch={self.config['net_arch']})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_htf_agent(
    env: gym.Env,
    config_path: Optional[str] = None,
    model_path: Optional[str] = None,
) -> HTFTradingAgent:
    """
    Factory function to create an HTFTradingAgent.

    Args:
        env:         A HTFTradingEnv (or compatible Gymnasium env).
        config_path: Optional path to a YAML config file.  The file should
                     have a top-level ``model`` key whose value is a dict of
                     hyperparameter overrides (same keys as _default_config).
        model_path:  Optional path to a pretrained model zip to resume from.

    Returns:
        A ready-to-use HTFTradingAgent instance.

    Example::

        env = HTFTradingEnv(df_15m, df_1h, df_4h, df_1d)
        agent = create_htf_agent(env, config_path="config/htf.yaml")
        metrics = agent.train(total_timesteps=1_500_000)
    """
    config: Optional[Dict[str, Any]] = None

    if config_path:
        import yaml  # optional dep — only required if config_path is used

        with open(config_path, "r") as fh:
            raw = yaml.safe_load(fh)
        config = raw.get("model", {}) if isinstance(raw, dict) else {}
        logger.info("Loaded agent config from %s: %s", config_path, config)

    return HTFTradingAgent(env=env, config=config, model_path=model_path)
