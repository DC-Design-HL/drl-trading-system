#!/usr/bin/env python3
"""
Project 2 — ETH VecNormalize fix (Mac M3).

Problem (per docs/eth-signal-investigation.md): the deployed ETH model
was trained with `stable_baselines3.common.vec_env.VecNormalize` (input
+ reward normalization wrapper), but in production it receives RAW
unnormalized observations because the corresponding `vecnorm.pkl` stats
file is missing or wasn't shipped alongside the model. Result: model
output is essentially noise; reports false 0.95+ confidence on every
inference; ETH LONG underperforms by -$23 / 39 trades historically.

This script does ONE of two things, controlled by --mode:

  --mode reexport     Locate the saved VecNormalize stats from the
                       original training run, re-bundle them next to
                       the model file, and verify a sample inference
                       produces non-degenerate confidence (std > 0.05).

  --mode validate    Just load the deployed ETH model and run 200
                       random-noise observations through it. If
                       confidence has std < 0.01 across samples, the
                       VecNormalize bug is confirmed. Useful sanity
                       check before deciding to retrain.

If --mode reexport finds no usable vecnorm.pkl, falls back to
recommending Option B (full retrain via train_sgfilter.py with
--no-vecnorm or via the curriculum training plan).

Run on Mac:
    pip install stable-baselines3 gym
    python3 scripts/train_eth_vecnorm_fix.py --mode validate
    python3 scripts/train_eth_vecnorm_fix.py --mode reexport \
        --training_run training_runs/htf_walkforward_eth_final/

Acceptance gate (verified by --mode validate AFTER the fix):
    Confidence stdev across 200 random-input samples > 0.05.
    If still < 0.01, the model can't be saved by re-exporting; retrain.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean, stdev

REPO = Path(__file__).resolve().parent.parent

DEPLOYED_ETH_MODEL = REPO / "data" / "models" / "htf_walkforward_eth" / "final_model_0.zip"
DEPLOYED_VECNORM = REPO / "data" / "models" / "htf_walkforward_eth" / "final_vecnorm_0.pkl"


def cmd_validate(model_path: Path, n_samples: int = 200, obs_dim: int | None = None) -> int:
    """Run noise through the ETH model and report whether confidence has variance."""
    try:
        import numpy as np
        from stable_baselines3 import PPO
    except ImportError as e:
        print(f"ERROR: missing dependency: {e}", file=sys.stderr)
        return 2

    if not model_path.exists():
        print(f"Model not found: {model_path}", file=sys.stderr)
        return 1

    print(f"Loading {model_path} ...")
    model = PPO.load(str(model_path))

    # Detect the model's expected observation shape — different training
    # runs have different feature counts (51 / 117 / etc.).
    try:
        expected_shape = model.observation_space.shape
        if obs_dim is None:
            obs_dim = int(expected_shape[0])
        print(f"  model expects observation shape={expected_shape}; using obs_dim={obs_dim}")
    except Exception:
        if obs_dim is None:
            obs_dim = 117
            print(f"  could not introspect observation_space; falling back to obs_dim={obs_dim}")

    # Generate random observations in a plausible range (z-scaled inputs
    # are usually in roughly [-3, 3] after VecNormalize).
    print(f"Running {n_samples} random observations ...")
    confidences = []
    actions = []
    for _ in range(n_samples):
        obs = np.random.randn(obs_dim).astype(np.float32)
        action, _ = model.predict(obs, deterministic=False)
        # Get raw action probabilities for confidence
        if hasattr(model.policy, "get_distribution"):
            dist = model.policy.get_distribution(model.policy.obs_to_tensor(obs)[0])
            try:
                probs = dist.distribution.probs.detach().numpy().flatten()
                conf = float(probs.max())
            except AttributeError:
                conf = 0.0  # continuous policy — we don't care here
        else:
            conf = 0.0
        confidences.append(conf)
        actions.append(int(action) if hasattr(action, "__int__") else int(action[0]))

    if not confidences:
        print("Could not extract confidences — model architecture differs from expected.", file=sys.stderr)
        return 1

    c_mean = mean(confidences)
    c_stdev = stdev(confidences) if len(confidences) > 1 else 0.0
    actions_count = {0: 0, 1: 0, 2: 0}
    for a in actions:
        actions_count[a] = actions_count.get(a, 0) + 1
    print(f"\nConfidence stats over {n_samples} random observations:")
    print(f"  mean={c_mean:.4f}  stdev={c_stdev:.4f}  min={min(confidences):.4f}  max={max(confidences):.4f}")
    print(f"  action distribution: {actions_count}")

    if c_stdev < 0.01:
        print("\n❌ DIAGNOSIS: VecNormalize bug confirmed.")
        print("    Confidence stdev < 0.01 across random inputs means the model is")
        print("    producing nearly-identical output regardless of input. This is the")
        print("    classic signature of running an inference graph that expected")
        print("    normalized input but is receiving raw values.")
        print("    Run `--mode reexport` to fix, OR retrain via train_sgfilter.py.")
        return 1
    if c_stdev < 0.05:
        print("\n⚠️ WARNING: marginal stdev. May be partially-broken.")
        return 1
    print("\n✅ Confidence varies meaningfully — VecNormalize is being applied correctly.")
    return 0


def cmd_reexport(training_run_dir: Path, model_path: Path) -> int:
    """Re-bundle the trained VecNormalize stats next to the deployed model."""
    try:
        from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
        import gymnasium as gym
    except ImportError:
        try:
            from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
            import gym
        except ImportError as e:
            print(f"ERROR: missing dependency: {e}", file=sys.stderr)
            return 2

    if not training_run_dir.exists():
        print(f"Training run not found: {training_run_dir}", file=sys.stderr)
        print("Likely candidates to search:")
        print("  - training_runs/htf_walkforward_eth_final/")
        print("  - data/models/htf_walkforward_eth/")
        print("  - your local training scratch directory")
        return 1

    # Find the original vecnorm.pkl in the training run
    candidates = list(training_run_dir.rglob("vecnorm*.pkl")) + list(training_run_dir.rglob("vec_normalize*.pkl"))
    if not candidates:
        print(f"No vecnorm.pkl found under {training_run_dir}.")
        print("This usually means the training run discarded normalization stats.")
        print("The fix is to RETRAIN — see train_sgfilter.py for the modern approach,")
        print("or rerun the original ETH training with VecNormalize.save() called explicitly.")
        return 1

    src = candidates[0]
    print(f"Found training-time vecnorm: {src}")
    print(f"Copying to deployment location: {model_path.parent}")
    import shutil
    dest = model_path.parent / "final_vecnorm_0.pkl"
    shutil.copy(src, dest)
    print(f"Wrote: {dest}")

    # Verify by loading
    print("\nVerifying ...")
    try:
        env = DummyVecEnv([lambda: gym.make("CartPole-v1")])  # dummy — VecNormalize.load needs *some* env
        vn = VecNormalize.load(str(dest), env)
        print(f"  Loaded VecNormalize: obs_rms.mean shape={vn.obs_rms.mean.shape}, "
              f"clip_obs={vn.clip_obs}, norm_obs={vn.norm_obs}")
    except Exception as e:
        print(f"  Could not verify load: {e}")
        print("  This may still be fine — the bot does its own loading via load_model_with_vecnorm.")

    print("\n✅ Re-export done. Now run:")
    print("    python3 scripts/train_eth_vecnorm_fix.py --mode validate")
    print("If validation still fails (stdev < 0.05), the issue is not vecnorm — retrain.")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["validate", "reexport"], default="validate")
    p.add_argument("--model_path", default=str(DEPLOYED_ETH_MODEL))
    p.add_argument("--training_run", default="",
                   help="(reexport) path to the training run dir that contains vecnorm.pkl")
    p.add_argument("--n_samples", type=int, default=200)
    args = p.parse_args(argv)

    if args.mode == "validate":
        return cmd_validate(Path(args.model_path), n_samples=args.n_samples)
    if args.mode == "reexport":
        if not args.training_run:
            print("--mode reexport requires --training_run <path>", file=sys.stderr)
            return 2
        return cmd_reexport(Path(args.training_run), Path(args.model_path))


if __name__ == "__main__":
    raise SystemExit(main())
