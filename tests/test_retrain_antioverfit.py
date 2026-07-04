"""Anti-overfit regression tests for the HTF walk-forward retrain pipeline.

Covers the three critical defects fixed 2026-07-03 that could otherwise
reproduce the fake +5515% overfit:

  1. HTF forming-parent-bar lookahead in HTFDataAligner.get_parent_idx
  2. evaluate_agent feeding RAW (un-normalized) obs to the policy
  3. best-on-val checkpoint machinery (structural)
"""

import numpy as np
import pandas as pd
import pytest

from src.features.htf_features import HTFDataAligner


def _ohlc(index):
    return pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
        index=index,
    )


# ---------------------------------------------------------------------------
# 1. Parent-bar alignment must never read a still-forming HTF bar
# ---------------------------------------------------------------------------

def test_parent_idx_excludes_forming_bar():
    aligner = HTFDataAligner()
    # 15m children spanning 08:00..09:45
    df_child = _ohlc(pd.date_range("2026-01-01 08:00", periods=8, freq="15min"))
    # 1h parents labeled by OPEN time (left): 07:00, 08:00, 09:00
    df_parent = _ohlc(pd.date_range("2026-01-01 07:00", periods=3, freq="1h"))

    # child 08:15 -> the 08:00 1h bar is still forming (closes 09:00);
    # must select the last CLOSED bar = 07:00 (idx 0)
    assert aligner.get_parent_idx(df_child, df_parent, 1) == 0
    # child 09:30 -> forming bar is 09:00; last closed = 08:00 (idx 1)
    assert aligner.get_parent_idx(df_child, df_parent, 6) == 1


def test_parent_idx_invariant_no_future_leak():
    """Selected parent bar must have CLOSED at or before the child timestamp."""
    aligner = HTFDataAligner()
    df_child = _ohlc(pd.date_range("2026-01-01 08:00", periods=12, freq="15min"))
    df_parent = _ohlc(pd.date_range("2026-01-01 06:00", periods=6, freq="1h"))
    one_hour = pd.Timedelta("1h")
    for ci in range(len(df_child)):
        pj = aligner.get_parent_idx(df_child, df_parent, ci)
        parent_close = df_parent.index[pj] + one_hour  # left-labeled bar closes at label+dur
        assert parent_close <= df_child.index[ci], (
            f"child {df_child.index[ci]} read parent closing {parent_close} (future leak)"
        )


def test_parent_idx_boundary_clamps_to_zero():
    aligner = HTFDataAligner()
    df_child = _ohlc(pd.date_range("2026-01-01 06:05", periods=2, freq="15min"))
    df_parent = _ohlc(pd.date_range("2026-01-01 06:00", periods=3, freq="1h"))
    # No fully-closed parent exists yet at 06:05 -> clamp to 0, never negative
    assert aligner.get_parent_idx(df_child, df_parent, 0) == 0


# ---------------------------------------------------------------------------
# 2. evaluate_agent must normalize observations before predict()
# ---------------------------------------------------------------------------

class _RecordingVecNorm:
    def __init__(self):
        self.calls = 0

    def normalize_obs(self, obs):
        self.calls += 1
        # sentinel: normalized obs are distinguishable from the raw env obs
        return np.full_like(np.asarray(obs, dtype=float), 999.0)


class _RecordingAgent:
    def __init__(self, vec):
        self.vec_env = vec
        self.received = []

    def predict(self, obs, deterministic=True):
        self.received.append(np.asarray(obs, dtype=float))
        return 0, None, 1.0


class _OneStepEnv:
    def reset(self):
        return np.zeros(3), {}

    def step(self, action):
        return np.ones(3), 0.0, True, False, {}

    def get_episode_metrics(self):
        return {"sharpe_ratio": 1.0, "total_return_pct": 2.0}


def test_evaluate_agent_normalizes_obs():
    from train_htf_walkforward import evaluate_agent

    vec = _RecordingVecNorm()
    agent = _RecordingAgent(vec)
    evaluate_agent(agent, _OneStepEnv(), n_episodes=1)

    assert vec.calls >= 1, "normalize_obs was never called — RAW obs reached the policy"
    assert agent.received, "predict was never called"
    assert all(np.all(r == 999.0) for r in agent.received), (
        "predict received un-normalized observations (the fake-return bug)"
    )


def test_evaluate_agent_survives_without_vecnorm():
    """No VecNormalize stats -> must not crash (falls back to raw obs + warns)."""
    from train_htf_walkforward import evaluate_agent

    class _NoVec:
        vec_env = None

        def predict(self, obs, deterministic=True):
            return 0, None, 1.0

    out = evaluate_agent(_NoVec(), _OneStepEnv(), n_episodes=1)
    assert isinstance(out, dict)


# ---------------------------------------------------------------------------
# 3. Best-on-val checkpoint callback (structural)
# ---------------------------------------------------------------------------

def test_best_vecnorm_callback_is_eval_callback():
    from stable_baselines3.common.callbacks import EvalCallback
    from src.brain.htf_agent import BestVecNormalizeEvalCallback

    assert issubclass(BestVecNormalizeEvalCallback, EvalCallback)
    # the vecnorm-save hook attribute exists on the class contract
    assert "vecnorm_save_path" in BestVecNormalizeEvalCallback.__init__.__code__.co_varnames


# ---------------------------------------------------------------------------
# 4. load_15m_csv accepts the downloader's date-stamped filename
# ---------------------------------------------------------------------------

def test_load_15m_csv_falls_back_to_datestamped_name(tmp_path):
    from train_htf_walkforward import load_15m_csv

    idx = pd.date_range("2026-01-01", periods=5, freq="15min", tz="UTC")
    df = pd.DataFrame({
        "open_time": idx, "open": 1.0, "high": 1.0, "low": 1.0,
        "close": 1.0, "volume": 1.0,
    })
    # write only the date-stamped file the downloader actually produces
    df.to_csv(tmp_path / "BTCUSDT_15m_20230101_20260101.csv", index=False)

    # caller passes the plain name (which does NOT exist) -> must resolve the glob
    out = load_15m_csv(str(tmp_path / "BTCUSDT_15m.csv"))
    assert len(out) == 5

    # genuinely-missing data still raises
    with pytest.raises(FileNotFoundError):
        load_15m_csv(str(tmp_path / "NOTHERE_15m.csv"))


# ---------------------------------------------------------------------------
# 5. Deploy verdict gates on OUT-OF-SAMPLE performance, not just the abs ratio
# ---------------------------------------------------------------------------

def test_deploy_verdict_gates_on_oos():
    from train_htf_walkforward import _deploy_verdict

    # the exact smoke-test case: val looked great, OOS negative, small abs ratio.
    # Old logic said "EXCELLENT"; it must now say NO EDGE.
    v = _deploy_verdict(oos_sharpe_mean=-2.13, positive_fold_pct=0.0,
                        avg_overfit_ratio=0.79)
    assert "NO EDGE" in v
    assert "EXCELLENT" not in v and "GOOD" not in v

    # positive & consistent OOS -> GOOD
    assert "GOOD" in _deploy_verdict(1.5, 100.0, 1.0)
    # profitable OOS but big val->test gap -> OVERFIT
    assert "OVERFIT" in _deploy_verdict(0.6, 60.0, 4.0)
    # zero OOS Sharpe is not deployable
    assert "NO EDGE" in _deploy_verdict(0.0, 100.0, 1.0)


if __name__ == "__main__":  # pragma: no cover
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
