"""Regression tests for the 2026-06-26 anti-overfit HTF reward reform
(RETRAINING_PLAN.md §4.1). The reformed reward credits ONLY realized PnL plus
small symmetric structural costs — no unrealized hold credit, no asymmetric
SL/TP shaping, no regime multipliers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from train_htf_walkforward import build_htf_dataframes
from src.env.htf_env import HTFTradingEnv


def _synth_15m(n: int = 8000, seed: int = 11) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=n, freq="15min")
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.004))
    op = np.roll(close, 1); op[0] = close[0]
    hi = np.maximum(op, close) * (1 + np.abs(rng.standard_normal(n)) * 0.001)
    lo = np.minimum(op, close) * (1 - np.abs(rng.standard_normal(n)) * 0.001)
    vol = np.abs(rng.standard_normal(n)) * 1000 + 100
    return pd.DataFrame({"open_time": ts, "open": op, "high": hi, "low": lo,
                         "close": close, "volume": vol})


@pytest.fixture(scope="module")
def env():
    merged, d1h, d4h, d1d = build_htf_dataframes(_synth_15m())
    return HTFTradingEnv(df_15m=merged, df_1h=d1h, df_4h=d4h, df_1d=d1d,
                         initial_balance=10_000, position_size=0.25,
                         training_mode=True)


def test_env_builds_and_rewards_finite(env):
    obs, _ = env.reset()
    rewards = []
    rng = np.random.default_rng(0)
    for _ in range(300):
        obs, r, term, trunc, _ = env.step(int(rng.integers(0, 3)))
        rewards.append(r)
        if term or trunc:
            break
    rewards = np.array(rewards)
    assert np.all(np.isfinite(rewards))
    # Reformed reward is small-magnitude (realized pnl + small costs), never the
    # old +0.10/+0.15 shaped spikes.
    assert rewards.max() < 0.10


def test_holding_a_winner_gives_no_unrealized_credit(env):
    """Open a LONG already ~1% in profit (inside the TP band) and HOLD one bar.
    Old reward added +0.2×pnl_pct (a positive unrealized credit); the reform
    removes it, so the hold step's reward must be ~0."""
    env.reset()
    # Jump to a clean interior bar and open a long priced 1% BELOW the next
    # bar's close → we'll be in profit but well inside SL/TP when we hold.
    env.current_step = 500
    next_close = float(env.df_15m.iloc[env.current_step + 1]["close"])
    assert env.take_profit_pct > 0.011 and env.stop_loss_pct > 0.011, \
        "test assumes a TP/SL band wider than the 1% test move"
    env._open_position(next_close * 0.99, 1)   # 1% in-the-money long
    env.position = 1
    env.position_entry_step = env.current_step
    equity_before = env._calculate_equity()
    if equity_before > env.max_balance:
        env.max_balance = equity_before  # no spurious drawdown penalty

    _, reward, _, _, _ = env.step(0)  # HOLD

    # New reward for a pure hold inside the band = 0 (no unrealized credit).
    # Old reward would have been +0.2 * ~0.0101 ≈ +0.002.
    assert abs(reward) < 5e-4, f"unexpected hold reward {reward} (unrealized credit not removed?)"
