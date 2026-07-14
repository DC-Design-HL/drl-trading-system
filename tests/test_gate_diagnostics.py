"""Tests for src/brain/gate_diagnostics.py (2026-07-13 SOL gate work).

Pure-logic tests plus a fake-model/fake-env episode walk. No SB3 training,
safe to run on the server.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.brain.gate_diagnostics import (
    gate_stats, join_trades, policy_step_records, run_gate_diagnostics,
)
from src.brain.htf_agent import EntropyAnnealCallback


# ---------------------------------------------------------------------------
# join_trades
# ---------------------------------------------------------------------------

def _rec(step, margin=1.0, max_prob=0.9):
    return {"step": step, "action": 2, "position_before": 0,
            "max_prob": max_prob, "margin": margin, "entropy": 0.3}


def _trade(entry_step, pnl):
    return {"direction": "short", "entry": 100.0, "exit": 99.0,
            "pnl": pnl, "pnl_pct": pnl / 1000.0, "entry_step": entry_step,
            "exit_step": entry_step + 5}


def test_join_trades_matches_entry_step():
    records = [_rec(0), _rec(1, margin=2.5), _rec(2)]
    trades = [_trade(1, 10.0)]
    joined = join_trades(records, trades)
    assert len(joined) == 1
    assert joined[0]["margin"] == 2.5
    assert joined[0]["pnl"] == 10.0


def test_join_trades_skips_unmatched_entry():
    joined = join_trades([_rec(0)], [_trade(99, 5.0)])
    assert joined == []


# ---------------------------------------------------------------------------
# gate_stats
# ---------------------------------------------------------------------------

def test_gate_stats_discriminating_margin_scores_usable():
    # winners get high margins, losers low — perfect ranking
    joined = ([{**_rec(i, margin=5.0 + i * 0.1), "pnl": 10, "pnl_pct": 0.01}
               for i in range(20)]
              + [{**_rec(100 + i, margin=0.5 + i * 0.01), "pnl": -10,
                  "pnl_pct": -0.01} for i in range(20)])
    stats = gate_stats(joined)
    assert stats["margin_auc"] == 1.0
    assert stats["verdict"].startswith("USABLE")
    # quartile table monotone: Q4 (highest margin) all winners
    assert stats["pnl_by_margin_quartile"][3]["win_rate"] == 1.0
    assert stats["pnl_by_margin_quartile"][0]["win_rate"] == 0.0


def test_gate_stats_uninformative_margin_scores_none():
    # identical margin for winners and losers — AUC 0.5 by construction
    joined = ([{**_rec(i, margin=1.0), "pnl": 10, "pnl_pct": 0.01}
               for i in range(15)]
              + [{**_rec(50 + i, margin=1.0), "pnl": -10, "pnl_pct": -0.01}
                 for i in range(15)])
    stats = gate_stats(joined)
    assert stats["margin_auc"] == 0.5
    assert stats["verdict"].startswith("NONE")


def test_gate_stats_insufficient_trades():
    stats = gate_stats([_rec(0)])
    assert stats["verdict"] == "insufficient trades"
    assert stats["n_trades"] == 1


def test_gate_stats_saturation_summary_from_all_records():
    records = [_rec(i, max_prob=0.99) for i in range(10)]
    stats = gate_stats([], all_records=records)
    assert stats["saturated_step_pct"] == 100.0
    assert stats["conf_median_all_steps"] == pytest.approx(0.99)


# ---------------------------------------------------------------------------
# policy_step_records with a fake torch policy + minimal env
# ---------------------------------------------------------------------------

class _FakeDistInner:
    def __init__(self, logits):
        import torch
        self.logits = torch.tensor([logits], dtype=torch.float32)
        self.probs = torch.softmax(self.logits, dim=-1)


class _FakeDist:
    def __init__(self, logits):
        self.distribution = _FakeDistInner(logits)


class _FakePolicy:
    """Emits a fixed logits sequence, one per step."""
    def __init__(self, logit_seq):
        self.logit_seq = logit_seq
        self.calls = 0

    def obs_to_tensor(self, arr):
        import torch
        return torch.tensor(arr), False

    def get_distribution(self, _tensor):
        logits = self.logit_seq[min(self.calls, len(self.logit_seq) - 1)]
        self.calls += 1
        return _FakeDist(logits)


class _FakeModel:
    def __init__(self, logit_seq):
        self.policy = _FakePolicy(logit_seq)
        self._seq = logit_seq
        self._predict_calls = 0

    def predict(self, _arr, deterministic=True):
        logits = self._seq[min(self._predict_calls, len(self._seq) - 1)]
        self._predict_calls += 1
        return int(np.argmax(logits)), None


class _FakeEnv:
    """3-step env that opens a short on action 2 at step 1, closes at step 2."""
    def __init__(self):
        self.current_step = 0
        self.position = 0
        self.trades = []

    def reset(self):
        self.current_step = 0
        self.position = 0
        self.trades = []
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        if action == 2 and self.position == 0:
            self.position = -1
            self._entry_step = self.current_step
        elif action == 0 and self.position == -1:
            self.position = 0
            self.trades.append(_trade(self._entry_step, 25.0))
        self.current_step += 1
        done = self.current_step >= 3
        return np.zeros(4, dtype=np.float32), 0.0, done, False, {}


def test_policy_step_records_and_join_end_to_end():
    # step 0: HOLD (argmax 0), step 1: SHORT (argmax 2), step 2: HOLD -> close
    seq = [[3.0, 0.0, 1.0], [0.0, 1.0, 4.0], [5.0, 0.0, 2.0]]
    env = _FakeEnv()
    model = _FakeModel(seq)
    records = policy_step_records(model, None, env, deterministic=True)
    assert len(records) == 3
    # margin at step 1 = 4.0 - 1.0
    assert records[1]["margin"] == pytest.approx(3.0)
    assert records[1]["action"] == 2
    joined = join_trades(records, env.trades)
    assert len(joined) == 1
    assert joined[0]["step"] == 1
    assert joined[0]["pnl"] == 25.0


def test_run_gate_diagnostics_wrapper():
    seq = [[3.0, 0.0, 1.0], [0.0, 1.0, 4.0], [5.0, 0.0, 2.0]]
    stats = run_gate_diagnostics(_FakeModel(seq), None, _FakeEnv())
    assert stats["n_trades"] == 1
    assert stats["verdict"] == "insufficient trades"
    assert "saturated_step_pct" in stats


# ---------------------------------------------------------------------------
# entropy floor
# ---------------------------------------------------------------------------

class _EntModel:
    ent_coef = 0.0


def test_entropy_anneal_respects_equal_bounds():
    # ent_floor plumbing produces start==end==floor when floor dominates;
    # the callback must then hold ent_coef constant at the floor.
    cb = EntropyAnnealCallback(start_ent=0.02, end_ent=0.02, total_steps=100)
    cb.model = _EntModel()
    cb.num_timesteps = 0
    cb._on_training_start()
    for ts in (0, 50, 100, 200):
        cb.num_timesteps = ts
        cb._on_step()
        assert cb.model.ent_coef == pytest.approx(0.02)


def test_entropy_anneal_linear_when_floor_below():
    cb = EntropyAnnealCallback(start_ent=0.05, end_ent=0.02, total_steps=100)
    cb.model = _EntModel()
    cb.num_timesteps = 0
    cb._on_training_start()
    cb.num_timesteps = 50
    cb._on_step()
    assert cb.model.ent_coef == pytest.approx(0.035)
    cb.num_timesteps = 100
    cb._on_step()
    assert cb.model.ent_coef == pytest.approx(0.02)
