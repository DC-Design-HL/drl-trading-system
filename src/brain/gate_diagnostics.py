"""
Gate diagnostics — can the policy's output DISCRIMINATE good entries from bad?

Motivation (2026-07-13): the SOL model-gate attempt failed because live
confidence = max softmax prob, which saturates (~0.99 everywhere) after the
Phase-2 entropy anneal — nothing to gate on. But saturation of the PROBABILITY
does not mean the signal is gone: the LOGIT MARGIN (top-1 minus top-2 logit)
still varies between decisions and survives softmax saturation. These helpers
run a trained policy over an eval env, record per-step logit margins, join them
to the env's trade log via entry_step, and score discrimination (does margin
rank winners above losers?).

Used by scripts/sol_gate_diagnostic.py (Mac, existing fold models) and by
train_htf_walkforward.py (per-fold gate metrics on every future retrain).
"""

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def policy_step_records(model, vecnorm, env, deterministic: bool = True) -> List[Dict]:
    """Run one episode; return one record per step with logits-level detail.

    Each record: {step, action, position_before, max_prob, margin, entropy}.
    ``step`` is env.current_step at decision time — the same value the env
    stamps into trades as entry_step, so records join 1:1 to trades.
    """
    import torch

    records: List[Dict] = []
    obs, _ = env.reset()
    done = truncated = False

    while not (done or truncated):
        obs_in = vecnorm.normalize_obs(obs) if vecnorm is not None else obs
        arr = np.asarray(obs_in, dtype=np.float32).reshape(1, -1)
        with torch.no_grad():
            obs_tensor = model.policy.obs_to_tensor(arr)[0]
            dist = model.policy.get_distribution(obs_tensor)
            logits = dist.distribution.logits.detach().cpu().numpy()[0]
            probs = dist.distribution.probs.detach().cpu().numpy()[0]

        action, _ = model.predict(arr, deterministic=deterministic)
        action = int(action)

        top = np.sort(logits)
        records.append({
            "step": int(env.current_step),
            "action": action,
            "position_before": int(getattr(env, "position", 0)),
            "max_prob": float(np.max(probs)),
            "margin": float(top[-1] - top[-2]),
            "entropy": float(-(probs * np.log(probs + 1e-12)).sum()),
        })

        obs, _, done, truncated, _ = env.step(action)

    return records


def join_trades(records: List[Dict], trades: List[Dict]) -> List[Dict]:
    """Attach the decision-time record to each trade via entry_step."""
    by_step = {r["step"]: r for r in records}
    joined = []
    for t in trades:
        r = by_step.get(t.get("entry_step"))
        if r is None:
            continue
        joined.append({**r, "pnl": t["pnl"], "pnl_pct": t["pnl_pct"],
                       "direction": t["direction"]})
    return joined


def _auc(scores: List[float], labels: List[int]) -> float:
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    w = sum(1 for p in pos for n in neg if p > n) \
        + 0.5 * sum(1 for p in pos for n in neg if p == n)
    return w / (len(pos) * len(neg))


def _pearson(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 3:
        return float("nan")
    x, y = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    sx, sy = x.std(), y.std()
    if sx == 0 or sy == 0:
        return float("nan")
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))


def gate_stats(joined: List[Dict], all_records: Optional[List[Dict]] = None) -> Dict:
    """Score gate viability from margin-joined trades.

    Key output: margin_auc — AUC of logit margin predicting a winning trade.
    ~0.5 = the policy's certainty carries no information about outcome
    (no gate possible without retrain); >=0.55 = usable raw material.
    max_prob_auc is reported alongside to show what saturation destroys.
    """
    out: Dict = {"n_trades": len(joined)}
    if all_records:
        mp = [r["max_prob"] for r in all_records]
        out["conf_median_all_steps"] = float(np.median(mp))
        out["conf_p10_all_steps"] = float(np.percentile(mp, 10))
        out["saturated_step_pct"] = float(np.mean([p > 0.95 for p in mp]) * 100)
    if len(joined) < 10:
        out["verdict"] = "insufficient trades"
        return out

    margins = [j["margin"] for j in joined]
    probs = [j["max_prob"] for j in joined]
    pnls = [j["pnl_pct"] for j in joined]
    wins = [1 if p > 0 else 0 for p in pnls]

    out.update({
        "win_rate": float(np.mean(wins)),
        "margin_auc": round(_auc(margins, wins), 4),
        "max_prob_auc": round(_auc(probs, wins), 4),
        "margin_pnl_pearson": round(_pearson(margins, pnls), 4),
        "margin_median": round(float(np.median(margins)), 3),
        "margin_iqr": round(float(np.percentile(margins, 75)
                                  - np.percentile(margins, 25)), 3),
    })

    # pnl by margin quartile — the shape a threshold gate would exploit
    order = np.argsort(margins)
    q = max(len(order) // 4, 1)
    quartiles = []
    for qi in range(4):
        idx = order[qi * q:(qi + 1) * q] if qi < 3 else order[3 * q:]
        quartiles.append({
            "q": qi + 1,
            "n": len(idx),
            "avg_pnl_pct": round(float(np.mean([pnls[i] for i in idx])) * 100, 3),
            "win_rate": round(float(np.mean([wins[i] for i in idx])), 3),
        })
    out["pnl_by_margin_quartile"] = quartiles

    auc = out["margin_auc"]
    if np.isnan(auc):
        out["verdict"] = "degenerate (all wins or all losses)"
    elif auc >= 0.55:
        out["verdict"] = "USABLE — margin ranks outcomes; gate has raw material"
    elif auc >= 0.52:
        out["verdict"] = "WEAK — marginal ranking; gate unlikely to clear costs"
    else:
        out["verdict"] = "NONE — policy certainty carries no outcome information"
    return out


def run_gate_diagnostics(model, vecnorm, env,
                         deterministic: bool = True) -> Dict:
    """One-call wrapper: episode -> join -> stats."""
    records = policy_step_records(model, vecnorm, env, deterministic)
    trades = list(getattr(env, "trades", []))
    joined = join_trades(records, trades)
    return gate_stats(joined, all_records=records)
