#!/usr/bin/env python3
"""
HTF Live Trading Bot
====================
Runs the Hierarchical Multi-Timeframe (HTF) PPO agent for BTCUSDT.

Features across 4 timeframes (1D, 4H, 1H, 15M) → 117-dim observation.
Mirrors all decisions to Binance testnet when TESTNET_MIRROR=true.
Saves trades to the same MongoDB/JSON storage as the main system.

Usage:
    python live_trading_htf.py --dry-run              # Paper trading only
    python live_trading_htf.py --interval 15          # 15-minute decision loop
    python live_trading_htf.py --balance 5000         # Custom starting balance
"""

import sys
import os
import time
import json
import logging
import argparse
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=False)
except ImportError:
    pass

import numpy as np
import pandas as pd

# Limit PyTorch thread pools — 4 bots on 2 CPUs, 1 thread each reduces RAM and CPU contention
try:
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.features.htf_features import HTFFeatureEngine, HTFDataAligner
from src.data.multi_asset_fetcher import MultiAssetDataFetcher
from src.data.storage import get_storage
from src.signals.bos_choch import MarketStructure

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s - %(message)s",
)
logger = logging.getLogger("htf_live")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SYMBOL = os.environ.get("HTF_SYMBOL", "BTCUSDT")
# STATE_FILE is set dynamically after --symbol is parsed (see _get_state_file())
TRADES_FILE = Path("logs/htf_trades.json")


def _get_state_file(symbol: str = None) -> Path:
    """Return symbol-specific state file path.

    Honors DRL_STATE_DIR env var so a dry-run consolidated process can
    isolate its state from the live 4-bot deployment writing the same
    symbol files.
    """
    sym = symbol or SYMBOL
    state_dir = os.environ.get("DRL_STATE_DIR")
    if state_dir:
        return Path(state_dir) / f"htf_trading_state_{sym}.json"
    if sym == "BTCUSDT":
        return Path("logs/htf_trading_state.json")  # backwards compat
    return Path(f"logs/htf_trading_state_{sym}.json")

# ── Risk Management (Fixed-Dollar-Risk Model) ──
# Risk pool = RISK_POOL_PCT of balance, divided into RISK_BUDGET_PARTS
# Each trade risks exactly (risk_pool / budget_parts) dollars.
# Margin is derived from liquidation buffer: liq must be SL + LIQ_BUFFER_PCT from entry.
#
# Aggressive-sizing rollout 2026-04-25 (target 30%/mo on $5K base):
#   - RISK_POOL_PCT 0.10 → 0.30 (3× per-trade dollar_risk: $25 → $75)
#   - FIXED_MAX_NOTIONAL 3000 → 6000 (allow the larger notional through)
# Backtested 32.4%/mo on the recent 20d regime, 10% max DD.
# Restore point: tag v-pre-aggressive-sizing-20260425.
#
# Stepped 0.30 → 0.20 (3× → 2×) on 2026-04-27 after equity drew down -8.18%
# in 42h. Per-symbol filters added in same commit (see SYMBOL_SIDE_BLOCKLIST
# below) to prevent the loser combos that drove the drawdown.
# Restore point for the 3×-no-filter state: tag v-3x-no-filters-20260427.
RISK_POOL_PCT = 0.20       # 20% of balance is the risk pool (was 0.30, originally 0.10)
RISK_BUDGET_PARTS = 20     # Pool divided into 20 equal risk slots (unchanged)
LIQ_BUFFER_PCT = 0.01      # Liquidation must be 1% beyond SL from entry
MAX_LEVERAGE = 50          # Hard cap on leverage (Binance testnet limit)

STOP_LOSS_PCT = 0.015  # 1.5% stop loss
TAKE_PROFIT_PCT = 0.030  # 3.0% take profit
TRADING_FEE = 0.0004   # 0.04% taker fee

# Trailing stop configuration
TRAILING_BREAKEVEN_PCT = 0.008   # At +0.8% profit, activate trailing stop (was 0.5% — too tight)
TRAILING_LOCK_PCT = 0.02         # At +2% profit, lock 50% of profit (legacy, kept for BOS overlay)
TRAILING_DISTANCE_PCT = 0.005    # Trail 0.5% behind peak (was 0.3% — was cutting winners too early)
TRAILING_DISTANCE_POST_TP1 = 0.008  # Trail 0.8% behind peak after TP1 hit — backtested +$181 (Apr 6-14)

# Anti-overtrading guards
COOLDOWN_SECONDS = 1800   # 30 min after a stopped-out trade
MIN_HOLD_SECONDS = 3600   # 1 hour minimum hold (HTF signal is slower)

# Time-based stagnant exit (hold > STAGNANT_HOURS, pnl in [STAGNANT_PCT_MIN, STAGNANT_PCT_MAX]).
# Widened lower bound from -0.3% → -1.0% on 2026-04-25 after 35-day backtest
# (456 reconstructed round-trips Mar 22 → Apr 25): widening to [-1.0%, +0.5%]
# improved net pnl by +$49 over baseline, the most robust improvement (positive
# in both 20d and 35d windows, 34 trades affected). Catches drifters that miss
# SL but are clearly off-thesis at the 6h mark.
STAGNANT_HOURS = 6.0
STAGNANT_PCT_MIN = -0.010
STAGNANT_PCT_MAX = 0.005

# Anti-whipsaw: block quick losing reversals (Strategy E from backtest_whipsaw.py)
# If previous trade was opposite direction, closed <2h ago, AND lost money → skip.
# Backtested +$75.92 improvement; reverse-close pattern is 1W/12L.
WHIPSAW_COOLDOWN_HOURS = 2.0

# ── Asymmetric REVERSE_CLOSE_LONG guard (canary rollout) ──
# Walk-forward counterfactual (2026-04-23, 66 trades over Apr 6-22) showed
# REVERSE_CLOSE_LONG is systematically −$297 over the window while the
# "held instead" counterfactual was +$1,462 — delta +$1,711 on LONG side,
# 95% CI [+$1,408, +$1,976], P(delta>0)=100% across 10k bootstrap resamples.
# Pattern is sign-consistent across all 4 symbols and both observed regimes
# (uptrend and range). SHORT reversals have the OPPOSITE sign (blocking them
# would cost ~$740), so guard applies to LONG only.
#
# Regime gate: the walk-forward window contained zero sustained BTC downtrend.
# In a downtrend the LONG-reversal-is-wrong logic could plausibly invert, so
# we only block when BTC 4h EMA slope over 20h > MIN_SLOPE_PCT (not in clear
# downtrend). Graceful degradation to current behavior otherwise.
#
# Starts as a canary on XRP only (biggest offender: 11 trades, −$109 actual,
# +$566 projected delta). Expand to BTC/ETH/SOL after 2 weeks of live signal.
REVERSAL_BLOCK_LONG_CANARY_SYMBOLS: set = {"XRPUSDT", "BTCUSDT", "ETHUSDT", "SOLUSDT"}
REVERSAL_BLOCK_LONG_REGIME_GATE_MIN_SLOPE_PCT = -0.5
_BTC_REGIME_CACHE_TTL_SECONDS = 900

# ── Open-Interest observational logging (Phase 2, 2026-05-01) ────────────
# Pre-test on 30d × 4 symbols showed a directional pattern (high OI ROC →
# negative forward 24h returns) that failed Bonferroni under non-overlap
# subsampling — autocorrelation artifact at our data window. Backtest of
# 5 candidate filter rules on 312 actual trades gave best directional rule
# +$58 (CI [-$184, +$339]). Insufficient power to deploy as a guard.
# Decision (Chen, 2026-05-01): log OI features at every trade open with
# zero decision-making impact. Re-test in 4-6 weeks against trades whose
# entry-time features are directly logged (no historical reconstruction).
_OI_CACHE_TTL_SECONDS = 900

# Actions from PPO (must match HTFTradingEnv)
ACTION_HOLD = 0
ACTION_LONG = 1
ACTION_SHORT = 2
ACTION_LABELS = {ACTION_HOLD: "HOLD", ACTION_LONG: "LONG", ACTION_SHORT: "SHORT"}

# Minimum confidence to act on a signal.
#
# 2026-05-03 — Raised 0.45 → 0.55 based on 90d btengine ablation:
# raising to 0.55 saves ~$150/month vs default; raising to 0.75 saves
# ~$320/month but cuts trades 85% (too aggressive for first step).
# 0.55 is the conservative canary; can move to 0.65/0.75 after 1-2
# weeks of live validation. See:
#   - docs/backtest_redesign_proposal.md
#   - data/training/btengine_signal_ablation_90d.json
#   - memory project_signal_ablation_90d_may02.md
# Structure-first SKIP wrapper removed at the same time — see comment
# at the use site (~line 2925) for why this gate now applies in both
# model-first and structure-first modes.
MIN_CONFIDENCE = 0.55

# Per-symbol minimum confidence overrides (symbols with poor low-conf performance)
SYMBOL_MIN_CONFIDENCE = {
    "ETHUSDT": 0.80,  # ETH low-conf trades had 0% SHORT WR, -$396 PnL
}

# Per-symbol directional confidence floors (temporary, until regime veto validated over 100+ trades)
# ETH LONGs: 43% WR, -$22.89 in Apr 7 analysis — require higher conviction for longs
# SOL LONGs (Apr 9-10): 1/3 wins, -$20.06 net — losers were conf 0.82/0.91, winner was 0.987
SYMBOL_DIRECTIONAL_CONF: dict = {
    "ETHUSDT": {"LONG": 0.95},
    "SOLUSDT": {"LONG": 0.95},
}

# Ranging regime filter: raise confidence threshold when ADX is low
RANGING_MIN_CONFIDENCE = 0.80  # Need higher conviction in ranging markets
RANGING_ADX_THRESHOLD = 20.0   # ADX below this = ranging

# Momentum exhaustion: skip entries when price is extended
EXHAUSTION_ATR_THRESHOLD = 3.0  # Skip if price > 3 ATR from 20-bar VWAP

# ── Market Signal Gate ──
# Tier 1: conf >= SIGNAL_GATE_AUTONOMOUS → model decides alone
# Tier 2: conf < SIGNAL_GATE_AUTONOMOUS → needs SIGNAL_GATE_MIN_CONFIRMS out of 4 signals
# Signals: MTF alignment, Order Flow, Regime, Orderbook Imbalance
SIGNAL_GATE_AUTONOMOUS = 0.80       # Above this, model acts alone
SIGNAL_GATE_MIN_CONFIRMS = 2        # Need at least 2/4 signals to agree
SIGNAL_GATE_OF_THRESHOLD = 0.20     # Order flow score magnitude to count as directional
SIGNAL_GATE_OB_THRESHOLD = 0.30     # Orderbook imbalance magnitude to count as directional
SIGNAL_GATE_REGIME_ADX_MIN = 25.0   # ADX must be above this for regime to count as opposing

# ── Funding-rate gate (LONG-only) ──
# Block LONG entries when 8h funding rate < 0 (longs paying shorts =
# market structure leans short). Backtested 2026-04-27 over 288 closes:
#   Blocked 51 trades that had 35.3% WR / -$9.24/trade in aggregate.
#   Incremental Δ on top of blocklist: +$284 over 20-day window
#   (~+$425/30d at 1×, ~+$850/30d at 2×).
# Asymmetric — does not affect SHORTs (they benefit from this regime).
# scripts/backtest_phase3_filters.py is the harness.
FUNDING_LONG_GUARD_ENABLED = True
FUNDING_LONG_GUARD_MIN_RATE = 0.0  # Block if funding strictly less than this

# ── Whale-NEUTRAL block ──
# When the whale-direction predictor returns NEUTRAL, the bot performs
# poorly: 37.3% WR / -$8.67 avg over 60 historical entries. The signal
# isn't bullish or bearish — it's "no confidence." Backtested incremental
# Δ on top of blocklist: +$413 over 20-day window.
# Block both LONGs and SHORTs when whale_dir=NEUTRAL.
WHALE_NEUTRAL_GUARD_ENABLED = True

# ── Extreme-positive news fade guard (LONG-only) ──
# Skip LONG entries when news in the trailing 4h has sentiment_score > 0.5
# (asset-tagged or untagged/global). Backed by independent forward-return
# correlation analysis: extreme-positive sentiment fades over 60-240m at
# p=2.5e-6 (Bonferroni-OK over 76 hypotheses). Adoption-event news fades
# even harder: -41 bps @ 240m (Bonferroni-OK).
#
# Backtest on production-filtered trades (8-day news window, Apr 21-29):
#   Blocked 7-10 LONGs, ALL LOSERS (0% WR), ~$158-224 of pnl prevented.
# Asymmetric — does not affect SHORTs.
# Reads news from data/trading.db news_events table populated by
# news_sentinel.py.
EXT_POS_NEWS_GUARD_ENABLED = True
EXT_POS_NEWS_SENTIMENT_THRESHOLD = 0.5  # >this counts as "extreme positive"
EXT_POS_NEWS_LOOKBACK_HOURS = 4
_EXT_POS_NEWS_CACHE_TTL_SECONDS = 60      # short cache; news arrives continuously
_ext_pos_news_cache: dict = {}            # symbol → (ts, hit_bool, latest_match_dict)


# ── Per-symbol-side blocklist (loser combos) ──
# Backtest 2026-04-26 over 288 round-trips (Apr 6 → Apr 26) showed clear
# asymmetric per-symbol/side performance:
#   Profitable: SOL SHORT (75.6% WR), XRP both, BTC LONG  → +$332
#   Losing:    BTC SHORT (40.7% WR -$68), ETH SHORT (-$11),
#              ETH LONG (-$23), SOL LONG (-$59)            → -$162
# Blocking the 4 losing combos at entry would have saved +$162 on the
# historical sample with no other change. Sample sizes 27-41 per bucket
# — meaningful, not bulletproof. Re-validate after 2 weeks of fresh data.
# Backtest harness: scripts/backtest_signal_filters.py.
SYMBOL_SIDE_BLOCKLIST: set = {
    ("BTCUSDT", "SHORT"),
    ("ETHUSDT", "SHORT"),
    ("ETHUSDT", "LONG"),
    ("SOLUSDT", "LONG"),
}

# ── fake_bos / fake_choch entry guard ──
# The structure detector flags suspicious BOS/CHOCH events as fake_bos /
# fake_choch (wick rejection > 70% of body, or rapid reversal in 3 bars).
# The bot already uses these flags to TIGHTEN SL post-entry.
#
# DISABLED 2026-04-27 19:30 UTC. Briefly enabled at 12:08 (commit 8eb762e)
# without backtest validation. In 5h of live traffic the guard blocked
# SOLUSDT SHORT (76% WR / +$177 historical — the bot's best combo) 4×
# and BTCUSDT LONG (also a winner) 2×, while the data showed it never
# blocked an actual losing combo. The fake-flag is firing on ~50% of
# all structure events; without a per-trade fake_bos audit log we cannot
# distinguish predictive fakes from noise. Re-enable only after the trade
# DB is instrumented to capture the flag and a real backtest is run.
# Code path retained so it can be flipped on without redeployment.
FAKE_BOS_ENTRY_GUARD_ENABLED = False


# ── USDT Dominance (USDT.D) Filter ──
# When stablecoin dominance is rising, capital is fleeing crypto → bad time
# to LONG. We use a synthetic proxy: inverse momentum of the 4-symbol crypto
# basket. If the basket dropped > USDT_D_THRESHOLD_PCT over the lookback
# window, treat USDT.D as "rising" and block LONG entries. SHORTs are
# unaffected (they benefit from this regime).
#
# Backtest evidence: docs/backtest-signal-combinations-2026-04-21.md and
# backtest_dominance_filter.py — over 248 trades the variant
# "ADX 15-40 + USDT.D 2h rising blocks LONG" lifted WR 58.5% → 65.8% and
# pnl +$265 → +$431 (+62%, +7.3% WR).
USDT_D_GUARD_ENABLED = True
USDT_D_LOOKBACK_HOURS = 2
# Threshold widened 0.5 → 0.7 on 2026-04-26 after threshold-sensitivity backtest
# (scripts/backtest_usdt_d_threshold.py, 152 LONG trades over 20d):
#   -0.5% deployed: blocked 17 trades, blocking netted +$3.84 (i.e. mild loss
#                   vs no-filter, blocking some net-winners)
#   -0.7% chosen:   blocks 13 trades, allowed-set WR rises 54.6% → 56.1%,
#                   net delta +$23 over 20d at 1× / ~$120/mo at 3× sizing
# -0.7% was the most robust positive option (-0.4% was higher EV but noisier,
# dominated by a single trade swing). All 5 worst LONG losers had basket
# < -0.9% so they're still blocked at the wider threshold.
USDT_D_THRESHOLD_PCT = 0.7   # 4-symbol basket must drop > 0.7% to flag "USDT.D rising"
USDT_D_PROXY_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
_USDT_D_CACHE_TTL_SECONDS = 600   # 10 min — share across all 4 bot threads
_usdt_d_cache: dict = {"ts": 0, "rising": False, "basket_change_pct": 0.0}


# ── Orderbook Guard (Golden Guard) ──
# Blocks trades when orderbook bias contradicts the trade direction.
# Applies to ALL tiers (including Tier 1 autonomous) — orderbook divergence
# is the strongest single predictor of whipsaw reversals.
# Backtested Mar 24-31: WR 46.3%→51.8%, PnL +217%, catches 3/3 whipsaw shorts.
ORDERBOOK_GUARD_ENABLED = True

# ── RSI Extreme Guard ──
# Blocks trades at overbought/oversold extremes on the 15m timeframe.
# Don't LONG when RSI > threshold (overbought), don't SHORT when RSI < threshold (oversold).
# Backtested Mar 24-31: catches RSI=18 shorts and RSI=83 longs that hit SL within minutes.
RSI_GUARD_ENABLED = True
RSI_GUARD_OB_THRESHOLD = 70         # Don't LONG above this RSI (overbought)
RSI_GUARD_OS_THRESHOLD = 30         # Don't SHORT below this RSI (oversold)
# Regime-aware loosened thresholds (added 2026-04-11): in confirmed strong
# trends (regime detector reports TRENDING_UP/DOWN with ADX >= TREND_ADX_MIN),
# RSI > 70 / < 30 is normal trend behavior, not exhaustion. The base 70/30
# ceiling was over-applying in trends and blocking high-conviction longs
# during the bull leg on 2026-04-11. Loosen the band only when the regime
# detector confirms we are NOT in a range. ADX >= 30 = "strong trend" per
# Wilder's classification; we already have ADX < 20 as the ranging block.
RSI_GUARD_OB_TREND = 80             # LONG ceiling when TRENDING_UP + ADX >= TREND_ADX_MIN
RSI_GUARD_OS_TREND = 20             # SHORT floor when TRENDING_DOWN + ADX >= TREND_ADX_MIN
RSI_GUARD_TREND_ADX_MIN = 25        # Wilder's "trending" cutoff (ADX >= 25 = real trend, not noise)

# ── ADX Ranging Guard ──
# Blocks trades when ADX is very low (no trend, choppy/ranging market).
# In ranging markets, trend-following signals produce whipsaws.
# Backtested Mar 24-31: catches ADX=10-13 losses in directionless markets.
ADX_GUARD_ENABLED = True
# 2026-05-03 — Tightened bounds [20, 60] → [25, 50] per 90d ablation.
# At [25, 50]: btengine showed +$433/90d (~$144/month) vs [20, 60].
# At [15, 70] (lax): -$199/90d. Monotonic — tighter is better in our
# data window. The original [20, 60] was conservative; this canary
# variant trades fewer entries (-30% expected) for higher per-trade EV.
# Reverts: change back to 20/60 in this file and restart.
# See: data/training/btengine_signal_ablation_90d.json
ADX_GUARD_MIN = 25                  # Block all trades when ADX below this
# ADX exhaustion: ADX>60 was deployed 2026-04-25 based on 53-trade study;
# 90d ablation (2026-05-03) showed [25, 50] cap is strictly better than
# [25, 60] in this data window — overextended-trend exits start losing
# above ~50, not 60. Tightened to 50.
ADX_GUARD_MAX = 50                  # Block all trades when ADX above this

# ── Rescue Rule ──
# Override RSI/ADX blocks when the model has HIGH confidence AND multiple signals agree.
# This saves ~7 winning trades that would otherwise be blocked, while only letting through ~2 losses.
# Backtested: $172→$227 PnL, 52.7%→56.2% WR.
RESCUE_ENABLED = False
RESCUE_MIN_CONFIDENCE = 0.90        # Model confidence threshold for rescue
RESCUE_MIN_AGREES = 2               # Minimum signal agreements for rescue (out of 4: MTF, OF, whale, OB)

# Phase 1 §3.5: Hard cap on notional per trade to prevent martingale compounding.
# Raised 3000 → 6000 on 2026-04-25 to accommodate the 3× position-size scaling.
FIXED_MAX_NOTIONAL = 6000.0  # USDT

# ── Structure-First Mode ──
# When True, BOS/CHOCH signals are the PRIMARY entry trigger.
# The DRL model is NOT used for entry decisions (only for exit management).
# Backtested over 31 days: 41.0% WR, +$7,146 vs model-first 27.2% WR, -$7,075.
STRUCTURE_FIRST_MODE = True

# Per-symbol structure config:
# "S1" = struct only (BOS/CHOCH triggers entry, no OB filter, no model)
# "S5" = struct + OB proximity + RSI + ADX directional confirmation
STRUCTURE_SYMBOL_CONFIG = {
    "BTCUSDT": "S1",   # 47.8% WR, +$3,798 in backtest
    "ETHUSDT": "S5",   # 43.5% WR, +$1,217 with extra filters
    "SOLUSDT": "S1",   # 37.0% WR, +$770
    "XRPUSDT": "S1",   # 44.4% WR, +$2,336
}
STRUCTURE_OB_PROXIMITY_PCT = 0.010  # 1% proximity for OB check (S5 mode)

# How many 15M bars to fetch (10 days = ~960 bars)
FETCH_DAYS = 12


# ---------------------------------------------------------------------------
# Model path resolution
# ---------------------------------------------------------------------------

def _find_best_fold_model(wf_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Find the best model (highest OOS Sharpe) in a walk-forward directory.

    Checks two formats:
      1. v3 ensemble: top3_models.json + final_model_X.zip (preferred)
      2. Legacy: fold_XX/best_model.zip + fold_XX/fold_result.json
    """
    # --- v3 ensemble format (from train_model.py) ---
    top3_file = wf_dir / "top3_models.json"
    if top3_file.exists():
        try:
            top3 = json.loads(top3_file.read_text())
            best = top3["top_folds"][0]  # rank 0 = best
            rank = best["rank"]
            model_zip = wf_dir / f"final_model_{rank}.zip"
            vecnorm = wf_dir / f"final_vecnorm_{rank}.pkl"
            if model_zip.exists():
                sharpe = best.get("val_sharpe", 0)
                logger.info(
                    "HTF model: v3 ensemble rank 0 (val Sharpe %.2f) → %s",
                    sharpe, model_zip,
                )
                return model_zip, vecnorm if vecnorm.exists() else None
        except Exception:
            pass

    # --- Legacy fold format ---
    best_sharpe = -999.0
    best_model_path: Optional[Path] = None
    best_vecnorm_path: Optional[Path] = None

    for fold_dir in sorted(wf_dir.iterdir()):
        result_file = fold_dir / "fold_result.json"
        model_zip = fold_dir / "best_model.zip"
        if not fold_dir.is_dir():
            continue
        if not (result_file.exists() and model_zip.exists()):
            continue
        try:
            result = json.loads(result_file.read_text())
            oos_sharpe = float(result.get("oos_sharpe", result.get("sharpe", result.get("test_metrics", {}).get("sharpe_ratio", -999))))
            if oos_sharpe > best_sharpe:
                best_sharpe = oos_sharpe
                best_model_path = model_zip
                vn = fold_dir / "vecnorm.pkl"
                if not vn.exists():
                    vn = fold_dir / "best_model_vecnorm.pkl"
                if not vn.exists():
                    vn = fold_dir / "fold_model_vecnorm.pkl"
                best_vecnorm_path = vn if vn.exists() else None
        except Exception:
            continue

    if best_model_path:
        logger.info(
            "HTF model: walk-forward best fold (OOS Sharpe %.2f) → %s",
            best_sharpe, best_model_path,
        )
    return best_model_path, best_vecnorm_path


def find_best_htf_model(symbol: str = "BTCUSDT") -> Tuple[Optional[Path], Optional[Path]]:
    """
    Return (model_path, vecnorm_path) for the best available HTF model.

    Search order (symbol-aware):
      1. data/models/htf_walkforward_<asset>/ — best OOS Sharpe fold (e.g. htf_walkforward_eth)
      2. data/models/htf_walkforward_50pct_v2/ — BTC default walk-forward
      3. data/models/htf/best_model.zip        — any saved HTF model
      4. data/models/wfv2/BTCUSDT/ppo/fold_00/ — walk-forward fallback
    """
    root = Path("data/models")

    # Derive asset name from symbol (e.g. BTCUSDT -> btc, ETHUSDT -> eth)
    asset = symbol.replace("USDT", "").lower()

    # 1. Symbol-specific walk-forward directory (e.g. htf_walkforward_eth)
    symbol_wf_dir = root / f"htf_walkforward_{asset}"
    if symbol_wf_dir.exists():
        model_path, vecnorm_path = _find_best_fold_model(symbol_wf_dir)
        if model_path:
            return model_path, vecnorm_path

    # 2. Walk-forward 50pct directory (8 folds) — BTC default
    wfv_dir = root / "htf_walkforward_50pct_v2"
    if wfv_dir.exists():
        model_path, vecnorm_path = _find_best_fold_model(wfv_dir)
        if model_path:
            return model_path, vecnorm_path

    # 2. Generic HTF model directory
    htf_best = root / "htf" / "best_model.zip"
    if htf_best.exists():
        vn = root / "htf" / "best_model_vecnorm.pkl"
        logger.info("HTF model: data/models/htf/best_model.zip")
        return htf_best, vn if vn.exists() else None

    # 3. Walk-forward v2 fold_00 (oldest fallback — older PPO, not HTF-specific)
    wfv2_model = root / "wfv2" / "BTCUSDT" / "ppo" / "fold_00" / "best_model.zip"
    wfv2_vecnorm = root / "wfv2" / "BTCUSDT" / "ppo" / "fold_00" / "vecnorm.pkl"
    if wfv2_model.exists():
        logger.warning(
            "HTF-specific model not found. Falling back to wfv2/fold_00 — "
            "this model was NOT trained on 117-dim HTF features; inference "
            "will run in heuristic-only mode."
        )
        return wfv2_model, wfv2_vecnorm if wfv2_vecnorm.exists() else None

    logger.error("No usable model found. Bot will HOLD on every step.")
    return None, None


# ---------------------------------------------------------------------------
# BTC regime slope helper — shared across all symbol threads
# ---------------------------------------------------------------------------

# Module-level cache so 4 consolidated-mode symbol threads share one fetch.
_btc_regime_cache = {"slope_pct": None, "ts": 0.0}


def _compute_btc_4h_ema_slope_pct(fetcher) -> Optional[float]:
    """Return BTC 4h EMA(10) slope over a trailing 20h window, as percent.

    Result is cached for _BTC_REGIME_CACHE_TTL_SECONDS so the 4 symbol
    threads running in the consolidated bot process share one fetch.
    Returns the last cached value (possibly stale) if the fetch fails;
    returns None only if no value has ever been computed.
    """
    now = time.time()
    cached = _btc_regime_cache.get("slope_pct")
    if cached is not None and (now - _btc_regime_cache["ts"]) < _BTC_REGIME_CACHE_TTL_SECONDS:
        return cached
    try:
        df = fetcher.fetch_asset("BTCUSDT", "4h", days=7)
        if df is None or len(df) < 15:
            return cached
        closes = df["close"].astype(float).values

        def _ema(vals, period: int) -> float:
            k = 2.0 / (period + 1)
            e = float(vals[0])
            for v in vals[1:]:
                e = float(v) * k + e * (1 - k)
            return e

        ema_now = _ema(closes[-10:], 10)
        ema_prev = _ema(closes[-15:-5], 10)
        if ema_prev == 0:
            return cached
        slope_pct = (ema_now - ema_prev) / ema_prev * 100.0
        _btc_regime_cache["slope_pct"] = slope_pct
        _btc_regime_cache["ts"] = now
        return slope_pct
    except Exception as exc:
        logger.debug("BTC regime slope fetch failed: %s", exc)
        return cached


# ── Open-Interest observational logging ─────────────────────────────────
# Per-symbol cache, shared across symbol threads in the consolidated bot.
_oi_features_cache: Dict[str, Dict] = {}


def _compute_oi_features(symbol: str) -> Optional[Dict]:
    """Fetch 1h Binance Futures OI hist + 1h klines for `symbol` and return
    the 5 candidate features for observational logging at trade-open time.

    Phase 2 OI investigation (2026-05-01): the pre-test found a directionally
    consistent but Bonferroni-failing signal at our 30d × hourly resolution.
    This helper exists purely to record entry-time OI state on every trade so
    we can re-test in 4-6 weeks with directly-logged data, no historical
    reconstruction required.

    Returns a dict with keys: oi_roc_4h, oi_roc_24h, oi_z_7d, oi_value_roc_4h,
    px_roc_4h, computed_at. Returns None on hard failure (e.g. fetch error
    with no cache yet) — callers should treat None as "no observation".
    Never raises.
    """
    now = time.time()
    cached = _oi_features_cache.get(symbol)
    if cached and (now - cached.get("ts", 0)) < _OI_CACHE_TTL_SECONDS:
        return cached.get("features")

    try:
        import urllib.request as _urlreq
        oi_url = (f"https://fapi.binance.com/futures/data/openInterestHist"
                  f"?symbol={symbol}&period=1h&limit=200")
        with _urlreq.urlopen(oi_url, timeout=8) as r:
            oi_data = json.loads(r.read())
        kl_url = (f"https://fapi.binance.com/fapi/v1/klines"
                  f"?symbol={symbol}&interval=1h&limit=200")
        with _urlreq.urlopen(kl_url, timeout=8) as r:
            kl_data = json.loads(r.read())

        if not oi_data or not kl_data or len(oi_data) < 25 or len(kl_data) < 25:
            return cached.get("features") if cached else None

        oi_data.sort(key=lambda d: int(d["timestamp"]))
        kl_data.sort(key=lambda k: int(k[0]))
        oi_arr = [float(d["sumOpenInterest"]) for d in oi_data]
        oi_v_arr = [float(d["sumOpenInterestValue"]) for d in oi_data]
        px_arr = [float(k[4]) for k in kl_data]

        def _pct(arr, k):
            if len(arr) <= k: return None
            base = arr[-(k + 1)]
            if base == 0: return None
            return (arr[-1] - base) / base * 100.0

        def _zscore(arr, w):
            if len(arr) < w: return None
            window = arr[-w:]
            m = sum(window) / w
            var = sum((x - m) ** 2 for x in window) / max(w - 1, 1)
            sd = var ** 0.5
            if sd == 0: return None
            return (arr[-1] - m) / sd

        feats = {
            "oi_roc_4h":       _pct(oi_arr, 4),
            "oi_roc_24h":      _pct(oi_arr, 24),
            "oi_z_7d":         _zscore(oi_arr, 168),
            "oi_value_roc_4h": _pct(oi_v_arr, 4),
            "px_roc_4h":       _pct(px_arr, 4),
            "computed_at":     int(now),
        }
        _oi_features_cache[symbol] = {"features": feats, "ts": now}
        return feats
    except Exception as exc:
        logger.debug("OI features fetch failed for %s: %s", symbol, exc)
        return cached.get("features") if cached else None


def _is_usdt_dominance_rising(fetcher) -> Optional[bool]:
    """Return True if USDT.D proxy is RISING over USDT_D_LOOKBACK_HOURS.

    Synthetic proxy: average % change of the 4-symbol crypto basket over
    the lookback window. If the basket dropped > USDT_D_THRESHOLD_PCT,
    USDT.D is treated as rising (capital fleeing crypto). Cached across
    all 4 bot threads. Returns None only on hard failure (no cache yet).
    """
    if not USDT_D_GUARD_ENABLED:
        return False
    now = time.time()
    cached = _usdt_d_cache.get("rising")
    cached_change = _usdt_d_cache.get("basket_change_pct")
    if _usdt_d_cache.get("ts") and (now - _usdt_d_cache["ts"]) < _USDT_D_CACHE_TTL_SECONDS:
        return cached
    try:
        per_sym = {}
        for sym in USDT_D_PROXY_SYMBOLS:
            df = fetcher.fetch_asset(sym, "1h", days=1)
            if df is None or len(df) < USDT_D_LOOKBACK_HOURS + 1:
                per_sym[sym] = None
                continue
            closes = df["close"].astype(float).values
            t_now = float(closes[-1])
            t_lookback = float(closes[-(USDT_D_LOOKBACK_HOURS + 1)])
            if t_lookback <= 0:
                per_sym[sym] = None
                continue
            per_sym[sym] = (t_now - t_lookback) / t_lookback * 100.0
        valid = [v for v in per_sym.values() if v is not None]
        if not valid:
            logger.warning(
                "USDT.D proxy: all %d symbol fetches returned no data — fail-open",
                len(USDT_D_PROXY_SYMBOLS),
            )
            return None
        avg_basket_change = sum(valid) / len(valid)
        rising = avg_basket_change <= -USDT_D_THRESHOLD_PCT
        # Diagnostic: visible in bots_live.log so we can see what the filter
        # observed on every cache refresh, even when it didn't fire.
        logger.info(
            "USDT.D proxy: basket %+.3f%% over %dh (threshold -%.2f%%) → rising=%s | "
            "BTC=%s ETH=%s SOL=%s XRP=%s",
            avg_basket_change, USDT_D_LOOKBACK_HOURS, USDT_D_THRESHOLD_PCT, rising,
            f"{per_sym.get('BTCUSDT'):+.3f}%" if per_sym.get('BTCUSDT') is not None else "n/a",
            f"{per_sym.get('ETHUSDT'):+.3f}%" if per_sym.get('ETHUSDT') is not None else "n/a",
            f"{per_sym.get('SOLUSDT'):+.3f}%" if per_sym.get('SOLUSDT') is not None else "n/a",
            f"{per_sym.get('XRPUSDT'):+.3f}%" if per_sym.get('XRPUSDT') is not None else "n/a",
        )
        _usdt_d_cache["rising"] = rising
        _usdt_d_cache["basket_change_pct"] = avg_basket_change
        _usdt_d_cache["ts"] = now
        return rising
    except Exception as exc:
        logger.warning("USDT.D proxy fetch failed: %s — fail-open", exc)
        return None


# ---------------------------------------------------------------------------
# HTF Live Bot
# ---------------------------------------------------------------------------

class HTFLiveBot:
    """
    Live trading bot using the Hierarchical Multi-Timeframe PPO agent.

    One instance trades BTCUSDT at 15-minute resolution.
    """

    def __init__(
        self,
        dry_run: bool = True,
        initial_balance: float = 5_000.0,
        interval_minutes: int = 15,
        symbol: str = None,
    ):
        self.symbol = symbol or SYMBOL
        self.dry_run = dry_run
        self.initial_balance = initial_balance
        self.interval_minutes = interval_minutes

        # Portfolio state
        self.balance = initial_balance
        self._last_exchange_balance = initial_balance  # cache for _get_real_balance fallback
        self.position = 0        # 1 = LONG, -1 = SHORT, 0 = FLAT
        self.position_price = 0.0
        self.position_units = 0.0
        self.sl_price = 0.0
        self.tp_price = 0.0
        self.peak_price = 0.0
        self.current_price = 0.0
        self.realized_pnl = 0.0

        # Phase 1 §3.5: Session balance — captured once at startup for stable position sizing
        self.session_balance = initial_balance

        # Phase 1 §3.9: MFE/MAE tracking per trade
        self.mfe_pct = 0.0   # Max Favorable Excursion (best unrealized %) since open
        self.mae_pct = 0.0   # Max Adverse Excursion (worst unrealized %, negative) since open

        # Phase 1 §3.3: Partial take-profit state
        self.sl_pct = 0.0               # SL % at open for R-multiple calculations
        self.initial_position_units = 0.0  # Total units at open (for partial size math)
        self.partial_tp_level = 0       # 0=none, 1=Level1 filled (40%), 2=Level2 filled (35%)
        self.partial_tp1_price = 0.0    # Level 1 TP price: entry ± 1.0 × sl_pct
        self.partial_tp2_price = 0.0    # Level 2 TP price: entry ± 2.0 × sl_pct

        # Phase 1 §3.4: Entry time tracking for time-based stagnant exit
        self.position_entry_time = 0.0

        # Wall-clock time of the most recent partial-TP fire. Used by
        # _sync_with_exchange to suppress upward position-units syncs for a
        # grace window after a partial close, since the exchange may not yet
        # reflect the reduce-only partial-close fill.
        self.last_partial_close_time = 0.0

        # Cache of last fetched 15m DataFrame (used by _open_position for regime detection)
        self._last_df = None

        # Anti-overtrading
        self.last_loss_time = 0.0
        self.last_entry_time = 0.0

        # Anti-whipsaw: track last closed trade for Strategy E filter
        self.last_close_direction = 0    # 1=was LONG, -1=was SHORT
        self.last_close_pnl = 0.0        # PnL of last closed trade
        self.last_close_time = 0.0       # epoch time of last close

        # Bookkeeping
        self.trades: List[Dict] = []
        self.start_time = datetime.now().isoformat()
        self._lock = threading.Lock()

        # Feature pipeline
        self.aligner = HTFDataAligner()
        self.feature_engine = HTFFeatureEngine()
        self.fetcher = MultiAssetDataFetcher()

        # BOS/CHOCH market structure detector
        # swing_lookback=8 for 5m candles (noisier than 15m, needs wider window)
        self.market_structure = MarketStructure(swing_lookback=8)
        self._last_structure_signals: Dict = {}
        self._last_signal_gate: Dict = {}  # Last signal gate decision for alerts

        # Phase 1 §3.2: Regime detector for adaptive SL/TP multipliers
        try:
            from src.features.regime_detector import MarketRegimeDetector
            self.regime_detector = MarketRegimeDetector()
        except Exception as _exc:
            logger.warning("Regime detector unavailable: %s — using fixed SL/TP", _exc)
            self.regime_detector = None

        # Storage (shared with main bot)
        self.storage = get_storage()

        # Model
        self.model: Optional[PPO] = None
        self.vec_normalize: Optional[VecNormalize] = None
        self._model_path: Optional[Path] = None
        self._load_model()

        # Testnet executor
        self.testnet_executor = None
        if os.getenv("TESTNET_MIRROR", "").lower() in ("1", "true", "yes"):
            self._init_testnet()

        # Symbol-specific state file
        self._state_file = _get_state_file(self.symbol)

        # Restore persisted state
        self._load_state()
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        TRADES_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Sync balance from exchange on startup for testnet bots
        # (handles case where no state file exists yet)
        if not self.dry_run and not self._state_file.exists():
            self._sync_balance_from_exchange()

        # Always verify position against exchange on startup
        if not self.dry_run:
            self._sync_position_from_exchange()

        logger.info(
            "HTFLiveBot ready | symbol=%s dry_run=%s balance=%.2f",
            self.symbol, self.dry_run, self.balance,
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        model_path, vecnorm_path = find_best_htf_model(self.symbol)
        if model_path is None:
            logger.warning("No model loaded — bot will HOLD on every step.")
            return

        try:
            self.model = PPO.load(str(model_path))
            self._model_path = model_path
            logger.info("PPO model loaded from %s", model_path)
        except Exception as exc:
            logger.error("Failed to load PPO model: %s", exc)
            self.model = None
            return

        if vecnorm_path and vecnorm_path.exists():
            try:
                import gymnasium as gym
                from gymnasium import spaces as gym_spaces
                # Create a dummy env whose observation space matches the model's
                # (117 dims for HTF features). Using CartPole (4 dims) caused a
                # shape mismatch that silently broke normalization.
                n_features = 117  # HTF feature count
                dummy_env = gym.Env()
                dummy_env.observation_space = gym_spaces.Box(
                    low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32,
                )
                dummy_env.action_space = gym_spaces.Discrete(3)  # HOLD/LONG/SHORT
                dummy_env.reset = lambda **kw: (np.zeros(n_features, dtype=np.float32), {})
                dummy_env.step = lambda a: (np.zeros(n_features, dtype=np.float32), 0.0, True, False, {})
                dummy_venv = DummyVecEnv([lambda: dummy_env])
                self.vec_normalize = VecNormalize.load(str(vecnorm_path), dummy_venv)
                self.vec_normalize.training = False
                self.vec_normalize.norm_reward = False
                logger.info("✅ VecNormalize stats loaded from %s (obs_shape=%s)",
                            vecnorm_path, self.vec_normalize.observation_space.shape)
            except Exception as exc:
                logger.warning("Could not load VecNormalize: %s — raw obs used", exc)
                self.vec_normalize = None

    # ------------------------------------------------------------------
    # Testnet executor
    # ------------------------------------------------------------------

    def _init_testnet(self) -> None:
        try:
            from src.api.testnet_executor import get_testnet_executor
            self.testnet_executor = get_testnet_executor()
            if self.testnet_executor:
                mode = "futures (real longs/shorts)" if getattr(self.testnet_executor, '_futures_executor', None) else "spot"
                logger.info("Testnet mirror enabled (%s)", mode)

                # On startup: verify SL/TP orders exist for all open exchange positions.
                # The FuturesTestnetExecutor already does this in __init__, but log it
                # explicitly so the HTF bot startup log shows the verification happened.
                futures_exec = getattr(self.testnet_executor, '_futures_executor', None)
                if futures_exec:
                    logger.info(
                        "Startup SL/TP verification: SL orders=%s, TP orders=%s",
                        dict(futures_exec._sl_orders),
                        dict(futures_exec._tp_orders),
                    )
            else:
                logger.warning("Testnet mirror: executor not available (keys missing?)")
        except Exception as exc:
            logger.warning("Testnet init failed: %s", exc)

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _save_state(self) -> None:
        state = {
            "symbol": self.symbol,
            "balance": self.balance,
            "position": self.position,
            "position_price": self.position_price,
            "position_units": self.position_units,
            "sl_price": self.sl_price,
            "tp_price": self.tp_price,
            "peak_price": self.peak_price,
            "realized_pnl": self.realized_pnl,
            "last_loss_time": self.last_loss_time,
            "last_entry_time": self.last_entry_time,
            "start_time": self.start_time,
            "model_path": str(self._model_path) if self._model_path else None,
            "session_balance": self.session_balance,
            "mfe_pct": self.mfe_pct,
            "mae_pct": self.mae_pct,
            "sl_pct": self.sl_pct,
            "initial_position_units": self.initial_position_units,
            "partial_tp_level": self.partial_tp_level,
            "partial_tp1_price": self.partial_tp1_price,
            "partial_tp2_price": self.partial_tp2_price,
            "position_entry_time": self.position_entry_time,
            "last_close_direction": self.last_close_direction,
            "last_close_pnl": self.last_close_pnl,
            "last_close_time": self.last_close_time,
            "updated_at": datetime.now().isoformat(),
        }
        # Save to local HTF state file
        try:
            self._state_file.write_text(json.dumps(state, indent=2))
        except Exception as exc:
            logger.error("Failed to save state: %s", exc)
        # Also update shared MongoDB state so Live Portfolio reflects HTF positions
        try:
            pos_label = {1: "LONG", -1: "SHORT", 0: "FLAT"}.get(self.position, "FLAT")
            shared_state = self.storage.load_state()
            if not shared_state:
                shared_state = {}
            assets = shared_state.get("assets", {})
            assets[self.symbol] = {
                "position": self.position,
                "position_label": pos_label,
                "entry_price": self.position_price,
                "units": self.position_units,
                "pnl": self.realized_pnl,
                "sl_price": self.sl_price,
                "tp_price": self.tp_price,
                "source": "htf_agent",
            }
            shared_state["assets"] = assets
            # Source-of-truth: query exchange for the SHARED USDT balance
            # rather than `self.balance + sum(per-symbol pnl)`. The latter is
            # buggy when 4 symbols share one wallet — each bot tracks its own
            # `self.balance` independently and the saved total_balance would
            # be whoever-saved-last's stale per-symbol view. (Fixed 2026-05-03.)
            try:
                real_balance = self._get_real_balance()
                if real_balance and real_balance > 0:
                    shared_state["total_balance"] = real_balance
                else:
                    # Fallback to legacy compute if exchange unreachable
                    shared_state["total_balance"] = self.balance + sum(
                        a.get("pnl", 0) for a in assets.values()
                    )
            except Exception:
                shared_state["total_balance"] = self.balance + sum(
                    a.get("pnl", 0) for a in assets.values()
                )
            self.storage.save_state(shared_state)
        except Exception as exc:
            logger.debug("Failed to update shared state: %s", exc)

    def _load_state(self) -> None:
        if not self._state_file.exists():
            return
        try:
            state = json.loads(self._state_file.read_text())
            # Guard: only load state that belongs to THIS symbol
            saved_symbol = state.get("symbol", self.symbol)
            if saved_symbol != self.symbol:
                logger.warning(
                    "State file symbol mismatch: file has %s, bot is %s — ignoring stale state",
                    saved_symbol, self.symbol,
                )
                return
            self.balance = float(state.get("balance", self.initial_balance))
            self.position = int(state.get("position", 0))
            self.position_price = float(state.get("position_price", 0.0))
            self.position_units = float(state.get("position_units", 0.0))
            self.sl_price = float(state.get("sl_price", 0.0))
            self.tp_price = float(state.get("tp_price", 0.0))
            self.peak_price = float(state.get("peak_price", 0.0))
            self.realized_pnl = float(state.get("realized_pnl", 0.0))
            self.last_loss_time = float(state.get("last_loss_time", 0.0))
            self.last_entry_time = float(state.get("last_entry_time", 0.0))
            self.start_time = state.get("start_time", self.start_time)
            # Phase 1 new fields — defaults keep existing state files loading correctly
            self.session_balance = float(state.get("session_balance", self.balance))
            self.mfe_pct = float(state.get("mfe_pct", 0.0))
            self.mae_pct = float(state.get("mae_pct", 0.0))
            self.sl_pct = float(state.get("sl_pct", 0.0))
            self.initial_position_units = float(state.get("initial_position_units", self.position_units))
            self.partial_tp_level = int(state.get("partial_tp_level", 0))
            self.partial_tp1_price = float(state.get("partial_tp1_price", 0.0))
            self.partial_tp2_price = float(state.get("partial_tp2_price", 0.0))
            self.position_entry_time = float(state.get("position_entry_time", 0.0))
            self.last_close_direction = int(state.get("last_close_direction", 0))
            self.last_close_pnl = float(state.get("last_close_pnl", 0.0))
            self.last_close_time = float(state.get("last_close_time", 0.0))
            logger.info(
                "State restored: pos=%d price=%.2f balance=%.2f",
                self.position, self.position_price, self.balance,
            )
            # Sync balance from exchange for testnet bots (not dry_run)
            if not self.dry_run:
                self._sync_balance_from_exchange()
            # If in a position but SL/TP are missing, recalculate from entry price
            if self.position != 0 and self.position_price > 0:
                if self.sl_price <= 0:
                    if self.position == 1:  # LONG: SL below entry
                        self.sl_price = round(self.position_price * (1 - STOP_LOSS_PCT), 2)
                    else:  # SHORT: SL above entry
                        self.sl_price = round(self.position_price * (1 + STOP_LOSS_PCT), 2)
                    logger.warning(
                        "⚠️ SL was $0 for %s position — recalculated from entry: SL=$%.2f",
                        "LONG" if self.position == 1 else "SHORT", self.sl_price,
                    )
                if self.tp_price <= 0:
                    if self.position == 1:  # LONG: TP above entry
                        self.tp_price = round(self.position_price * (1 + TAKE_PROFIT_PCT), 2)
                    else:  # SHORT: TP below entry
                        self.tp_price = round(self.position_price * (1 - TAKE_PROFIT_PCT), 2)
                    logger.warning(
                        "⚠️ TP was $0 for %s position — recalculated from entry: TP=$%.2f",
                        "LONG" if self.position == 1 else "SHORT", self.tp_price,
                    )
                if self.peak_price <= 0:
                    self.peak_price = self.position_price
                self._save_state()
            # Check liquidation safety on state restore (if in a position)
            if self.position != 0:
                self._check_liquidation_safety()
        except Exception as exc:
            logger.warning("Could not restore state: %s", exc)

    def _sync_balance_from_exchange(self) -> None:
        """
        Fetch real USDT balance from Binance Futures testnet and override
        the internally tracked balance.

        This prevents the bot from using inflated balances accumulated during
        training/simulation. Only called for testnet bots (not dry_run).
        """
        try:
            from src.api.futures_executor import get_futures_executor
            executor = get_futures_executor()
            if executor is None:
                logger.warning("Balance sync: futures executor unavailable, keeping state balance")
                return

            real_balance = executor.get_account_balance('USDT')
            if real_balance <= 0:
                logger.warning(
                    "Balance sync: exchange returned $%.2f — keeping state balance $%.2f",
                    real_balance, self.balance,
                )
                return

            old_balance = self.balance
            if abs(old_balance - real_balance) > 1.0:  # Only log if meaningful difference
                logger.info(
                    "💰 Balance sync: state=$%.2f → exchange=$%.2f (corrected)",
                    old_balance, real_balance,
                )
            else:
                logger.info(
                    "💰 Balance sync: state=$%.2f ≈ exchange=$%.2f (in sync)",
                    old_balance, real_balance,
                )

            self.balance = real_balance
            self._last_exchange_balance = real_balance  # Update cache for _get_real_balance fallback
            # Keep initial_balance as the original testnet starting balance ($5,000)
            # so that PnL % in alerts reflects total performance, not per-session.
            # session_balance is used for position sizing only.
            self.session_balance = real_balance  # Update session balance for position sizing cap
            self._save_state()
        except Exception as exc:
            logger.warning("Balance sync failed: %s — keeping state balance $%.2f", exc, self.balance)

    def _sync_position_from_exchange(self) -> None:
        """
        Sync position state with the actual exchange position.
        The exchange is the source of truth — covers both cases:
        1. Bot has position but exchange is FLAT → reset to flat
        2. Bot is FLAT but exchange has position → adopt exchange position
        """
        if self.dry_run:
            return
        try:
            from src.api.futures_executor import get_futures_executor
            executor = get_futures_executor()
            if executor is None:
                return
            positions = executor.connector.get_positions()
            # Find our symbol's position on the exchange
            exchange_pos = None
            for p in positions:
                if p.get("symbol") == self.symbol:
                    exchange_pos = p
                    break

            if exchange_pos is None:
                real_amt = 0.0
            else:
                real_amt = abs(float(exchange_pos.get("positionAmt", 0)))

            # Case 2: Bot thinks FLAT but exchange has a position — adopt it
            if real_amt > 0 and self.position == 0:
                raw_amt = float(exchange_pos.get("positionAmt", 0))
                exchange_dir = 1 if raw_amt > 0 else -1
                exchange_entry = float(exchange_pos.get("entryPrice", 0))
                logger.warning(
                    "⚠️ POSITION RECOVERY: bot was FLAT but exchange has %s %s "
                    "(entry=$%.2f, qty=%.6f). Adopting exchange position.",
                    "LONG" if exchange_dir == 1 else "SHORT",
                    self.symbol, exchange_entry, real_amt,
                )
                self.position = exchange_dir
                self.position_price = exchange_entry
                self.position_units = real_amt
                self.peak_price = exchange_entry
                # Set SL/TP from exchange orders if available, else recalculate
                if exchange_dir == 1:  # LONG
                    self.sl_price = round(exchange_entry * (1 - STOP_LOSS_PCT), 2)
                    self.tp_price = round(exchange_entry * (1 + TAKE_PROFIT_PCT), 2)
                else:  # SHORT
                    self.sl_price = round(exchange_entry * (1 + STOP_LOSS_PCT), 2)
                    self.tp_price = round(exchange_entry * (1 - TAKE_PROFIT_PCT), 2)
                self._save_state()
                logger.info(
                    "✅ Position recovered from exchange: %s %s @ $%.2f, SL=$%.2f, TP=$%.2f",
                    "LONG" if exchange_dir == 1 else "SHORT",
                    self.symbol, exchange_entry, self.sl_price, self.tp_price,
                )
                return

            if real_amt == 0.0 and self.position != 0:
                # Exchange says flat but bot thinks it has a position — reset
                logger.warning(
                    "⚠️ STALE POSITION DETECTED: bot has %s %s but exchange is FLAT. "
                    "Resetting to flat (likely closed on exchange side).",
                    "LONG" if self.position == 1 else "SHORT",
                    self.symbol,
                )
                # Record a synthetic close trade for tracking
                close_action = "CLOSE_LONG" if self.position == 1 else "CLOSE_SHORT"
                self.position = 0
                self.position_price = 0.0
                self.position_units = 0.0
                self.sl_price = 0.0
                self.tp_price = 0.0
                self.peak_price = 0.0
                self.partial_tp_level = 0
                self.partial_tp1_price = 0.0
                self.partial_tp2_price = 0.0
                self.mfe_pct = 0.0
                self.mae_pct = 0.0
                self._save_state()
                logger.info("✅ Bot state reset to FLAT for %s", self.symbol)
                return

            if real_amt > 0:
                # Check direction mismatch (bot says LONG but exchange has SHORT, or vice versa)
                raw_amt = float(exchange_pos.get("positionAmt", 0))
                exchange_dir = 1 if raw_amt > 0 else -1  # positive = LONG, negative = SHORT
                if exchange_dir != self.position:
                    exchange_entry = float(exchange_pos.get("entryPrice", 0))
                    logger.warning(
                        "⚠️ DIRECTION MISMATCH: bot has %s but exchange has %s %s (entry=$%.2f, qty=%.6f). "
                        "Syncing to exchange state.",
                        "LONG" if self.position == 1 else "SHORT",
                        "LONG" if exchange_dir == 1 else "SHORT",
                        self.symbol, exchange_entry, real_amt,
                    )
                    self.position = exchange_dir
                    self.position_price = exchange_entry if exchange_entry > 0 else self.position_price
                    self.position_units = real_amt
                    self.peak_price = self.position_price
                    # Recalculate SL/TP for the correct direction
                    if exchange_dir == 1:  # LONG
                        self.sl_price = round(self.position_price * (1 - STOP_LOSS_PCT), 2)
                        self.tp_price = round(self.position_price * (1 + TAKE_PROFIT_PCT), 2)
                    else:  # SHORT
                        self.sl_price = round(self.position_price * (1 + STOP_LOSS_PCT), 2)
                        self.tp_price = round(self.position_price * (1 - TAKE_PROFIT_PCT), 2)
                    self._save_state()
                    logger.info(
                        "✅ Synced to exchange: %s %s @ $%.2f, SL=$%.2f, TP=$%.2f",
                        "LONG" if exchange_dir == 1 else "SHORT",
                        self.symbol, self.position_price, self.sl_price, self.tp_price,
                    )
                elif abs(real_amt - self.position_units) > 0.0001:
                    # Only accept DOWNWARD corrections. Upward jumps are almost
                    # always a partial-TP timing artifact: the bot has just
                    # decremented position_units for a reduce-only partial close
                    # that hasn't yet reflected on the exchange side. Trusting
                    # the exchange here re-adds the closed units and causes the
                    # eventual full-close pnl to double-count (see
                    # memory/project_pnl_accounting_bug_apr23.md).
                    grace = 60.0
                    within_grace = (time.time() - self.last_partial_close_time) < grace
                    if real_amt > self.position_units + 1e-4:
                        logger.warning(
                            "📐 Position sync skipped: exchange reports larger "
                            "size (bot=%.6f, exchange=%.6f) — suspected partial-"
                            "close settlement lag%s. Keeping bot-tracked units.",
                            self.position_units, real_amt,
                            " (within 60s grace after partial)" if within_grace else "",
                        )
                    elif within_grace:
                        logger.info(
                            "📐 Position sync skipped: within %.0fs grace after "
                            "partial close (bot=%.6f, exchange=%.6f).",
                            grace, self.position_units, real_amt,
                        )
                    else:
                        logger.info(
                            "📐 Position sync: units=%.6f → exchange=%.6f (corrected downward)",
                            self.position_units, real_amt,
                        )
                        self.position_units = real_amt
                        self._save_state()
        except Exception as exc:
            logger.warning("Position sync failed: %s", exc)

    def _get_real_balance(self) -> float:
        """Fetch real USDT wallet balance from exchange.

        Never falls back to self.balance (which drifts from reality).
        Returns exchange balance, or last cached exchange balance on failure.
        """
        try:
            if self.testnet_executor and not self.dry_run:
                bal = self.testnet_executor.get_account_balance("USDT")
                if bal > 0:
                    self._last_exchange_balance = bal
                    return bal
                logger.warning("Exchange returned 0 balance — using last cached: %.2f",
                               getattr(self, '_last_exchange_balance', 0))
        except Exception as exc:
            logger.warning("Real balance fetch failed: %s — using last cached: %.2f",
                           exc, getattr(self, '_last_exchange_balance', 0))
        # Return last known exchange balance, never the internal tracker
        return getattr(self, '_last_exchange_balance', self.balance)

    def _log_trade(self, trade: Dict) -> None:
        """Append trade to line-delimited JSON file and shared storage."""
        # Always tag testnet trades — this bot only runs on Binance Futures testnet
        trade["is_testnet"] = True
        self.trades.append(trade)
        try:
            with open(TRADES_FILE, "a") as f:
                f.write(json.dumps(trade) + "\n")
        except Exception as exc:
            logger.error("Failed to write trade log: %s", exc)
        try:
            self.storage.log_trade(trade)
        except Exception as exc:
            logger.debug("storage.log_trade failed: %s", exc)
        # Send Telegram alert immediately
        self._send_telegram_alert(trade)

    def _fetch_market_signals(self, symbol: str = "BTCUSDT") -> Dict:
        """Fetch current market analysis signals from the local API server."""
        try:
            import requests as req
            resp = req.get(f"http://127.0.0.1:5001/api/market?symbol={symbol}", timeout=10)
            if resp.ok:
                return resp.json()
        except Exception as exc:
            logger.debug("Failed to fetch market signals: %s", exc)
        return {}

    def _build_signal_summary(self, market: Dict) -> Dict:
        """Extract a compact signal summary from market analysis data."""
        summary = {}

        # MTF Alignment (key signal for gate)
        mtf = market.get("mtf", {})
        if mtf:
            summary["mtf"] = {
                "bias": mtf.get("bias", "NEUTRAL"),
                "aligned": mtf.get("aligned", False),
                "strength": mtf.get("strength", 0),
                "signals": mtf.get("signals", {}),
            }

        # Order flow (full data for gate)
        of = market.get("order_flow", {})
        if of:
            summary["order_flow"] = {
                "bias": of.get("bias", "neutral"),
                "score": of.get("score", 0),
                "large_buys": of.get("large_buys", of.get("notable", {}).get("large_buys", 0)),
                "large_sells": of.get("large_sells", of.get("notable", {}).get("large_sells", 0)),
            }

        # Regime
        regime = market.get("regime", {})
        if regime:
            summary["regime"] = {
                "type": regime.get("type", "UNKNOWN"),
                "state": regime.get("regime", regime.get("type", "unknown")),
                "adx": regime.get("adx"),
                "trend": regime.get("trend_strength"),
            }

        # Orderbook Imbalance (key signal for gate)
        ob = of.get("orderbook", {}) if of else {}
        if ob:
            summary["orderbook"] = {
                "bias": ob.get("bias", "neutral"),
                "imbalance_10": ob.get("imbalance_10", 0),
            }

        # Funding rate
        funding = market.get("funding", {})
        if funding:
            summary["funding"] = {
                "rate": funding.get("rate"),
                "bias": funding.get("bias"),
            }

        # Whale signals
        whale = market.get("whale", {})
        if whale:
            summary["whale"] = {
                "direction": whale.get("direction", "NEUTRAL"),
                "score": whale.get("score", 0),
                "confidence": whale.get("confidence", 0),
            }

        # Price
        if market.get("price"):
            summary["price"] = market["price"]

        return summary

    def _check_orderbook_guard(self, action: int) -> bool:
        """
        Orderbook Guard (Golden Guard): block trades when orderbook bias contradicts direction.

        This guard applies to ALL confidence tiers, including Tier 1 autonomous.
        The orderbook reflects real-time buy/sell pressure and is the strongest
        single predictor of whipsaw reversals in our backtesting.

        Returns True if the trade is ALLOWED, False if BLOCKED.
        """
        if not ORDERBOOK_GUARD_ENABLED:
            return True

        if action == ACTION_HOLD:
            return True

        direction = "LONG" if action == ACTION_LONG else "SHORT"

        try:
            market = self._fetch_market_signals(self.symbol)
            if not market:
                logger.debug("Orderbook guard: no market data — allowing trade (fail-open)")
                return True
        except Exception as exc:
            logger.debug("Orderbook guard: fetch failed (%s) — allowing trade (fail-open)", exc)
            return True

        # Get orderbook bias from order_flow.orderbook
        of = market.get("order_flow", {})
        ob = of.get("orderbook", {}) if of else {}
        ob_bias = (ob.get("bias") or "neutral").lower()

        blocked = False
        if direction == "LONG" and ob_bias == "bearish":
            blocked = True
        elif direction == "SHORT" and ob_bias == "bullish":
            blocked = True

        if blocked:
            logger.info(
                "🛡️ ORDERBOOK GUARD BLOCK: %s blocked — orderbook bias is %s (contradicts %s)",
                self.symbol, ob_bias.upper(), direction,
            )
        else:
            logger.debug(
                "🛡️ Orderbook guard PASS: %s %s — orderbook bias: %s",
                self.symbol, direction, ob_bias,
            )

        return not blocked

    def _check_usdt_d_guard(self, action: int) -> bool:
        """USDT.D filter: block LONGs when stablecoin dominance is rising.

        Returns True if ALLOWED, False if BLOCKED. Only affects LONG entries.
        Fail-open: if the proxy can't be computed, allow the trade.
        """
        if not USDT_D_GUARD_ENABLED or action != ACTION_LONG:
            return True
        rising = _is_usdt_dominance_rising(self.fetcher)
        basket_pct = _usdt_d_cache.get("basket_change_pct")
        if rising is None:
            logger.warning(
                "🛡️ USDT.D guard: proxy unavailable for %s LONG — allowing (fail-open)",
                self.symbol,
            )
            return True
        if rising:
            logger.info(
                "🛡️ USDT.D FILTER BLOCK: %s LONG blocked — crypto basket %+.3f%% over %dh "
                "(below -%.2f%% threshold = USDT.D rising)",
                self.symbol, basket_pct or 0.0, USDT_D_LOOKBACK_HOURS, USDT_D_THRESHOLD_PCT,
            )
            return False
        # PASS — also log so we can confirm the guard is being checked at all.
        logger.info(
            "🛡️ USDT.D guard PASS: %s LONG — basket %+.3f%% over %dh (above -%.2f%% threshold)",
            self.symbol, basket_pct if basket_pct is not None else 0.0,
            USDT_D_LOOKBACK_HOURS, USDT_D_THRESHOLD_PCT,
        )
        return True

    def _check_funding_long_guard(self, action: int) -> bool:
        """Block LONG entries when 8h funding rate < FUNDING_LONG_GUARD_MIN_RATE.
        Negative funding = market structure leans short. Backtest 2026-04-27:
        51 historical LONGs with funding<0 had 35.3% WR / -$9.24 avg pnl.
        Returns True if ALLOWED, False if BLOCKED. Affects LONG only.
        Fail-open if funding rate is unavailable.
        """
        if not FUNDING_LONG_GUARD_ENABLED or action != ACTION_LONG:
            return True
        try:
            market = self._fetch_market_signals(self.symbol)
        except Exception as exc:
            logger.debug("Funding guard: fetch failed (%s) — allowing (fail-open)", exc)
            return True
        funding = (market.get("funding") or {}).get("rate")
        if funding is None:
            logger.debug("Funding guard: rate unavailable for %s — allowing (fail-open)", self.symbol)
            return True
        if funding < FUNDING_LONG_GUARD_MIN_RATE:
            logger.info(
                "🛡️ FUNDING-LONG BLOCK: %s LONG blocked — funding rate %+.4f%% (< %+.4f%% threshold, "
                "structure leans short)",
                self.symbol, funding * 100, FUNDING_LONG_GUARD_MIN_RATE * 100,
            )
            return False
        return True

    def _check_whale_neutral_guard(self, action: int) -> bool:
        """Block any entry when whale-direction signal is NEUTRAL.
        Backtest 2026-04-27: 60 historical entries with whale=NEUTRAL had
        37.3% WR / -$8.67 avg pnl. The "no confidence" state is a regime
        stress indicator, not a directional one.
        Returns True if ALLOWED, False if BLOCKED. Affects both directions.
        Fail-open if whale signal is unavailable.
        """
        if not WHALE_NEUTRAL_GUARD_ENABLED or action == ACTION_HOLD:
            return True
        try:
            market = self._fetch_market_signals(self.symbol)
        except Exception as exc:
            logger.debug("Whale guard: fetch failed (%s) — allowing (fail-open)", exc)
            return True
        whale_dir = ((market.get("whale") or {}).get("direction") or "").upper()
        if not whale_dir:
            logger.debug("Whale guard: direction unavailable for %s — allowing (fail-open)", self.symbol)
            return True
        if whale_dir == "NEUTRAL":
            logger.info(
                "🛡️ WHALE-NEUTRAL BLOCK: %s %s blocked — whale.direction=NEUTRAL "
                "(historical regime: 37%% WR, -$8.67 avg)",
                self.symbol, "LONG" if action == ACTION_LONG else "SHORT",
            )
            return False
        return True

    def _check_ext_pos_news_guard(self, action: int) -> bool:
        """Block LONG entries when news in trailing 4h has sentiment > 0.5.
        Asset-tagged to symbol's base coin OR untagged/global news.
        Backed by Bonferroni-OK forward-return analysis (extreme-positive
        sentiment fades over 60-240m). Affects LONG only.
        Fail-open if news DB is unavailable.
        """
        if not EXT_POS_NEWS_GUARD_ENABLED or action != ACTION_LONG:
            return True
        import sqlite3
        from datetime import datetime as _dt, timedelta as _td, timezone as _tz
        # Per-symbol short-TTL cache (news arrives continuously; refresh fast)
        cache_key = self.symbol
        cached = _ext_pos_news_cache.get(cache_key)
        now = time.time()
        if cached and (now - cached[0]) < _EXT_POS_NEWS_CACHE_TTL_SECONDS:
            hit, info = cached[1], cached[2]
        else:
            try:
                asset = self.symbol.replace("USDT", "").upper()
                lookback_start = _dt.now(_tz.utc) - _td(hours=EXT_POS_NEWS_LOOKBACK_HOURS)
                conn = sqlite3.connect("data/trading.db")
                cur = conn.cursor()
                cur.execute(
                    "SELECT published_at, sentiment_score, assets, event_type, title "
                    "FROM news_events "
                    "WHERE sentiment_score IS NOT NULL "
                    "AND sentiment_score > ? "
                    "AND published_at >= ? "
                    "ORDER BY published_at DESC",
                    (EXT_POS_NEWS_SENTIMENT_THRESHOLD, lookback_start.isoformat()),
                )
                rows = cur.fetchall()
                conn.close()
                hit = False
                info = None
                for ts, sent, assets_json, et, title in rows:
                    try:
                        tags = json.loads(assets_json) if assets_json else []
                    except Exception:
                        tags = []
                    if tags and not any(asset in (t or "").upper() for t in tags):
                        continue  # asset-tagged but not for this symbol
                    hit = True
                    info = {"ts": ts[:19], "sentiment": float(sent),
                            "event_type": et or "", "title": (title or "")[:80]}
                    break
                _ext_pos_news_cache[cache_key] = (now, hit, info)
            except Exception as exc:
                logger.debug("Ext-pos news guard: DB read failed (%s) — fail-open", exc)
                return True
        if hit and info:
            logger.info(
                "🛡️ EXT-POS-NEWS BLOCK: %s LONG blocked — sentiment %+.2f at %s "
                "(%s) %s",
                self.symbol, info["sentiment"], info["ts"], info["event_type"], info["title"],
            )
            return False
        return True

    def _check_rsi_adx_guard(self, action: int, confidence: float) -> bool:
        """
        RSI Extreme + ADX Ranging Guard with Rescue Override.

        Blocks trades when:
        1. RSI extreme: LONG with 15m RSI > 70 (overbought) or SHORT with RSI < 30 (oversold)
        2. ADX too low: ADX < ADX_GUARD_MIN means no trend (ranging/choppy market)
        3. ADX too high: ADX > ADX_GUARD_MAX means trend exhaustion (overextended)

        Rescue Override: If blocked by RSI or ADX, but model confidence >= 0.90
        AND >= 2 market signals agree with the direction, ALLOW the trade anyway.

        Returns True if ALLOWED, False if BLOCKED.
        """
        if action == ACTION_HOLD:
            return True

        if not RSI_GUARD_ENABLED and not ADX_GUARD_ENABLED:
            return True

        direction = "LONG" if action == ACTION_LONG else "SHORT"

        try:
            market = self._fetch_market_signals(self.symbol)
            if not market:
                return True  # Fail-open
        except Exception:
            return True  # Fail-open

        # --- Check RSI ---
        rsi_blocked = False
        rsi_reason = ""
        if RSI_GUARD_ENABLED:
            mtf = market.get("mtf", {})
            rsi_15m = mtf.get("signals", {}).get("15m", {}).get("rsi", 50)
            # Regime-aware threshold: when the regime detector reports a
            # confirmed strong trend in the SAME direction we're trading,
            # loosen the RSI band so we stop fighting the regime detector.
            # In ranges or mismatched regimes, fall back to the strict 70/30.
            regime_for_rsi = market.get("regime", {}) or {}
            regime_type_rsi = (regime_for_rsi.get("type") or "UNKNOWN").upper()
            regime_adx_rsi = regime_for_rsi.get("adx", 0) or 0
            ob_threshold = RSI_GUARD_OB_THRESHOLD
            os_threshold = RSI_GUARD_OS_THRESHOLD
            if isinstance(regime_adx_rsi, (int, float)) and regime_adx_rsi >= RSI_GUARD_TREND_ADX_MIN:
                if direction == "LONG" and regime_type_rsi == "TRENDING_UP":
                    ob_threshold = RSI_GUARD_OB_TREND
                elif direction == "SHORT" and regime_type_rsi == "TRENDING_DOWN":
                    os_threshold = RSI_GUARD_OS_TREND
            if isinstance(rsi_15m, (int, float)):
                if direction == "LONG" and rsi_15m > ob_threshold:
                    rsi_blocked = True
                    rsi_reason = f"RSI 15m={rsi_15m:.0f} > {ob_threshold} (overbought)"
                elif direction == "SHORT" and rsi_15m < os_threshold:
                    rsi_blocked = True
                    rsi_reason = f"RSI 15m={rsi_15m:.0f} < {os_threshold} (oversold)"

        # --- Check ADX ---
        adx_blocked = False
        adx_reason = ""
        if ADX_GUARD_ENABLED:
            regime = market.get("regime", {})
            adx = regime.get("adx", 30) or 30
            if isinstance(adx, (int, float)):
                if adx < ADX_GUARD_MIN:
                    adx_blocked = True
                    adx_reason = f"ADX={adx:.0f} < {ADX_GUARD_MIN} (ranging)"
                elif adx > ADX_GUARD_MAX:
                    adx_blocked = True
                    adx_reason = f"ADX={adx:.0f} > {ADX_GUARD_MAX} (exhaustion / overextended trend)"

        if not rsi_blocked and not adx_blocked:
            return True  # No block needed

        # --- Rescue Override ---
        if RESCUE_ENABLED and confidence >= RESCUE_MIN_CONFIDENCE:
            # Count how many signals agree with the direction
            agrees = 0

            # 1. MTF alignment
            mtf = market.get("mtf", {})
            mtf_bias = (mtf.get("bias") or "NEUTRAL").upper()
            mtf_aligned = mtf.get("aligned", False)
            if direction == "LONG" and mtf_bias == "BULLISH" and mtf_aligned:
                agrees += 1
            elif direction == "SHORT" and mtf_bias == "BEARISH" and mtf_aligned:
                agrees += 1

            # 2. Order Flow
            of = market.get("order_flow", {})
            of_bias = (of.get("bias") or "neutral").lower()
            if direction == "LONG" and of_bias == "bullish":
                agrees += 1
            elif direction == "SHORT" and of_bias == "bearish":
                agrees += 1

            # 3. Whale
            whale = market.get("whale", {})
            whale_dir = (whale.get("direction") or "NEUTRAL").upper()
            if direction == "LONG" and whale_dir == "BULLISH":
                agrees += 1
            elif direction == "SHORT" and whale_dir == "BEARISH":
                agrees += 1

            # 4. Orderbook
            ob = of.get("orderbook", {})
            ob_bias = (ob.get("bias") or "neutral").lower()
            if direction == "LONG" and ob_bias == "bullish":
                agrees += 1
            elif direction == "SHORT" and ob_bias == "bearish":
                agrees += 1

            if agrees >= RESCUE_MIN_AGREES:
                reasons = []
                if rsi_blocked:
                    reasons.append(rsi_reason)
                if adx_blocked:
                    reasons.append(adx_reason)
                logger.info(
                    "🆘 RESCUE OVERRIDE: %s %s — blocked by [%s] but RESCUED "
                    "(conf=%.2f >= %.2f, agrees=%d >= %d)",
                    self.symbol, direction, " + ".join(reasons),
                    confidence, RESCUE_MIN_CONFIDENCE, agrees, RESCUE_MIN_AGREES,
                )
                return True  # Rescued!

        # --- Block ---
        reasons = []
        if rsi_blocked:
            reasons.append(rsi_reason)
        if adx_blocked:
            reasons.append(adx_reason)
        logger.info(
            "🚫 RSI/ADX GUARD BLOCK: %s %s — %s (conf=%.2f, no rescue)",
            self.symbol, direction, " + ".join(reasons), confidence,
        )
        return False

    def _should_block_long_reversal(self) -> bool:
        """Canary guard: block REVERSE_CLOSE_LONG for specific symbols when the
        BTC 4h regime is not a clear downtrend.

        Returns True → skip the reversal close (position stays LONG, SL/TP
        continues to protect it).  Returns False → let the reversal fire as
        normal.

        Regime gate ensures we fall back to current behavior in downtrend
        regimes where the counterfactual evidence is absent.  Fail-open on
        fetch errors so an outage of the klines endpoint never silently
        disables trading.
        """
        if self.symbol not in REVERSAL_BLOCK_LONG_CANARY_SYMBOLS:
            return False
        slope_pct = _compute_btc_4h_ema_slope_pct(self.fetcher)
        if slope_pct is None:
            logger.warning(
                "[%s] BTC regime slope unavailable — allowing LONG reversal (fail-open)",
                self.symbol,
            )
            return False
        if slope_pct <= REVERSAL_BLOCK_LONG_REGIME_GATE_MIN_SLOPE_PCT:
            logger.info(
                "[%s] LONG reversal allowed — BTC 4h slope %+.2f%% <= gate %+.2f%% (downtrend)",
                self.symbol, slope_pct, REVERSAL_BLOCK_LONG_REGIME_GATE_MIN_SLOPE_PCT,
            )
            return False
        logger.info(
            "🛡️  [%s] REVERSE_CLOSE_LONG blocked (canary) — BTC 4h slope %+.2f%% > gate %+.2f%%",
            self.symbol, slope_pct, REVERSAL_BLOCK_LONG_REGIME_GATE_MIN_SLOPE_PCT,
        )
        return True

    def _check_signal_gate(self, action: int, confidence: float) -> bool:
        """
        Market Signal Gate: validate trades against real market signals.

        Returns True if the trade is ALLOWED, False if BLOCKED.

        Regime veto (ALL tiers): Hard block when trading against a strong trend (ADX 25-60).
        Tier 1 (conf >= 0.80): autonomous after regime veto passes.
        Tier 2 (conf < 0.80): needs at least 2/4 market signals to agree with direction.

        Signals checked:
          1. MTF Alignment — all timeframes agree with direction
          2. Order Flow Score — net buying/selling pressure matches direction
          3. Regime — not trading against a strong trend
          4. Orderbook Imbalance — bid/ask pressure matches direction
        """
        if action == ACTION_HOLD:
            return True  # Not opening, no gate needed

        direction = "LONG" if action == ACTION_LONG else "SHORT"

        # Fetch market signals once — used by both regime veto and Tier 2 checks
        try:
            market = self._fetch_market_signals(self.symbol)
            if not market:
                logger.warning("Signal gate: no market data — allowing trade (fail-open)")
                if confidence >= SIGNAL_GATE_AUTONOMOUS:
                    self._last_signal_gate = {"result": "AUTONOMOUS", "tier": "Tier 1", "confirmations": 4, "details": "conf >= 0.80"}
                    return True
                return True
        except Exception as exc:
            logger.warning("Signal gate: fetch failed (%s) — allowing trade (fail-open)", exc)
            if confidence >= SIGNAL_GATE_AUTONOMOUS:
                self._last_signal_gate = {"result": "AUTONOMOUS", "tier": "Tier 1", "confirmations": 4, "details": "conf >= 0.80"}
                return True
            return True

        # ── Regime hard-veto (applies to ALL tiers, including Tier 1) ──
        # A confirmed strong trend against direction blocks the trade regardless of confidence
        regime = market.get("regime", {})
        regime_type = (regime.get("type") or "UNKNOWN").upper()
        regime_adx = regime.get("adx", 0) or 0
        if regime_adx >= SIGNAL_GATE_REGIME_ADX_MIN and regime_adx < 60:
            if direction == 'LONG' and 'DOWN' in regime_type:
                logger.info("🚫 Regime HARD-VETO: LONG vs %s ADX=%.0f (all tiers)", regime_type, regime_adx)
                self._last_signal_gate = {"result": "REGIME_VETO", "tier": "ALL", "details": f"LONG vs {regime_type} ADX={regime_adx:.0f}"}
                return False
            elif direction == 'SHORT' and 'UP' in regime_type:
                logger.info("🚫 Regime HARD-VETO: SHORT vs %s ADX=%.0f (all tiers)", regime_type, regime_adx)
                self._last_signal_gate = {"result": "REGIME_VETO", "tier": "ALL", "details": f"SHORT vs {regime_type} ADX={regime_adx:.0f}"}
                return False

        # ── Tier 1: autonomous after regime veto ──
        if confidence >= SIGNAL_GATE_AUTONOMOUS:
            self._last_signal_gate = {"result": "AUTONOMOUS", "tier": "Tier 1", "confirmations": 4, "details": "conf >= 0.80"}
            return True

        confirmations = 0
        vetoes = 0
        details = []

        # 1. MTF Alignment
        mtf = market.get("mtf", {})
        mtf_bias = (mtf.get("bias") or "NEUTRAL").upper()
        mtf_aligned = mtf.get("aligned", False)
        if direction == "LONG" and mtf_bias == "BULLISH" and mtf_aligned:
            confirmations += 1
            details.append("MTF=✅ bullish aligned")
        elif direction == "SHORT" and mtf_bias == "BEARISH" and mtf_aligned:
            confirmations += 1
            details.append("MTF=✅ bearish aligned")
        elif mtf_bias == "NEUTRAL" or not mtf_aligned:
            details.append(f"MTF=➖ {mtf_bias.lower()} (not aligned)")
        else:
            vetoes += 1
            details.append(f"MTF=❌ {mtf_bias.lower()} vs {direction}")

        # 2. Order Flow Score
        of = market.get("order_flow", {})
        of_score = of.get("score", 0)
        if isinstance(of_score, (int, float)):
            if direction == "LONG" and of_score > SIGNAL_GATE_OF_THRESHOLD:
                confirmations += 1
                details.append(f"OF=✅ {of_score:+.2f}")
            elif direction == "SHORT" and of_score < -SIGNAL_GATE_OF_THRESHOLD:
                confirmations += 1
                details.append(f"OF=✅ {of_score:+.2f}")
            elif abs(of_score) <= SIGNAL_GATE_OF_THRESHOLD:
                details.append(f"OF=➖ neutral ({of_score:+.2f})")
            else:
                vetoes += 1
                details.append(f"OF=❌ {of_score:+.2f} vs {direction}")
        else:
            details.append("OF=➖ no data")

        # 3. Regime — check we're not fighting a strong trend
        regime = market.get("regime", {})
        regime_type = (regime.get("type") or "UNKNOWN").upper()
        regime_adx = regime.get("adx", 0) or 0
        if direction == "LONG" and "DOWN" in regime_type and regime_adx >= SIGNAL_GATE_REGIME_ADX_MIN:
            vetoes += 1
            details.append(f"REG=❌ {regime_type} ADX={regime_adx:.0f}")
        elif direction == "SHORT" and "UP" in regime_type and regime_adx >= SIGNAL_GATE_REGIME_ADX_MIN:
            vetoes += 1
            details.append(f"REG=❌ {regime_type} ADX={regime_adx:.0f}")
        elif direction == "LONG" and "UP" in regime_type and regime_adx >= SIGNAL_GATE_REGIME_ADX_MIN:
            confirmations += 1
            details.append(f"REG=✅ {regime_type} ADX={regime_adx:.0f}")
        elif direction == "SHORT" and "DOWN" in regime_type and regime_adx >= SIGNAL_GATE_REGIME_ADX_MIN:
            confirmations += 1
            details.append(f"REG=✅ {regime_type} ADX={regime_adx:.0f}")
        else:
            details.append(f"REG=➖ {regime_type} ADX={regime_adx:.0f}")

        # 4. Orderbook Imbalance
        ob = of.get("orderbook", {})
        ob_imbalance = ob.get("imbalance_10", 0)
        ob_bias = (ob.get("bias") or "neutral").lower()
        if isinstance(ob_imbalance, (int, float)):
            if direction == "LONG" and ob_imbalance > SIGNAL_GATE_OB_THRESHOLD:
                confirmations += 1
                details.append(f"OB=✅ {ob_imbalance:+.2f} bullish")
            elif direction == "SHORT" and ob_imbalance < -SIGNAL_GATE_OB_THRESHOLD:
                confirmations += 1
                details.append(f"OB=✅ {ob_imbalance:+.2f} bearish")
            elif abs(ob_imbalance) <= SIGNAL_GATE_OB_THRESHOLD:
                details.append(f"OB=➖ neutral ({ob_imbalance:+.2f})")
            else:
                vetoes += 1
                details.append(f"OB=❌ {ob_imbalance:+.2f} vs {direction}")
        else:
            details.append("OB=➖ no data")

        allowed = confirmations >= SIGNAL_GATE_MIN_CONFIRMS
        detail_str = " | ".join(details)

        self._last_signal_gate = {
            "result": "PASS" if allowed else "BLOCKED",
            "tier": "Tier 2",
            "confirmations": confirmations,
            "vetoes": vetoes,
            "details": detail_str,
        }

        if allowed:
            logger.info(
                "✅ Signal gate PASS: %s conf=%.2f | %d/%d confirms | %s",
                direction, confidence, confirmations, SIGNAL_GATE_MIN_CONFIRMS, detail_str,
            )
        else:
            logger.info(
                "🚫 Signal gate BLOCK: %s conf=%.2f | %d/%d confirms (%d vetoes) | %s",
                direction, confidence, confirmations, SIGNAL_GATE_MIN_CONFIRMS, vetoes, detail_str,
            )

        return allowed

    def _get_calibrated_confidence(self, raw_confidence: float, action_probs=None,
                                      regime: str = "unknown", adx: float = 0) -> float:
        """Get v2 calibrated confidence. Returns raw if calibrator not available."""
        try:
            if not hasattr(self, '_confidence_calibrator'):
                from src.brain.confidence_calibrator import ConfidenceCalibrator
                self._confidence_calibrator = ConfidenceCalibrator()
                if not self._confidence_calibrator.load():
                    self._confidence_calibrator = None
                    return raw_confidence
            if self._confidence_calibrator is None:
                return raw_confidence
            return self._confidence_calibrator.calibrate(
                raw_confidence, action_probs=action_probs,
                regime=regime, adx=adx)
        except Exception:
            return raw_confidence

    def _get_whale_behavior_signal(self) -> Dict:
        """Get whale behavior model signal for trade logging. Fail-safe."""
        try:
            from src.whale_behavior.models.predictor import WhaleIntentPredictor
            if not hasattr(self, '_whale_predictor'):
                self._whale_predictor = WhaleIntentPredictor()
            signal = self._whale_predictor.get_signal()
            # Compact version for logging — keep per-wallet SELL probs
            wallet_sells = {}
            for name, d in signal.get("wallet_details", {}).items():
                sell_p = d.get("probs", {}).get("SELL", 0)
                if sell_p >= 0.30:  # Only log wallets with meaningful sell signal
                    wallet_sells[name] = round(sell_p, 3)
            return {
                "intent": signal.get("intent", "unavailable"),
                "sell_confidence": round(signal.get("sell_confidence", 0), 3),
                "buy_confidence": round(signal.get("buy_confidence", 0), 3),
                "direction": round(signal.get("direction", 0.5), 3),
                "active_wallets": signal.get("active_wallets", 0),
                "top_sellers": wallet_sells,
            }
        except Exception as exc:
            logger.debug("Whale behavior signal unavailable: %s", exc)
            return {"intent": "unavailable", "sell_confidence": 0, "buy_confidence": 0}

    def _log_whale_shadow(self, trade: Dict, whale_signal: Dict) -> None:
        """Log whale signal + v2 confidence alongside trade for shadow analysis.
        
        This creates a separate log at logs/whale_shadow.jsonl that tracks:
        - Every trade action (OPEN/CLOSE) with the whale signal at that moment
        - v1 confidence (raw) vs v2 confidence (calibrated)
        - Used later to analyze correlation between signals and trade outcomes
        """
        try:
            raw_conf = trade.get("confidence", 0)
            # Get regime info for v2 calibration
            market = self._fetch_market_signals(trade.get("symbol", self.symbol))
            regime = market.get("regime", {}).get("type",
                     market.get("regime", {}).get("state", "unknown"))
            adx = market.get("regime", {}).get("adx", 0) or 0
            v2_conf = self._get_calibrated_confidence(raw_conf, regime=regime, adx=adx)

            shadow_file = Path("logs/whale_shadow.jsonl")
            shadow_file.parent.mkdir(parents=True, exist_ok=True)
            with open(shadow_file, "a") as f:
                f.write(json.dumps({
                    "timestamp": datetime.now().isoformat(),
                    "symbol": trade.get("symbol", "?"),
                    "action": trade.get("action", "?"),
                    "price": trade.get("price", 0),
                    "confidence_v1": raw_conf,
                    "confidence_v2": round(v2_conf, 4),
                    "regime": regime,
                    "adx": adx,
                    "entry_price": trade.get("entry_price", self.position_price),
                    "pnl": trade.get("pnl", None),
                    "whale": whale_signal,
                }, default=str) + "\n")
        except Exception as exc:
            logger.debug("Failed to write whale shadow log: %s", exc)

    def _send_telegram_alert(self, trade: Dict) -> None:
        """Write trade alert with market signal context to pending alerts file."""
        try:
            # Fetch current market signals for context
            market = self._fetch_market_signals(trade.get("symbol", "BTCUSDT"))
            signal_summary = self._build_signal_summary(market)

            # Include signal gate decision in alerts
            if self._last_signal_gate:
                signal_summary["signal_gate"] = self._last_signal_gate

            # Get whale behavior signal
            whale_signal = self._get_whale_behavior_signal()

            # Log whale shadow data for analysis
            self._log_whale_shadow(trade, whale_signal)

            alert_file = Path("logs/htf_pending_alerts.jsonl")
            alert_file.parent.mkdir(parents=True, exist_ok=True)
            with open(alert_file, "a") as f:
                f.write(json.dumps({
                    "timestamp": datetime.now().isoformat(),
                    "strategy": "htf",
                    "trade": trade,
                    "signals": signal_summary,
                    "whale_behavior": whale_signal,
                    "position": {
                        "entry_price": self.position_price,
                        "sl_price": self.sl_price,
                        "tp_price": self.tp_price,
                        "units": self.position_units,
                        "direction": "LONG" if self.position == 1 else "SHORT" if self.position == -1 else "FLAT",
                    },
                }, default=str) + "\n")
            logger.info("Trade alert queued: %s %s @ $%.2f (whale_sell=%.0f%%)",
                        trade.get("action", "?"), trade.get("symbol", "?"),
                        trade.get("price", 0), whale_signal.get("sell_confidence", 0) * 100)
        except Exception as exc:
            logger.warning("Failed to queue trade alert: %s", exc)

    def _write_system_alert(self, alert_type: str, details: str) -> None:
        """Write a system/connectivity alert to the shared alerts file."""
        try:
            alert_file = Path("logs/htf_pending_alerts.jsonl")
            alert_file.parent.mkdir(parents=True, exist_ok=True)
            with open(alert_file, "a") as f:
                f.write(json.dumps({
                    "timestamp": datetime.now().isoformat(),
                    "strategy": "htf",
                    "trade": {
                        "type": alert_type,
                        "symbol": self.symbol,
                        "details": details,
                        "has_open_position": self.position != 0,
                        "position_direction": "LONG" if self.position == 1 else "SHORT" if self.position == -1 else "FLAT",
                    },
                    "signals": {},
                    "position": {
                        "entry_price": self.position_price,
                        "sl_price": self.sl_price,
                        "tp_price": self.tp_price,
                        "units": self.position_units,
                        "direction": "LONG" if self.position == 1 else "SHORT" if self.position == -1 else "FLAT",
                    },
                }) + "\n")
        except Exception as exc:
            logger.error("Failed to write system alert: %s", exc)

    def _write_sltp_update_alert(
        self, sl_changed: bool, tp_changed: bool,
        old_sl: float, new_sl: float,
        old_tp: float, new_tp: float,
        reason: str,
    ) -> None:
        """Write SL/TP update alert to the shared alerts file."""
        try:
            alert_file = Path("logs/htf_pending_alerts.jsonl")
            alert_file.parent.mkdir(parents=True, exist_ok=True)
            direction = "LONG" if self.position == 1 else "SHORT" if self.position == -1 else "FLAT"

            # Compute current profit %
            profit_pct = 0.0
            if self.position_price > 0 and self.current_price > 0:
                if self.position == 1:
                    profit_pct = ((self.current_price - self.position_price) / self.position_price) * 100
                elif self.position == -1:
                    profit_pct = ((self.position_price - self.current_price) / self.position_price) * 100

            with open(alert_file, "a") as f:
                if sl_changed and old_sl != new_sl:
                    f.write(json.dumps({
                        "timestamp": datetime.now().isoformat(),
                        "strategy": "htf",
                        "trade": {
                            "type": "SL_UPDATE",
                            "update_type": "SL",
                            "symbol": self.symbol,
                            "old_price": old_sl,
                            "new_price": new_sl,
                            "reason": reason,
                            "direction": direction,
                            "profit_pct": round(profit_pct, 2),
                            "current_price": self.current_price,
                        },
                        "signals": {},
                        "position": {
                            "entry_price": self.position_price,
                            "sl_price": new_sl,
                            "tp_price": self.tp_price,
                            "units": self.position_units,
                            "direction": direction,
                        },
                    }) + "\n")

                if tp_changed and old_tp != new_tp:
                    f.write(json.dumps({
                        "timestamp": datetime.now().isoformat(),
                        "strategy": "htf",
                        "trade": {
                            "type": "TP_UPDATE",
                            "update_type": "TP",
                            "symbol": self.symbol,
                            "old_price": old_tp,
                            "new_price": new_tp,
                            "reason": reason,
                            "direction": direction,
                            "profit_pct": round(profit_pct, 2),
                            "current_price": self.current_price,
                        },
                        "signals": {},
                        "position": {
                            "entry_price": self.position_price,
                            "sl_price": self.sl_price,
                            "tp_price": new_tp,
                            "units": self.position_units,
                            "direction": direction,
                        },
                    }) + "\n")

            logger.info("SL/TP update alert queued for %s", self.symbol)
        except Exception as exc:
            logger.warning("Failed to queue SL/TP update alert: %s", exc)

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def fetch_data(self) -> Optional[pd.DataFrame]:
        """Fetch recent 15M OHLCV data for BTCUSDT."""
        try:
            df = self.fetcher.fetch_asset(self.symbol, "15m", FETCH_DAYS)
            if df is None or df.empty:
                logger.warning("Empty 15M data returned")
                return None
            # Ensure DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.set_index("timestamp")
                else:
                    df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            logger.info("Fetched %d 15M bars for %s", len(df), self.symbol)
            return df
        except Exception as exc:
            logger.error("fetch_data failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Observation building
    # ------------------------------------------------------------------

    def compute_observation(self, df_15m: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Compute the 117-dim HTF observation from a 15M DataFrame.

        Layout: [20 1D | 25 4H | 30 1H | 35 15M | 4 align | 3 pos]
        """
        try:
            frames = self.aligner.align_timestamps(df_15m)
            df_1d = frames["1d"]
            df_4h = frames["4h"]
            df_1h = frames["1h"]
            df_15 = frames["15m"]

            # Need at least MIN_BARS in each timeframe
            if len(df_1d) < 5 or len(df_4h) < 10 or len(df_1h) < 20 or len(df_15) < 30:
                logger.warning(
                    "Insufficient bars: 1d=%d 4h=%d 1h=%d 15m=%d",
                    len(df_1d), len(df_4h), len(df_1h), len(df_15),
                )
                return None

            # Compute features at the last available bar
            f1d = self.feature_engine.compute_1d_features(df_1d, len(df_1d) - 1)
            f4h = self.feature_engine.compute_4h_features(df_4h, len(df_4h) - 1)
            f1h = self.feature_engine.compute_1h_features(df_1h, len(df_1h) - 1)
            f15m = self.feature_engine.compute_15m_features(df_15, len(df_15) - 1)

            # Cross-TF alignment (uses last scores from each TF feature vector)
            sig_1d = float(f1d[-1])
            sig_4h = float(f4h[-1])
            sig_1h = float(f1h[-1])
            sig_15m = float(f15m[-1])
            f_align = self.feature_engine.compute_alignment_full(
                sig_1d, sig_4h, sig_1h, sig_15m
            )

            # Concatenate 114 market features
            feats_114 = np.concatenate([f1d, f4h, f1h, f15m, f_align])

            # Position state (3 dims — must match HTFTradingEnv._get_observation)
            current_price = float(df_15.iloc[-1]["close"])
            self.current_price = current_price

            if self.position != 0 and self.position_price > 0:
                if self.position == 1:
                    unrealized_pnl = (current_price - self.position_price) / (self.position_price + 1e-10)
                else:
                    unrealized_pnl = (self.position_price - current_price) / (self.position_price + 1e-10)
            else:
                unrealized_pnl = 0.0

            balance_ratio = (self.balance - self.initial_balance) / (self.initial_balance + 1e-10)

            pos_state = np.array([
                float(self.position),
                float(np.clip(unrealized_pnl, -0.5, 0.5)),
                float(np.clip(balance_ratio, -0.5, 0.5)),
            ], dtype=np.float32)

            obs = np.concatenate([feats_114, pos_state]).astype(np.float32)
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

            return obs

        except Exception as exc:
            logger.error("compute_observation failed: %s", exc, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def get_action(self, obs: np.ndarray) -> Tuple[int, float]:
        """
        Run PPO inference on a 117-dim observation.

        Returns (action, confidence) where action ∈ {0, 1, 2}.
        Falls back to HOLD if no model is loaded.
        """
        if self.model is None:
            return ACTION_HOLD, 0.0

        try:
            obs_2d = obs.reshape(1, -1)

            # Apply VecNormalize if available
            if self.vec_normalize is not None:
                try:
                    obs_2d = self.vec_normalize.normalize_obs(obs_2d)
                except Exception as exc:
                    logger.debug("VecNormalize failed: %s", exc)

            action, _ = self.model.predict(obs_2d, deterministic=True)
            action = int(action.item() if hasattr(action, "item") else action)

            # Compute confidence (max action probability)
            confidence = self._compute_confidence(obs_2d)

            return action, confidence

        except Exception as exc:
            logger.error("get_action failed: %s", exc)
            return ACTION_HOLD, 0.0

    def _compute_confidence(self, obs_2d: np.ndarray) -> float:
        """Return max action probability from the policy distribution."""
        try:
            import torch
            with torch.no_grad():
                obs_tensor = self.model.policy.obs_to_tensor(obs_2d)[0]
                dist = self.model.policy.get_distribution(obs_tensor)
                probs = dist.distribution.probs.detach().cpu().numpy()[0]
            return float(np.max(probs))
        except Exception:
            return 1.0 / 3.0

    # ------------------------------------------------------------------
    # BOS/CHOCH market structure data
    # ------------------------------------------------------------------

    def _fetch_structure_candles(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Fetch 5m, 1H and 4H OHLCV candles for BOS/CHOCH multi-timeframe analysis.

        Returns (df_5m, df_1h, df_4h).  Any may be None on fetch failure.
        """
        df_5m = None
        df_1h = None
        df_4h = None

        # 5m candles — primary BOS/CHOCH timeframe (faster signal detection)
        try:
            df_5m = self.fetcher.fetch_asset(self.symbol, "5m", days=FETCH_DAYS)
            if df_5m is not None and not df_5m.empty:
                if not isinstance(df_5m.index, pd.DatetimeIndex):
                    if "timestamp" in df_5m.columns:
                        df_5m["timestamp"] = pd.to_datetime(df_5m["timestamp"])
                        df_5m = df_5m.set_index("timestamp")
                    else:
                        df_5m.index = pd.to_datetime(df_5m.index)
                df_5m = df_5m.sort_index()
                logger.info("Fetched %d 5m bars for BOS/CHOCH", len(df_5m))
        except Exception as exc:
            logger.warning("Failed to fetch 5m candles for BOS/CHOCH: %s", exc)

        try:
            df_1h = self.fetcher.fetch_asset(self.symbol, "1h", days=FETCH_DAYS)
            if df_1h is not None and not df_1h.empty:
                if not isinstance(df_1h.index, pd.DatetimeIndex):
                    if "timestamp" in df_1h.columns:
                        df_1h["timestamp"] = pd.to_datetime(df_1h["timestamp"])
                        df_1h = df_1h.set_index("timestamp")
                    else:
                        df_1h.index = pd.to_datetime(df_1h.index)
                df_1h = df_1h.sort_index()
                logger.debug("Fetched %d 1H bars for BOS/CHOCH", len(df_1h))
        except Exception as exc:
            logger.debug("Failed to fetch 1H candles: %s", exc)

        try:
            df_4h = self.fetcher.fetch_asset(self.symbol, "4h", days=FETCH_DAYS)
            if df_4h is not None and not df_4h.empty:
                if not isinstance(df_4h.index, pd.DatetimeIndex):
                    if "timestamp" in df_4h.columns:
                        df_4h["timestamp"] = pd.to_datetime(df_4h["timestamp"])
                        df_4h = df_4h.set_index("timestamp")
                    else:
                        df_4h.index = pd.to_datetime(df_4h.index)
                df_4h = df_4h.sort_index()
                logger.debug("Fetched %d 4H bars for BOS/CHOCH", len(df_4h))
        except Exception as exc:
            logger.debug("Failed to fetch 4H candles: %s", exc)

        return df_5m, df_1h, df_4h

    def _get_structure_signals(self, trigger: bool = False) -> Dict:
        """
        Get the latest BOS/CHOCH signals.  Caches the result so it is only
        recomputed once per iteration (not on every WS tick).

        If *trigger* is True, a fresh analysis is run using 5m candles as
        the primary timeframe and 1H/4H for multi-timeframe confirmation.
        Otherwise the cached result from the last iteration is returned.
        """
        if trigger:
            try:
                df_5m, df_1h, df_4h = self._fetch_structure_candles()
                if df_5m is None or df_5m.empty:
                    logger.warning("BOS/CHOCH: no 5m data available — skipping")
                    self._last_structure_signals = {}
                else:
                    _struct_result = self.market_structure.get_signals(
                        df_5m, df_1h=df_1h, df_4h=df_4h,
                    )
                    # Convert to dict for .get() access throughout the codebase
                    self._last_structure_signals = (
                        _struct_result.to_dict()
                        if hasattr(_struct_result, 'to_dict')
                        else _struct_result
                    )
                    # Log BOS/CHOCH signal state every iteration at INFO level
                    sig = self._last_structure_signals
                    logger.info(
                        "BOS/CHOCH signals: bos_bull=%s bos_bear=%s choch_bull=%s "
                        "choch_bear=%s fake_bos=%s fake_choch=%s trend=%s "
                        "swing_high=$%.2f swing_low=$%.2f conf=%.2f",
                        sig.get("bos_bullish", False),
                        sig.get("bos_bearish", False),
                        sig.get("choch_bullish", False),
                        sig.get("choch_bearish", False),
                        sig.get("fake_bos", False),
                        sig.get("fake_choch", False),
                        sig.get("trend", "unknown"),
                        sig.get("last_swing_high", 0.0),
                        sig.get("last_swing_low", 0.0),
                        sig.get("confidence", 0.0),
                    )
            except Exception as exc:
                logger.warning("BOS/CHOCH signal computation failed: %s", exc)
                self._last_structure_signals = {}
        return self._last_structure_signals

    def _get_structure_direction(self, current_price: float, df_15m: pd.DataFrame) -> Optional[int]:
        """
        Structure-first entry: derive trade direction from BOS/CHOCH signals.
        Returns ACTION_LONG, ACTION_SHORT, or None (no entry).

        Per-symbol config controls which extra filters apply:
          S1: BOS/CHOCH only (no OB, no ADX directional check)
          S5: BOS/CHOCH + OB proximity + RSI guard + ADX directional confirmation
        """
        sig = self._last_structure_signals
        if not sig:
            return None

        trend = sig.get("trend", "ranging")
        last_dir = sig.get("last_signal_direction", "none")

        # Most-recent-signal-wins: use last_signal_direction (the direction of
        # the most recent BOS/CHOCH signal) instead of cumulative booleans.
        # The old boolean approach caused ALL four flags to be permanently True
        # over any non-trivial lookback, blocking every entry.
        #
        # Entry logic:
        #   1. trend must be bullish or bearish (not ranging)
        #   2. last_signal_direction must agree with trend
        if trend == "bullish" and last_dir == "bullish":
            direction = ACTION_LONG
            side_str = "LONG"
        elif trend == "bearish" and last_dir == "bearish":
            direction = ACTION_SHORT
            side_str = "SHORT"
        elif trend in ("bullish", "bearish") and last_dir != trend:
            logger.info(
                "Structure-first: trend=%s but last_signal=%s — waiting for alignment",
                trend, last_dir,
            )
            return None
        else:
            # Ranging or no signal — skip
            return None

        # Per-symbol-side blocklist: skip entries on combos that historically lose.
        # See SYMBOL_SIDE_BLOCKLIST near the top of this file for backtest evidence.
        if (self.symbol, side_str) in SYMBOL_SIDE_BLOCKLIST:
            logger.info(
                "🚫 Structure-first: skipping %s %s — combo is in SYMBOL_SIDE_BLOCKLIST "
                "(historical net-negative expectancy; see scripts/backtest_signal_filters.py)",
                self.symbol, side_str,
            )
            return None

        # fake_bos / fake_choch entry guard: the detector already produces these
        # flags. Use them at entry time, not just for SL adjustments.
        if FAKE_BOS_ENTRY_GUARD_ENABLED:
            fake_bos = bool(sig.get("fake_bos"))
            fake_choch = bool(sig.get("fake_choch"))
            # Direction-specific check: only block when the fake aligns with the
            # direction we'd take. e.g. don't block a LONG just because there's a
            # bearish fake_choch in the data — only if the bullish signal we're
            # acting on was itself flagged fake.
            if direction == ACTION_LONG and (sig.get("bos_bullish") and fake_bos
                                              or sig.get("choch_bullish") and fake_choch):
                logger.info(
                    "🚫 Structure-first: skipping %s LONG — fake_bos=%s fake_choch=%s, "
                    "direction-aligned signal flagged as fake",
                    self.symbol, fake_bos, fake_choch,
                )
                return None
            if direction == ACTION_SHORT and (sig.get("bos_bearish") and fake_bos
                                               or sig.get("choch_bearish") and fake_choch):
                logger.info(
                    "🚫 Structure-first: skipping %s SHORT — fake_bos=%s fake_choch=%s, "
                    "direction-aligned signal flagged as fake",
                    self.symbol, fake_bos, fake_choch,
                )
                return None

        # Per-symbol config
        sym_cfg = STRUCTURE_SYMBOL_CONFIG.get(self.symbol, "S1")

        if sym_cfg == "S5":
            # Extra filter: Order Block proximity
            if df_15m is not None and len(df_15m) >= 40:
                ob_window = df_15m.tail(40)
                opn = ob_window["open"].values
                close = ob_window["close"].values
                high = ob_window["high"].values
                low = ob_window["low"].values
                n = len(close)
                bull_obs, bear_obs = [], []
                atr_proxy = np.mean(high - low) + 1e-10
                for idx in range(max(0, n - 30), n - 2):
                    body_i = close[idx] - opn[idx]
                    body_i1 = close[idx + 1] - opn[idx + 1]
                    move = abs(close[idx + 2] - close[idx + 1]) if (idx + 2) < n else 0.0
                    if body_i < 0 and body_i1 > 0 and move > atr_proxy * 0.5:
                        bull_obs.append((high[idx] + low[idx]) / 2.0)
                    if body_i > 0 and body_i1 < 0 and move > atr_proxy * 0.5:
                        bear_obs.append((high[idx] + low[idx]) / 2.0)

                ob_levels = bull_obs if direction == ACTION_LONG else bear_obs
                near_ob = any(
                    abs(current_price - lvl) / (lvl + 1e-10) < STRUCTURE_OB_PROXIMITY_PCT
                    for lvl in ob_levels
                )
                if not near_ob:
                    logger.info("Structure-first S5: blocked by OB proximity (no nearby %s OB)",
                                "bullish" if direction == ACTION_LONG else "bearish")
                    return None

            # Extra filter: ADX directional confirmation (from 15m candles)
            if df_15m is not None and len(df_15m) >= 30:
                try:
                    _close = df_15m["close"].values
                    _high = df_15m["high"].values
                    _low = df_15m["low"].values
                    _period = 14
                    # Quick ADX/DI from last 30 bars
                    _plus_dm = np.diff(_high[-30:])
                    _minus_dm = -np.diff(_low[-30:])
                    _plus_dm = np.where((_plus_dm > _minus_dm) & (_plus_dm > 0), _plus_dm, 0)
                    _minus_dm = np.where((_minus_dm > _plus_dm) & (_minus_dm > 0), _minus_dm, 0)
                    _tr = np.maximum(_high[-29:] - _low[-29:],
                                     np.maximum(np.abs(_high[-29:] - _close[-30:-1]),
                                                np.abs(_low[-29:] - _close[-30:-1])))
                    _atr = np.mean(_tr[-_period:])
                    if _atr > 0:
                        _plus_di = 100 * np.mean(_plus_dm[-_period:]) / _atr
                        _minus_di = 100 * np.mean(_minus_dm[-_period:]) / _atr
                        _adx_val = 100 * abs(_plus_di - _minus_di) / (_plus_di + _minus_di + 1e-10)
                        if _adx_val >= ADX_GUARD_MIN:
                            if direction == ACTION_LONG and _minus_di > _plus_di:
                                logger.info("Structure-first S5: ADX directional block (LONG but -DI > +DI, ADX=%.1f)", _adx_val)
                                return None
                            if direction == ACTION_SHORT and _plus_di > _minus_di:
                                logger.info("Structure-first S5: ADX directional block (SHORT but +DI > -DI, ADX=%.1f)", _adx_val)
                                return None
                except Exception as exc:
                    logger.debug("Structure-first S5 ADX check failed: %s", exc)

        logger.info(
            "Structure-first %s: %s | trend=%s last_signal=%s",
            sym_cfg,
            "LONG" if direction == ACTION_LONG else "SHORT",
            trend, last_dir,
        )
        return direction

    # ------------------------------------------------------------------
    # Liquidation price safety check
    # ------------------------------------------------------------------

    def _get_liquidation_price(self) -> float:
        """
        Fetch the current liquidation price from the exchange via the
        testnet executor. Returns 0.0 if unavailable, dry-run, or no
        exchange connection.
        """
        if self.dry_run or not self.testnet_executor:
            return 0.0
        try:
            futures_exec = getattr(self.testnet_executor, '_futures_executor', None)
            if futures_exec and hasattr(futures_exec, 'get_liquidation_price'):
                return futures_exec.get_liquidation_price(self.symbol)
        except Exception as exc:
            logger.debug("Failed to fetch liquidation price: %s", exc)
        return 0.0

    def _check_liquidation_safety(self) -> bool:
        """
        Validate that liquidation price is safely beyond the SL.

        Only alerts if liquidation is CLOSER to entry than SL (i.e., SL won't
        trigger before liquidation). With our 1% buffer design (LIQ_BUFFER_PCT),
        liquidation is intentionally close to SL — that's expected and safe.

        TRUE danger: liq is between entry and SL (would get liquidated before SL fires).

        Returns True if safe, False if at risk.
        """
        if self.position == 0 or self.position_price <= 0 or self.sl_price <= 0:
            return True

        liq_price = self._get_liquidation_price()

        # Skip if liquidation is 0 (1x leverage — impossible to liquidate)
        if liq_price <= 0:
            return True

        # Determine actual direction from liquidation price vs entry price.
        if liq_price < self.position_price:
            # Liq below entry → LONG position
            # Danger: liq is ABOVE SL (would be liquidated before SL fires)
            if liq_price > self.sl_price:
                buffer_pct = (self.sl_price - liq_price) / self.position_price * 100
                logger.critical(
                    "⚠️ LIQUIDATION RISK: %s LONG liq=$%.2f is ABOVE SL=$%.2f! "
                    "Will be liquidated BEFORE SL fires! (entry=$%.2f, gap=%.2f%%)",
                    self.symbol, liq_price, self.sl_price,
                    self.position_price, buffer_pct,
                )
                self._write_liquidation_alert(liq_price, 0)
                return False
        else:
            # Liq above entry → SHORT position
            # Danger: liq is BELOW SL (would be liquidated before SL fires)
            if liq_price < self.sl_price:
                buffer_pct = (liq_price - self.sl_price) / self.position_price * 100
                logger.critical(
                    "⚠️ LIQUIDATION RISK: %s SHORT liq=$%.2f is BELOW SL=$%.2f! "
                    "Will be liquidated BEFORE SL fires! (entry=$%.2f, gap=%.2f%%)",
                    self.symbol, liq_price, self.sl_price,
                    self.position_price, buffer_pct,
                )
                self._write_liquidation_alert(liq_price, 0)
                return False

        return True

    def _write_liquidation_alert(self, liq_price: float, delta: float) -> None:
        """Write a LIQUIDATION_RISK alert to the shared alerts file for Telegram."""
        try:
            alert_file = Path("logs/htf_pending_alerts.jsonl")
            alert_file.parent.mkdir(parents=True, exist_ok=True)
            direction = "LONG" if self.position == 1 else "SHORT" if self.position == -1 else "FLAT"

            if self.position == 1:
                buffer = self.sl_price - liq_price
            else:
                buffer = liq_price - self.sl_price
            buffer_pct = buffer / self.position_price * 100 if self.position_price > 0 else 0

            with open(alert_file, "a") as f:
                f.write(json.dumps({
                    "timestamp": datetime.now().isoformat(),
                    "strategy": "htf",
                    "trade": {
                        "type": "LIQUIDATION_RISK",
                        "symbol": self.symbol,
                        "direction": direction,
                        "liquidation_price": liq_price,
                        "sl_price": self.sl_price,
                        "entry_price": self.position_price,
                        "delta": delta,
                        "buffer": buffer,
                        "buffer_pct": buffer_pct,
                    },
                    "signals": {},
                    "position": {
                        "entry_price": self.position_price,
                        "sl_price": self.sl_price,
                        "tp_price": self.tp_price,
                        "units": self.position_units,
                        "direction": direction,
                    },
                }) + "\n")
            logger.info("Liquidation risk alert queued for %s %s", self.symbol, direction)
        except Exception as exc:
            logger.warning("Failed to queue liquidation risk alert: %s", exc)

    # ------------------------------------------------------------------
    # SL/TP check
    # ------------------------------------------------------------------

    def _update_peak_price(self, current_price: float) -> None:
        """Track the most-favorable price since entry (highest for LONG, lowest for SHORT)."""
        if self.position == 0:
            return
        if self.position == 1:  # LONG — track highest
            if current_price > self.peak_price:
                self.peak_price = current_price
        elif self.position == -1:  # SHORT — track lowest
            if self.peak_price == 0.0 or current_price < self.peak_price:
                self.peak_price = current_price

    def _check_sl_tp(self, current_price: float) -> Optional[str]:
        """
        Dynamic SL/TP logic enhanced with BOS/CHOCH market structure signals.

        Layered approach:
          1. Hard TP check (unchanged at configured TP level).
          2. Basic trailing stop (break-even at +1%, lock 50% at +2%).
          3. BOS/CHOCH overlay — **only when the position is currently
             profitable** (unrealized PnL > 0).

        BOS/CHOCH rules (profitable positions only):
          LONG:
            - BOS bullish  → trail SL to last swing low + buffer, extend TP
            - CHOCH bearish → tighten SL to lock 75% of profit
            - Fake BOS bearish / Fake CHOCH bullish → ignore
          SHORT:
            - BOS bearish  → trail SL to last swing high - buffer, extend TP
            - CHOCH bullish → tighten SL to lock 75% of profit
            - Fake BOS bullish / Fake CHOCH bearish → ignore

        CRITICAL RULES:
          - SL only moves *toward* profit (never to a worse position).
          - BOS/CHOCH adjustments are ONLY applied when profitable.

        Returns 'SL' or 'TP' if the current position should be closed, else None.
        """
        if self.position == 0:
            return None

        entry = self.position_price
        if entry <= 0:
            return None

        # --- Update peak price ---
        self._update_peak_price(current_price)

        # --- Compute unrealized profit % ---
        if self.position == 1:  # LONG
            profit_pct = (current_price - entry) / entry
        else:  # SHORT
            profit_pct = (entry - current_price) / entry

        is_profitable = profit_pct > 0

        # --- Hard TP check ---
        if self.position == 1 and self.tp_price > 0 and current_price >= self.tp_price:
            return "TP"
        if self.position == -1 and self.tp_price > 0 and current_price <= self.tp_price:
            return "TP"

        # --- Baseline trailing stop adjustments (always active) ---
        new_sl = self.sl_price
        new_tp = self.tp_price
        adjustment_reason = ""

        if profit_pct >= TRAILING_BREAKEVEN_PCT:
            # After TP1, widen trailing distance to let winners run longer.
            # Backtested Apr 6-14: 0.5%→0.8% post-TP1 added +$181 across 120 trades.
            trail_dist = TRAILING_DISTANCE_POST_TP1 if self.partial_tp_level >= 1 else TRAILING_DISTANCE_PCT
            if self.position == 1:
                trailing_sl = self.peak_price * (1.0 - trail_dist)
                # Never let trailing SL go below entry (minimum breakeven)
                trailing_sl = max(trailing_sl, entry)
                if trailing_sl > new_sl:
                    new_sl = trailing_sl
                    adjustment_reason = "trailing_continuous"
            else:
                trailing_sl = self.peak_price * (1.0 + trail_dist)
                # Never let trailing SL go above entry (minimum breakeven)
                trailing_sl = min(trailing_sl, entry)
                if new_sl <= 0 or trailing_sl < new_sl:
                    new_sl = trailing_sl
                    adjustment_reason = "trailing_continuous"

        # --- BOS/CHOCH overlay (ONLY when profitable) ---
        if is_profitable and self._last_structure_signals:
            sig = self._last_structure_signals
            swing_high = sig.get("last_swing_high", 0.0)
            swing_low = sig.get("last_swing_low", 0.0)
            confidence = sig.get("confidence", 0.0)

            # Small buffer to avoid SL sitting exactly on the swing level
            buffer_pct = 0.002  # 0.2%

            if self.position == 1:  # ── LONG ──
                # BOS bullish (trend continuation) → trail SL to swing low
                if sig.get("bos_bullish") and not sig.get("fake_bos") and swing_low > 0:
                    bos_sl = swing_low * (1.0 - buffer_pct)
                    if bos_sl > new_sl:
                        logger.info(
                            "🔄 BOS bullish → trailing SL to swing low $%.2f (was $%.2f, conf=%.2f)",
                            bos_sl, new_sl, confidence,
                        )
                        new_sl = bos_sl
                        adjustment_reason = f"BOS_bullish(conf={confidence:.2f})"
                    # Extend TP to next swing high if it's above current TP
                    if swing_high > 0 and swing_high > new_tp:
                        new_tp = swing_high
                        logger.info(
                            "📈 BOS bullish: TP extended → $%.2f (swing high)",
                            new_tp,
                        )

                # CHOCH bearish (reversal warning) → lock 75% of profit
                if sig.get("choch_bearish") and not sig.get("fake_choch"):
                    choch_sl = entry + 0.75 * (current_price - entry)
                    if choch_sl > new_sl:
                        logger.info(
                            "🔄 CHOCH bearish → tightening SL to lock 75%% profit $%.2f (was $%.2f, conf=%.2f)",
                            choch_sl, new_sl, confidence,
                        )
                        new_sl = choch_sl
                        adjustment_reason = f"CHOCH_bearish(conf={confidence:.2f})"

                # Fake BOS bearish → explicitly ignore (hold position)
                # Logged at DEBUG — fires every tick, would spam INFO logs
                if sig.get("fake_bos") and sig.get("bos_bearish"):
                    logger.debug("⚠️ Fake BOS bearish detected — ignoring, holding LONG")

                # Fake CHOCH bullish → ignore (don't over-tighten)
                if sig.get("fake_choch") and sig.get("choch_bullish"):
                    logger.debug("⚠️ Fake CHOCH bullish detected — ignoring")

            elif self.position == -1:  # ── SHORT ──
                # BOS bearish (continuation) → trail SL to swing high
                if sig.get("bos_bearish") and not sig.get("fake_bos") and swing_high > 0:
                    bos_sl = swing_high * (1.0 + buffer_pct)
                    if new_sl <= 0 or bos_sl < new_sl:
                        logger.info(
                            "🔄 BOS bearish → trailing SL to swing high $%.2f (was $%.2f, conf=%.2f)",
                            bos_sl, new_sl, confidence,
                        )
                        new_sl = bos_sl
                        adjustment_reason = f"BOS_bearish(conf={confidence:.2f})"
                    # Extend TP to next swing low
                    if swing_low > 0 and (new_tp <= 0 or swing_low < new_tp):
                        new_tp = swing_low
                        logger.info(
                            "📉 BOS bearish: TP extended → $%.2f (swing low)",
                            new_tp,
                        )

                # CHOCH bullish (reversal warning) → lock 75% of profit
                if sig.get("choch_bullish") and not sig.get("fake_choch"):
                    choch_sl = entry - 0.75 * (entry - current_price)
                    if new_sl <= 0 or choch_sl < new_sl:
                        logger.info(
                            "🔄 CHOCH bullish → tightening SL to lock 75%% profit $%.2f (was $%.2f, conf=%.2f)",
                            choch_sl, new_sl, confidence,
                        )
                        new_sl = choch_sl
                        adjustment_reason = f"CHOCH_bullish(conf={confidence:.2f})"

                # Fake BOS bullish → ignore
                # Logged at DEBUG — fires every tick, would spam INFO logs
                if sig.get("fake_bos") and sig.get("bos_bullish"):
                    logger.debug("⚠️ Fake BOS bullish detected — ignoring, holding SHORT")

                # Fake CHOCH bearish → ignore
                if sig.get("fake_choch") and sig.get("choch_bearish"):
                    logger.debug("⚠️ Fake CHOCH bearish detected — ignoring")

        # --- Safety: SL must never move to a worse position ---
        # Bot-side SL is updated on every tick for precision (WS monitor checks
        # exact sl_price). But LOGGING, ALERTS, and EXCHANGE UPDATES only fire
        # when the change exceeds 0.1% to avoid noise/API spam.
        MIN_SLTP_LOG_PCT = 0.001  # 0.1% threshold for logging/alerts/exchange
        _min_change = current_price * MIN_SLTP_LOG_PCT

        # ── Step 1: Always update bot-side SL to best trailing value ──
        # This is the PRIMARY SL — checked on every WS tick for instant exit.
        # No threshold here — precision matters for the actual exit check.
        sl_moved = False  # did the internal sl_price actually change?
        if self.position == 1:
            if new_sl > self.sl_price:
                sl_moved = True
            else:
                new_sl = self.sl_price  # revert — SL only moves toward profit
        elif self.position == -1:
            if self.sl_price <= 0 or (new_sl > 0 and new_sl < self.sl_price):
                sl_moved = True
            else:
                new_sl = self.sl_price  # revert

        tp_moved = new_tp != self.tp_price and new_tp > 0

        # ── Step 2: Apply internal SL/TP (always, no threshold) ──
        old_sl_price = self.sl_price
        old_tp_price = self.tp_price

        if sl_moved:
            self.sl_price = new_sl
        if tp_moved:
            self.tp_price = new_tp
        if sl_moved or tp_moved:
            self._save_state()

        # ── Step 3: Log/alert/exchange sync ONLY when change is significant ──
        # This prevents the 10-updates-in-8-seconds spam.
        sl_significant = sl_moved and abs(self.sl_price - old_sl_price) >= _min_change
        tp_significant = tp_moved and abs(self.tp_price - old_tp_price) >= _min_change

        if sl_significant or tp_significant:
            if sl_significant:
                logger.info(
                    "🔄 SL adjusted: $%.2f → $%.2f (reason=%s, profit=%.2f%%, peak=$%.2f)",
                    old_sl_price, self.sl_price, adjustment_reason or "trailing",
                    profit_pct * 100, self.peak_price,
                )

            if tp_significant:
                logger.info(
                    "🔄 TP adjusted: $%.2f → $%.2f (reason=%s)",
                    old_tp_price, self.tp_price, adjustment_reason or "structure",
                )

            # Write SL/TP update alerts to the shared alerts file
            self._write_sltp_update_alert(
                sl_significant, tp_significant,
                old_sl_price, self.sl_price,
                old_tp_price, self.tp_price,
                adjustment_reason or "trailing",
            )

            # Sync updated SL/TP to exchange crashguard (throttled: 0.1% + 60s cooldown)
            if not self.dry_run and self.testnet_executor:
                try:
                    self.testnet_executor.update_sl_tp(
                        self.symbol, self.sl_price, self.tp_price
                    )
                except Exception as _exc:
                    logger.warning("Testnet SL/TP update failed: %s", _exc)
                    self._write_system_alert(
                        "REST_API_ERROR",
                        f"SL/TP exchange sync failed for {self.symbol}: {_exc}"
                    )

        # --- Liquidation safety check (max once per 5 min to avoid spam) ---
        _now = time.time()
        if not hasattr(self, '_last_liq_check') or (_now - self._last_liq_check) > 300:
            self._check_liquidation_safety()
            self._last_liq_check = _now

        # --- Check if current price hits the (possibly adjusted) SL ---
        if self.position == 1:
            if self.sl_price > 0 and current_price <= self.sl_price:
                return "SL"
        elif self.position == -1:
            if self.sl_price > 0 and current_price >= self.sl_price:
                return "SL"

        return None

    # ------------------------------------------------------------------
    # Trade execution
    # ------------------------------------------------------------------

    def execute_trade(self, action: int, confidence: float, current_price: float) -> Optional[Dict]:
        """
        Execute a trading decision.

        Returns a trade record dict if a trade was executed, else None.
        """
        now = time.time()

        # ── Guard: cooldown after loss ──
        if self.last_loss_time > 0 and (now - self.last_loss_time) < COOLDOWN_SECONDS:
            remaining = COOLDOWN_SECONDS - (now - self.last_loss_time)
            logger.info("Cooldown active: %.0fs remaining — HOLD", remaining)
            return None

        # ── Guard: anti-whipsaw (block quick losing reversals) ──
        # Case 1: Flat and last trade was opposite direction, lost, closed <2h ago.
        # Case 2: In position and model wants to REVERSE — check if current position
        #         was itself a reversal from a recent loss (whipsaw chain).
        if action != ACTION_HOLD and self.last_close_direction != 0:
            # Determine if this action would be opposite to the last closed direction
            would_reverse_last_close = (
                (self.last_close_direction == 1 and action == ACTION_SHORT) or
                (self.last_close_direction == -1 and action == ACTION_LONG)
            )
            # Also catch same-iteration reversals: bot is LONG and model says SHORT
            # (the close hasn't happened yet, but we know the new entry would be
            # opposite to the CURRENT position which is the last close's direction)
            is_live_reversal = (
                (self.position == 1 and action == ACTION_SHORT) or
                (self.position == -1 and action == ACTION_LONG)
            )
            check_whipsaw = (
                (self.position == 0 and would_reverse_last_close) or
                (is_live_reversal and self.last_close_pnl < 0)
            )
            if check_whipsaw:
                hours_since_close = (now - self.last_close_time) / 3600.0 if self.last_close_time > 0 else 999
                if self.last_close_pnl < 0 and hours_since_close < WHIPSAW_COOLDOWN_HOURS:
                    logger.info(
                        "🚫 Anti-whipsaw: last %s lost $%.2f, closed %.1fh ago — blocking %s reversal",
                        "LONG" if self.last_close_direction == 1 else "SHORT",
                        self.last_close_pnl,
                        hours_since_close,
                        ACTION_LABELS.get(action, "?"),
                    )
                    return None

        # ── Guard: minimum hold time ──
        # Exception: if the model wants to REVERSE direction (e.g. SHORT→LONG)
        # with high confidence (≥0.75), let it flip — the model should decide.
        is_reversal = (
            (self.position == 1 and action == ACTION_SHORT) or
            (self.position == -1 and action == ACTION_LONG)
        )
        if self.position != 0 and (now - self.last_entry_time) < MIN_HOLD_SECONDS:
            # In structure-first mode, allow reversal if BOS/CHOCH confidence >= 0.65
            reversal_conf_threshold = 0.65 if STRUCTURE_FIRST_MODE else 0.82
            if is_reversal and confidence >= reversal_conf_threshold:
                logger.info(
                    "Min hold override: reversal %s→%s with conf=%.2f ≥ %.2f — allowing flip",
                    "LONG" if self.position == 1 else "SHORT",
                    ACTION_LABELS.get(action, "?"), confidence, reversal_conf_threshold,
                )
            else:
                remaining = MIN_HOLD_SECONDS - (now - self.last_entry_time)
                logger.info("Min hold: %.0fs remaining — HOLD", remaining)
                return None

        # ── Guard: confidence threshold (per-symbol or global) ──
        # 2026-05-03: Now applies in BOTH model-first and structure-first
        # modes. Previously skipped in structure-first because the
        # original threshold (0.45) was calibrated for model output and
        # below the BOS-confidence base (0.5), making the gate a no-op
        # in practice. With MIN_CONFIDENCE raised to 0.55 (from 90d
        # ablation finding) the gate is now meaningful in both modes —
        # it filters BOS+HTF signals where HTF confirmation is weak
        # (the 0.5-0.55 band corresponds to "BOS exists but no strong
        # multi-timeframe alignment").
        min_conf = SYMBOL_MIN_CONFIDENCE.get(self.symbol, MIN_CONFIDENCE)
        if action != ACTION_HOLD and confidence < min_conf:
            logger.info("Low confidence %.2f < %.2f (%s) — HOLD", confidence, min_conf, self.symbol)
            return None

        # ── Guard: per-symbol directional confidence floor ──
        # SKIP in structure-first mode — these are model-confidence calibrated
        if not STRUCTURE_FIRST_MODE:
            if action != ACTION_HOLD:
                direction_str = "LONG" if action == ACTION_LONG else "SHORT"
                dir_floor = SYMBOL_DIRECTIONAL_CONF.get(self.symbol, {}).get(direction_str)
                if dir_floor is not None and confidence < dir_floor:
                    logger.info("🚫 Directional floor: %s %s conf=%.2f < %.2f — HOLD", self.symbol, direction_str, confidence, dir_floor)
                    return None

        # ── Guard: ranging regime filter ──
        # In structure-first mode, use ADX as hard block (no confidence comparison)
        if action != ACTION_HOLD and self.position == 0 and self.regime_detector is not None and self._last_df is not None:
            try:
                regime_info = self.regime_detector.detect_regime(self._last_df)
                regime_name = regime_info.regime.value
                adx_val = getattr(regime_info, 'trend_strength', None) or 0.0
                if STRUCTURE_FIRST_MODE:
                    # In structure-first mode, just block on very low ADX
                    if adx_val < ADX_GUARD_MIN:
                        logger.info(
                            "🚫 Structure-first ADX block: ADX=%.1f < %.1f — SKIP entry",
                            adx_val, ADX_GUARD_MIN,
                        )
                        return None
                elif adx_val < RANGING_ADX_THRESHOLD and confidence < RANGING_MIN_CONFIDENCE:
                    logger.info(
                        "🚫 Ranging regime filter: ADX=%.1f < %.1f, conf=%.2f < %.2f — SKIP entry",
                        adx_val, RANGING_ADX_THRESHOLD, confidence, RANGING_MIN_CONFIDENCE,
                    )
                    return None
            except Exception as exc:
                logger.debug("Regime filter check failed: %s", exc)

        # ── Guard: momentum exhaustion filter ──
        # Don't enter when price is extended far from VWAP (likely to revert)
        if action != ACTION_HOLD and self.position == 0 and self._last_df is not None:
            try:
                df = self._last_df
                if len(df) >= 20:
                    closes = df["close"].values[-20:]
                    volumes = df["volume"].values[-20:]
                    # Compute VWAP over last 20 bars
                    typical_price = (df["high"].values[-20:] + df["low"].values[-20:] + closes) / 3.0
                    vwap_20 = float(np.sum(typical_price * volumes) / (np.sum(volumes) + 1e-10))
                    # Compute ATR (14-period) from last 20 bars
                    highs = df["high"].values[-20:]
                    lows = df["low"].values[-20:]
                    tr = np.maximum(highs - lows, np.maximum(
                        np.abs(highs - np.roll(closes, 1)),
                        np.abs(lows - np.roll(closes, 1))
                    ))
                    atr_14 = float(np.mean(tr[-14:]))
                    if atr_14 > 0:
                        extension = abs(current_price - vwap_20) / atr_14
                        if extension > EXHAUSTION_ATR_THRESHOLD:
                            direction_str = "LONG" if action == ACTION_LONG else "SHORT"
                            logger.info(
                                "🚫 Exhaustion filter: price $%.2f is %.1f ATR from VWAP $%.2f — SKIP %s entry",
                                current_price, extension, vwap_20, direction_str,
                            )
                            return None
            except Exception as exc:
                logger.debug("Exhaustion filter check failed: %s", exc)

        # ── Guard: Orderbook Guard (Golden Guard) ──
        # Blocks trades when orderbook bias contradicts trade direction.
        # Applies to ALL tiers — this is a hard veto based on real-time order flow.
        if action != ACTION_HOLD and (self.position == 0 or
            (self.position == 1 and action == ACTION_SHORT) or
            (self.position == -1 and action == ACTION_LONG)):
            if not self._check_orderbook_guard(action):
                return None

        # ── Guard: RSI Extreme + ADX Ranging (with Rescue Override) ──
        # Blocks overbought/oversold entries and ranging-market entries.
        # Rescue: if conf >= 0.90 AND >= 2 signals agree, override the block.
        if action != ACTION_HOLD and (self.position == 0 or
            (self.position == 1 and action == ACTION_SHORT) or
            (self.position == -1 and action == ACTION_LONG)):
            if not self._check_rsi_adx_guard(action, confidence):
                return None

        # ── Guard: USDT.D Filter (LONG only) ──
        # Block LONG entries when stablecoin dominance is rising (capital
        # fleeing crypto). SHORTs unaffected. See USDT_D_* constants for
        # backtest evidence and threshold reasoning.
        if action != ACTION_HOLD and (self.position == 0 or
            (self.position == 1 and action == ACTION_SHORT) or
            (self.position == -1 and action == ACTION_LONG)):
            if not self._check_usdt_d_guard(action):
                return None

        # ── Guard: Funding-rate (LONG only) — phase 3 ──
        # Block LONG entries when funding rate is negative. See FUNDING_LONG_*
        # constants. Backtested +$284 incremental over 20d on top of blocklist.
        if action != ACTION_HOLD and (self.position == 0 or
            (self.position == 1 and action == ACTION_SHORT) or
            (self.position == -1 and action == ACTION_LONG)):
            if not self._check_funding_long_guard(action):
                return None

        # ── Guard: Whale=NEUTRAL block — phase 3 ──
        # Block any entry when the whale-direction predictor returns NEUTRAL.
        # Stress regime indicator. Backtested +$413 incremental over 20d.
        if action != ACTION_HOLD and (self.position == 0 or
            (self.position == 1 and action == ACTION_SHORT) or
            (self.position == -1 and action == ACTION_LONG)):
            if not self._check_whale_neutral_guard(action):
                return None

        # ── Guard: Extreme-positive news fade (LONG only) ──
        # Block LONG entries when news in trailing 4h has sentiment > 0.5.
        # Backed by Bonferroni-OK forward-return analysis showing euphoric
        # news fades over 60-240m. Backtested incremental Δ over 8 days
        # of news coverage: +$158 prevented losses (10 blocked, 0/10 wins).
        if action != ACTION_HOLD and (self.position == 0 or
            (self.position == 1 and action == ACTION_SHORT) or
            (self.position == -1 and action == ACTION_LONG)):
            if not self._check_ext_pos_news_guard(action):
                return None

        # ── Guard: Market Signal Gate ──
        # Low-confidence entries need confirmation from real market signals
        # SKIP in structure-first mode — structure IS the primary signal, no confidence tiers
        if not STRUCTURE_FIRST_MODE:
            if action != ACTION_HOLD and (self.position == 0 or
                (self.position == 1 and action == ACTION_SHORT) or
                (self.position == -1 and action == ACTION_LONG)):
                if not self._check_signal_gate(action, confidence):
                    # For reversals: close the existing position flat (don't open opposite)
                    # This avoids double-loss from closing + immediately entering wrong direction
                    if is_reversal and self.position != 0:
                        logger.info("Reversal gate BLOCKED — closing existing position flat, not flipping")
                        return self._close_position(current_price, "REVERSAL_BLOCKED_CLOSE", confidence)
                    return None

        trade: Optional[Dict] = None

        # ── CLOSE existing position if direction reverses ──
        if self.position == 1 and action == ACTION_SHORT:
            if self._should_block_long_reversal():
                return None
            trade = self._close_position(current_price, "REVERSE_CLOSE_LONG", confidence)
        elif self.position == -1 and action == ACTION_LONG:
            trade = self._close_position(current_price, "REVERSE_CLOSE_SHORT", confidence)
        elif self.position != 0 and action == ACTION_HOLD:
            # Just hold; SL/TP handled separately
            pass

        # ── OPEN new position ──
        if action == ACTION_LONG and self.position == 0:
            if trade:  # Already closed — open after brief gap
                pass
            trade = self._open_position(current_price, 1, confidence)
        elif action == ACTION_SHORT and self.position == 0:
            trade = self._open_position(current_price, -1, confidence)

        return trade

    def _open_position(self, price: float, direction: int, confidence: float) -> Dict:
        # ── Fixed-Dollar-Risk Position Sizing ──
        # Step 1: Calculate dollar risk per trade
        risk_pool = self.session_balance * RISK_POOL_PCT
        dollar_risk = risk_pool / RISK_BUDGET_PARTS

        # Step 2: Calculate notional from dollar risk and SL%
        # (regime-adaptive SL is calculated below, use base for sizing)
        sl_pct_for_sizing = STOP_LOSS_PCT
        notional = dollar_risk / sl_pct_for_sizing

        # Step 3: Cap notional to hard max
        notional = min(notional, FIXED_MAX_NOTIONAL)

        # Step 4: Derive leverage from liquidation buffer
        # Liq distance = SL% + buffer% → leverage = 1 / liq_distance
        liq_distance = sl_pct_for_sizing + LIQ_BUFFER_PCT
        raw_leverage = 1.0 / liq_distance if liq_distance > 0 else 1
        leverage = max(1, min(int(raw_leverage), MAX_LEVERAGE))

        # Step 5: Calculate margin from leverage
        actual_margin = notional / leverage

        # Step 7: Validate we have enough balance for the margin
        if actual_margin > self.session_balance * 0.95:
            # Scale down notional to fit
            notional = self.session_balance * 0.90 * leverage
            logger.warning(
                "⚠️ Margin $%.2f exceeds balance — scaled notional to $%.2f",
                actual_margin, notional,
            )
            actual_margin = notional / leverage

        trade_value = notional
        logger.info(
            "📐 %s risk sizing: bal=$%.0f | risk_pool=$%.0f | risk/trade=$%.2f | "
            "notional=$%.0f | margin=$%.2f | leverage=%dx | liq_buffer=%.1f%%",
            "LONG" if direction == 1 else "SHORT",
            self.session_balance, risk_pool, dollar_risk,
            notional, actual_margin, leverage, LIQ_BUFFER_PCT * 100,
        )

        fee = trade_value * TRADING_FEE
        self.balance -= fee
        self.position = direction
        self.position_price = price
        self.position_units = (trade_value - fee) / (price + 1e-10)
        self.peak_price = price  # Reset peak price for trailing stop tracking
        self.last_entry_time = time.time()
        self.position_entry_time = time.time()  # Phase 1 §3.4: for time-based exit

        # Phase 1 §3.2: Regime-adaptive SL/TP multipliers
        sl_pct = STOP_LOSS_PCT
        tp_pct = TAKE_PROFIT_PCT
        try:
            if self.regime_detector is not None and self._last_df is not None:
                regime_info = self.regime_detector.detect_regime(self._last_df)
                regime_name = regime_info.regime.value
                if regime_name == 'high_volatility':
                    sl_pct *= 1.5   # Was 2.0× — reduced to limit loss per trade
                    tp_pct *= 1.3   # Was 1.5× — scaled proportionally
                    logger.info("📊 HIGH VOL regime: SL ×1.5, TP ×1.3")
                elif regime_name == 'trending_up' and direction == 1:
                    tp_pct *= 1.8   # With-trend LONG: let runners run
                    logger.info("📊 TRENDING_UP regime (LONG): TP ×1.8, SL tight")
                elif regime_name == 'trending_down' and direction == -1:
                    tp_pct *= 1.8   # With-trend SHORT: let runners run
                    logger.info("📊 TRENDING_DOWN regime (SHORT): TP ×1.8, SL tight")
                elif regime_name == 'trending_down' and direction == 1:
                    sl_pct *= 1.3   # Counter-trend LONG: wider SL
                    logger.info("📊 TRENDING_DOWN regime (counter-trend LONG): SL ×1.3")
                elif regime_name == 'trending_up' and direction == -1:
                    sl_pct *= 1.3   # Counter-trend SHORT: wider SL
                    logger.info("📊 TRENDING_UP regime (counter-trend SHORT): SL ×1.3")
                elif regime_name == 'ranging':
                    sl_pct *= 1.2   # Tighter to limit chop loss
                    tp_pct *= 0.8   # Mean-reversion moves are smaller
                    logger.info("📊 RANGING regime: SL ×1.2, TP ×0.8")
        except Exception as exc:
            logger.warning("Regime-adaptive SL/TP failed: %s — using base values", exc)

        self.sl_pct = sl_pct  # Phase 1 §3.3: store for partial TP R-multiple calculations

        if direction == 1:
            self.sl_price = price * (1.0 - sl_pct)
            self.tp_price = price * (1.0 + tp_pct)
            action_str = "OPEN_LONG"
            # Phase 1 §3.3: Partial TP prices (1R and 2R targets)
            self.partial_tp1_price = price * (1.0 + 1.0 * sl_pct)
            self.partial_tp2_price = price * (1.0 + 2.0 * sl_pct)
        else:
            self.sl_price = price * (1.0 + sl_pct)
            self.tp_price = price * (1.0 - tp_pct)
            action_str = "OPEN_SHORT"
            # Phase 1 §3.3: Partial TP prices (1R and 2R targets)
            self.partial_tp1_price = price * (1.0 - 1.0 * sl_pct)
            self.partial_tp2_price = price * (1.0 - 2.0 * sl_pct)

        # Phase 1 §3.3: Reset partial TP tracking for new trade
        self.initial_position_units = self.position_units
        self.partial_tp_level = 0

        # Phase 1 §3.9: Reset MFE/MAE for new trade
        self.mfe_pct = 0.0
        self.mae_pct = 0.0

        # ── Liquidation vs SL validation ──
        # Verify: liq distance (1/leverage) ≥ SL% + buffer%
        actual_liq_dist = 1.0 / leverage if leverage > 0 else 1.0
        required_liq_dist = sl_pct + LIQ_BUFFER_PCT
        if actual_liq_dist < required_liq_dist:
            logger.warning(
                "⚠️ Liquidation too close to SL! liq_dist=%.2f%% need=%.2f%% "
                "(SL=%.2f%% + buffer=%.2f%%) — reducing leverage",
                actual_liq_dist * 100, required_liq_dist * 100,
                sl_pct * 100, LIQ_BUFFER_PCT * 100,
            )
            safe_leverage = int(1.0 / required_liq_dist)
            leverage = max(1, min(safe_leverage, MAX_LEVERAGE))
            actual_margin = notional / leverage
            logger.info("📐 Adjusted leverage to %dx for safety", leverage)

        trade = {
            "action": action_str,
            "symbol": self.symbol,
            "price": price,
            "units": self.position_units,
            "trade_value": trade_value,
            "confidence": confidence,
            "sl": self.sl_price,
            "tp": self.tp_price,
            "partial_tp1": self.partial_tp1_price,
            "partial_tp2": self.partial_tp2_price,
            "leverage": leverage,
            "dollar_risk": dollar_risk,
            "margin": actual_margin,
            "pnl": 0.0,
            "timestamp": datetime.now().isoformat(),
            "agent": "htf",
        }

        logger.info(
            "📈 %s @ $%.2f | units=%.5f | SL=$%.2f (%.2f%%) | TP=$%.2f (%.2f%%) | "
            "TP1=$%.2f | TP2=$%.2f | conf=%.2f",
            action_str, price, self.position_units,
            self.sl_price, sl_pct * 100,
            self.tp_price, tp_pct * 100,
            self.partial_tp1_price, self.partial_tp2_price, confidence,
        )

        # Phase 2 OI logging — observational only, never blocks. See
        # _compute_oi_features for context. Helper never raises.
        try:
            trade["oi_features"] = _compute_oi_features(self.symbol)
        except Exception:
            trade["oi_features"] = None

        if not self.dry_run:
            self._mirror_testnet(trade)
            # Sync balance AND position size from exchange after opening
            self._sync_balance_from_exchange()
            self._sync_position_from_exchange()
            # Update trade dict with real exchange units BEFORE logging/alerting
            trade["units"] = self.position_units
            trade["trade_value"] = self.position_units * price
            self._log_trade(trade)
        else:
            self._log_trade(trade)
            logger.info("[DRY RUN] Trade not executed")

        return trade

    def _close_position(self, price: float, reason: str, confidence: float) -> Dict:
        if self.position == 0:
            return {}

        if self.position == 1:
            raw_pnl = (price - self.position_price) * self.position_units
            action_str = "CLOSE_LONG"
        else:
            raw_pnl = (self.position_price - price) * self.position_units
            action_str = "CLOSE_SHORT"

        fee = price * self.position_units * TRADING_FEE
        net_pnl = raw_pnl - fee
        self.balance += (price * self.position_units) + net_pnl - raw_pnl
        self.realized_pnl += net_pnl

        # Track loss time for cooldown
        if net_pnl < 0:
            self.last_loss_time = time.time()

        # Track last close for anti-whipsaw filter
        self.last_close_direction = self.position  # 1=LONG, -1=SHORT
        self.last_close_pnl = net_pnl
        self.last_close_time = time.time()

        trade = {
            "action": action_str,
            "reason": reason,
            "symbol": self.symbol,
            "entry_price": self.position_price,
            "exit_price": price,
            "units": self.position_units,
            "pnl": net_pnl,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "agent": "htf",
            "balance_after": self._get_real_balance(),
            "realized_pnl_total": self.realized_pnl,
            "initial_balance": 5000.0,  # Testnet starting deposit
            "balance_pnl_pct": ((self._get_real_balance() - 5000.0) / 5000.0 * 100)
                if True else 0.0,
        }

        logger.info(
            "📉 %s @ $%.2f | pnl=$%.2f | reason=%s | balance=$%.2f",
            action_str, price, net_pnl, reason, self.balance,
        )

        # Phase 1 §3.9: Log MFE/MAE on close
        logger.info(
            "📊 Trade MFE/MAE: MFE=%+.3f%% MAE=%+.3f%% (entry=$%.2f exit=$%.2f)",
            self.mfe_pct * 100, self.mae_pct * 100, self.position_price, price,
        )
        trade["mfe_pct"] = self.mfe_pct
        trade["mae_pct"] = self.mae_pct

        # Reset position
        self.position = 0
        self.position_price = 0.0
        self.position_units = 0.0
        self.sl_price = 0.0
        self.tp_price = 0.0
        self.peak_price = 0.0
        # Phase 1: Reset partial TP and MFE/MAE state
        self.mfe_pct = 0.0
        self.mae_pct = 0.0
        self.partial_tp_level = 0
        self.partial_tp1_price = 0.0
        self.partial_tp2_price = 0.0
        self.sl_pct = 0.0
        self.initial_position_units = 0.0
        self.position_entry_time = 0.0

        # Always log the close (dry_run or live) — needed for learning from outcomes
        self._log_trade(trade)

        if not self.dry_run:
            self._mirror_testnet(trade)
            # Sync balance from exchange after closing position
            self._sync_balance_from_exchange()
        else:
            logger.info("[DRY RUN] Close logged but not executed on exchange")

        return trade

    # ------------------------------------------------------------------
    # Testnet mirroring
    # ------------------------------------------------------------------

    def _mirror_testnet(self, trade: Dict) -> None:
        """
        Mirror a bot decision to futures testnet.

        OPEN_LONG / OPEN_SHORT → places real futures order + SL/TP exchange orders.
        CLOSE_LONG / CLOSE_SHORT → MARKET close + cancel remaining SL/TP orders.
        """
        if not self.testnet_executor:
            return
        action = trade.get("action", "")
        symbol = trade.get("symbol", self.symbol)

        # ALL close actions: close via MARKET order + cleanup SL/TP
        if "CLOSE" in action:
            try:
                from src.api.futures_executor import get_futures_executor
                executor = get_futures_executor()
                if executor:
                    side = "LONG" if "CLOSE_LONG" in action else "SHORT"
                    result = executor.close_position_market(symbol, side)
                    if result.get("executed"):
                        logger.info(
                            "✅ Testnet mirror: %s %s via MARKET close (qty=%s, orderId=%s)",
                            action, symbol, result.get("quantity"), result.get("order_id"),
                        )
                    else:
                        logger.warning(
                            "Testnet mirror: %s %s — %s",
                            action, symbol, result.get("error", "unknown error"),
                        )
            except Exception as exc:
                logger.warning("Testnet mirror close failed for %s: %s", symbol, exc)
                self._write_system_alert(
                    "REST_API_ERROR",
                    f"Failed to close {symbol} position on exchange: {exc}"
                )
            return
        try:
            result = self.testnet_executor.mirror_trade(trade, {})
            if result and result.get("executed"):
                logger.info(
                    "Testnet mirror: %s @ $%.2f (order=%s sl=%s tp=%s)",
                    action,
                    trade.get("price", 0),
                    result.get("order_id", ""),
                    result.get("sl_order_id", ""),
                    result.get("tp_order_id", ""),
                )
        except Exception as exc:
            logger.warning("Testnet mirror failed: %s", exc)
            self._write_system_alert(
                "REST_API_ERROR",
                f"Testnet mirror failed for {action} {symbol}: {exc}"
            )

    # ------------------------------------------------------------------
    # Main iteration
    # ------------------------------------------------------------------

    def run_iteration(self) -> Dict:
        """Execute one full decision cycle. Returns a status dict."""
        t0 = time.time()
        status: Dict = {
            "timestamp": datetime.now().isoformat(),
            "symbol": self.symbol,
            "action": "HOLD",
            "price": 0.0,
            "confidence": 0.0,
            "position": self.position,
            "balance": self.balance,
            "realized_pnl": self.realized_pnl,
            "error": None,
        }

        try:
            # 1. Fetch data
            df_15m = self.fetch_data()
            if df_15m is None or df_15m.empty:
                status["error"] = "No data"
                return status

            current_price = float(df_15m.iloc[-1]["close"])
            self.current_price = current_price
            status["price"] = current_price

            # Cache data for regime detection in _open_position
            self._last_df = df_15m

            # Exchange position sync: detect stale positions (exchange closed
            # but bot still thinks it's in a trade → ghost SL/TP alerts).
            # Runs every iteration when bot thinks it has a position.
            if self.position != 0 and not self.dry_run:
                self._sync_position_from_exchange()
                # If sync reset us to flat, skip the rest of position logic
                if self.position == 0:
                    status["action"] = "SYNC_FLAT"
                    logger.info("Position was stale — synced to FLAT, skipping to next iteration")
                    return status

            # Phase 1 §3.9: Update MFE/MAE every iteration while in position
            if self.position != 0 and self.position_price > 0:
                if self.position == 1:
                    _unrealized_pct = (current_price - self.position_price) / self.position_price
                else:
                    _unrealized_pct = (self.position_price - current_price) / self.position_price
                self.mfe_pct = max(self.mfe_pct, _unrealized_pct)
                self.mae_pct = min(self.mae_pct, _unrealized_pct)

            # Phase 1 §3.3: Partial Take Profit check (40% at 1R, 35% at 2R)
            if self.position != 0 and self.position_price > 0 and self.initial_position_units > 0:
                if self.position == 1 and self.partial_tp_level < 1 and self.partial_tp1_price > 0:
                    if current_price >= self.partial_tp1_price:
                        partial_units = self.initial_position_units * 0.40
                        partial_units = min(partial_units, self.position_units)
                        if partial_units > 0:
                            partial_pnl = (self.partial_tp1_price - self.position_price) * partial_units
                            self.realized_pnl += partial_pnl
                            self.position_units -= partial_units
                            self.last_partial_close_time = time.time()
                            self.partial_tp_level = 1
                            old_sl = self.sl_price
                            self.sl_price = self.position_price  # Move SL to break-even
                            logger.info(
                                "✅ PARTIAL TP1 (LONG): closed %.5f units @ $%.2f | "
                                "PnL=$%+.2f | SL → break-even $%.2f (was $%.2f) | "
                                "remaining=%.5f",
                                partial_units, self.partial_tp1_price, partial_pnl,
                                self.sl_price, old_sl, self.position_units,
                            )
                            self._log_trade({
                                "action": "PARTIAL_CLOSE_LONG",
                                "reason": "PARTIAL_TP1",
                                "symbol": self.symbol,
                                "entry_price": self.position_price,
                                "exit_price": self.partial_tp1_price,
                                "units": partial_units,
                                "remaining_units": self.position_units,
                                "original_units": self.initial_position_units,
                                "pnl": partial_pnl,
                                "partial_exit_num": 1,
                                "old_sl": old_sl,
                                "new_sl": self.sl_price,
                                "timestamp": datetime.now().isoformat(),
                                "agent": "htf",
                                "balance_after": self._get_real_balance(),
                            })
                            self._sync_balance_from_exchange()
                            self._save_state()
                            # Sync updated SL to exchange
                            if not self.dry_run and self.testnet_executor:
                                try:
                                    self.testnet_executor.update_sl_tp(
                                        self.symbol, self.sl_price, self.tp_price
                                    )
                                except Exception as _exc:
                                    logger.warning("Testnet SL update after TP1 failed: %s", _exc)

                elif self.position == 1 and self.partial_tp_level == 1 and self.partial_tp2_price > 0:
                    if current_price >= self.partial_tp2_price:
                        partial_units = self.initial_position_units * 0.35
                        partial_units = min(partial_units, self.position_units)
                        if partial_units > 0:
                            partial_pnl = (self.partial_tp2_price - self.position_price) * partial_units
                            self.realized_pnl += partial_pnl
                            self.position_units -= partial_units
                            self.last_partial_close_time = time.time()
                            self.partial_tp_level = 2
                            level2_gain = self.partial_tp2_price - self.position_price
                            new_trailing_sl = self.position_price + level2_gain * 0.50
                            if new_trailing_sl > self.sl_price:
                                old_sl = self.sl_price
                                self.sl_price = new_trailing_sl
                                logger.info(
                                    "✅ PARTIAL TP2 (LONG): closed %.5f units @ $%.2f | "
                                    "PnL=$%+.2f | trailing SL locked → $%.2f (was $%.2f) | "
                                    "remaining=%.5f",
                                    partial_units, self.partial_tp2_price, partial_pnl,
                                    self.sl_price, old_sl, self.position_units,
                                )
                            else:
                                logger.info(
                                    "✅ PARTIAL TP2 (LONG): closed %.5f units @ $%.2f | "
                                    "PnL=$%+.2f | remaining=%.5f (SL already ahead)",
                                    partial_units, self.partial_tp2_price, partial_pnl,
                                    self.position_units,
                                )
                            self._log_trade({
                                "action": "PARTIAL_CLOSE_LONG",
                                "reason": "PARTIAL_TP2",
                                "symbol": self.symbol,
                                "entry_price": self.position_price,
                                "exit_price": self.partial_tp2_price,
                                "units": partial_units,
                                "remaining_units": self.position_units,
                                "original_units": self.initial_position_units,
                                "pnl": partial_pnl,
                                "partial_exit_num": 2,
                                "new_sl": self.sl_price,
                                "timestamp": datetime.now().isoformat(),
                                "agent": "htf",
                                "balance_after": self._get_real_balance(),
                            })
                            self._sync_balance_from_exchange()
                            self._save_state()
                            if not self.dry_run and self.testnet_executor:
                                try:
                                    self.testnet_executor.update_sl_tp(
                                        self.symbol, self.sl_price, self.tp_price
                                    )
                                except Exception as _exc:
                                    logger.warning("Testnet SL update after TP2 failed: %s", _exc)

                elif self.position == -1 and self.partial_tp_level < 1 and self.partial_tp1_price > 0:
                    if current_price <= self.partial_tp1_price:
                        partial_units = self.initial_position_units * 0.40
                        partial_units = min(partial_units, self.position_units)
                        if partial_units > 0:
                            partial_pnl = (self.position_price - self.partial_tp1_price) * partial_units
                            self.realized_pnl += partial_pnl
                            self.position_units -= partial_units
                            self.last_partial_close_time = time.time()
                            self.partial_tp_level = 1
                            old_sl = self.sl_price
                            self.sl_price = self.position_price  # Move SL to break-even
                            logger.info(
                                "✅ PARTIAL TP1 (SHORT): closed %.5f units @ $%.2f | "
                                "PnL=$%+.2f | SL → break-even $%.2f (was $%.2f) | "
                                "remaining=%.5f",
                                partial_units, self.partial_tp1_price, partial_pnl,
                                self.sl_price, old_sl, self.position_units,
                            )
                            self._log_trade({
                                "action": "PARTIAL_CLOSE_SHORT",
                                "reason": "PARTIAL_TP1",
                                "symbol": self.symbol,
                                "entry_price": self.position_price,
                                "exit_price": self.partial_tp1_price,
                                "units": partial_units,
                                "remaining_units": self.position_units,
                                "original_units": self.initial_position_units,
                                "pnl": partial_pnl,
                                "partial_exit_num": 1,
                                "old_sl": old_sl,
                                "new_sl": self.sl_price,
                                "timestamp": datetime.now().isoformat(),
                                "agent": "htf",
                                "balance_after": self._get_real_balance(),
                            })
                            self._sync_balance_from_exchange()
                            self._save_state()
                            if not self.dry_run and self.testnet_executor:
                                try:
                                    self.testnet_executor.update_sl_tp(
                                        self.symbol, self.sl_price, self.tp_price
                                    )
                                except Exception as _exc:
                                    logger.warning("Testnet SL update after TP1 failed: %s", _exc)

                elif self.position == -1 and self.partial_tp_level == 1 and self.partial_tp2_price > 0:
                    if current_price <= self.partial_tp2_price:
                        partial_units = self.initial_position_units * 0.35
                        partial_units = min(partial_units, self.position_units)
                        if partial_units > 0:
                            partial_pnl = (self.position_price - self.partial_tp2_price) * partial_units
                            self.realized_pnl += partial_pnl
                            self.position_units -= partial_units
                            self.last_partial_close_time = time.time()
                            self.partial_tp_level = 2
                            level2_gain = self.position_price - self.partial_tp2_price
                            new_trailing_sl = self.position_price - level2_gain * 0.50
                            if new_trailing_sl < self.sl_price:
                                old_sl = self.sl_price
                                self.sl_price = new_trailing_sl
                                logger.info(
                                    "✅ PARTIAL TP2 (SHORT): closed %.5f units @ $%.2f | "
                                    "PnL=$%+.2f | trailing SL locked → $%.2f (was $%.2f) | "
                                    "remaining=%.5f",
                                    partial_units, self.partial_tp2_price, partial_pnl,
                                    self.sl_price, old_sl, self.position_units,
                                )
                            else:
                                logger.info(
                                    "✅ PARTIAL TP2 (SHORT): closed %.5f units @ $%.2f | "
                                    "PnL=$%+.2f | remaining=%.5f (SL already ahead)",
                                    partial_units, self.partial_tp2_price, partial_pnl,
                                    self.position_units,
                                )
                            self._log_trade({
                                "action": "PARTIAL_CLOSE_SHORT",
                                "reason": "PARTIAL_TP2",
                                "symbol": self.symbol,
                                "entry_price": self.position_price,
                                "exit_price": self.partial_tp2_price,
                                "units": partial_units,
                                "remaining_units": self.position_units,
                                "original_units": self.initial_position_units,
                                "pnl": partial_pnl,
                                "partial_exit_num": 2,
                                "new_sl": self.sl_price,
                                "timestamp": datetime.now().isoformat(),
                                "agent": "htf",
                                "balance_after": self._get_real_balance(),
                            })
                            self._sync_balance_from_exchange()
                            self._save_state()
                            if not self.dry_run and self.testnet_executor:
                                try:
                                    self.testnet_executor.update_sl_tp(
                                        self.symbol, self.sl_price, self.tp_price
                                    )
                                except Exception as _exc:
                                    logger.warning("Testnet SL update after TP2 failed: %s", _exc)

            # 1b. Update peak price for trailing stop tracking
            self._update_peak_price(current_price)

            # 1c. Compute BOS/CHOCH structure signals on 5m candles (cached for this iteration)
            self._get_structure_signals(trigger=True)

            # 2. Check SL/TP before computing new action (uses cached BOS/CHOCH)
            exit_reason = self._check_sl_tp(current_price)
            if exit_reason and self.position != 0:
                logger.info("SL/TP triggered (%s) @ $%.2f", exit_reason, current_price)
                trade = self._close_position(current_price, exit_reason, 1.0)
                status["action"] = trade.get("action", "CLOSE")
                status["position"] = self.position
                status["balance"] = self.balance
                status["realized_pnl"] = self.realized_pnl
                return status

            # Phase 1 §3.4: Time-based stagnant exit (see STAGNANT_* constants).
            # BUT: keep position if model confidence ≥ 0.80 OR 2/4 market signals agree
            if self.position != 0 and self.position_price > 0 and self.position_entry_time > 0:
                _time_in_pos = time.time() - self.position_entry_time
                if _time_in_pos > STAGNANT_HOURS * 3600:
                    if self.position == 1:
                        _stagnant_pct = (current_price - self.position_price) / self.position_price
                    else:
                        _stagnant_pct = (self.position_price - current_price) / self.position_price
                    if STAGNANT_PCT_MIN <= _stagnant_pct <= STAGNANT_PCT_MAX:
                        # Check if we should KEEP the position despite stagnation
                        _keep_position = False
                        _keep_reason = ""

                        # Check 1: Model confidence — if agent is still confident, hold
                        try:
                            obs = self.compute_observation(df_15m)
                            if obs is not None:
                                _action, _conf = self.predict(obs)
                                # Keep if model confidence ≥ 0.80 AND model still agrees with direction
                                if _conf >= 0.80:
                                    _model_wants_long = (_action == ACTION_LONG)
                                    _model_wants_short = (_action == ACTION_SHORT)
                                    if (self.position == 1 and _model_wants_long) or \
                                       (self.position == -1 and _model_wants_short):
                                        _keep_position = True
                                        _keep_reason = f"model conf={_conf:.2f} agrees with position"
                        except Exception as _exc:
                            logger.debug("Stagnant gate model check failed: %s", _exc)

                        # Check 2: Market signal gate — same logic as entry gate
                        if not _keep_position:
                            try:
                                _direction = "LONG" if self.position == 1 else "SHORT"
                                _market = self._fetch_market_signals(self.symbol)
                                if _market:
                                    _confirmations = 0

                                    # MTF
                                    _mtf = _market.get("mtf", {})
                                    _mtf_bias = (_mtf.get("bias") or "NEUTRAL").upper()
                                    _mtf_aligned = _mtf.get("aligned", False)
                                    if _direction == "LONG" and _mtf_bias == "BULLISH" and _mtf_aligned:
                                        _confirmations += 1
                                    elif _direction == "SHORT" and _mtf_bias == "BEARISH" and _mtf_aligned:
                                        _confirmations += 1

                                    # Order Flow
                                    _of = _market.get("order_flow", {})
                                    _of_score = _of.get("score", 0)
                                    if isinstance(_of_score, (int, float)):
                                        if _direction == "LONG" and _of_score > 0.3:
                                            _confirmations += 1
                                        elif _direction == "SHORT" and _of_score < -0.3:
                                            _confirmations += 1

                                    # Regime
                                    _regime = _market.get("regime", {})
                                    _regime_type = (_regime.get("type") or "").upper()
                                    _regime_adx = _regime.get("adx", 0) or 0
                                    if _direction == "LONG" and "UP" in _regime_type and _regime_adx >= 20:
                                        _confirmations += 1
                                    elif _direction == "SHORT" and "DOWN" in _regime_type and _regime_adx >= 20:
                                        _confirmations += 1

                                    # Orderbook
                                    _ob = _of.get("orderbook", {})
                                    _ob_imb = _ob.get("imbalance_10", 0)
                                    if isinstance(_ob_imb, (int, float)):
                                        if _direction == "LONG" and _ob_imb > 0.15:
                                            _confirmations += 1
                                        elif _direction == "SHORT" and _ob_imb < -0.15:
                                            _confirmations += 1

                                    if _confirmations >= 2:
                                        _keep_position = True
                                        _keep_reason = f"{_confirmations}/4 market signals agree with {_direction}"
                            except Exception as _exc:
                                logger.debug("Stagnant gate market check failed: %s", _exc)

                        if _keep_position:
                            logger.info(
                                "⏱️ STAGNANT but HOLDING: %s in position %.1fh, "
                                "PnL=%+.3f%% — keeping because %s",
                                self.symbol, _time_in_pos / 3600, _stagnant_pct * 100,
                                _keep_reason,
                            )
                        else:
                            logger.info(
                                "⏱️ TIME-BASED STAGNANT EXIT: %s in position %.1fh, "
                                "PnL=%+.3f%% (within stagnant band [%+.1f%%, %+.1f%%]) "
                                "— no model confidence or market signal support → closing",
                                self.symbol, _time_in_pos / 3600, _stagnant_pct * 100,
                                STAGNANT_PCT_MIN * 100, STAGNANT_PCT_MAX * 100,
                            )
                            stagnant_trade = self._close_position(current_price, "STAGNANT_EXIT", 1.0)
                            status["action"] = stagnant_trade.get("action", "CLOSE")
                            status["position"] = self.position
                            status["balance"] = self.balance
                            status["realized_pnl"] = self.realized_pnl
                            return status

            # 3. Determine action — structure-first or model-first
            if STRUCTURE_FIRST_MODE:
                # Structure-first: BOS/CHOCH signals drive entry direction
                struct_action = self._get_structure_direction(current_price, df_15m)
                if struct_action is None:
                    action = ACTION_HOLD
                    confidence = 0.0
                else:
                    action = struct_action
                    # Use BOS/CHOCH confidence from the structure signal
                    sig = self._last_structure_signals or {}
                    confidence = sig.get("confidence", 0.5)
            else:
                # Model-first (legacy): PPO model drives entry direction
                obs = self.compute_observation(df_15m)
                if obs is None:
                    status["error"] = "Observation failed"
                    return status
                action, confidence = self.get_action(obs)

            status["action"] = ACTION_LABELS[action]
            status["confidence"] = confidence

            logger.info(
                "Step: price=$%.2f action=%s conf=%.2f pos=%d bal=$%.2f mode=%s",
                current_price, ACTION_LABELS[action], confidence, self.position, self.balance,
                "STRUCT" if STRUCTURE_FIRST_MODE else "MODEL",
            )

            # 4. Execute trade (guards still apply: RSI, ADX, orderbook, exhaustion)
            trade = self.execute_trade(action, confidence, current_price)
            if trade:
                status["action"] = trade.get("action", ACTION_LABELS[action])

        except Exception as exc:
            logger.error("run_iteration error: %s", exc, exc_info=True)
            status["error"] = str(exc)

        status["position"] = self.position
        status["balance"] = self.balance
        status["realized_pnl"] = self.realized_pnl
        status["elapsed_s"] = round(time.time() - t0, 2)
        return status

    # ------------------------------------------------------------------
    # WebSocket price monitor
    # ------------------------------------------------------------------

    def _start_ws_monitor(self) -> None:
        """
        Start a background daemon thread that streams aggTrade prices from
        Binance and triggers SL/TP closes in real-time (between 15-min loops).
        """
        import websocket as _websocket

        ws_symbol = self.symbol.lower()  # e.g. "ethusdt"
        url = f"wss://stream.binance.com:9443/ws/{ws_symbol}@aggTrade"

        # Track WS state for disconnect alerting
        ws_state = {
            "connected": False,
            "last_tick_time": 0,
            "disconnect_alerted": False,
            "reconnect_count": 0,
        }

        def _write_connectivity_alert(alert_type: str, details: str) -> None:
            """Write a connectivity alert to the shared alerts file."""
            try:
                alert_file = Path("logs/htf_pending_alerts.jsonl")
                alert_file.parent.mkdir(parents=True, exist_ok=True)
                with open(alert_file, "a") as f:
                    f.write(json.dumps({
                        "timestamp": datetime.now().isoformat(),
                        "strategy": "htf",
                        "trade": {
                            "type": alert_type,
                            "symbol": self.symbol,
                            "details": details,
                            "has_open_position": self.position != 0,
                            "position_direction": "LONG" if self.position == 1 else "SHORT" if self.position == -1 else "FLAT",
                        },
                        "signals": {},
                        "position": {
                            "entry_price": self.position_price,
                            "sl_price": self.sl_price,
                            "tp_price": self.tp_price,
                            "units": self.position_units,
                            "direction": "LONG" if self.position == 1 else "SHORT" if self.position == -1 else "FLAT",
                        },
                    }) + "\n")
            except Exception as exc:
                logger.error("Failed to write connectivity alert: %s", exc)

        def _on_open(ws):
            logger.info("WS price monitor connected to aggTrade stream")
            if ws_state["reconnect_count"] > 0:
                _write_connectivity_alert(
                    "WS_RECONNECTED",
                    f"WebSocket reconnected after {ws_state['reconnect_count']} retries"
                )
            # Skip WS_CONNECTED alert on initial startup — it's just noise (4 symbols × 1 msg each)
            ws_state["connected"] = True
            ws_state["disconnect_alerted"] = False
            ws_state["last_tick_time"] = time.time()

        def _on_error(ws, error):
            logger.warning("WS price monitor error: %s", error)
            if not ws_state["disconnect_alerted"]:
                pos_ctx = f"while {self.symbol} position is open" if self.position != 0 else f"({self.symbol} flat)"
                _write_connectivity_alert(
                    "WS_ERROR",
                    f"WebSocket error {pos_ctx}: {error}"
                )
                ws_state["disconnect_alerted"] = True

        def _on_close(ws, code, msg):
            logger.info("WS price monitor closed (code=%s)", code)
            ws_state["connected"] = False
            if not ws_state["disconnect_alerted"]:
                if self.position != 0:
                    detail = (f"WebSocket disconnected (code={code}) while {self.symbol} position is open! "
                              f"Exchange crashguard SL is active as backup.")
                else:
                    detail = f"WebSocket disconnected (code={code}) — {self.symbol} currently flat."
                _write_connectivity_alert("WS_DISCONNECTED", detail)
                ws_state["disconnect_alerted"] = True

        first_price_logged = [False]  # mutable container for nonlocal in Python 2-compat closure

        def _on_message(ws, message):
            try:
                data = json.loads(message)
                price = float(data.get("p", 0))
                if price <= 0:
                    return

                self.current_price = price

                # Phase 1 §3.9: Update MFE/MAE from real-time price ticks
                if self.position != 0 and self.position_price > 0:
                    if self.position == 1:
                        _tick_unrealized = (price - self.position_price) / self.position_price
                    else:
                        _tick_unrealized = (self.position_price - price) / self.position_price
                    if _tick_unrealized > self.mfe_pct:
                        self.mfe_pct = _tick_unrealized
                    if _tick_unrealized < self.mae_pct:
                        self.mae_pct = _tick_unrealized

                ws_state["last_tick_time"] = time.time()

                if not first_price_logged[0]:
                    logger.info("WS first price tick: $%.2f — stream is live", price)
                    first_price_logged[0] = True

                with self._lock:
                    if self.position == 0:
                        return
                    exit_reason = self._check_sl_tp(price)
                    if exit_reason:
                        logger.info("WS: SL/TP triggered (%s) @ $%.2f", exit_reason, price)
                        self._close_position(price, exit_reason, 1.0)

            except Exception as exc:
                logger.debug("WS message handler error: %s", exc)

        def _run():
            backoff = 1
            while True:
                try:
                    ws = _websocket.WebSocketApp(
                        url,
                        on_open=_on_open,
                        on_message=_on_message,
                        on_error=_on_error,
                        on_close=_on_close,
                    )
                    ws.run_forever()
                except Exception as exc:
                    logger.warning("WS run_forever exception: %s", exc)
                ws_state["reconnect_count"] += 1
                logger.info("WS reconnecting in %ds... (attempt #%d)", backoff, ws_state["reconnect_count"])
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)

        t = threading.Thread(target=_run, name="ws-price-monitor", daemon=True)
        t.start()
        logger.info("WebSocket price monitor thread started")

    # ------------------------------------------------------------------
    # Continuous loop
    # ------------------------------------------------------------------

    def run_loop(self, stop_event: Optional[threading.Event] = None) -> None:
        """Run the trading loop at `interval_minutes` cadence until stopped."""
        interval_secs = self.interval_minutes * 60
        logger.info(
            "HTF trading loop started | interval=%dmin dry_run=%s",
            self.interval_minutes, self.dry_run,
        )
        self._start_ws_monitor()

        while True:
            if stop_event and stop_event.is_set():
                logger.info("Stop event received — shutting down.")
                break

            with self._lock:
                status = self.run_iteration()

            logger.info("Iteration complete: %s", json.dumps(status, default=str))

            # Sleep until next bar
            if stop_event:
                stop_event.wait(timeout=interval_secs)
            else:
                time.sleep(interval_secs)

    # ------------------------------------------------------------------
    # Status summary (for API)
    # ------------------------------------------------------------------

    def get_status(self) -> Dict:
        """Return current agent status (safe to call from API thread)."""
        with self._lock:
            unrealized_pnl = 0.0
            if self.position != 0 and self.position_price > 0 and self.current_price > 0:
                if self.position == 1:
                    unrealized_pnl = (self.current_price - self.position_price) * self.position_units
                else:
                    unrealized_pnl = (self.position_price - self.current_price) * self.position_units

            return {
                "symbol": self.symbol,
                "position": self.position,
                "position_label": {1: "LONG", -1: "SHORT", 0: "FLAT"}.get(self.position, "FLAT"),
                "position_price": self.position_price,
                "position_units": self.position_units,
                "sl_price": self.sl_price,
                "tp_price": self.tp_price,
                "current_price": self.current_price,
                "balance": self.balance,
                "realized_pnl": self.realized_pnl,
                "unrealized_pnl": unrealized_pnl,
                "total_pnl": self.realized_pnl + unrealized_pnl,
                "model_path": str(self._model_path) if self._model_path else None,
                "start_time": self.start_time,
                "trade_count": len(self.trades),
                "dry_run": self.dry_run,
            }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HTF Live Trading Bot")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Paper trading only (default: True)")
    parser.add_argument("--live", action="store_true",
                        help="Enable real trade execution (overrides --dry-run)")
    parser.add_argument("--balance", type=float, default=10_000.0,
                        help="Initial balance in USDT (default: 10000)")
    parser.add_argument("--interval", type=int, default=15,
                        help="Decision interval in minutes (default: 15)")
    parser.add_argument("--symbol", type=str, default=None,
                        help="Trading symbol (e.g. ETHUSDT). Overrides HTF_SYMBOL env var.")
    parser.add_argument("--once", action="store_true",
                        help="Run a single iteration and exit")
    args = parser.parse_args()

    # Symbol override: CLI flag > env var > default
    if args.symbol:
        global SYMBOL
        SYMBOL = args.symbol

    dry_run = not args.live  # live flag disables dry_run

    bot = HTFLiveBot(
        dry_run=dry_run,
        initial_balance=args.balance,
        interval_minutes=args.interval,
    )

    if args.once:
        status = bot.run_iteration()
        print(json.dumps(status, indent=2, default=str))
        return

    stop_event = threading.Event()
    try:
        bot.run_loop(stop_event=stop_event)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — stopping.")
        stop_event.set()


if __name__ == "__main__":
    main()
