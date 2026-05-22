"""Self-improvement plane for the DRL trading system.

See PLAN.md at the repo root for the full design. The self-improvement
plane is strictly additive — it reads from the live trading database
(data/trading.db) and writes to its own new tables (decisions,
experiments, metrics_snapshots, agent_runs). It does not modify the
live trading processes directly; any change to live behavior goes
through the experiment pipeline (backtest → paper → canary → live)
with Risk Officer gates and automated rollback.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("drl-trading-system")
except PackageNotFoundError:
    __version__ = "0.0.0+self_improve"

DEFAULT_DB_PATH = "data/trading.db"
