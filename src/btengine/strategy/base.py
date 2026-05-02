"""Strategy ABC + composable primitives.

A strategy is `EntryRule + GuardChain + ExitPolicy + Sizing`. Each piece
is a pluggable component that the runner consults at well-defined hooks
(on_bar, on_position_update). Phase 1 defines the surface; concrete
implementations land in M2 (entry) / M3 (exits) / M4 (guards).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


# ── Decisions ──────────────────────────────────────────────────────────

@dataclass
class Intent:
    """What the strategy wants to do at this bar.

    `action` is one of: HOLD, OPEN_LONG, OPEN_SHORT, CLOSE.
    Additional fields are opaque to the runner; they propagate to the
    broker (sl/tp prices, sizing hints) and to logs.
    """
    action: str
    confidence: float = 0.0
    sl_pct: float = 0.0
    tp_pct: float = 0.0
    reason: str = ""
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GuardResult:
    """Block or allow. If blocked, `reason` is logged for ablation analysis."""
    allowed: bool
    reason: str = ""
    snapshot: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def allow(cls) -> "GuardResult":
        return cls(allowed=True)

    @classmethod
    def block(cls, reason: str, **snap) -> "GuardResult":
        return cls(allowed=False, reason=reason, snapshot=dict(snap))


# ── Components ─────────────────────────────────────────────────────────

class EntryRule(ABC):
    """Produces an Intent (or HOLD) at each bar based on signals."""

    @abstractmethod
    def __call__(self, ctx) -> Intent: ...


class Guard(ABC):
    """Either allows or blocks an Intent. Multiple guards form a chain."""

    name: str = ""

    @abstractmethod
    def __call__(self, intent: Intent, ctx) -> GuardResult: ...


class GuardChain:
    """Apply guards in order; first blocker wins. Empty chain always allows."""

    def __init__(self, guards: Sequence[Guard]):
        self._guards = list(guards)

    def __call__(self, intent: Intent, ctx) -> GuardResult:
        for g in self._guards:
            r = g(intent, ctx)
            if not r.allowed:
                return r
        return GuardResult.allow()

    def __iter__(self):
        return iter(self._guards)

    def __len__(self):
        return len(self._guards)


class ExitPolicy(ABC):
    """Owns partial-TP, trailing, stagnant, REVERSE_CLOSE behavior.

    Lifecycle:
        on_open(position, ctx)
        on_bar(position, ctx) -> Optional[ExitDecision]
    """

    @abstractmethod
    def on_open(self, position, ctx) -> None: ...

    @abstractmethod
    def on_bar(self, position, ctx) -> Optional["ExitDecision"]: ...


@dataclass
class ExitDecision:
    """What the exit policy wants the broker to do."""
    kind: str   # "sl" | "tp" | "tp_partial" | "trail" | "stagnant" | "reverse_close"
    price: float
    fraction: float = 1.0    # 1.0 = close all remaining
    reason: str = ""


class Sizing(ABC):
    """Translates an entry intent into a position size."""

    @abstractmethod
    def units_for(self, intent: Intent, ctx) -> float: ...


# ── Top-level strategy ──────────────────────────────────────────────────

class Strategy(ABC):
    """Composition root. Concrete subclasses live under
    src/btengine/strategy/library/ and register via @register_strategy."""

    name: str = ""

    entry: Optional[EntryRule] = None
    guards: GuardChain = GuardChain([])
    exits: Optional[ExitPolicy] = None
    sizing: Optional[Sizing] = None

    def on_bar(self, ctx) -> Intent:
        """Default: defer to entry rule."""
        if self.entry is None:
            return Intent(action="HOLD")
        return self.entry(ctx)


# ── Registry ───────────────────────────────────────────────────────────

_REGISTRY: Dict[str, type] = {}


def register_strategy(name: str):
    """Decorator: associate a Strategy subclass with a name for config lookup."""
    def _wrap(cls: type) -> type:
        if name in _REGISTRY:
            raise ValueError(f"strategy {name!r} already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return _wrap


def get_strategy(name: str) -> type:
    if name not in _REGISTRY:
        # Lazy-load the library so strategies register on first lookup
        from . import library  # noqa: F401
    if name not in _REGISTRY:
        raise KeyError(f"strategy {name!r} not registered. Known: {list(_REGISTRY)}")
    return _REGISTRY[name]


def list_strategies() -> List[str]:
    from . import library  # noqa: F401
    return sorted(_REGISTRY)
