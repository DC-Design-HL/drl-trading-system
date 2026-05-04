"""Built-in strategy library.

Each module registers a strategy via the @register_strategy decorator.
Importing this package side-effect-loads them.
"""

# Side-effect imports — register strategies on package load
from . import structure_first  # noqa: F401
from . import model_first      # noqa: F401

