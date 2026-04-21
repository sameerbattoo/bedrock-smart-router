"""Custom strategy plugin interface.

Users can implement their own routing strategy by subclassing
``RoutingStrategy`` and registering it with the strategy engine.

Example::

    from bedrock_smart_router.custom_strategy import register_strategy
    from bedrock_smart_router.strategy_engine import RoutingStrategy, StrategyResult

    class MyStrategy(RoutingStrategy):
        name = "my-custom"

        def select(self, candidates, analysis):
            # Your custom logic here
            best = candidates[0]  # Simplest possible: pick first
            return StrategyResult(
                selected_model=best,
                scores={best.model_id: {"composite": 1.0}},
                fallback_chain=candidates[1:3],
            )

    register_strategy("my-custom", MyStrategy)

    # Now usable in config:
    router = BedrockRouter.create({"strategy": "my-custom"})
"""

from __future__ import annotations

from typing import Any

from bedrock_smart_router.strategy_engine import (
    BUILTIN_STRATEGIES,
    RoutingStrategy,
)


def register_strategy(name: str, cls: type[RoutingStrategy]) -> None:
    """Register a custom strategy class so it can be resolved by name.

    Args:
        name: The strategy name to use in config (e.g. ``"my-custom"``).
        cls: A subclass of ``RoutingStrategy``.

    Raises:
        TypeError: If *cls* is not a subclass of ``RoutingStrategy``.
    """
    if not (isinstance(cls, type) and issubclass(cls, RoutingStrategy)):
        raise TypeError(
            f"{cls} is not a subclass of RoutingStrategy"
        )
    BUILTIN_STRATEGIES[name] = cls


def unregister_strategy(name: str) -> bool:
    """Remove a custom strategy registration.

    Returns True if the strategy was found and removed.
    """
    return BUILTIN_STRATEGIES.pop(name, None) is not None


def list_strategies() -> list[str]:
    """Return all registered strategy names."""
    return list(BUILTIN_STRATEGIES.keys()) + ["quality-optimized"]
