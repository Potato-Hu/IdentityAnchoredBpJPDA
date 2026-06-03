from __future__ import annotations

__all__ = [
    "StepResult",
    "SwarmEnv",
    "SwarmWorld",
]


def __getattr__(name: str):
    if name in {"StepResult", "SwarmWorld"}:
        from src.env.world import StepResult, SwarmWorld

        exports = {
            "StepResult": StepResult,
            "SwarmWorld": SwarmWorld,
        }
        return exports[name]

    if name in {"SwarmEnv"}:
        from src.env.swarm_env import SwarmEnv

        return SwarmEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
