"""Space builders for the swarm RL project.

This file defines action, observation, and critic-state spaces used by the env skeleton.
The local-only baseline exposes legal self state and anonymous local target candidates;
the groundtruth debug mode exposes full world-truth target features.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

try:
    from gymnasium import spaces as gym_spaces
except ModuleNotFoundError:
    @dataclass(frozen=True)
    class _FallbackDiscrete:
        n: int

    @dataclass(frozen=True)
    class _FallbackBox:
        low: np.ndarray
        high: np.ndarray
        shape: tuple[int, ...]
        dtype: Any

    gym_spaces = SimpleNamespace(Discrete=_FallbackDiscrete, Box=_FallbackBox)

spaces = gym_spaces


@dataclass(frozen=True)
class SwarmSpaceSpec:
    """Compact summary of the current multi-agent space layout."""

    num_friendlies: int
    num_enemies: int
    per_agent_action_dim: int
    per_agent_obs_dim: int
    global_state_dim: int


def build_agent_ids(num_friendlies: int) -> list[str]:
    return [f"agent_{idx}" for idx in range(num_friendlies)]


def build_action_spaces(num_friendlies: int, action_dim: int) -> dict[str, Any]:
    return {agent_id: spaces.Discrete(action_dim) for agent_id in build_agent_ids(num_friendlies)}


def local_only_observation_dim(max_local_candidates: int) -> int:
    return 3 + 3 * max_local_candidates + max_local_candidates + max_local_candidates


def groundtruth_observation_dim(num_friendlies: int, num_enemies: int, num_key_enemies: int) -> int:
    assigned_target_dim = 4 + num_key_enemies if num_key_enemies > 0 else 0
    return 3 + 4 * num_enemies + num_key_enemies + assigned_target_dim + 3 * num_friendlies


def policy_observation_dim(
    policy_input_mode: str,
    num_friendlies: int,
    num_enemies: int,
    num_key_enemies: int,
    max_local_candidates: int,
) -> int:
    if policy_input_mode == "groundtruth":
        return groundtruth_observation_dim(num_friendlies, num_key_enemies, num_key_enemies)
    return local_only_observation_dim(max_local_candidates)


def policy_global_state_dim(
    policy_input_mode: str,
    num_friendlies: int,
    num_enemies: int,
    num_key_enemies: int,
    max_local_candidates: int,
) -> int:
    if policy_input_mode == "groundtruth":
        return 3 * num_key_enemies + 3 * num_friendlies + num_key_enemies + num_friendlies
    return num_friendlies * local_only_observation_dim(max_local_candidates)


def build_observation_spaces(
    num_friendlies: int,
    num_enemies: int,
    num_key_enemies: int,
    max_local_candidates: int,
    bounds: np.ndarray,
    policy_input_mode: str,
) -> dict[str, Any]:
    per_agent_obs_dim = policy_observation_dim(
        policy_input_mode,
        num_friendlies,
        num_enemies,
        num_key_enemies,
        max_local_candidates,
    )
    high = np.ones(per_agent_obs_dim, dtype=np.float32)
    low = -np.ones(per_agent_obs_dim, dtype=np.float32)
    return {
        agent_id: spaces.Box(low=low, high=high, shape=(per_agent_obs_dim,), dtype=np.float32)
        for agent_id in build_agent_ids(num_friendlies)
    }


def build_global_state_space(
    num_friendlies: int,
    num_enemies: int,
    num_key_enemies: int,
    max_local_candidates: int,
    bounds: np.ndarray,
    policy_input_mode: str,
) -> Any:
    global_state_dim = policy_global_state_dim(
        policy_input_mode,
        num_friendlies,
        num_enemies,
        num_key_enemies,
        max_local_candidates,
    )
    if policy_input_mode == "groundtruth":
        position_high = np.ones(3 * num_key_enemies + 3 * num_friendlies, dtype=np.float32)
        power_high = np.full(num_key_enemies + num_friendlies, np.finfo(np.float32).max, dtype=np.float32)
        high = np.concatenate([position_high, power_high], dtype=np.float32)
        low = np.concatenate(
            [
                np.zeros(3 * num_key_enemies + 3 * num_friendlies, dtype=np.float32),
                np.full(num_key_enemies + num_friendlies, -np.finfo(np.float32).max, dtype=np.float32),
            ],
            dtype=np.float32,
        )
    else:
        high = np.ones(global_state_dim, dtype=np.float32)
        low = -np.ones(global_state_dim, dtype=np.float32)
    return spaces.Box(low=low, high=high, shape=(global_state_dim,), dtype=np.float32)


def build_space_spec(
    num_friendlies: int,
    num_enemies: int,
    num_key_enemies: int,
    max_local_candidates: int,
    action_dim: int,
    policy_input_mode: str,
) -> SwarmSpaceSpec:
    per_agent_obs_dim = policy_observation_dim(
        policy_input_mode,
        num_friendlies,
        num_enemies,
        num_key_enemies,
        max_local_candidates,
    )
    global_state_dim = policy_global_state_dim(
        policy_input_mode,
        num_friendlies,
        num_enemies,
        num_key_enemies,
        max_local_candidates,
    )
    return SwarmSpaceSpec(
        num_friendlies=num_friendlies,
        num_enemies=num_enemies,
        per_agent_action_dim=action_dim,
        per_agent_obs_dim=per_agent_obs_dim,
        global_state_dim=global_state_dim,
    )
