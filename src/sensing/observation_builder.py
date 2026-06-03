from __future__ import annotations

from typing import Sequence

import numpy as np


def replace_key_enemy_positions_for_tracking_view(
    enemy_positions: np.ndarray,
    enemy_roles: Sequence[str],
    key_target_positions: np.ndarray,
) -> np.ndarray:
    """Return enemy positions with key enemies replaced by tracking-visible positions."""

    tracking_enemy_positions = np.asarray(enemy_positions, dtype=float).copy()
    key_positions = np.asarray(key_target_positions, dtype=float)
    key_idx = 0
    for enemy_idx, role in enumerate(enemy_roles):
        if role == "key":
            tracking_enemy_positions[enemy_idx] = key_positions[key_idx]
            key_idx += 1
    return tracking_enemy_positions


def build_tracking_enemy_view(
    enemy_positions: np.ndarray,
    enemy_roles: Sequence[str],
    key_target_positions: np.ndarray,
    *,
    include_non_key: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the enemy-position view exposed to the tracking layer."""

    key_positions = np.asarray(key_target_positions, dtype=float)
    if not include_non_key:
        return key_positions.copy(), np.ones(key_positions.shape[0], dtype=float)

    tracking_enemy_positions = replace_key_enemy_positions_for_tracking_view(
        enemy_positions=enemy_positions,
        enemy_roles=enemy_roles,
        key_target_positions=key_positions,
    )
    enemy_is_key = np.asarray([1.0 if role == "key" else 0.0 for role in enemy_roles], dtype=float)
    return tracking_enemy_positions, enemy_is_key


def replace_key_enemy_positions(
    enemy_positions: np.ndarray,
    enemy_roles: Sequence[str],
    key_target_positions: np.ndarray,
) -> np.ndarray:
    """Backward-compatible alias for the tracking-view helper."""

    return replace_key_enemy_positions_for_tracking_view(
        enemy_positions=enemy_positions,
        enemy_roles=enemy_roles,
        key_target_positions=key_target_positions,
    )


def build_policy_enemy_view(
    enemy_positions: np.ndarray,
    enemy_roles: Sequence[str],
    key_target_positions: np.ndarray,
    *,
    include_non_key: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible alias for the tracking-view helper."""

    return build_tracking_enemy_view(
        enemy_positions=enemy_positions,
        enemy_roles=enemy_roles,
        key_target_positions=key_target_positions,
        include_non_key=include_non_key,
    )
