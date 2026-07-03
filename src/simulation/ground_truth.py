from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from src.entities.enemy_node import EnemyNode
from src.entities.friendly_uav import FriendlyUAV
from src.utils.config_loader import (
    get_default_seed,
    get_enemy_mobility_config,
    get_enemy_vmax,
    get_friendly_vmax,
    get_initialization_config,
    should_print_initial_positions,
)

PositionDict = Mapping[str, Sequence[Sequence[float]]]
NearbyDistanceLimits = Mapping[str, Sequence[float]]


@dataclass
class ScenarioGroundTruth:
    """Container for scenario initialization, simulation, and plotting."""

    bounds: np.ndarray
    enemy_nodes: list[EnemyNode]
    friendly_uavs: list[FriendlyUAV]
    dt: float = 1.0

    @property
    def key_enemy_nodes(self) -> list[EnemyNode]:
        return [node for node in self.enemy_nodes if node.role == "key"]

    @property
    def non_key_enemy_nodes(self) -> list[EnemyNode]:
        return [node for node in self.enemy_nodes if node.role != "key"]

    def reset(self) -> None:
        for node in self.enemy_nodes:
            node.reset()
        for uav in self.friendly_uavs:
            uav.reset()


def _as_bounds(bounds: Sequence[float] | None) -> np.ndarray:
    mobility_config = get_enemy_mobility_config()
    scene_bounds = np.asarray(bounds if bounds is not None else mobility_config.get("bounds", [1000.0, 1000.0, 300.0]), dtype=float)
    if scene_bounds.shape != (3,):
        raise ValueError(f"bounds must be a 3D vector, got shape {scene_bounds.shape}.")
    return scene_bounds


def _validate_positions(name: str, positions: Sequence[Sequence[float]], expected_count: int, bounds: np.ndarray) -> np.ndarray:
    coords = np.asarray(positions, dtype=float)
    if coords.shape != (expected_count, 3):
        raise ValueError(f"{name} must contain {expected_count} 3D positions, got shape {coords.shape}.")
    if np.any(coords < 0.0) or np.any(coords > bounds):
        raise ValueError(f"{name} positions must stay within bounds {bounds.tolist()}.")
    return coords


def _random_positions(count: int, bounds: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if count == 0:
        return np.zeros((0, 3), dtype=float)
    return rng.uniform(low=np.zeros(3, dtype=float), high=bounds, size=(count, 3))


def _read_grid_splits(raw_value: object, *, default: Sequence[int] = (4, 4, 4)) -> np.ndarray:
    grid_splits = np.asarray(default if raw_value is None else raw_value, dtype=int)
    if grid_splits.ndim == 0:
        grid_splits = np.full(3, int(grid_splits), dtype=int)
    if grid_splits.shape != (3,):
        raise ValueError(f"grid_splits must be a scalar or 3D vector, got shape {grid_splits.shape}.")
    if np.any(grid_splits <= 0):
        raise ValueError("grid_splits must contain strictly positive integers.")
    return grid_splits.astype(int)


def _movement_partition_bounds_for_positions(
    positions: np.ndarray,
    scene_bounds: np.ndarray,
    *,
    grid_splits: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    if positions.shape[0] == 0:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=float)

    partition_counts = _read_grid_splits(grid_splits)
    cell_sizes = scene_bounds / partition_counts.astype(float)
    partition_indices = np.floor(positions / cell_sizes.reshape(1, 3)).astype(int)
    partition_indices = np.clip(partition_indices, 0, partition_counts.reshape(1, 3) - 1)
    lower_bounds = partition_indices.astype(float) * cell_sizes.reshape(1, 3)
    upper_bounds = (partition_indices.astype(float) + 1.0) * cell_sizes.reshape(1, 3)
    upper_bounds = np.minimum(upper_bounds, scene_bounds.reshape(1, 3))
    return lower_bounds.astype(float), upper_bounds.astype(float)


def _nearby_distance_limits(bounds: np.ndarray, limits: Sequence[float] | None = None) -> tuple[float, float]:
    if limits is None:
        min_distance = float(bounds[0] / 40.0)
        max_distance = float(bounds[0] / 10.0)
    else:
        distance_limits = np.asarray(limits, dtype=float)
        if distance_limits.shape != (2,):
            raise ValueError(f"nearby distance limits must be [min, max], got shape {distance_limits.shape}.")
        min_distance = float(distance_limits[0])
        max_distance = float(distance_limits[1])
    if min_distance < 0.0:
        raise ValueError("nearby minimum distance must be non-negative.")
    if max_distance < min_distance:
        raise ValueError("nearby maximum distance must be greater than or equal to minimum distance.")
    if max_distance <= 0.0:
        raise ValueError("x boundary must be positive to sample nearby initial positions.")
    return min_distance, max_distance


def _sample_near_anchor(
    anchor: np.ndarray,
    bounds: np.ndarray,
    rng: np.random.Generator,
    min_distance: float,
    max_distance: float,
) -> np.ndarray:
    for _ in range(2048):
        direction = rng.normal(size=3)
        norm = float(np.linalg.norm(direction))
        if norm == 0.0:
            continue
        distance = rng.uniform(min_distance, max_distance)
        candidate = anchor + direction / norm * distance
        if np.all(candidate >= 0.0) and np.all(candidate <= bounds):
            return candidate.astype(float)

    candidates = rng.uniform(low=np.zeros(3, dtype=float), high=bounds, size=(8192, 3))
    distances = np.linalg.norm(candidates - anchor, axis=1)
    valid_indices = np.flatnonzero((distances >= min_distance) & (distances <= max_distance))
    if valid_indices.size:
        return candidates[int(rng.choice(valid_indices))].astype(float)

    raise ValueError(
        "Unable to sample a nearby initial position within "
        f"[{min_distance:.3f}, {max_distance:.3f}] and bounds {bounds.tolist()}."
    )


def _anchor_indices_for_nearby_positions(
    count: int,
    anchor_count: int,
    rng: np.random.Generator,
    *,
    entity_name: str,
) -> np.ndarray:
    if count == 0:
        return np.zeros(0, dtype=int)
    if anchor_count == 0:
        raise ValueError(f"Cannot initialize {entity_name} around enemy UAVs because key_num is 0.")
    if count < anchor_count:
        raise ValueError(
            f"{entity_name} count ({count}) must be at least key_num ({anchor_count}) "
            "so every key enemy UAV has one nearby unit."
        )

    required = np.arange(anchor_count, dtype=int)
    extra_count = count - anchor_count
    if extra_count == 0:
        return required
    extras = rng.integers(low=0, high=anchor_count, size=extra_count, dtype=int)
    return np.concatenate([required, extras])


def _sample_near_anchors(
    anchors: np.ndarray,
    count: int,
    bounds: np.ndarray,
    rng: np.random.Generator,
    *,
    entity_name: str,
    distance_limits: Sequence[float] | None = None,
) -> np.ndarray:
    positions, _ = _sample_near_anchors_with_indices(
        anchors,
        count,
        bounds,
        rng,
        entity_name=entity_name,
        distance_limits=distance_limits,
    )
    return positions


def _sample_random_positions_with_min_separation(
    count: int,
    bounds: np.ndarray,
    rng: np.random.Generator,
    *,
    min_separation: float,
) -> np.ndarray:
    if count == 0:
        return np.zeros((0, 3), dtype=float)
    if min_separation <= 0.0:
        return _random_positions(count, bounds, rng)

    positions: list[np.ndarray] = []
    for _ in range(count):
        for _ in range(4096):
            candidate = rng.uniform(low=np.zeros(3, dtype=float), high=bounds, size=3)
            if not positions:
                positions.append(candidate.astype(float))
                break
            distances = np.linalg.norm(np.asarray(positions, dtype=float) - candidate.reshape(1, 3), axis=1)
            if np.all(distances >= float(min_separation)):
                positions.append(candidate.astype(float))
                break
        else:
            raise ValueError(
                "Unable to sample friendly UAV initial positions with minimum separation "
                f"{min_separation:.3f} inside bounds {bounds.tolist()}."
            )
    return np.asarray(positions, dtype=float)


def _sample_overlapping_friendlies_near_keys(
    key_positions: np.ndarray,
    count: int,
    bounds: np.ndarray,
    rng: np.random.Generator,
    *,
    distance_limits: Sequence[float] | None = None,
    min_separation: float = 0.0,
) -> np.ndarray:
    """Sample friendly UAVs near the key-target cluster so their local views overlap."""

    if count == 0:
        return np.zeros((0, 3), dtype=float)
    if key_positions.shape[0] == 0:
        return _sample_random_positions_with_min_separation(
            count,
            bounds,
            rng,
            min_separation=min_separation,
        )

    min_distance, max_distance = _nearby_distance_limits(bounds, distance_limits)
    cluster_center = np.mean(np.asarray(key_positions, dtype=float), axis=0)
    shared_anchor = np.clip(cluster_center, np.zeros(3, dtype=float), bounds)
    minimum_separation = max(float(min_separation), 0.0)

    friendly_positions: list[np.ndarray] = []
    for _ in range(count):
        for _ in range(4096):
            candidate = _sample_near_anchor(
                anchor=shared_anchor,
                bounds=bounds,
                rng=rng,
                min_distance=min_distance,
                max_distance=max_distance,
            )
            if minimum_separation <= 0.0 or not friendly_positions:
                friendly_positions.append(candidate.astype(float))
                break
            distances = np.linalg.norm(
                np.asarray(friendly_positions, dtype=float) - candidate.reshape(1, 3),
                axis=1,
            )
            if np.all(distances >= minimum_separation):
                friendly_positions.append(candidate.astype(float))
                break
        else:
            raise ValueError(
                "Unable to sample friendly UAV initial positions near the key-target cluster "
                f"with minimum separation {minimum_separation:.3f} and distance limits "
                f"[{min_distance:.3f}, {max_distance:.3f}]."
            )
    return np.asarray(friendly_positions, dtype=float)


def _sample_near_anchors_with_indices(
    anchors: np.ndarray,
    count: int,
    bounds: np.ndarray,
    rng: np.random.Generator,
    *,
    entity_name: str,
    distance_limits: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if count == 0:
        return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=int)

    min_distance, max_distance = _nearby_distance_limits(bounds, distance_limits)
    anchor_indices = _anchor_indices_for_nearby_positions(
        count,
        len(anchors),
        rng,
        entity_name=entity_name,
    )
    nearby_positions = [
        _sample_near_anchor(
            anchor=anchors[int(anchor_idx)],
            bounds=bounds,
            rng=rng,
            min_distance=min_distance,
            max_distance=max_distance,
        )
        for anchor_idx in anchor_indices
    ]
    return np.asarray(nearby_positions, dtype=float), anchor_indices


def _nearest_anchor_indices(positions: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    if positions.shape[0] == 0:
        return np.zeros(0, dtype=int)
    if anchors.shape[0] == 0:
        raise ValueError("Cannot assign nonkey_enemy anchors because key_num is 0.")
    distances = np.linalg.norm(positions[:, None, :] - anchors[None, :, :], axis=-1)
    return np.argmin(distances, axis=1).astype(int)


def _format_coord(coord: Sequence[float]) -> str:
    vector = np.asarray(coord, dtype=float)
    return f"({vector[0]:.2f}, {vector[1]:.2f}, {vector[2]:.2f})"


def format_scene_positions(
    scene: ScenarioGroundTruth,
    enemy_positions: np.ndarray | None = None,
    friendly_positions: np.ndarray | None = None,
) -> list[str]:
    enemy_coords = enemy_positions if enemy_positions is not None else np.array([node.coords for node in scene.enemy_nodes], dtype=float)
    friendly_coords = friendly_positions if friendly_positions is not None else np.array([uav.coords for uav in scene.friendly_uavs], dtype=float)

    lines: list[str] = []
    for idx, coord in enumerate(enemy_coords):
        lines.append(f"E{idx + 1}:{_format_coord(coord)}")
    for idx, coord in enumerate(friendly_coords):
        lines.append(f"F{idx + 1}:{_format_coord(coord)}")
    return lines


def print_scene_positions(
    scene: ScenarioGroundTruth,
    title: str,
    enemy_positions: np.ndarray | None = None,
    friendly_positions: np.ndarray | None = None,
) -> None:
    print(title)
    for line in format_scene_positions(scene, enemy_positions=enemy_positions, friendly_positions=friendly_positions):
        print(line)


def assign_initial_positions(
    key_num: int,
    nonkey_num: int,
    mydrone_num: int,
    bounds: Sequence[float] | None = None,
    initial_positions: PositionDict | None = None,
    nearby_distance_limits: NearbyDistanceLimits | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """
    Resolve initial positions for the scenario.

    `initial_positions` defaults to an empty dict and supports manual coordinates.
    Keys are:
    - `key_enemy`
    - `nonkey_enemy`
    - `friend_drone`

    Each value should be a list of 3D coordinates. Example:
    {
        "key_enemy": [[260.0, 280.0, 150.0], [420.0, 520.0, 150.0]],
        "nonkey_enemy": [[340.0, 400.0, 150.0], [500.0, 360.0, 150.0]],
        "friend_drone": [[250.0, 500.0, 0.0], [500.0, 500.0, 0.0], [750.0, 500.0, 0.0]],
    }
    """

    if key_num < 0 or nonkey_num < 0 or mydrone_num < 0:
        raise ValueError("key_num, nonkey_num, and mydrone_num must be non-negative.")

    scene_bounds = _as_bounds(bounds)
    manual_positions = dict(initial_positions or {})
    configured_distance_limits = dict(get_initialization_config())
    configured_distance_limits.update(dict(nearby_distance_limits or {}))
    position_rng = rng if rng is not None else np.random.default_rng()
    nonkey_initialization_mode = str(
        configured_distance_limits.get("nonkey_initialization_mode", "near_key")
    ).lower()

    key_positions = (
        _validate_positions("key_enemy", manual_positions["key_enemy"], key_num, scene_bounds)
        if manual_positions.get("key_enemy") is not None
        else _random_positions(key_num, scene_bounds, position_rng)
    )
    if manual_positions.get("nonkey_enemy") is not None:
        nonkey_positions = _validate_positions("nonkey_enemy", manual_positions["nonkey_enemy"], nonkey_num, scene_bounds)
        nonkey_anchor_indices = _nearest_anchor_indices(nonkey_positions, key_positions)
    elif nonkey_initialization_mode == "uniform_random":
        nonkey_positions = _random_positions(nonkey_num, scene_bounds, position_rng)
        nonkey_anchor_indices = _nearest_anchor_indices(nonkey_positions, key_positions)
    else:
        nonkey_positions, nonkey_anchor_indices = _sample_near_anchors_with_indices(
            key_positions,
            nonkey_num,
            scene_bounds,
            position_rng,
            entity_name="nonkey_enemy",
            distance_limits=configured_distance_limits.get("nonkey_enemy"),
        )
    friendly_positions = (
        _validate_positions("friend_drone", manual_positions["friend_drone"], mydrone_num, scene_bounds)
        if manual_positions.get("friend_drone") is not None
        else _sample_overlapping_friendlies_near_keys(
            key_positions,
            mydrone_num,
            scene_bounds,
            position_rng,
            distance_limits=configured_distance_limits.get("friend_drone"),
            min_separation=float(configured_distance_limits.get("friend_drone_min_separation_m", 0.0)),
        )
    )

    enemy_positions = np.vstack([key_positions, nonkey_positions])
    key_indices = np.arange(key_num, dtype=int)

    return {
        "key_enemy": np.asarray(key_positions, dtype=float),
        "nonkey_enemy": np.asarray(nonkey_positions, dtype=float),
        "nonkey_anchor_indices": np.asarray(nonkey_anchor_indices, dtype=int),
        "enemy": enemy_positions,
        "key_indices": key_indices,
        "friend_drone": np.asarray(friendly_positions, dtype=float),
    }


def initialize_scene(
    key_num: int,
    nonkey_num: int,
    mydrone_num: int,
    bounds: Sequence[float] | None = None,
    dt: float = 1.0,
    seed: int | None = None,
    enemy_vmax: float | None = None,
    friendly_vmax: float | None = None,
    initial_positions: PositionDict | None = None,
    nearby_distance_limits: NearbyDistanceLimits | None = None,
) -> ScenarioGroundTruth:
    scene_bounds = _as_bounds(bounds)
    base_seed = get_default_seed() if seed is None else int(seed)
    resolved_enemy_vmax = get_enemy_vmax() if enemy_vmax is None else float(enemy_vmax)
    resolved_friendly_vmax = get_friendly_vmax() if friendly_vmax is None else float(friendly_vmax)
    mobility_config = get_enemy_mobility_config()
    companion_config = mobility_config.get("non_key_companion", {})
    if not isinstance(companion_config, dict):
        companion_config = {}
    nonkey_local_partition_config = mobility_config.get("non_key_local_partition", {})
    if not isinstance(nonkey_local_partition_config, dict):
        nonkey_local_partition_config = {}
    use_nonkey_companion = bool(companion_config.get("enabled", False))
    companion_direction_jitter_std = float(companion_config.get("direction_jitter_std", 0.15))
    companion_pitch_jitter_std = float(companion_config.get("pitch_jitter_std", 0.03))
    companion_rng = np.random.default_rng(base_seed + 10_000)
    positions = assign_initial_positions(
        key_num=key_num,
        nonkey_num=nonkey_num,
        mydrone_num=mydrone_num,
        bounds=scene_bounds,
        initial_positions=initial_positions,
        nearby_distance_limits=nearby_distance_limits,
        rng=np.random.default_rng(base_seed),
    )

    enemy_nodes: list[EnemyNode] = []
    key_indices = set(np.asarray(positions["key_indices"], dtype=int).tolist())
    nonkey_anchor_indices = np.asarray(positions["nonkey_anchor_indices"], dtype=int)
    nonkey_positions = np.asarray(positions["nonkey_enemy"], dtype=float)
    nonkey_local_partition_enabled = bool(nonkey_local_partition_config.get("enabled", False))
    nonkey_partition_grid_splits = _read_grid_splits(
        nonkey_local_partition_config.get("grid_splits", [4, 4, 4])
    )
    nonkey_lower_bounds, nonkey_upper_bounds = _movement_partition_bounds_for_positions(
        nonkey_positions,
        scene_bounds,
        grid_splits=nonkey_partition_grid_splits,
    )
    key_directions = {
        key_idx: 0.0 if key_idx % 2 == 0 else np.pi
        for key_idx in key_indices
    }
    key_pitches = {key_idx: 0.04 for key_idx in key_indices}
    for node_id, coords in enumerate(positions["enemy"]):
        role = "key" if node_id in key_indices else "non_key"
        if role == "key":
            direction = key_directions[node_id]
            pitch = key_pitches[node_id]
        elif use_nonkey_companion and key_num > 0:
            anchor_idx = int(nonkey_anchor_indices[node_id - key_num])
            direction = key_directions[anchor_idx] + float(companion_rng.normal(0.0, companion_direction_jitter_std))
            pitch = key_pitches[anchor_idx] + float(companion_rng.normal(0.0, companion_pitch_jitter_std))
        else:
            direction = 0.0 if node_id % 2 == 0 else np.pi
            pitch = -0.04
        local_lower_bounds = None
        local_upper_bounds = None
        if role != "key" and nonkey_local_partition_enabled:
            nonkey_idx = node_id - key_num
            local_lower_bounds = nonkey_lower_bounds[nonkey_idx]
            local_upper_bounds = nonkey_upper_bounds[nonkey_idx]
        enemy_nodes.append(
            EnemyNode(
                node_id=node_id,
                role=role,
                position=coords,
                speed=resolved_enemy_vmax,
                seed=base_seed + node_id,
                direction=direction,
                pitch=pitch,
                movement_lower_bounds=local_lower_bounds,
                movement_upper_bounds=local_upper_bounds,
            )
        )

    friendly_uavs: list[FriendlyUAV] = []
    for idx, coords in enumerate(positions["friend_drone"]):
        friendly_uavs.append(
            FriendlyUAV(
                node_id=idx,
                position=coords,
                vmax=resolved_friendly_vmax,
                seed=base_seed + 100 + idx,
            )
        )

    scene = ScenarioGroundTruth(
        bounds=scene_bounds,
        enemy_nodes=enemy_nodes,
        friendly_uavs=friendly_uavs,
        dt=float(dt),
    )
    if should_print_initial_positions():
        print_scene_positions(scene, title="Initialization positions:")
    return scene


def create_demo_ground_truth(
    bounds: Sequence[float] | None = None,
    dt: float = 1.0,
    seed: int | None = None,
    enemy_speed: float | None = None,
    friendly_vmax: float | None = None,
) -> ScenarioGroundTruth:
    """Backward-compatible wrapper around `initialize_scene`."""

    scene_bounds = _as_bounds(bounds)
    enemy_y = float(scene_bounds[1] * 0.68)
    enemy_z = float(scene_bounds[2] * 0.58)
    friendly_y = float(scene_bounds[1] * 0.32)
    friendly_z = float(scene_bounds[2] * 0.42)

    return initialize_scene(
        key_num=3,
        nonkey_num=3,
        mydrone_num=3,
        bounds=scene_bounds,
        dt=dt,
        seed=seed,
        enemy_vmax=enemy_speed,
        friendly_vmax=friendly_vmax,
        initial_positions={
            "key_enemy": [
                [scene_bounds[0] * 0.18, enemy_y, enemy_z],
                [scene_bounds[0] * 0.436, enemy_y, enemy_z],
                [scene_bounds[0] * 0.692, enemy_y, enemy_z],
            ],
            "nonkey_enemy": [
                [scene_bounds[0] * 0.308, enemy_y, enemy_z],
                [scene_bounds[0] * 0.564, enemy_y, enemy_z],
                [scene_bounds[0] * 0.82, enemy_y, enemy_z],
            ],
            "friend_drone": [
                [scene_bounds[0] * 0.28, friendly_y, friendly_z],
                [scene_bounds[0] * 0.5, friendly_y, friendly_z],
                [scene_bounds[0] * 0.72, friendly_y, friendly_z],
            ],
        },
    )
