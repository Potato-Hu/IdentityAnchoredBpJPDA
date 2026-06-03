"""World state container for the swarm RL project.

This file owns the real-time ground-truth state of all enemy nodes and friendly UAVs,
and provides the single step() entry used by rollout generation and future RL env logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.entities.enemy_node import EnemyNode
from src.entities.friendly_uav import FriendlyUAV
from src.interference.channel import InterferenceSnapshot, compute_interference_snapshot
from src.simulation.ground_truth import ScenarioGroundTruth


def clip_to_bounds(position: Sequence[float], bounds: Sequence[float], margin: float = 1.0) -> np.ndarray:
    """Clip a 3D position into the valid map region."""

    bounds_array = np.asarray(bounds, dtype=float)
    lower = np.full(3, margin, dtype=float)
    upper = bounds_array - margin
    return np.clip(np.asarray(position, dtype=float), lower, upper)


@dataclass
class StepResult:
    """Output snapshot returned by a single world step."""

    enemy_positions: np.ndarray
    friendly_positions: np.ndarray
    key_enemy_interference_watts: np.ndarray
    key_enemy_interference_dbm: np.ndarray
    friendly_interference_watts: np.ndarray
    friendly_interference_dbm: np.ndarray
    friendly_actions: np.ndarray
    time_step: int
    sim_time: float


class SwarmWorld:
    """Single source of truth for real-time swarm state evolution."""

    def __init__(
        self,
        enemy_nodes: list[EnemyNode],
        friendly_uavs: list[FriendlyUAV],
        bounds: Sequence[float],
        dt: float = 1.0,
    ) -> None:
        self.enemy_nodes = enemy_nodes
        self.friendly_uavs = friendly_uavs
        self.bounds = np.asarray(bounds, dtype=float).copy()
        self.dt = float(dt)
        self.time_step = 0
        self.sim_time = 0.0
        self.interference = self._compute_interference()

    @classmethod
    def from_scene(cls, scene: ScenarioGroundTruth) -> "SwarmWorld":
        """Clone a scene container into an isolated mutable world state."""

        enemy_nodes = [EnemyNode.from_dict(node.as_dict()) for node in scene.enemy_nodes]
        friendly_uavs = [FriendlyUAV.from_dict(uav.as_dict()) for uav in scene.friendly_uavs]
        return cls(
            enemy_nodes=enemy_nodes,
            friendly_uavs=friendly_uavs,
            bounds=scene.bounds,
            dt=scene.dt,
        )

    def reset(self) -> None:
        for node in self.enemy_nodes:
            node.reset()
        for uav in self.friendly_uavs:
            uav.reset()
        self.time_step = 0
        self.sim_time = 0.0
        self.interference = self._compute_interference()

    def sample_friendly_actions(self) -> np.ndarray:
        if not self.friendly_uavs:
            return np.zeros(0, dtype=int)
        return np.asarray(
            [int(uav.rng.integers(0, uav.action_dim)) for uav in self.friendly_uavs],
            dtype=int,
        )

    def get_enemy_positions(self) -> np.ndarray:
        if not self.enemy_nodes:
            return np.zeros((0, 3), dtype=float)
        return np.asarray([node.coords.copy() for node in self.enemy_nodes], dtype=float)

    def get_enemy_velocities(self) -> np.ndarray:
        if not self.enemy_nodes:
            return np.zeros((0, 3), dtype=float)
        return np.asarray([node.velocity_vector.copy() for node in self.enemy_nodes], dtype=float)

    def get_enemy_states(self) -> np.ndarray:
        if not self.enemy_nodes:
            return np.zeros((0, 6), dtype=float)
        return np.concatenate([self.get_enemy_positions(), self.get_enemy_velocities()], axis=1)

    def get_friendly_positions(self) -> np.ndarray:
        if not self.friendly_uavs:
            return np.zeros((0, 3), dtype=float)
        return np.asarray([uav.coords.copy() for uav in self.friendly_uavs], dtype=float)

    def get_friendly_velocities(self) -> np.ndarray:
        if not self.friendly_uavs:
            return np.zeros((0, 3), dtype=float)
        return np.asarray([uav.velocity_vector.copy() for uav in self.friendly_uavs], dtype=float)

    def get_friendly_states(self) -> np.ndarray:
        if not self.friendly_uavs:
            return np.zeros((0, 6), dtype=float)
        return np.concatenate([self.get_friendly_positions(), self.get_friendly_velocities()], axis=1)

    def get_key_enemy_nodes(self) -> list[EnemyNode]:
        return [node for node in self.enemy_nodes if node.role == "key"]

    def get_key_enemy_positions(self) -> np.ndarray:
        key_nodes = self.get_key_enemy_nodes()
        if not key_nodes:
            return np.zeros((0, 3), dtype=float)
        return np.asarray([node.coords.copy() for node in key_nodes], dtype=float)

    def get_key_enemy_velocities(self) -> np.ndarray:
        key_nodes = self.get_key_enemy_nodes()
        if not key_nodes:
            return np.zeros((0, 3), dtype=float)
        return np.asarray([node.velocity_vector.copy() for node in key_nodes], dtype=float)

    def get_key_enemy_states(self) -> np.ndarray:
        if not self.get_key_enemy_nodes():
            return np.zeros((0, 6), dtype=float)
        return np.concatenate([self.get_key_enemy_positions(), self.get_key_enemy_velocities()], axis=1)

    def snapshot(self) -> dict[str, np.ndarray | float | int]:
        """Return the current world state as plain arrays."""

        return {
            "enemy_positions": self.get_enemy_positions(),
            "friendly_positions": self.get_friendly_positions(),
            "key_enemy_interference_watts": self.interference.key_enemy_received_watts.copy(),
            "key_enemy_interference_dbm": self.interference.key_enemy_received_dbm.copy(),
            "friendly_interference_watts": self.interference.friendly_received_watts.copy(),
            "friendly_interference_dbm": self.interference.friendly_received_dbm.copy(),
            "time_step": self.time_step,
            "sim_time": self.sim_time,
        }

    def step(self, friendly_actions: Sequence[int] | None = None, dt: float | None = None) -> StepResult:
        """Advance enemy mobility and friendly actions by one simulation step."""

        step_dt = self.dt if dt is None else float(dt)

        if friendly_actions is None:
            resolved_actions = self.sample_friendly_actions()
        else:
            resolved_actions = np.asarray(friendly_actions, dtype=int)
            if resolved_actions.shape != (len(self.friendly_uavs),):
                raise ValueError(
                    f"friendly_actions must have shape ({len(self.friendly_uavs)},), got {resolved_actions.shape}."
                )

        enemy_positions = np.zeros((len(self.enemy_nodes), 3), dtype=float)
        for idx, node in enumerate(self.enemy_nodes):
            enemy_positions[idx] = node.update(step_dt)

        friendly_positions = np.zeros((len(self.friendly_uavs), 3), dtype=float)
        for idx, (uav, action) in enumerate(zip(self.friendly_uavs, resolved_actions, strict=False)):
            next_position = uav.update(step_dt, action=int(action))
            uav.set_position(clip_to_bounds(next_position, self.bounds, margin=1.0))
            friendly_positions[idx] = uav.coords.copy()

        self.interference = self._compute_interference()
        self.time_step += 1
        self.sim_time += step_dt
        return StepResult(
            enemy_positions=enemy_positions,
            friendly_positions=friendly_positions,
            key_enemy_interference_watts=self.interference.key_enemy_received_watts.copy(),
            key_enemy_interference_dbm=self.interference.key_enemy_received_dbm.copy(),
            friendly_interference_watts=self.interference.friendly_received_watts.copy(),
            friendly_interference_dbm=self.interference.friendly_received_dbm.copy(),
            friendly_actions=resolved_actions.copy(),
            time_step=self.time_step,
            sim_time=self.sim_time,
        )

    def _compute_interference(self) -> InterferenceSnapshot:
        return compute_interference_snapshot(
            friendly_positions=self.get_friendly_positions(),
            key_enemy_positions=self.get_key_enemy_positions(),
        )
