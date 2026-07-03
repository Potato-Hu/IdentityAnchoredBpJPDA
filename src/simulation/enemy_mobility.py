from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from src.utils.config_loader import get_enemy_mobility_config


class MobilityModel(Protocol):
    def step(self, node: object, dt: float) -> np.ndarray:
        ...


@dataclass
class BaseEnemyMobility:
    bounds: np.ndarray | None = None
    safety_margin: float = 1.0

    def __post_init__(self) -> None:
        if self.bounds is None:
            config = get_enemy_mobility_config()
            configured_bounds = config.get("bounds", [1000.0, 1000.0, 300.0])
            self.bounds = np.asarray(configured_bounds, dtype=float)
        else:
            self.bounds = np.asarray(self.bounds, dtype=float)

        if self.bounds.shape != (3,):
            raise ValueError(f"bounds must be a 3D vector, got shape {self.bounds.shape}.")

    def _integrate(self, node: object, dt: float) -> np.ndarray:
        next_coords = np.asarray(node.coords, dtype=float) + np.asarray(node.velocity_vector, dtype=float) * float(dt)
        return self._clip_and_reflect(node, next_coords)

    def _resolve_active_bounds(self, node: object) -> tuple[np.ndarray, np.ndarray]:
        lower_override = getattr(node, "local_lower_bounds", None)
        upper_override = getattr(node, "local_upper_bounds", None)
        if lower_override is None or upper_override is None:
            lower = np.full(3, self.safety_margin, dtype=float)
            upper = self.bounds - self.safety_margin
            return lower, upper

        lower_bounds = np.asarray(lower_override, dtype=float)
        upper_bounds = np.asarray(upper_override, dtype=float)
        if lower_bounds.shape != (3,) or upper_bounds.shape != (3,):
            raise ValueError("node-local mobility bounds must be 3D vectors.")
        if np.any(upper_bounds <= lower_bounds):
            raise ValueError("node-local mobility upper bounds must exceed lower bounds.")

        lower = lower_bounds + self.safety_margin
        upper = upper_bounds - self.safety_margin
        if np.any(upper <= lower):
            raise ValueError(
                "node-local mobility bounds are too small after applying safety_margin; "
                "increase the local region or reduce safety_margin."
            )
        return lower, upper

    def _clip_and_reflect(self, node: object, coords: np.ndarray) -> np.ndarray:
        lower, upper = self._resolve_active_bounds(node)
        velocity = np.asarray(node.velocity_vector, dtype=float).copy()

        for axis in range(3):
            if coords[axis] < lower[axis] or coords[axis] > upper[axis]:
                velocity[axis] *= -1.0

        coords = np.clip(coords, lower, upper)
        node.set_velocity(velocity)
        return coords


@dataclass
class GaussMarkov3DMobility(BaseEnemyMobility):
    alpha: float = 0.85
    direction_update_interval: float = 0.5
    direction_noise_std: float = 1.0
    pitch_noise_std: float = 0.2

    def step(self, node: object, dt: float) -> np.ndarray:
        node.motion_elapsed += float(dt)
        if node.motion_elapsed >= self.direction_update_interval:
            node.motion_elapsed = 0.0
            alpha2 = 1.0 - self.alpha
            alpha3 = math.sqrt(max(0.0, 1.0 - self.alpha * self.alpha))

            node.direction = (
                self.alpha * node.direction
                + alpha2 * node.direction_mean
                + alpha3 * node.rng.normal(0.0, self.direction_noise_std)
            )
            node.pitch = (
                self.alpha * node.pitch
                + alpha2 * node.pitch_mean
                + alpha3 * node.rng.normal(0.0, self.pitch_noise_std)
            )
            node.pitch = float(np.clip(node.pitch, -math.pi / 2.0, math.pi / 2.0))
            node.set_velocity(node.velocity_from_angles(node.direction, node.pitch, node.speed))

        return self._integrate(node, dt)


@dataclass
class RandomWalk3DMobility(BaseEnemyMobility):
    travel_duration: float = 4.0
    pitch_limit: float = math.pi / 3.0

    def step(self, node: object, dt: float) -> np.ndarray:
        node.motion_elapsed += float(dt)
        if node.motion_elapsed >= self.travel_duration:
            node.motion_elapsed = 0.0
            node.direction = float(node.rng.uniform(0.0, 2.0 * math.pi))
            node.pitch = float(node.rng.uniform(-self.pitch_limit, self.pitch_limit))
            node.set_velocity(node.velocity_from_angles(node.direction, node.pitch, node.speed))

        return self._integrate(node, dt)


def build_enemy_mobility(model_name: str | None = None) -> MobilityModel:
    config = get_enemy_mobility_config()
    selected_model = (model_name or config.get("model", "gauss_markov")).lower()
    bounds = config.get("bounds", [1000.0, 1000.0, 300.0])
    safety_margin = float(config.get("safety_margin", 1.0))

    common_kwargs = {
        "bounds": bounds,
        "safety_margin": safety_margin,
    }

    if selected_model == "gauss_markov":
        gm_cfg = config.get("gauss_markov", {})
        return GaussMarkov3DMobility(
            alpha=float(gm_cfg.get("alpha", 0.85)),
            direction_update_interval=float(gm_cfg.get("direction_update_interval", 0.5)),
            direction_noise_std=float(gm_cfg.get("direction_noise_std", 1.0)),
            pitch_noise_std=float(gm_cfg.get("pitch_noise_std", 0.2)),
            **common_kwargs,
        )

    if selected_model == "random_walk":
        rw_cfg = config.get("random_walk", {})
        return RandomWalk3DMobility(
            travel_duration=float(rw_cfg.get("travel_duration", 4.0)),
            pitch_limit=float(rw_cfg.get("pitch_limit", math.pi / 3.0)),
            **common_kwargs,
        )

    raise ValueError(f"Unsupported enemy mobility model: {selected_model}.")
