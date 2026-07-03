from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from src.simulation.enemy_mobility import build_enemy_mobility
from src.utils.config_loader import get_default_seed
from src.utils.seed import make_rng


def _as_vector3(value: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{name} must be a 3D vector, got shape {array.shape}.")
    return array


@dataclass
class EnemyNode:
    """Enemy node with selectable lightweight 3D mobility."""

    node_id: int
    role: str
    position: Sequence[float]
    speed: float
    velocity: Sequence[float] | None = None
    seed: int | None = None
    mobility_model: str | None = None
    direction: float | None = None
    pitch: float | None = None
    movement_lower_bounds: Sequence[float] | None = None
    movement_upper_bounds: Sequence[float] | None = None

    rng: np.random.Generator = field(init=False, repr=False)
    coords: np.ndarray = field(init=False, repr=False)
    velocity_vector: np.ndarray = field(init=False, repr=False)
    initial_coords: np.ndarray = field(init=False, repr=False)
    initial_velocity: np.ndarray = field(init=False, repr=False)
    local_lower_bounds: np.ndarray | None = field(init=False, repr=False, default=None)
    local_upper_bounds: np.ndarray | None = field(init=False, repr=False, default=None)
    direction_mean: float = field(init=False)
    pitch_mean: float = field(init=False)
    motion_elapsed: float = field(init=False, default=0.0, repr=False)
    mobility: object = field(init=False, repr=False)
    initial_direction: float = field(init=False, repr=False)
    initial_pitch: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.speed < 0.0:
            raise ValueError("speed must be non-negative.")
        if not self.role:
            raise ValueError("role must be a non-empty string.")

        self.rng = make_rng(self.seed, offset=self.node_id, fallback=get_default_seed())
        self.coords = _as_vector3(self.position, name="position")
        self.local_lower_bounds = (
            _as_vector3(self.movement_lower_bounds, name="movement_lower_bounds")
            if self.movement_lower_bounds is not None
            else None
        )
        self.local_upper_bounds = (
            _as_vector3(self.movement_upper_bounds, name="movement_upper_bounds")
            if self.movement_upper_bounds is not None
            else None
        )
        self.direction = self._init_direction(self.direction)
        self.pitch = self._init_pitch(self.pitch)
        self.direction_mean = self.direction
        self.pitch_mean = self.pitch
        self.mobility = build_enemy_mobility(self.mobility_model)

        if self.velocity is None:
            self.velocity_vector = self.velocity_from_angles(self.direction, self.pitch, self.speed)
        else:
            self.velocity_vector = _as_vector3(self.velocity, name="velocity")
            self.speed = float(np.linalg.norm(self.velocity_vector))
            self.direction, self.pitch = self.angles_from_velocity(self.velocity_vector)
            self.direction_mean = self.direction
            self.pitch_mean = self.pitch

        self.initial_coords = self.coords.copy()
        self.initial_velocity = self.velocity_vector.copy()
        self.initial_direction = self.direction
        self.initial_pitch = self.pitch

    @property
    def x(self) -> float:
        return float(self.coords[0])

    @property
    def y(self) -> float:
        return float(self.coords[1])

    @property
    def z(self) -> float:
        return float(self.coords[2])

    @property
    def vx(self) -> float:
        return float(self.velocity_vector[0])

    @property
    def vy(self) -> float:
        return float(self.velocity_vector[1])

    @property
    def vz(self) -> float:
        return float(self.velocity_vector[2])

    def reset(self) -> None:
        self.coords = self.initial_coords.copy()
        self.velocity_vector = self.initial_velocity.copy()
        self.direction = self.initial_direction
        self.pitch = self.initial_pitch
        self.direction_mean = self.initial_direction
        self.pitch_mean = self.initial_pitch
        self.motion_elapsed = 0.0
        self.speed = float(np.linalg.norm(self.velocity_vector))

    def set_position(self, position: Sequence[float]) -> None:
        self.coords = _as_vector3(position, name="position")

    def set_velocity(self, velocity: Sequence[float]) -> None:
        self.velocity_vector = _as_vector3(velocity, name="velocity")
        self.speed = float(np.linalg.norm(self.velocity_vector))
        self.direction, self.pitch = self.angles_from_velocity(self.velocity_vector)

    def set_mobility_model(self, mobility_model: str) -> None:
        self.mobility_model = mobility_model
        self.mobility = build_enemy_mobility(mobility_model)
        self.motion_elapsed = 0.0

    def update(
        self,
        dt: float,
        velocity: Sequence[float] | None = None,
        speed: float | None = None,
    ) -> np.ndarray:
        """Advance the node state by one step."""

        if dt < 0.0:
            raise ValueError("dt must be non-negative.")

        if velocity is not None:
            self.set_velocity(velocity)
            self.coords = self.coords + self.velocity_vector * float(dt)
            return self.coords.copy()

        if speed is not None:
            if speed < 0.0:
                raise ValueError("speed must be non-negative.")
            self.speed = float(speed)
            self.velocity_vector = self.velocity_from_angles(self.direction, self.pitch, self.speed)

        self.coords = self.mobility.step(self, dt)
        return self.coords.copy()

    def velocity_from_angles(self, direction: float, pitch: float, speed: float) -> np.ndarray:
        return np.array(
            [
                speed * math.cos(direction) * math.cos(pitch),
                speed * math.sin(direction) * math.cos(pitch),
                speed * math.sin(pitch),
            ],
            dtype=float,
        )

    def angles_from_velocity(self, velocity: Sequence[float]) -> tuple[float, float]:
        vector = _as_vector3(velocity, name="velocity")
        speed = float(np.linalg.norm(vector))
        if speed == 0.0:
            return 0.0, 0.0

        direction = math.atan2(vector[1], vector[0])
        pitch = math.asin(np.clip(vector[2] / speed, -1.0, 1.0))
        return direction, pitch

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "role": self.role,
            "coords": self.coords.copy(),
            "speed": self.speed,
            "velocity": self.velocity_vector.copy(),
            "direction": self.direction,
            "pitch": self.pitch,
            "mobility_model": self.mobility_model,
            "movement_lower_bounds": None if self.local_lower_bounds is None else self.local_lower_bounds.copy(),
            "movement_upper_bounds": None if self.local_upper_bounds is None else self.local_upper_bounds.copy(),
            "seed": self.seed if self.seed is not None else get_default_seed(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EnemyNode":
        return cls(
            node_id=int(data["node_id"]),
            role=str(data["role"]),
            position=data["coords"],
            speed=float(data.get("speed", 0.0)),
            velocity=data.get("velocity"),
            seed=data.get("seed"),
            mobility_model=data.get("mobility_model"),
            direction=data.get("direction"),
            pitch=data.get("pitch"),
            movement_lower_bounds=data.get("movement_lower_bounds"),
            movement_upper_bounds=data.get("movement_upper_bounds"),
        )

    def _init_direction(self, direction: float | None) -> float:
        if direction is not None:
            return float(direction)
        return float(self.rng.uniform(0.0, 2.0 * math.pi))

    def _init_pitch(self, pitch: float | None) -> float:
        if pitch is not None:
            return float(pitch)
        return float(self.rng.uniform(-0.05, 0.05))
