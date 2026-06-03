from __future__ import annotations

from dataclasses import dataclass

import numpy as np


STATE_DIM = 6


@dataclass(frozen=True)
class LocalSensingConfig:
    detection_radius_m: float
    max_candidates: int
    measurement_position_std_m: np.ndarray
    measurement_velocity_std_mps: np.ndarray
    detection_probability: float
    clutter_rate_per_second: float
    dt_seconds: float
    relative_velocity_limit_mps: float

    @property
    def covariance_diagonal(self) -> np.ndarray:
        return np.concatenate(
            [
                np.square(np.asarray(self.measurement_position_std_m, dtype=float).reshape(3)),
                np.square(np.asarray(self.measurement_velocity_std_mps, dtype=float).reshape(3)),
            ]
        )


@dataclass(frozen=True)
class LocalTargetObservation:
    relative_states: np.ndarray
    mask: np.ndarray
    covariance_diagonal: np.ndarray


class LocalTargetSensor:
    """Omnidirectional local sensor returning anonymous 6D relative-state candidates."""

    def __init__(self, config: LocalSensingConfig, seed: int | None = None) -> None:
        self.config = config
        if self.config.detection_radius_m < 0.0:
            raise ValueError(f"local_sensing.detection_radius_m must be >= 0, got {self.config.detection_radius_m}.")
        if self.config.max_candidates < 0:
            raise ValueError(f"local_sensing.max_candidates must be >= 0, got {self.config.max_candidates}.")

        position_std = np.asarray(self.config.measurement_position_std_m, dtype=float).reshape(-1)
        velocity_std = np.asarray(self.config.measurement_velocity_std_mps, dtype=float).reshape(-1)
        if position_std.shape != (3,):
            raise ValueError(
                "local_sensing.measurement_position_std_m must resolve to a 3D vector, "
                f"got shape {position_std.shape}."
            )
        if velocity_std.shape != (3,):
            raise ValueError(
                "local_sensing.measurement_velocity_std_mps must resolve to a 3D vector, "
                f"got shape {velocity_std.shape}."
            )
        if np.any(position_std < 0.0):
            raise ValueError(
                "local_sensing.measurement_position_std_m must be >= 0 on every axis, "
                f"got {position_std.tolist()}."
            )
        if np.any(velocity_std < 0.0):
            raise ValueError(
                "local_sensing.measurement_velocity_std_mps must be >= 0 on every axis, "
                f"got {velocity_std.tolist()}."
            )
        if not 0.0 <= self.config.detection_probability <= 1.0:
            raise ValueError(
                "local_sensing.detection_probability must be in [0, 1], "
                f"got {self.config.detection_probability}."
            )
        if self.config.clutter_rate_per_second < 0.0:
            raise ValueError(
                "local_sensing.clutter_rate_per_second must be >= 0, "
                f"got {self.config.clutter_rate_per_second}."
            )
        if self.config.dt_seconds <= 0.0:
            raise ValueError(f"local_sensing.dt_seconds must be > 0, got {self.config.dt_seconds}.")
        if self.config.relative_velocity_limit_mps < 0.0:
            raise ValueError(
                "local_sensing.relative_velocity_limit_mps must be >= 0, "
                f"got {self.config.relative_velocity_limit_mps}."
            )

        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._seed = seed
        self._rng = np.random.default_rng(self._seed)

    def observe(self, friendly_state: np.ndarray, target_states: np.ndarray) -> LocalTargetObservation:
        max_candidates = int(self.config.max_candidates)
        relative_states = np.zeros((max_candidates, STATE_DIM), dtype=np.float32)
        mask = np.zeros(max_candidates, dtype=np.float32)
        covariance_diagonal = self.config.covariance_diagonal.astype(np.float32, copy=True)
        if max_candidates == 0:
            return LocalTargetObservation(
                relative_states=relative_states,
                mask=mask,
                covariance_diagonal=covariance_diagonal,
            )

        friendly = np.asarray(friendly_state, dtype=float).reshape(STATE_DIM)
        targets = np.asarray(target_states, dtype=float)
        if targets.size == 0:
            targets = np.zeros((0, STATE_DIM), dtype=float)
        elif targets.ndim != 2 or targets.shape[1] != STATE_DIM:
            raise ValueError(f"target_states must have shape (N, {STATE_DIM}), got {targets.shape}.")

        true_relative_states = targets - friendly
        distances = np.linalg.norm(true_relative_states[:, :3], axis=1)
        visible_indices = np.flatnonzero(distances <= self.config.detection_radius_m)
        detected_relative_states = np.zeros((0, STATE_DIM), dtype=float)
        if visible_indices.size:
            detection_draws = self._rng.random(visible_indices.size)
            detected_mask = detection_draws < self.config.detection_probability
            detected_indices = visible_indices[detected_mask]
            if detected_indices.size:
                detected_relative_states = true_relative_states[detected_indices].copy()
                noise_std = np.concatenate(
                    [
                        np.asarray(self.config.measurement_position_std_m, dtype=float).reshape(3),
                        np.asarray(self.config.measurement_velocity_std_mps, dtype=float).reshape(3),
                    ]
                )
                if np.any(noise_std > 0.0):
                    detected_relative_states = detected_relative_states + self._rng.normal(
                        loc=0.0,
                        scale=noise_std,
                        size=detected_relative_states.shape,
                    )
                velocity_limit = float(self.config.relative_velocity_limit_mps)
                detected_relative_states[:, 3:] = np.clip(
                    detected_relative_states[:, 3:],
                    -velocity_limit,
                    velocity_limit,
                )

        clutter_relative_states = self._sample_clutter_relative_states()
        candidate_relative_states = np.concatenate(
            [detected_relative_states, clutter_relative_states],
            axis=0,
        )
        if candidate_relative_states.shape[0] == 0:
            return LocalTargetObservation(
                relative_states=relative_states,
                mask=mask,
                covariance_diagonal=covariance_diagonal,
            )

        candidate_distances = np.linalg.norm(candidate_relative_states[:, :3], axis=1)
        selected = np.argsort(candidate_distances)[:max_candidates]
        observed_relative_states = candidate_relative_states[selected]

        count = observed_relative_states.shape[0]
        relative_states[:count] = observed_relative_states.astype(np.float32)
        mask[:count] = 1.0
        return LocalTargetObservation(
            relative_states=relative_states,
            mask=mask,
            covariance_diagonal=covariance_diagonal,
        )

    def _sample_clutter_relative_states(self) -> np.ndarray:
        mean_clutter_count = self.config.clutter_rate_per_second * self.config.dt_seconds
        clutter_count = int(self._rng.poisson(mean_clutter_count))
        if clutter_count <= 0:
            return np.zeros((0, STATE_DIM), dtype=float)

        directions = self._rng.normal(size=(clutter_count, 3))
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        zero_norms = norms.squeeze(-1) == 0.0
        while np.any(zero_norms):
            directions[zero_norms] = self._rng.normal(size=(int(np.sum(zero_norms)), 3))
            norms = np.linalg.norm(directions, axis=1, keepdims=True)
            zero_norms = norms.squeeze(-1) == 0.0
        unit_directions = directions / norms
        radii = self.config.detection_radius_m * np.cbrt(self._rng.random(clutter_count))
        positions = unit_directions * radii[:, None]

        velocity_limit = float(self.config.relative_velocity_limit_mps)
        velocities = self._rng.uniform(
            low=-velocity_limit,
            high=velocity_limit,
            size=(clutter_count, 3),
        )
        return np.concatenate([positions, velocities], axis=1)
