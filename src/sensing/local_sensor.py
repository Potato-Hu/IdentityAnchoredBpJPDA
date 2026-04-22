from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LocalSensingConfig:
    detection_radius_m: float
    max_candidates: int
    position_noise_std_m: float
    detection_probability: float
    clutter_rate_per_second: float
    dt_seconds: float


@dataclass(frozen=True)
class LocalTargetObservation:
    relative_positions: np.ndarray
    mask: np.ndarray


class LocalTargetSensor:
    """Omnidirectional local target sensor with fixed-length candidate output.

    The public observation deliberately contains no target id, role, key flag, or matching
    metadata. Candidates are sorted only by current range and padded with a mask.
    """

    def __init__(self, config: LocalSensingConfig, seed: int | None = None) -> None:
        self.config = config
        if self.config.detection_radius_m < 0.0:
            raise ValueError(f"local_sensing.detection_radius_m must be >= 0, got {self.config.detection_radius_m}.")
        if self.config.max_candidates < 0:
            raise ValueError(f"local_sensing.max_candidates must be >= 0, got {self.config.max_candidates}.")
        if self.config.position_noise_std_m < 0.0:
            raise ValueError(
                "local_sensing.local_position_noise_std_m must be >= 0, "
                f"got {self.config.position_noise_std_m}."
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

        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._seed = seed
        self._rng = np.random.default_rng(self._seed)

    def observe(self, friendly_position: np.ndarray, target_positions: np.ndarray) -> LocalTargetObservation:
        max_candidates = int(self.config.max_candidates)
        relative_positions = np.zeros((max_candidates, 3), dtype=np.float32)
        mask = np.zeros(max_candidates, dtype=np.float32)
        if max_candidates == 0:
            return LocalTargetObservation(relative_positions=relative_positions, mask=mask)

        friendly = np.asarray(friendly_position, dtype=float).reshape(3)
        targets = np.asarray(target_positions, dtype=float)
        if targets.size == 0:
            targets = np.zeros((0, 3), dtype=float)
        elif targets.ndim != 2 or targets.shape[1] != 3:
            raise ValueError(f"target_positions must have shape (N, 3), got {targets.shape}.")

        true_relative_positions = targets - friendly
        distances = np.linalg.norm(true_relative_positions, axis=1)
        visible_indices = np.flatnonzero(distances <= self.config.detection_radius_m)
        detected_relative_positions = np.zeros((0, 3), dtype=float)
        if visible_indices.size:
            detection_draws = self._rng.random(visible_indices.size)
            detected_mask = detection_draws < self.config.detection_probability
            detected_indices = visible_indices[detected_mask]
            if detected_indices.size:
                detected_relative_positions = true_relative_positions[detected_indices].copy()
                if self.config.position_noise_std_m > 0.0:
                    detected_relative_positions = detected_relative_positions + self._rng.normal(
                        loc=0.0,
                        scale=self.config.position_noise_std_m,
                        size=detected_relative_positions.shape,
                    )

        clutter_relative_positions = self._sample_clutter_relative_positions()
        candidate_relative_positions = np.concatenate(
            [detected_relative_positions, clutter_relative_positions],
            axis=0,
        )
        if candidate_relative_positions.shape[0] == 0:
            return LocalTargetObservation(relative_positions=relative_positions, mask=mask)

        candidate_distances = np.linalg.norm(candidate_relative_positions, axis=1)
        selected = np.argsort(candidate_distances)[:max_candidates]
        observed_relative_positions = candidate_relative_positions[selected]

        count = observed_relative_positions.shape[0]
        relative_positions[:count] = observed_relative_positions.astype(np.float32)
        mask[:count] = 1.0
        return LocalTargetObservation(relative_positions=relative_positions, mask=mask)

    def _sample_clutter_relative_positions(self) -> np.ndarray:
        mean_clutter_count = self.config.clutter_rate_per_second * self.config.dt_seconds
        clutter_count = int(self._rng.poisson(mean_clutter_count))
        if clutter_count <= 0:
            return np.zeros((0, 3), dtype=float)

        directions = self._rng.normal(size=(clutter_count, 3))
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        zero_norms = norms.squeeze(-1) == 0.0
        while np.any(zero_norms):
            directions[zero_norms] = self._rng.normal(size=(int(np.sum(zero_norms)), 3))
            norms = np.linalg.norm(directions, axis=1, keepdims=True)
            zero_norms = norms.squeeze(-1) == 0.0
        unit_directions = directions / norms
        radii = self.config.detection_radius_m * np.cbrt(self._rng.random(clutter_count))
        return unit_directions * radii[:, None]
