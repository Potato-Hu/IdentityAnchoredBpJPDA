from __future__ import annotations

from dataclasses import dataclass

import numpy as np


STATE_DIM = 6


@dataclass(frozen=True)
class LocalSensingConfig:
    detection_radius_m: float
    measurement_position_std_m: np.ndarray
    measurement_velocity_std_mps: np.ndarray
    detection_probability: float
    detection_probability_min: float
    detection_full_probability_range_m: float
    detection_density_penalty_alpha: float
    clutter_rate_per_second: float
    clutter_hotspot_probability: float
    clutter_hotspot_position_std_m: np.ndarray
    clutter_hotspot_velocity_std_mps: np.ndarray
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
        if not 0.0 <= self.config.detection_probability_min <= 1.0:
            raise ValueError(
                "local_sensing.detection_probability_min must be in [0, 1], "
                f"got {self.config.detection_probability_min}."
            )
        if self.config.detection_probability_min > self.config.detection_probability:
            raise ValueError(
                "local_sensing.detection_probability_min must be <= detection_probability, "
                f"got {self.config.detection_probability_min} and {self.config.detection_probability}."
            )
        if self.config.detection_full_probability_range_m < 0.0:
            raise ValueError(
                "local_sensing.detection_full_probability_range_m must be >= 0, "
                f"got {self.config.detection_full_probability_range_m}."
            )
        if self.config.detection_density_penalty_alpha < 0.0:
            raise ValueError(
                "local_sensing.detection_density_penalty_alpha must be >= 0, "
                f"got {self.config.detection_density_penalty_alpha}."
            )
        if self.config.clutter_rate_per_second < 0.0:
            raise ValueError(
                "local_sensing.clutter_rate_per_second must be >= 0, "
                f"got {self.config.clutter_rate_per_second}."
            )
        if not 0.0 <= self.config.clutter_hotspot_probability <= 1.0:
            raise ValueError(
                "local_sensing.clutter_hotspot_probability must be in [0, 1], "
                f"got {self.config.clutter_hotspot_probability}."
            )
        if self.config.dt_seconds <= 0.0:
            raise ValueError(f"local_sensing.dt_seconds must be > 0, got {self.config.dt_seconds}.")
        if self.config.relative_velocity_limit_mps < 0.0:
            raise ValueError(
                "local_sensing.relative_velocity_limit_mps must be >= 0, "
                f"got {self.config.relative_velocity_limit_mps}."
            )
        clutter_position_std = np.asarray(self.config.clutter_hotspot_position_std_m, dtype=float).reshape(-1)
        clutter_velocity_std = np.asarray(self.config.clutter_hotspot_velocity_std_mps, dtype=float).reshape(-1)
        if clutter_position_std.shape != (3,):
            raise ValueError(
                "local_sensing.clutter_hotspot_position_std_m must resolve to a 3D vector, "
                f"got shape {clutter_position_std.shape}."
            )
        if clutter_velocity_std.shape != (3,):
            raise ValueError(
                "local_sensing.clutter_hotspot_velocity_std_mps must resolve to a 3D vector, "
                f"got shape {clutter_velocity_std.shape}."
            )
        if np.any(clutter_position_std < 0.0) or np.any(clutter_velocity_std < 0.0):
            raise ValueError("local_sensing clutter hotspot stds must be >= 0 on every axis.")

        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._seed = seed
        self._rng = np.random.default_rng(self._seed)

    def observe(self, friendly_state: np.ndarray, target_states: np.ndarray) -> LocalTargetObservation:
        covariance_diagonal = self.config.covariance_diagonal.astype(np.float32, copy=True)

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
            visible_relative_states = true_relative_states[visible_indices]
            density_scores = self._local_density_scores(visible_relative_states)
            effective_detection_probabilities = np.asarray(
                [
                    self._effective_detection_probability(
                        distance_m=float(np.linalg.norm(relative_state[:3])),
                        density_score=float(density_score),
                    )
                    for relative_state, density_score in zip(visible_relative_states, density_scores)
                ],
                dtype=float,
            )
            detection_draws = self._rng.random(visible_indices.size)
            detected_mask = detection_draws < effective_detection_probabilities
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
        else:
            visible_relative_states = np.zeros((0, STATE_DIM), dtype=float)
            density_scores = np.zeros(0, dtype=float)

        clutter_relative_states = self._sample_clutter_relative_states(
            reference_relative_states=visible_relative_states,
            reference_density_scores=density_scores,
        )
        candidate_relative_states = np.concatenate(
            [detected_relative_states, clutter_relative_states],
            axis=0,
        )
        if candidate_relative_states.shape[0] == 0:
            return LocalTargetObservation(
                relative_states=np.zeros((0, STATE_DIM), dtype=np.float32),
                mask=np.zeros(0, dtype=np.float32),
                covariance_diagonal=covariance_diagonal,
            )

        relative_states = candidate_relative_states.astype(np.float32)
        mask = np.ones(relative_states.shape[0], dtype=np.float32)
        return LocalTargetObservation(
            relative_states=relative_states,
            mask=mask,
            covariance_diagonal=covariance_diagonal,
        )

    def _range_detection_probability(self, distance_m: float) -> float:
        detection_radius = max(float(self.config.detection_radius_m), 1.0e-6)
        full_probability_range = float(
            np.clip(self.config.detection_full_probability_range_m, 0.0, detection_radius)
        )
        probability_near = float(self.config.detection_probability)
        probability_min = float(self.config.detection_probability_min)
        if distance_m <= full_probability_range or full_probability_range >= detection_radius:
            return probability_near
        remaining_range = max(detection_radius - full_probability_range, 1.0e-6)
        normalized_distance = np.clip((distance_m - full_probability_range) / remaining_range, 0.0, 1.0)
        return probability_near - (probability_near - probability_min) * normalized_distance

    def _local_density_scores(self, relative_states: np.ndarray) -> np.ndarray:
        if relative_states.shape[0] == 0:
            return np.zeros(0, dtype=float)
        density_radius = max(
            min(float(self.config.detection_full_probability_range_m) * 0.5, float(self.config.detection_radius_m)),
            1.0,
        )
        position_deltas = relative_states[:, None, :3] - relative_states[None, :, :3]
        distances = np.linalg.norm(position_deltas, axis=-1)
        neighbor_counts = np.sum((distances <= density_radius) & (distances > 0.0), axis=1)
        return np.clip(neighbor_counts.astype(float) / 4.0, 0.0, 1.0)

    def _effective_detection_probability(self, *, distance_m: float, density_score: float) -> float:
        range_probability = self._range_detection_probability(distance_m)
        penalty = np.exp(-float(self.config.detection_density_penalty_alpha) * float(np.clip(density_score, 0.0, 1.0)))
        probability = range_probability * penalty
        return float(
            np.clip(
                probability,
                float(self.config.detection_probability_min),
                float(self.config.detection_probability),
            )
        )

    def _sample_uniform_clutter_relative_states(self, clutter_count: int) -> np.ndarray:
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

    def _sample_clutter_relative_states(
        self,
        *,
        reference_relative_states: np.ndarray,
        reference_density_scores: np.ndarray,
    ) -> np.ndarray:
        mean_clutter_count = self.config.clutter_rate_per_second * self.config.dt_seconds
        clutter_count = int(self._rng.poisson(mean_clutter_count))
        if clutter_count <= 0:
            return np.zeros((0, STATE_DIM), dtype=float)
        if reference_relative_states.shape[0] == 0 or float(self.config.clutter_hotspot_probability) <= 0.0:
            return self._sample_uniform_clutter_relative_states(clutter_count)

        hotspot_mask = self._rng.random(clutter_count) < float(self.config.clutter_hotspot_probability)
        hotspot_count = int(np.sum(hotspot_mask))
        clutter_states = np.zeros((clutter_count, STATE_DIM), dtype=float)
        if hotspot_count < clutter_count:
            clutter_states[~hotspot_mask] = self._sample_uniform_clutter_relative_states(clutter_count - hotspot_count)

        if hotspot_count > 0:
            anchor_weights = 1.0 + np.asarray(reference_density_scores, dtype=float).reshape(-1)
            anchor_weights = anchor_weights / np.sum(anchor_weights)
            anchor_indices = self._rng.choice(reference_relative_states.shape[0], size=hotspot_count, p=anchor_weights)
            anchors = reference_relative_states[anchor_indices]
            position_noise = self._rng.normal(
                loc=0.0,
                scale=np.asarray(self.config.clutter_hotspot_position_std_m, dtype=float).reshape(1, 3),
                size=(hotspot_count, 3),
            )
            velocity_noise = self._rng.normal(
                loc=0.0,
                scale=np.asarray(self.config.clutter_hotspot_velocity_std_mps, dtype=float).reshape(1, 3),
                size=(hotspot_count, 3),
            )
            hotspot_positions = anchors[:, :3] + position_noise
            position_norms = np.linalg.norm(hotspot_positions, axis=1)
            out_of_range = position_norms > float(self.config.detection_radius_m)
            if np.any(out_of_range):
                hotspot_positions[out_of_range] *= (
                    (0.98 * float(self.config.detection_radius_m))
                    / np.maximum(position_norms[out_of_range], 1.0e-6)
                ).reshape(-1, 1)
            velocity_limit = float(self.config.relative_velocity_limit_mps)
            hotspot_velocities = np.clip(
                anchors[:, 3:] + velocity_noise,
                -velocity_limit,
                velocity_limit,
            )
            clutter_states[hotspot_mask] = np.concatenate([hotspot_positions, hotspot_velocities], axis=1)

        return clutter_states
