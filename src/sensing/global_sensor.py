from __future__ import annotations

from dataclasses import dataclass

import numpy as np


STATE_DIM = 6


@dataclass(frozen=True)
class GlobalKeyTargetSensingConfig:
    prior_position_std_m: np.ndarray
    prior_velocity_std_mps: np.ndarray

    @property
    def covariance_diagonal(self) -> np.ndarray:
        return np.concatenate(
            [
                np.square(np.asarray(self.prior_position_std_m, dtype=float).reshape(3)),
                np.square(np.asarray(self.prior_velocity_std_mps, dtype=float).reshape(3)),
            ]
        )


@dataclass(frozen=True)
class GlobalKeyTargetStatePrior:
    states: np.ndarray
    covariance_diagonal: np.ndarray


class GlobalKeyTargetStatePriorSensor:
    """External radar-like sensor that provides a noisy 6D key-target state prior."""

    def __init__(self, config: GlobalKeyTargetSensingConfig, seed: int | None = None) -> None:
        self.config = config
        position_std = np.asarray(self.config.prior_position_std_m, dtype=float).reshape(-1)
        velocity_std = np.asarray(self.config.prior_velocity_std_mps, dtype=float).reshape(-1)
        if position_std.shape != (3,):
            raise ValueError(
                "global_sensing.prior_position_std_m must resolve to a 3D vector, "
                f"got shape {position_std.shape}."
            )
        if velocity_std.shape != (3,):
            raise ValueError(
                "global_sensing.prior_velocity_std_mps must resolve to a 3D vector, "
                f"got shape {velocity_std.shape}."
            )
        if np.any(position_std < 0.0):
            raise ValueError(
                "global_sensing.prior_position_std_m must be >= 0 on every axis, "
                f"got {position_std.tolist()}."
            )
        if np.any(velocity_std < 0.0):
            raise ValueError(
                "global_sensing.prior_velocity_std_mps must be >= 0 on every axis, "
                f"got {velocity_std.tolist()}."
            )

        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._cache: GlobalKeyTargetStatePrior | None = None
        self._cache_step: int | None = None

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._seed = seed
        self._rng = np.random.default_rng(self._seed)
        self._cache = None
        self._cache_step = None

    def observe(self, *, time_step: int, key_target_states: np.ndarray) -> GlobalKeyTargetStatePrior:
        if self._cache_step == int(time_step) and self._cache is not None:
            return GlobalKeyTargetStatePrior(
                states=self._cache.states.copy(),
                covariance_diagonal=self._cache.covariance_diagonal.copy(),
            )

        states = np.asarray(key_target_states, dtype=float)
        if states.size == 0:
            states = np.zeros((0, STATE_DIM), dtype=float)
        elif states.ndim != 2 or states.shape[1] != STATE_DIM:
            raise ValueError(f"key_target_states must have shape (N, {STATE_DIM}), got {states.shape}.")

        noisy_states = states.copy()
        if noisy_states.size:
            noise_std = np.concatenate(
                [
                    np.asarray(self.config.prior_position_std_m, dtype=float).reshape(3),
                    np.asarray(self.config.prior_velocity_std_mps, dtype=float).reshape(3),
                ]
            )
            if np.any(noise_std > 0.0):
                noisy_states = noisy_states + self._rng.normal(
                    loc=0.0,
                    scale=noise_std,
                    size=noisy_states.shape,
                )

        covariance_diagonal = self.config.covariance_diagonal.astype(float, copy=True)
        self._cache = GlobalKeyTargetStatePrior(
            states=noisy_states.astype(float, copy=True),
            covariance_diagonal=covariance_diagonal,
        )
        self._cache_step = int(time_step)
        return GlobalKeyTargetStatePrior(
            states=self._cache.states.copy(),
            covariance_diagonal=self._cache.covariance_diagonal.copy(),
        )
