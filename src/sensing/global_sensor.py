from __future__ import annotations

from dataclasses import dataclass

import numpy as np


STATE_DIM = 6


@dataclass(frozen=True)
class GlobalKeyTargetSensingConfig:
    update_interval_seconds: float
    latency_mean_seconds: float
    latency_std_seconds: float
    dt_seconds: float
    prior_position_std_m: np.ndarray
    prior_velocity_std_mps: np.ndarray
    prior_bias_position_std_m: np.ndarray
    prior_bias_velocity_std_mps: np.ndarray
    prior_bias_correlation_steps: float
    prior_missed_probability: float

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
    valid_mask: np.ndarray


@dataclass(frozen=True)
class _PendingGlobalReport:
    sampled_time_seconds: float
    arrival_time_seconds: float
    states: np.ndarray
    valid_mask: np.ndarray


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
        bias_position_std = np.asarray(self.config.prior_bias_position_std_m, dtype=float).reshape(-1)
        bias_velocity_std = np.asarray(self.config.prior_bias_velocity_std_mps, dtype=float).reshape(-1)
        if bias_position_std.shape != (3,):
            raise ValueError(
                "global_sensing.prior_bias_position_std_m must resolve to a 3D vector, "
                f"got shape {bias_position_std.shape}."
            )
        if bias_velocity_std.shape != (3,):
            raise ValueError(
                "global_sensing.prior_bias_velocity_std_mps must resolve to a 3D vector, "
                f"got shape {bias_velocity_std.shape}."
            )
        if np.any(bias_position_std < 0.0) or np.any(bias_velocity_std < 0.0):
            raise ValueError("global_sensing prior bias stds must be >= 0 on every axis.")
        if float(self.config.prior_bias_correlation_steps) < 0.0:
            raise ValueError(
                "global_sensing.prior_bias_correlation_steps must be >= 0, "
                f"got {self.config.prior_bias_correlation_steps}."
            )
        if not 0.0 <= float(self.config.prior_missed_probability) <= 1.0:
            raise ValueError(
                "global_sensing.prior_missed_probability must be in [0, 1], "
                f"got {self.config.prior_missed_probability}."
            )
        if float(self.config.update_interval_seconds) <= 0.0:
            raise ValueError("global_sensing.update_interval_seconds must be > 0.")
        if float(self.config.latency_mean_seconds) < 0.0:
            raise ValueError("global_sensing.latency_mean_seconds must be >= 0.")
        if float(self.config.latency_std_seconds) < 0.0:
            raise ValueError("global_sensing.latency_std_seconds must be >= 0.")
        if float(self.config.dt_seconds) <= 0.0:
            raise ValueError("global_sensing.dt_seconds must be > 0.")

        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._cache: GlobalKeyTargetStatePrior | None = None
        self._cache_step: int | None = None
        self._bias_state = np.zeros(STATE_DIM, dtype=float)
        self._next_sample_time_seconds = 0.0
        self._pending_reports: list[_PendingGlobalReport] = []

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._seed = seed
        self._rng = np.random.default_rng(self._seed)
        self._cache = None
        self._cache_step = None
        self._bias_state = np.zeros(STATE_DIM, dtype=float)
        self._next_sample_time_seconds = 0.0
        self._pending_reports = []

    def _advance_bias_state(self) -> None:
        bias_std = np.concatenate(
            [
                np.asarray(self.config.prior_bias_position_std_m, dtype=float).reshape(3),
                np.asarray(self.config.prior_bias_velocity_std_mps, dtype=float).reshape(3),
            ]
        )
        if not np.any(bias_std > 0.0):
            self._bias_state = np.zeros(STATE_DIM, dtype=float)
            return
        correlation_steps = float(self.config.prior_bias_correlation_steps)
        if correlation_steps <= 0.0:
            self._bias_state = self._rng.normal(loc=0.0, scale=bias_std, size=STATE_DIM)
            return
        correlation = float(np.exp(-1.0 / max(correlation_steps, 1.0e-6)))
        innovation_scale = np.sqrt(max(1.0 - correlation * correlation, 0.0))
        innovation = self._rng.normal(loc=0.0, scale=bias_std, size=STATE_DIM)
        self._bias_state = correlation * self._bias_state + innovation_scale * innovation

    def _sample_report(self, *, sampled_time_seconds: float, states: np.ndarray) -> None:
        noisy_states = states.copy()
        valid_mask = np.ones(states.shape[0], dtype=bool)
        if noisy_states.size:
            missed_probability = float(self.config.prior_missed_probability)
            if missed_probability > 0.0:
                valid_mask = self._rng.random(states.shape[0]) >= missed_probability
            self._advance_bias_state()
            valid_indices = np.flatnonzero(valid_mask)
            noisy_states[valid_indices] += self._bias_state.reshape(1, STATE_DIM)
            noise_std = np.concatenate(
                [
                    np.asarray(self.config.prior_position_std_m, dtype=float).reshape(3),
                    np.asarray(self.config.prior_velocity_std_mps, dtype=float).reshape(3),
                ]
            )
            if np.any(noise_std > 0.0) and valid_indices.size:
                noisy_states[valid_indices] += self._rng.normal(
                    loc=0.0,
                    scale=noise_std,
                    size=(valid_indices.size, STATE_DIM),
                )
            noisy_states[~valid_mask] = 0.0

        latency_seconds = max(
            float(
                self._rng.normal(
                    loc=float(self.config.latency_mean_seconds),
                    scale=float(self.config.latency_std_seconds),
                )
            ),
            0.0,
        )
        self._pending_reports.append(
            _PendingGlobalReport(
                sampled_time_seconds=float(sampled_time_seconds),
                arrival_time_seconds=float(sampled_time_seconds) + latency_seconds,
                states=noisy_states,
                valid_mask=valid_mask,
            )
        )

    def observe(self, *, time_step: int, key_target_states: np.ndarray) -> GlobalKeyTargetStatePrior:
        if self._cache_step == int(time_step) and self._cache is not None:
            return GlobalKeyTargetStatePrior(
                states=self._cache.states.copy(),
                covariance_diagonal=self._cache.covariance_diagonal.copy(),
                valid_mask=self._cache.valid_mask.copy(),
            )

        states = np.asarray(key_target_states, dtype=float)
        if states.size == 0:
            states = np.zeros((0, STATE_DIM), dtype=float)
        elif states.ndim != 2 or states.shape[1] != STATE_DIM:
            raise ValueError(f"key_target_states must have shape (N, {STATE_DIM}), got {states.shape}.")

        current_time_seconds = float(time_step) * float(self.config.dt_seconds)
        if current_time_seconds + 1.0e-9 >= self._next_sample_time_seconds:
            self._sample_report(sampled_time_seconds=current_time_seconds, states=states)
            while self._next_sample_time_seconds <= current_time_seconds + 1.0e-9:
                self._next_sample_time_seconds += float(self.config.update_interval_seconds)

        arrived_reports = [
            report
            for report in self._pending_reports
            if report.arrival_time_seconds <= current_time_seconds + 1.0e-9
        ]
        self._pending_reports = [
            report
            for report in self._pending_reports
            if report.arrival_time_seconds > current_time_seconds + 1.0e-9
        ]

        covariance_diagonal = self.config.covariance_diagonal.astype(float, copy=True)
        noisy_states = np.zeros_like(states)
        valid_mask = np.zeros(states.shape[0], dtype=bool)
        if arrived_reports:
            latest_report = max(arrived_reports, key=lambda report: report.sampled_time_seconds)
            noisy_states = latest_report.states.copy()
            valid_mask = latest_report.valid_mask.copy()
            report_age_seconds = max(current_time_seconds - latest_report.sampled_time_seconds, 0.0)
            noisy_states[valid_mask, :3] += noisy_states[valid_mask, 3:] * report_age_seconds
            covariance_diagonal[:3] += covariance_diagonal[3:] * report_age_seconds * report_age_seconds

        self._cache = GlobalKeyTargetStatePrior(
            states=noisy_states.astype(float, copy=True),
            covariance_diagonal=covariance_diagonal,
            valid_mask=valid_mask.astype(bool, copy=True),
        )
        self._cache_step = int(time_step)
        return GlobalKeyTargetStatePrior(
            states=self._cache.states.copy(),
            covariance_diagonal=self._cache.covariance_diagonal.copy(),
            valid_mask=self._cache.valid_mask.copy(),
        )
