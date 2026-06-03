from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from src.sensing.association import KeyTrackPosterior, MultiTargetTrackOutput, STATE_DIM


@dataclass(frozen=True)
class FusedTrack:
    global_target_index: int
    key_label: str
    fused_state_mean: np.ndarray
    fused_state_covariance: np.ndarray
    contributing_agents: tuple[str, ...]
    fused_confidence: float
    fused_existence: float
    valid: bool


@dataclass(frozen=True)
class FusedTrackSet:
    timestamp: int
    tracks: tuple[FusedTrack, ...]


@dataclass(frozen=True)
class _GlobalTrackCandidate:
    agent_id: str
    global_target_index: int
    state_mean: np.ndarray
    state_covariance: np.ndarray
    confidence: float
    existence_probability: float


def _as_state_vector(state: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(state, dtype=float).reshape(-1)
    if vector.shape != (STATE_DIM,):
        raise ValueError(f"{name} must have shape ({STATE_DIM},), got {vector.shape}.")
    return vector


def _as_covariance_matrix(covariance: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(covariance, dtype=float)
    if matrix.shape == (STATE_DIM,):
        matrix = np.diag(matrix)
    if matrix.shape != (STATE_DIM, STATE_DIM):
        raise ValueError(f"{name} must have shape ({STATE_DIM}, {STATE_DIM}), got {matrix.shape}.")
    stabilized = 0.5 * (matrix + matrix.T)
    diagonal = np.maximum(np.diag(stabilized), 1.0e-6)
    np.fill_diagonal(stabilized, diagonal)
    return stabilized


def _self_localization_covariance(self_position_std_m: np.ndarray | Sequence[float]) -> np.ndarray:
    position_std = np.asarray(self_position_std_m, dtype=float).reshape(-1)
    if position_std.shape != (3,):
        raise ValueError(
            "self_position_std_m must resolve to a 3D vector, "
            f"got shape {position_std.shape}."
        )
    covariance = np.zeros(STATE_DIM, dtype=float)
    covariance[:3] = np.square(np.maximum(position_std, 0.0))
    return covariance


def _to_global_candidate(
    *,
    agent_id: str,
    friendly_state: np.ndarray,
    track: KeyTrackPosterior,
    self_position_std_m: np.ndarray,
    use_prior: bool,
) -> _GlobalTrackCandidate:
    friendly = _as_state_vector(friendly_state, name="friendly_state")
    state = track.prior_relative_state if use_prior else track.posterior_relative_state
    covariance_diagonal = (
        track.prior_covariance_diagonal if use_prior else track.posterior_covariance_diagonal
    )
    global_mean = friendly + _as_state_vector(state, name="track_state")
    global_covariance_diagonal = np.asarray(covariance_diagonal, dtype=float).reshape(STATE_DIM).copy()
    global_covariance_diagonal += _self_localization_covariance(self_position_std_m)
    return _GlobalTrackCandidate(
        agent_id=str(agent_id),
        global_target_index=int(track.global_target_index),
        state_mean=global_mean.astype(float, copy=True),
        state_covariance=np.diag(np.maximum(global_covariance_diagonal, 1.0e-6)),
        confidence=float(track.confidence),
        existence_probability=float(track.existence_probability),
    )


def _mahalanobis_distance(
    mean_a: np.ndarray,
    covariance_a: np.ndarray,
    mean_b: np.ndarray,
    covariance_b: np.ndarray,
) -> float:
    delta = _as_state_vector(mean_a, name="mean_a") - _as_state_vector(mean_b, name="mean_b")
    innovation_covariance = _as_covariance_matrix(covariance_a, name="covariance_a") + _as_covariance_matrix(
        covariance_b,
        name="covariance_b",
    )
    inv_innovation_covariance = np.linalg.pinv(innovation_covariance)
    mahalanobis_sq = float(delta.T @ inv_innovation_covariance @ delta)
    return float(np.sqrt(max(mahalanobis_sq, 0.0)))


def _covariance_intersection_pair(
    mean_a: np.ndarray,
    covariance_a: np.ndarray,
    mean_b: np.ndarray,
    covariance_b: np.ndarray,
    *,
    search_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    state_a = _as_state_vector(mean_a, name="mean_a")
    state_b = _as_state_vector(mean_b, name="mean_b")
    cov_a = _as_covariance_matrix(covariance_a, name="covariance_a")
    cov_b = _as_covariance_matrix(covariance_b, name="covariance_b")
    precision_a = np.linalg.inv(cov_a)
    precision_b = np.linalg.inv(cov_b)

    candidate_steps = max(int(search_steps), 2)
    candidate_weights = np.linspace(0.0, 1.0, candidate_steps)
    best_mean = state_a
    best_covariance = cov_a
    best_trace = np.inf

    for raw_weight in candidate_weights:
        weight = float(np.clip(raw_weight, 1.0e-3, 1.0 - 1.0e-3))
        precision = weight * precision_a + (1.0 - weight) * precision_b
        fused_covariance = _as_covariance_matrix(np.linalg.inv(precision), name="fused_covariance")
        fused_mean = fused_covariance @ (
            weight * (precision_a @ state_a) + (1.0 - weight) * (precision_b @ state_b)
        )
        trace_value = float(np.trace(fused_covariance))
        if trace_value < best_trace:
            best_trace = trace_value
            best_mean = fused_mean
            best_covariance = fused_covariance

    return best_mean, best_covariance


def _build_invalid_fused_track(
    *,
    global_target_index: int,
    fallback_mean: np.ndarray,
    fallback_covariance: np.ndarray,
) -> FusedTrack:
    return FusedTrack(
        global_target_index=int(global_target_index),
        key_label=f"E{int(global_target_index) + 1}",
        fused_state_mean=_as_state_vector(fallback_mean, name="fallback_mean").astype(float, copy=True),
        fused_state_covariance=_as_covariance_matrix(
            fallback_covariance,
            name="fallback_covariance",
        ).astype(float, copy=True),
        contributing_agents=tuple(),
        fused_confidence=0.0,
        fused_existence=0.0,
        valid=False,
    )


def _fuse_group(
    *,
    global_target_index: int,
    candidates: Sequence[_GlobalTrackCandidate],
    fallback: _GlobalTrackCandidate,
    mahalanobis_distance_threshold: float,
    ci_search_steps: int,
) -> FusedTrack:
    if not candidates:
        return _build_invalid_fused_track(
            global_target_index=global_target_index,
            fallback_mean=fallback.state_mean,
            fallback_covariance=fallback.state_covariance,
        )

    ordered_candidates = sorted(candidates, key=lambda candidate: candidate.confidence, reverse=True)
    fused_mean = ordered_candidates[0].state_mean.astype(float, copy=True)
    fused_covariance = ordered_candidates[0].state_covariance.astype(float, copy=True)
    contributing_agents = [ordered_candidates[0].agent_id]
    accepted_candidates = [ordered_candidates[0]]

    for candidate in ordered_candidates[1:]:
        distance = _mahalanobis_distance(
            fused_mean,
            fused_covariance,
            candidate.state_mean,
            candidate.state_covariance,
        )
        if distance > float(mahalanobis_distance_threshold):
            continue
        fused_mean, fused_covariance = _covariance_intersection_pair(
            fused_mean,
            fused_covariance,
            candidate.state_mean,
            candidate.state_covariance,
            search_steps=ci_search_steps,
        )
        contributing_agents.append(candidate.agent_id)
        accepted_candidates.append(candidate)

    confidence_weights = np.asarray(
        [max(candidate.confidence, 1.0e-6) for candidate in accepted_candidates],
        dtype=float,
    )
    confidence_weights /= float(np.sum(confidence_weights))
    fused_confidence = float(
        np.clip(
            np.sum(
                confidence_weights * np.asarray([candidate.confidence for candidate in accepted_candidates], dtype=float)
            ),
            0.0,
            1.0,
        )
    )
    fused_existence = float(
        np.clip(
            np.sum(
                confidence_weights
                * np.asarray([candidate.existence_probability for candidate in accepted_candidates], dtype=float)
            ),
            0.0,
            1.0,
        )
    )

    return FusedTrack(
        global_target_index=int(global_target_index),
        key_label=f"E{int(global_target_index) + 1}",
        fused_state_mean=_as_state_vector(fused_mean, name="fused_mean").astype(float, copy=True),
        fused_state_covariance=_as_covariance_matrix(
            fused_covariance,
            name="fused_covariance",
        ).astype(float, copy=True),
        contributing_agents=tuple(contributing_agents),
        fused_confidence=fused_confidence,
        fused_existence=fused_existence,
        valid=True,
    )


def build_fused_track_set(
    *,
    timestamp: int,
    track_outputs: Mapping[str, MultiTargetTrackOutput],
    friendly_states: np.ndarray,
    agent_ids: Sequence[str],
    self_position_std_m: np.ndarray | Sequence[float],
    min_confidence: float,
    mahalanobis_distance_threshold: float,
    ci_search_steps: int,
) -> FusedTrackSet:
    friendly_array = np.asarray(friendly_states, dtype=float)
    if friendly_array.ndim != 2 or friendly_array.shape[1] != STATE_DIM:
        raise ValueError(f"friendly_states must have shape (N, {STATE_DIM}), got {friendly_array.shape}.")
    if friendly_array.shape[0] != len(agent_ids):
        raise ValueError("friendly_states row count must match agent_ids length.")

    self_position_std = np.asarray(self_position_std_m, dtype=float).reshape(3)
    min_conf = float(np.clip(min_confidence, 0.0, 1.0))
    gate = max(float(mahalanobis_distance_threshold), 0.0)

    candidate_groups: dict[int, list[_GlobalTrackCandidate]] = {}
    fallback_groups: dict[int, list[_GlobalTrackCandidate]] = {}
    for agent_idx, agent_id in enumerate(agent_ids):
        output = track_outputs.get(str(agent_id))
        if output is None:
            continue
        for track in output.tracks:
            fallback_candidate = _to_global_candidate(
                agent_id=str(agent_id),
                friendly_state=friendly_array[agent_idx],
                track=track,
                self_position_std_m=self_position_std,
                use_prior=True,
            )
            fallback_groups.setdefault(int(track.global_target_index), []).append(fallback_candidate)

            if not track.valid or float(track.confidence) < min_conf:
                continue
            candidate = _to_global_candidate(
                agent_id=str(agent_id),
                friendly_state=friendly_array[agent_idx],
                track=track,
                self_position_std_m=self_position_std,
                use_prior=False,
            )
            candidate_groups.setdefault(int(track.global_target_index), []).append(candidate)

    fused_tracks: list[FusedTrack] = []
    all_target_indices = sorted(set(fallback_groups) | set(candidate_groups))
    for target_idx in all_target_indices:
        fallback_candidates = fallback_groups.get(target_idx, [])
        if not fallback_candidates:
            continue
        fused_tracks.append(
            _fuse_group(
                global_target_index=target_idx,
                candidates=candidate_groups.get(target_idx, []),
                fallback=fallback_candidates[0],
                mahalanobis_distance_threshold=gate,
                ci_search_steps=ci_search_steps,
            )
        )

    return FusedTrackSet(
        timestamp=int(timestamp),
        tracks=tuple(fused_tracks),
    )
