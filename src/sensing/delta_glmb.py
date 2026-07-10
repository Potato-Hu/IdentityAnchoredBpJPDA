from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.sensing.association import (
    BPJPDAFilterState,
    IMMMotionState,
    KeyTrackPosterior,
    MultiTargetTrackOutput,
    STATE_DIM,
    _as_state_batch,
    _bp_jpda_predict_track_state,
    _build_prior_relative_states,
    _collapse_motion_state,
    _gaussian_mixture_moments,
    _gaussian_pdf_zero_mean_diag,
    _is_filter_state_expired,
    _merge_motion_hypotheses,
    _nearest_measurement_support,
    _next_missed_count,
    _position_gate_margin,
    _previous_missed_count,
    _sanitize_max_missed_steps,
    _update_motion_state_with_measurement,
    _validate_local_observation,
    empty_multi_target_track_output,
)
from src.sensing.global_sensor import GlobalKeyTargetStatePrior
from src.sensing.local_sensor import LocalTargetObservation


@dataclass(frozen=True)
class DeltaGLMBTrackState:
    existence_probability: float
    mean: np.ndarray
    covariance: np.ndarray
    motion_state: IMMMotionState | None = None
    missed_count: int = 0


@dataclass(frozen=True)
class DeltaGLMBHypothesis:
    weight: float
    track_states: tuple[DeltaGLMBTrackState, ...]
    assigned_candidate_offsets: tuple[int, ...]
    innovation_distances: tuple[float, ...]


@dataclass(frozen=True)
class DeltaGLMBFilterState:
    hypotheses: tuple[DeltaGLMBHypothesis, ...]


@dataclass(frozen=True)
class _AssociationBranch:
    weight_factor: float
    track_state: DeltaGLMBTrackState
    candidate_offset: int
    innovation_distance: float


def _normalize_and_prune_hypotheses(
    hypotheses: Sequence[DeltaGLMBHypothesis],
    *,
    max_hypotheses: int,
    weight_prune_threshold: float,
) -> tuple[DeltaGLMBHypothesis, ...]:
    if not hypotheses:
        return tuple()

    valid_hypotheses = [hypothesis for hypothesis in hypotheses if float(hypothesis.weight) > 0.0]
    if not valid_hypotheses:
        return tuple()

    total_weight = float(sum(float(hypothesis.weight) for hypothesis in valid_hypotheses))
    normalized = [
        DeltaGLMBHypothesis(
            weight=float(hypothesis.weight) / total_weight,
            track_states=hypothesis.track_states,
            assigned_candidate_offsets=hypothesis.assigned_candidate_offsets,
            innovation_distances=hypothesis.innovation_distances,
        )
        for hypothesis in valid_hypotheses
    ]
    normalized.sort(key=lambda hypothesis: float(hypothesis.weight), reverse=True)
    pruned = [
        hypothesis for hypothesis in normalized if float(hypothesis.weight) >= float(max(weight_prune_threshold, 0.0))
    ]
    if not pruned:
        pruned = [normalized[0]]
    if int(max_hypotheses) > 0:
        pruned = pruned[: int(max_hypotheses)]

    total_weight = float(sum(float(hypothesis.weight) for hypothesis in pruned))
    return tuple(
        DeltaGLMBHypothesis(
            weight=float(hypothesis.weight) / total_weight,
            track_states=hypothesis.track_states,
            assigned_candidate_offsets=hypothesis.assigned_candidate_offsets,
            innovation_distances=hypothesis.innovation_distances,
        )
        for hypothesis in pruned
    )


def _predict_hypotheses(
    *,
    previous_filter_state: DeltaGLMBFilterState | None,
    prior_relative_states: np.ndarray,
    prior_covariance_diagonal: np.ndarray,
    recursive_motion_model: str,
    imm_model_types: Sequence[str],
    imm_model_transition_matrix: np.ndarray | Sequence[Sequence[float]] | None,
    imm_initial_model_probabilities: np.ndarray | Sequence[float] | None,
    dt_seconds: float,
    survival_probability: float,
    birth_weight: float,
    process_position_std: np.ndarray,
    process_velocity_std: np.ndarray,
    imm_turn_rate_radps: float,
    imm_ct_process_noise_scale: float,
    max_hypotheses: int,
    max_missed_steps: int,
) -> tuple[DeltaGLMBHypothesis, ...]:
    base_hypotheses = previous_filter_state.hypotheses if previous_filter_state is not None else tuple()
    if not base_hypotheses:
        base_hypotheses = (
            DeltaGLMBHypothesis(
                weight=1.0,
                track_states=tuple(),
                assigned_candidate_offsets=tuple(),
                innovation_distances=tuple(),
            ),
        )

    predicted_hypotheses: list[DeltaGLMBHypothesis] = []
    for hypothesis in base_hypotheses[: max(int(max_hypotheses), 1)]:
        predicted_track_states: list[DeltaGLMBTrackState] = []
        for target_idx in range(prior_relative_states.shape[0]):
            previous_track_state = hypothesis.track_states[target_idx] if target_idx < len(hypothesis.track_states) else None
            if _is_filter_state_expired(previous_track_state, max_missed_steps):
                previous_track_state = None
            previous_filter = None
            previous_existence = 0.0
            if previous_track_state is not None:
                previous_filter = BPJPDAFilterState(
                    mean=np.asarray(previous_track_state.mean, dtype=float).copy(),
                    covariance=np.asarray(previous_track_state.covariance, dtype=float).copy(),
                    motion_state=previous_track_state.motion_state,
                    missed_count=previous_track_state.missed_count,
                )
                previous_existence = float(previous_track_state.existence_probability)

            predicted_mean, predicted_covariance, predicted_motion_state = _bp_jpda_predict_track_state(
                previous_filter_state=previous_filter,
                prior_relative_state=prior_relative_states[target_idx],
                prior_covariance_diagonal=prior_covariance_diagonal,
                recursive_motion_model=recursive_motion_model,
                imm_model_types=imm_model_types,
                imm_model_transition_matrix=imm_model_transition_matrix,
                imm_initial_model_probabilities=imm_initial_model_probabilities,
                dt_seconds=dt_seconds,
                survival_probability=survival_probability,
                birth_weight=birth_weight,
                process_position_std=process_position_std,
                process_velocity_std=process_velocity_std,
                imm_turn_rate_radps=imm_turn_rate_radps,
                imm_ct_process_noise_scale=imm_ct_process_noise_scale,
            )
            if previous_filter is None:
                predicted_existence = float(np.clip(birth_weight, 0.0, 1.0))
            else:
                predicted_existence = float(
                    np.clip(
                        survival_probability * previous_existence + (1.0 - previous_existence) * float(birth_weight),
                        0.0,
                        1.0,
                    )
                )
            predicted_track_states.append(
                DeltaGLMBTrackState(
                    existence_probability=predicted_existence,
                    mean=predicted_mean.astype(float, copy=True),
                    covariance=predicted_covariance.astype(float, copy=True),
                    motion_state=predicted_motion_state,
                    missed_count=_previous_missed_count(previous_track_state),
                )
            )

        predicted_hypotheses.append(
            DeltaGLMBHypothesis(
                weight=float(hypothesis.weight),
                track_states=tuple(predicted_track_states),
                assigned_candidate_offsets=tuple(-1 for _ in predicted_track_states),
                innovation_distances=tuple(np.inf for _ in predicted_track_states),
            )
        )
    return _normalize_and_prune_hypotheses(
        predicted_hypotheses,
        max_hypotheses=max_hypotheses,
        weight_prune_threshold=0.0,
    )


def _build_target_branches(
    *,
    predicted_track_state: DeltaGLMBTrackState,
    candidate_states: np.ndarray,
    local_covariance: np.ndarray,
    detection_radius_m: float,
    local_detection_probability: float,
    clutter_intensity: float,
    gate_sigma: float,
    max_missed_steps: int,
) -> list[_AssociationBranch]:
    predicted_mean = np.asarray(predicted_track_state.mean, dtype=float).reshape(STATE_DIM)
    predicted_covariance = np.asarray(predicted_track_state.covariance, dtype=float).reshape(STATE_DIM, STATE_DIM)
    predicted_covariance_diagonal = np.diag(predicted_covariance)
    predicted_motion_state = predicted_track_state.motion_state
    predicted_existence = float(np.clip(predicted_track_state.existence_probability, 0.0, 1.0))

    innovation_covariance_diagonal = predicted_covariance_diagonal + local_covariance
    prior_position_distance = float(np.linalg.norm(predicted_mean[:3]))
    track_active = prior_position_distance <= float(detection_radius_m) + _position_gate_margin(
        innovation_covariance_diagonal,
        gate_sigma,
    )

    if not track_active:
        return [
            _AssociationBranch(
                weight_factor=1.0,
                track_state=DeltaGLMBTrackState(
                    existence_probability=0.0,
                    mean=predicted_mean.astype(float, copy=True),
                    covariance=predicted_covariance.astype(float, copy=True),
                    motion_state=predicted_motion_state,
                    missed_count=_next_missed_count(
                        predicted_track_state.missed_count,
                        has_measurement_support=False,
                        max_missed_steps=max_missed_steps,
                    ),
                ),
                candidate_offset=-1,
                innovation_distance=np.inf,
            )
        ]

    missed_factor = (1.0 - predicted_existence) + predicted_existence * (1.0 - local_detection_probability)
    missed_factor = max(float(missed_factor), 1.0e-12)
    missed_existence = float(
        np.clip(
            (predicted_existence * (1.0 - local_detection_probability)) / missed_factor,
            0.0,
            1.0,
        )
    )
    branches = [
        _AssociationBranch(
            weight_factor=missed_factor,
            track_state=DeltaGLMBTrackState(
                existence_probability=missed_existence,
                mean=predicted_mean.astype(float, copy=True),
                covariance=predicted_covariance.astype(float, copy=True),
                motion_state=predicted_motion_state,
                missed_count=_next_missed_count(
                    predicted_track_state.missed_count,
                    has_measurement_support=False,
                    max_missed_steps=max_missed_steps,
                ),
            ),
            candidate_offset=-1,
            innovation_distance=np.inf,
        )
    ]

    if candidate_states.size == 0 or predicted_existence <= 0.0:
        return branches

    residuals = candidate_states - predicted_mean.reshape(1, STATE_DIM)
    squared_mahalanobis = np.sum(
        np.square(residuals) / innovation_covariance_diagonal.reshape(1, STATE_DIM),
        axis=1,
    )
    gated_offsets = np.flatnonzero(squared_mahalanobis <= float(gate_sigma) ** 2)
    if gated_offsets.size == 0:
        return branches

    gaussian_likelihoods = _gaussian_pdf_zero_mean_diag(
        residuals[gated_offsets],
        innovation_covariance_diagonal,
    )
    for gated_offset, gaussian_likelihood in zip(gated_offsets, gaussian_likelihoods):
        branch_weight = float(
            predicted_existence
            * local_detection_probability
            * float(gaussian_likelihood)
            / max(float(clutter_intensity), 1.0e-12)
        )
        if branch_weight <= 0.0:
            continue
        updated_motion_state, innovation_distance = _update_motion_state_with_measurement(
            predicted_motion_state,
            measurement_state=candidate_states[gated_offset],
            measurement_covariance_diagonal=local_covariance,
        )
        updated_mean, updated_covariance = _collapse_motion_state(updated_motion_state)
        branches.append(
            _AssociationBranch(
                weight_factor=branch_weight,
                track_state=DeltaGLMBTrackState(
                    existence_probability=1.0,
                    mean=updated_mean.astype(float, copy=True),
                    covariance=updated_covariance.astype(float, copy=True),
                    motion_state=updated_motion_state,
                    missed_count=0,
                ),
                candidate_offset=int(gated_offset),
                innovation_distance=float(innovation_distance),
            )
        )
    return branches


def _expand_joint_hypotheses(
    *,
    hypothesis_weight: float,
    branches_by_target: Sequence[Sequence[_AssociationBranch]],
    target_idx: int,
    used_candidates: set[int],
    current_track_states: list[DeltaGLMBTrackState],
    current_candidate_offsets: list[int],
    current_innovations: list[float],
    output_hypotheses: list[DeltaGLMBHypothesis],
) -> None:
    if target_idx >= len(branches_by_target):
        output_hypotheses.append(
            DeltaGLMBHypothesis(
                weight=float(hypothesis_weight),
                track_states=tuple(current_track_states),
                assigned_candidate_offsets=tuple(current_candidate_offsets),
                innovation_distances=tuple(current_innovations),
            )
        )
        return

    for branch in branches_by_target[target_idx]:
        candidate_offset = int(branch.candidate_offset)
        if candidate_offset >= 0 and candidate_offset in used_candidates:
            continue
        next_used_candidates = set(used_candidates)
        if candidate_offset >= 0:
            next_used_candidates.add(candidate_offset)
        current_track_states.append(branch.track_state)
        current_candidate_offsets.append(candidate_offset)
        current_innovations.append(float(branch.innovation_distance))
        _expand_joint_hypotheses(
            hypothesis_weight=float(hypothesis_weight) * float(branch.weight_factor),
            branches_by_target=branches_by_target,
            target_idx=target_idx + 1,
            used_candidates=next_used_candidates,
            current_track_states=current_track_states,
            current_candidate_offsets=current_candidate_offsets,
            current_innovations=current_innovations,
            output_hypotheses=output_hypotheses,
        )
        current_track_states.pop()
        current_candidate_offsets.pop()
        current_innovations.pop()


def _aggregate_track_posteriors(
    *,
    hypotheses: Sequence[DeltaGLMBHypothesis],
    prior_relative_states: np.ndarray,
    prior_covariance_diagonal: np.ndarray,
    local_relative_states: np.ndarray,
    valid_indices: np.ndarray,
    confidence_threshold: float,
    max_missed_steps: int,
) -> tuple[tuple[KeyTrackPosterior, ...], dict[int, DeltaGLMBTrackState]]:
    if prior_relative_states.shape[0] == 0:
        return tuple(), {}

    output_tracks: list[KeyTrackPosterior] = []
    collapsed_track_states: dict[int, DeltaGLMBTrackState] = {}
    for target_idx in range(prior_relative_states.shape[0]):
        full_weights = np.zeros(local_relative_states.shape[0], dtype=np.float32)
        mixture_means: list[np.ndarray] = []
        mixture_covariances: list[np.ndarray] = []
        mixture_motion_states: list[IMMMotionState] = []
        mixture_weights: list[float] = []
        mixture_missed_counts: list[int] = []
        detection_weight = 0.0
        innovation_weighted_sum = 0.0

        for hypothesis in hypotheses:
            track_state = hypothesis.track_states[target_idx]
            marginal_weight = float(hypothesis.weight) * float(track_state.existence_probability)
            if marginal_weight > 0.0:
                mixture_means.append(np.asarray(track_state.mean, dtype=float))
                mixture_covariances.append(np.asarray(track_state.covariance, dtype=float))
                if track_state.motion_state is not None:
                    mixture_motion_states.append(track_state.motion_state)
                mixture_weights.append(marginal_weight)
                mixture_missed_counts.append(_previous_missed_count(track_state))
            candidate_offset = int(hypothesis.assigned_candidate_offsets[target_idx])
            if candidate_offset >= 0 and candidate_offset < len(valid_indices):
                local_index = int(valid_indices[candidate_offset])
                full_weights[local_index] += float(hypothesis.weight)
                detection_weight += float(hypothesis.weight)
                innovation_distance = float(hypothesis.innovation_distances[target_idx])
                if np.isfinite(innovation_distance):
                    innovation_weighted_sum += float(hypothesis.weight) * innovation_distance

        posterior_existence = float(np.clip(np.sum(mixture_weights), 0.0, 1.0))
        if mixture_weights:
            posterior_mean, posterior_covariance = _gaussian_mixture_moments(
                mixture_means,
                mixture_covariances,
                mixture_weights,
            )
            if mixture_motion_states:
                posterior_motion_state = _merge_motion_hypotheses(
                    mixture_motion_states,
                    mixture_weights[: len(mixture_motion_states)],
                )
            else:
                posterior_motion_state = None
        else:
            posterior_mean = np.asarray(prior_relative_states[target_idx], dtype=float).copy()
            posterior_covariance = np.diag(np.asarray(prior_covariance_diagonal, dtype=float).reshape(STATE_DIM))
            posterior_motion_state = None

        confidence = float(np.clip(detection_weight, 0.0, 1.0))
        if confidence > 0.0:
            collapsed_missed_count = 0
        elif mixture_missed_counts:
            collapsed_missed_count = min(max(mixture_missed_counts), _sanitize_max_missed_steps(max_missed_steps))
        else:
            collapsed_missed_count = _sanitize_max_missed_steps(max_missed_steps)
        best_local_index = -1
        best_innovation_distance = np.inf
        if np.any(full_weights > 0.0):
            best_local_index = int(np.argmax(full_weights))
            if detection_weight > 0.0:
                best_innovation_distance = innovation_weighted_sum / detection_weight
        elif valid_indices.size:
            weight_support, nearest_index, nearest_distance = _nearest_measurement_support(
                posterior_mean,
                local_relative_states,
                valid_indices,
            )
            if confidence <= 0.0:
                full_weights = weight_support
            if nearest_index >= 0:
                best_local_index = int(nearest_index)
                best_innovation_distance = float(nearest_distance)

        valid = bool(posterior_existence >= confidence_threshold and np.any(full_weights > 0.0))
        output_tracks.append(
            KeyTrackPosterior(
                global_target_index=target_idx,
                prior_relative_state=np.asarray(prior_relative_states[target_idx], dtype=np.float32),
                prior_covariance_diagonal=np.asarray(prior_covariance_diagonal, dtype=np.float32),
                posterior_relative_state=np.asarray(posterior_mean, dtype=np.float32),
                posterior_covariance_diagonal=np.asarray(np.diag(posterior_covariance), dtype=np.float32),
                confidence=confidence,
                existence_probability=posterior_existence,
                valid=valid,
                local_candidate_weights=full_weights,
                local_candidate_index=best_local_index,
                innovation_distance=float(best_innovation_distance),
            )
        )
        collapsed_track_states[target_idx] = DeltaGLMBTrackState(
            existence_probability=posterior_existence,
            mean=np.asarray(posterior_mean, dtype=float).copy(),
            covariance=np.asarray(posterior_covariance, dtype=float).copy(),
            motion_state=posterior_motion_state,
            missed_count=collapsed_missed_count,
        )

    return tuple(output_tracks), collapsed_track_states


def delta_glmb_association_for_friendlies(
    *,
    friendly_states: np.ndarray,
    global_target_priors: GlobalKeyTargetStatePrior,
    local_observations: dict[str, LocalTargetObservation],
    agent_ids: Sequence[str],
    previous_filter_states: dict[str, DeltaGLMBFilterState] | None,
    detection_radius_m: float,
    relative_velocity_limit_mps: float,
    local_detection_probability: float,
    clutter_rate_per_second: float,
    recursive_motion_model: str = "imm",
    imm_model_types: Sequence[str] = ("cv", "ca", "ct_left", "ct_right"),
    imm_model_transition_matrix: np.ndarray | Sequence[Sequence[float]] | None = None,
    imm_initial_model_probabilities: np.ndarray | Sequence[float] | None = None,
    dt_seconds: float = 1.0,
    gate_sigma: float = 3.0,
    survival_probability: float = 0.98,
    birth_weight: float = 0.25,
    process_position_std_m: np.ndarray | Sequence[float] = (2.0, 2.0, 1.0),
    process_velocity_std_mps: np.ndarray | Sequence[float] = (1.0, 1.0, 0.5),
    imm_turn_rate_radps: float = 0.35,
    imm_ct_process_noise_scale: float = 1.25,
    confidence_threshold: float = 0.1,
    max_hypotheses: int = 48,
    hypothesis_weight_prune_threshold: float = 1.0e-4,
    max_missed_steps: int = 5,
) -> tuple[dict[str, MultiTargetTrackOutput], dict[str, DeltaGLMBFilterState]]:
    """Run a compact delta-GLMB-style single-sensor baseline for each friendly UAV."""

    friendly_array = _as_state_batch(friendly_states, name="friendly_states")
    if friendly_array.shape[0] != len(agent_ids):
        raise ValueError("friendly_states row count must match agent_ids length.")

    previous_states = previous_filter_states or {}
    process_position_std = np.asarray(process_position_std_m, dtype=float).reshape(3)
    process_velocity_std = np.asarray(process_velocity_std_mps, dtype=float).reshape(3)
    local_detection_probability = float(np.clip(local_detection_probability, 0.0, 1.0))
    confidence_threshold = float(np.clip(confidence_threshold, 0.0, 1.0))
    position_volume = (4.0 / 3.0) * np.pi * max(float(detection_radius_m), 1.0e-6) ** 3
    velocity_support = max(float(relative_velocity_limit_mps), 1.0e-6)
    velocity_volume = np.power(2.0 * velocity_support, 3)
    measurement_volume = max(position_volume * velocity_volume, 1.0e-9)
    clutter_rate_per_step = max(float(clutter_rate_per_second), 0.0) * max(float(dt_seconds), 1.0e-6)
    clutter_intensity = max(clutter_rate_per_step / measurement_volume, 1.0e-9)

    outputs: dict[str, MultiTargetTrackOutput] = {}
    next_filter_states: dict[str, DeltaGLMBFilterState] = {}
    for friendly_idx, agent_id in enumerate(agent_ids):
        prior_relative_states, prior_covariance = _build_prior_relative_states(
            friendly_array[friendly_idx],
            global_target_priors,
        )
        local_relative_states, local_mask, local_covariance = _validate_local_observation(local_observations[str(agent_id)])
        valid_indices = np.flatnonzero(local_mask > 0.0)
        candidate_states = local_relative_states[valid_indices] if valid_indices.size else np.zeros((0, STATE_DIM), dtype=float)

        if prior_relative_states.shape[0] == 0:
            outputs[str(agent_id)] = empty_multi_target_track_output(
                agent_id=str(agent_id),
                prior_relative_states=prior_relative_states,
                prior_covariance_diagonal=prior_covariance,
                mode="local_only",
            )
            next_filter_states[str(agent_id)] = DeltaGLMBFilterState(hypotheses=tuple())
            continue

        predicted_hypotheses = _predict_hypotheses(
            previous_filter_state=previous_states.get(str(agent_id)),
            prior_relative_states=prior_relative_states,
            prior_covariance_diagonal=prior_covariance,
            recursive_motion_model=recursive_motion_model,
            imm_model_types=imm_model_types,
            imm_model_transition_matrix=imm_model_transition_matrix,
            imm_initial_model_probabilities=imm_initial_model_probabilities,
            dt_seconds=dt_seconds,
            survival_probability=survival_probability,
            birth_weight=birth_weight,
            process_position_std=process_position_std,
            process_velocity_std=process_velocity_std,
            imm_turn_rate_radps=imm_turn_rate_radps,
            imm_ct_process_noise_scale=imm_ct_process_noise_scale,
            max_hypotheses=max_hypotheses,
            max_missed_steps=max_missed_steps,
        )

        updated_hypotheses: list[DeltaGLMBHypothesis] = []
        for predicted_hypothesis in predicted_hypotheses:
            branches_by_target = [
                _build_target_branches(
                    predicted_track_state=predicted_hypothesis.track_states[target_idx],
                    candidate_states=candidate_states,
                    local_covariance=local_covariance,
                    detection_radius_m=detection_radius_m,
                    local_detection_probability=local_detection_probability,
                    clutter_intensity=clutter_intensity,
                    gate_sigma=gate_sigma,
                    max_missed_steps=max_missed_steps,
                )
                for target_idx in range(len(predicted_hypothesis.track_states))
            ]
            _expand_joint_hypotheses(
                hypothesis_weight=float(predicted_hypothesis.weight),
                branches_by_target=branches_by_target,
                target_idx=0,
                used_candidates=set(),
                current_track_states=[],
                current_candidate_offsets=[],
                current_innovations=[],
                output_hypotheses=updated_hypotheses,
            )

        normalized_hypotheses = _normalize_and_prune_hypotheses(
            updated_hypotheses,
            max_hypotheses=max_hypotheses,
            weight_prune_threshold=hypothesis_weight_prune_threshold,
        )
        if not normalized_hypotheses:
            normalized_hypotheses = predicted_hypotheses

        agent_tracks, collapsed_track_states = _aggregate_track_posteriors(
            hypotheses=normalized_hypotheses,
            prior_relative_states=prior_relative_states,
            prior_covariance_diagonal=prior_covariance,
            local_relative_states=local_relative_states,
            valid_indices=valid_indices,
            confidence_threshold=confidence_threshold,
            max_missed_steps=max_missed_steps,
        )
        outputs[str(agent_id)] = MultiTargetTrackOutput(
            agent_id=str(agent_id),
            tracks=agent_tracks,
            mode="local_only",
        )
        next_filter_states[str(agent_id)] = DeltaGLMBFilterState(hypotheses=normalized_hypotheses)

    return outputs, next_filter_states
