from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.sensing.local_sensor import LocalTargetObservation


@dataclass(frozen=True)
class AssociationOutput:
    """Target association result for one friendly UAV."""

    target_est_relative_position: np.ndarray
    confidence: float
    valid: bool
    local_candidate_weights: np.ndarray | None = None
    local_candidate_index: int = -1
    global_target_index: int = -1
    association_distance_m: float = np.inf


def empty_association_output() -> AssociationOutput:
    return AssociationOutput(
        target_est_relative_position=np.zeros(3, dtype=np.float32),
        confidence=0.0,
        valid=False,
    )


def association_output_to_weight_vector(
    association_output: AssociationOutput,
    *,
    max_local_candidates: int,
) -> np.ndarray:
    """Encode association output as a fixed-length candidate weight vector."""

    candidate_count = int(max_local_candidates)
    if candidate_count < 0:
        raise ValueError(f"max_local_candidates must be >= 0, got {candidate_count}.")

    weights = np.zeros(candidate_count, dtype=np.float32)
    if association_output.local_candidate_weights is not None:
        raw_weights = np.asarray(association_output.local_candidate_weights, dtype=np.float32).reshape(-1)
        copy_count = min(candidate_count, raw_weights.shape[0])
        weights[:copy_count] = np.clip(raw_weights[:copy_count], 0.0, 1.0)
        return weights

    if not association_output.valid:
        return weights

    selected_index = int(association_output.local_candidate_index)
    if 0 <= selected_index < candidate_count:
        weights[selected_index] = 1.0
    return weights


def nearest_neighbor_association(
    *,
    friendly_position: np.ndarray,
    global_target_position: np.ndarray,
    local_observation: LocalTargetObservation,
    global_target_index: int = -1,
) -> AssociationOutput:
    """Match one noisy global target prior to the nearest local candidate."""

    friendly = np.asarray(friendly_position, dtype=float).reshape(3)
    global_target = np.asarray(global_target_position, dtype=float).reshape(3)
    local_relative_positions = np.asarray(local_observation.relative_positions, dtype=float)
    local_mask = np.asarray(local_observation.mask, dtype=float).reshape(-1)
    if local_relative_positions.ndim != 2 or local_relative_positions.shape[1] != 3:
        raise ValueError(
            "local_observation.relative_positions must have shape (N, 3), "
            f"got {local_relative_positions.shape}."
        )
    if local_relative_positions.shape[0] != local_mask.shape[0]:
        raise ValueError(
            "local_observation.mask length must match relative_positions rows, "
            f"got {local_mask.shape[0]} and {local_relative_positions.shape[0]}."
        )

    valid_indices = np.flatnonzero(local_mask > 0.0)
    if valid_indices.size == 0:
        return empty_association_output()

    global_relative = global_target - friendly
    distances = np.linalg.norm(local_relative_positions[valid_indices] - global_relative, axis=1)
    nearest_offset = int(np.argmin(distances))
    nearest_index = int(valid_indices[nearest_offset])
    nearest_distance = float(distances[nearest_offset])
    return AssociationOutput(
        target_est_relative_position=local_relative_positions[nearest_index].astype(np.float32),
        confidence=1.0,
        valid=True,
        local_candidate_weights=np.eye(local_relative_positions.shape[0], dtype=np.float32)[nearest_index],
        local_candidate_index=nearest_index,
        global_target_index=int(global_target_index),
        association_distance_m=nearest_distance,
    )


def nearest_local_candidate_association(local_observation: LocalTargetObservation) -> AssociationOutput:
    """Select the nearest valid local candidate without using global priors."""

    local_relative_positions = np.asarray(local_observation.relative_positions, dtype=float)
    local_mask = np.asarray(local_observation.mask, dtype=float).reshape(-1)
    if local_relative_positions.ndim != 2 or local_relative_positions.shape[1] != 3:
        raise ValueError(
            "local_observation.relative_positions must have shape (N, 3), "
            f"got {local_relative_positions.shape}."
        )
    if local_relative_positions.shape[0] != local_mask.shape[0]:
        raise ValueError(
            "local_observation.mask length must match relative_positions rows, "
            f"got {local_mask.shape[0]} and {local_relative_positions.shape[0]}."
        )

    valid_indices = np.flatnonzero(local_mask > 0.0)
    if valid_indices.size == 0:
        return empty_association_output()

    distances = np.linalg.norm(local_relative_positions[valid_indices], axis=1)
    nearest_offset = int(np.argmin(distances))
    nearest_index = int(valid_indices[nearest_offset])
    return AssociationOutput(
        target_est_relative_position=local_relative_positions[nearest_index].astype(np.float32),
        confidence=1.0,
        valid=True,
        local_candidate_weights=np.eye(local_relative_positions.shape[0], dtype=np.float32)[nearest_index],
        local_candidate_index=nearest_index,
        global_target_index=-1,
        association_distance_m=float(distances[nearest_offset]),
    )


def nearest_local_candidate_association_for_friendlies(
    *,
    local_observations: dict[str, LocalTargetObservation],
    agent_ids: list[str],
) -> dict[str, AssociationOutput]:
    """Select the nearest local candidate for each friendly UAV."""

    return {
        agent_id: nearest_local_candidate_association(local_observations[agent_id])
        for agent_id in agent_ids
    }


def nearest_neighbor_association_for_friendlies(
    *,
    friendly_positions: np.ndarray,
    global_target_positions: np.ndarray,
    local_observations: dict[str, LocalTargetObservation],
    agent_ids: list[str],
    detection_radius_m: float,
) -> dict[str, AssociationOutput]:
    """Run global-to-local nearest-neighbor association for each friendly UAV."""

    friendly_array = np.asarray(friendly_positions, dtype=float)
    global_targets = np.asarray(global_target_positions, dtype=float)
    detection_radius = float(detection_radius_m)
    if detection_radius < 0.0:
        raise ValueError(f"detection_radius_m must be >= 0, got {detection_radius}.")
    if friendly_array.ndim != 2 or friendly_array.shape[1] != 3:
        raise ValueError(f"friendly_positions must have shape (N, 3), got {friendly_array.shape}.")
    if global_targets.ndim != 2 or global_targets.shape[1] != 3:
        raise ValueError(f"global_target_positions must have shape (M, 3), got {global_targets.shape}.")
    if friendly_array.shape[0] != len(agent_ids):
        raise ValueError("friendly_positions row count must match agent_ids length.")

    outputs: dict[str, AssociationOutput] = {}
    for friendly_idx, agent_id in enumerate(agent_ids):
        if global_targets.shape[0] == 0:
            outputs[agent_id] = empty_association_output()
            continue

        friendly_position = friendly_array[friendly_idx]
        prior_distances = np.linalg.norm(global_targets - friendly_position, axis=1)
        in_range_target_indices = np.flatnonzero(prior_distances <= detection_radius)
        if in_range_target_indices.size == 0:
            outputs[agent_id] = empty_association_output()
            continue

        best_output = empty_association_output()
        for target_idx in in_range_target_indices:
            candidate_output = nearest_neighbor_association(
                friendly_position=friendly_position,
                global_target_position=global_targets[int(target_idx)],
                local_observation=local_observations[agent_id],
                global_target_index=int(target_idx),
            )
            if not candidate_output.valid:
                continue
            if not best_output.valid or candidate_output.association_distance_m < best_output.association_distance_m:
                best_output = candidate_output
        outputs[agent_id] = best_output
    return outputs


def _gaussian_pdf_zero_mean(samples: np.ndarray, variance: float) -> np.ndarray:
    if variance <= 0.0:
        raise ValueError(f"variance must be > 0, got {variance}.")
    squared_norm = np.sum(np.square(samples), axis=1)
    normalizer = np.power(2.0 * np.pi * variance, samples.shape[1] / 2.0)
    return np.exp(-0.5 * squared_norm / variance) / normalizer


def _run_bp_jpda(
    likelihood_ratios: np.ndarray,
    *,
    max_iterations: int,
    convergence_tol: float,
) -> np.ndarray:
    """Approximate JPDA marginals with the bipartite BP fixed-point updates."""

    psi = np.asarray(likelihood_ratios, dtype=float)
    if psi.size == 0:
        return np.zeros_like(psi, dtype=float)

    num_targets, num_measurements = psi.shape
    incoming_messages = np.ones((num_measurements, num_targets), dtype=float)
    eps = 1.0e-9
    for _ in range(max_iterations):
        previous_messages = incoming_messages.copy()
        outgoing_messages = np.zeros((num_targets, num_measurements), dtype=float)
        for target_idx in range(num_targets):
            weighted = psi[target_idx] * incoming_messages[:, target_idx]
            total_weight = float(np.sum(weighted))
            denominators = 1.0 + np.maximum(total_weight - weighted, 0.0)
            outgoing_messages[target_idx] = 1.0 / np.maximum(denominators, eps)

        for measurement_idx in range(num_measurements):
            weighted = psi[:, measurement_idx] * outgoing_messages[:, measurement_idx]
            total_weight = float(np.sum(weighted))
            denominators = 1.0 + np.maximum(total_weight - weighted, 0.0)
            incoming_messages[measurement_idx] = 1.0 / np.maximum(denominators, eps)

        max_delta = float(np.max(np.abs(incoming_messages - previous_messages)))
        if max_delta <= convergence_tol:
            break

    marginals = np.zeros_like(psi, dtype=float)
    for target_idx in range(num_targets):
        weighted = psi[target_idx] * incoming_messages[:, target_idx]
        denominator = 1.0 + float(np.sum(weighted))
        marginals[target_idx] = weighted / max(denominator, eps)
    return marginals


def bp_jpda_association_for_friendlies(
    *,
    friendly_positions: np.ndarray,
    global_target_positions: np.ndarray,
    local_observations: dict[str, LocalTargetObservation],
    agent_ids: list[str],
    detection_radius_m: float,
    local_position_noise_std_m: float,
    global_position_noise_std_m: float,
    prior_delay_seconds: float,
    target_speed_std_mps: float,
    local_detection_probability: float,
    clutter_rate_per_second: float,
    dt_seconds: float,
    gate_sigma: float = 3.0,
    max_iterations: int = 20,
    convergence_tol: float = 1.0e-3,
    min_missed_detection_probability: float = 0.05,
) -> dict[str, AssociationOutput]:
    """Run BP-JPDA using delayed global priors as per-target predicted states.

    Each friendly UAV solves an independent local association problem:
    - targets are the delayed/noisy global key priors projected into that UAV's frame
    - measurements are the anonymous local candidates returned by the onboard sensor
    - BP approximates JPDA marginal probabilities over the target-measurement bipartite graph
    """

    friendly_array = np.asarray(friendly_positions, dtype=float)
    global_targets = np.asarray(global_target_positions, dtype=float)
    detection_radius = float(detection_radius_m)
    if detection_radius < 0.0:
        raise ValueError(f"detection_radius_m must be >= 0, got {detection_radius}.")
    if friendly_array.ndim != 2 or friendly_array.shape[1] != 3:
        raise ValueError(f"friendly_positions must have shape (N, 3), got {friendly_array.shape}.")
    if global_targets.ndim != 2 or global_targets.shape[1] != 3:
        raise ValueError(f"global_target_positions must have shape (M, 3), got {global_targets.shape}.")
    if friendly_array.shape[0] != len(agent_ids):
        raise ValueError("friendly_positions row count must match agent_ids length.")
    if max_iterations <= 0:
        raise ValueError(f"max_iterations must be > 0, got {max_iterations}.")
    if convergence_tol <= 0.0:
        raise ValueError(f"convergence_tol must be > 0, got {convergence_tol}.")
    if gate_sigma <= 0.0:
        raise ValueError(f"gate_sigma must be > 0, got {gate_sigma}.")

    local_detection_probability = float(np.clip(local_detection_probability, 0.0, 1.0))
    missed_detection_probability = max(1.0 - local_detection_probability, float(min_missed_detection_probability))
    delay_position_std = max(float(target_speed_std_mps), 0.0) * max(float(prior_delay_seconds), 0.0) / np.sqrt(3.0)
    fused_variance = (
        float(local_position_noise_std_m) ** 2
        + float(global_position_noise_std_m) ** 2
        + delay_position_std**2
    )
    fused_variance = max(fused_variance, 1.0e-6)
    gate_distance = float(gate_sigma) * np.sqrt(fused_variance)
    detection_volume = (4.0 / 3.0) * np.pi * max(detection_radius, 1.0e-6) ** 3
    clutter_rate_per_step = max(float(clutter_rate_per_second), 0.0) * max(float(dt_seconds), 1.0e-6)
    clutter_intensity = max(clutter_rate_per_step / detection_volume, 1.0e-9)

    outputs: dict[str, AssociationOutput] = {}
    for friendly_idx, agent_id in enumerate(agent_ids):
        local_observation = local_observations[agent_id]
        local_relative_positions = np.asarray(local_observation.relative_positions, dtype=float)
        local_mask = np.asarray(local_observation.mask, dtype=float).reshape(-1)
        candidate_count = local_relative_positions.shape[0]
        if candidate_count == 0 or global_targets.shape[0] == 0:
            outputs[agent_id] = empty_association_output()
            continue

        valid_local_indices = np.flatnonzero(local_mask > 0.0)
        if valid_local_indices.size == 0:
            outputs[agent_id] = empty_association_output()
            continue

        friendly_position = friendly_array[friendly_idx]
        local_valid_positions = local_relative_positions[valid_local_indices]
        prior_distances = np.linalg.norm(global_targets - friendly_position, axis=1)
        prior_gate_margin = detection_radius + gate_distance
        usable_target_indices = np.flatnonzero(prior_distances <= prior_gate_margin)
        if usable_target_indices.size == 0:
            outputs[agent_id] = empty_association_output()
            continue

        global_relative_positions = global_targets[usable_target_indices] - friendly_position
        likelihood_ratios = np.zeros((usable_target_indices.size, valid_local_indices.size), dtype=float)
        for local_offset, local_relative in enumerate(local_valid_positions):
            residuals = local_relative.reshape(1, 3) - global_relative_positions
            distances = np.linalg.norm(residuals, axis=1)
            gated = distances <= gate_distance
            if not np.any(gated):
                continue
            gaussian_likelihood = _gaussian_pdf_zero_mean(residuals[gated], fused_variance)
            likelihood_ratios[gated, local_offset] = (
                local_detection_probability * gaussian_likelihood / clutter_intensity
            ) / missed_detection_probability

        if not np.any(likelihood_ratios > 0.0):
            outputs[agent_id] = empty_association_output()
            continue

        marginals = _run_bp_jpda(
            likelihood_ratios,
            max_iterations=max_iterations,
            convergence_tol=convergence_tol,
        )
        aggregated_weights = np.sum(marginals, axis=0)
        aggregated_weights = np.clip(aggregated_weights, 0.0, 1.0)
        full_weights = np.zeros(candidate_count, dtype=np.float32)
        full_weights[valid_local_indices] = aggregated_weights.astype(np.float32)

        best_local_offset = int(np.argmax(aggregated_weights))
        best_weight = float(aggregated_weights[best_local_offset])
        if best_weight <= 0.0:
            outputs[agent_id] = empty_association_output()
            continue

        winning_target_offset = int(np.argmax(marginals[:, best_local_offset]))
        best_local_index = int(valid_local_indices[best_local_offset])
        weighted_relative_position = np.average(
            local_valid_positions,
            axis=0,
            weights=np.maximum(aggregated_weights, 1.0e-9),
        )
        target_residual = local_valid_positions[best_local_offset] - global_relative_positions[winning_target_offset]
        outputs[agent_id] = AssociationOutput(
            target_est_relative_position=weighted_relative_position.astype(np.float32),
            confidence=best_weight,
            valid=True,
            local_candidate_weights=full_weights,
            local_candidate_index=best_local_index,
            global_target_index=int(usable_target_indices[winning_target_offset]),
            association_distance_m=float(np.linalg.norm(target_residual)),
        )
    return outputs
