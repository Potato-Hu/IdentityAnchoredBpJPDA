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
    local_candidate_index: int = -1
    global_target_index: int = -1
    association_distance_m: float = np.inf


def empty_association_output() -> AssociationOutput:
    return AssociationOutput(
        target_est_relative_position=np.zeros(3, dtype=np.float32),
        confidence=0.0,
        valid=False,
    )


def association_output_to_one_hot(
    association_output: AssociationOutput,
    *,
    max_local_candidates: int,
) -> np.ndarray:
    """Encode the selected local candidate as a one-hot vector."""

    candidate_count = int(max_local_candidates)
    if candidate_count < 0:
        raise ValueError(f"max_local_candidates must be >= 0, got {candidate_count}.")

    one_hot = np.zeros(candidate_count, dtype=np.float32)
    if not association_output.valid:
        return one_hot

    selected_index = int(association_output.local_candidate_index)
    if 0 <= selected_index < candidate_count:
        one_hot[selected_index] = 1.0
    return one_hot


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
