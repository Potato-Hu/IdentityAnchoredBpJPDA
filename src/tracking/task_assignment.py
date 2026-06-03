from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Sequence

import numpy as np

from src.tracking.fusion import FusedTrackSet


@dataclass(frozen=True)
class AssignedTrack:
    agent_id: str
    key_track_id: str | None = None
    key_label: int | None = None
    assigned: bool = False
    note: str = "placeholder"
    assignment_cost: float = np.inf


@dataclass(frozen=True)
class TaskAssignmentResult:
    timestamp: int
    assignments: tuple[AssignedTrack, ...]


def _solve_exact_assignment(cost_matrix: np.ndarray) -> list[int]:
    costs = np.asarray(cost_matrix, dtype=float)
    if costs.ndim != 2:
        raise ValueError(f"cost_matrix must be 2D, got {costs.shape}.")

    num_rows, num_cols = costs.shape
    assignments = [-1] * num_rows
    if num_rows == 0 or num_cols == 0:
        return assignments
    if num_cols < num_rows or num_rows > 7:
        return _solve_greedy_assignment(costs)

    best_cost = np.inf
    best_perm: tuple[int, ...] | None = None
    for perm in permutations(range(num_cols), num_rows):
        total_cost = 0.0
        valid = True
        for row_idx, col_idx in enumerate(perm):
            cost = float(costs[row_idx, col_idx])
            if not np.isfinite(cost):
                valid = False
                break
            total_cost += cost
        if valid and total_cost < best_cost:
            best_cost = total_cost
            best_perm = perm

    if best_perm is None:
        return _solve_greedy_assignment(costs)
    return [int(col_idx) for col_idx in best_perm]


def _solve_greedy_assignment(cost_matrix: np.ndarray) -> list[int]:
    costs = np.asarray(cost_matrix, dtype=float)
    num_rows, num_cols = costs.shape
    assignments = [-1] * num_rows
    used_cols: set[int] = set()
    flat_pairs = sorted(
        (
            (float(costs[row_idx, col_idx]), row_idx, col_idx)
            for row_idx in range(num_rows)
            for col_idx in range(num_cols)
            if np.isfinite(costs[row_idx, col_idx])
        ),
        key=lambda item: item[0],
    )
    for _, row_idx, col_idx in flat_pairs:
        if assignments[row_idx] >= 0 or col_idx in used_cols:
            continue
        assignments[row_idx] = int(col_idx)
        used_cols.add(int(col_idx))
    return assignments


def build_empty_task_assignment(agent_ids: Sequence[str], *, timestamp: int) -> TaskAssignmentResult:
    """Return an explicit placeholder assignment result with no assigned key tracks."""

    return TaskAssignmentResult(
        timestamp=int(timestamp),
        assignments=tuple(AssignedTrack(agent_id=str(agent_id)) for agent_id in agent_ids),
    )


def build_hungarian_task_assignment(
    agent_ids: Sequence[str],
    *,
    timestamp: int,
    friendly_states: np.ndarray,
    fused_track_set: FusedTrackSet,
    prediction_dt: float = 0.0,
    switch_penalty: float = 0.0,
    previous_assignment: TaskAssignmentResult | None = None,
) -> TaskAssignmentResult:
    """Assign friendly UAVs to fused key tracks using predicted target positions."""

    friendly_array = np.asarray(friendly_states, dtype=float)
    if friendly_array.ndim != 2 or friendly_array.shape[1] < 3:
        raise ValueError(f"friendly_states must have shape (N, >=3), got {friendly_array.shape}.")
    if friendly_array.shape[0] != len(agent_ids):
        raise ValueError("friendly_states row count must match agent_ids length.")

    valid_tracks = [track for track in fused_track_set.tracks if bool(track.valid)]
    candidate_tracks = valid_tracks if valid_tracks else list(fused_track_set.tracks)
    if not candidate_tracks:
        return build_empty_task_assignment(agent_ids, timestamp=timestamp)

    target_positions = []
    for track in candidate_tracks:
        track_state = np.asarray(track.fused_state_mean, dtype=float).reshape(-1)
        predicted_position = track_state[:3].copy()
        if track_state.shape[0] >= 6 and prediction_dt > 0.0:
            predicted_position += float(prediction_dt) * track_state[3:6]
        target_positions.append(predicted_position)
    target_position_array = np.asarray(target_positions, dtype=float)
    pairwise_distances = np.linalg.norm(
        friendly_array[:, None, :3] - target_position_array[None, :, :],
        axis=-1,
    )

    pairwise_costs = pairwise_distances.copy()
    if previous_assignment is not None and switch_penalty > 0.0:
        previous_target_by_agent = {
            str(item.agent_id): str(item.key_track_id)
            for item in previous_assignment.assignments
            if bool(item.assigned) and item.key_track_id is not None
        }
        for friendly_idx, agent_id in enumerate(agent_ids):
            previous_target_id = previous_target_by_agent.get(str(agent_id))
            if previous_target_id is None:
                continue
            for target_idx, track in enumerate(candidate_tracks):
                if str(track.key_label) == previous_target_id:
                    continue
                pairwise_costs[friendly_idx, target_idx] += float(switch_penalty)

    num_friendlies = len(agent_ids)
    num_targets = len(candidate_tracks)
    agent_to_target = np.full(num_friendlies, -1, dtype=int)
    agent_assignment_costs = np.full(num_friendlies, np.inf, dtype=float)

    primary_assignments = _solve_exact_assignment(pairwise_costs.T)
    for target_idx, friendly_idx in enumerate(primary_assignments):
        if friendly_idx < 0:
            continue
        agent_to_target[friendly_idx] = int(target_idx)
        agent_assignment_costs[friendly_idx] = float(pairwise_costs[friendly_idx, target_idx])

    unassigned_friendlies = np.flatnonzero(agent_to_target < 0)
    if num_targets > 0:
        for friendly_idx in unassigned_friendlies:
            nearest_target_idx = int(np.argmin(pairwise_costs[friendly_idx]))
            agent_to_target[friendly_idx] = nearest_target_idx
            agent_assignment_costs[friendly_idx] = float(pairwise_costs[friendly_idx, nearest_target_idx])

    assignments: list[AssignedTrack] = []
    for friendly_idx, agent_id in enumerate(agent_ids):
        target_idx = int(agent_to_target[friendly_idx])
        if target_idx < 0:
            assignments.append(AssignedTrack(agent_id=str(agent_id)))
            continue
        track = candidate_tracks[target_idx]
        assignments.append(
            AssignedTrack(
                agent_id=str(agent_id),
                key_track_id=str(track.key_label),
                key_label=int(track.global_target_index),
                assigned=True,
                note="hungarian_predicted_switch",
                assignment_cost=float(agent_assignment_costs[friendly_idx]),
            )
        )

    return TaskAssignmentResult(
        timestamp=int(timestamp),
        assignments=tuple(assignments),
    )
