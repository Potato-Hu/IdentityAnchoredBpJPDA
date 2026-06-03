from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.tracking.task_assignment import _solve_exact_assignment


def _compute_step_track_matches(
    truth_positions: np.ndarray,
    estimated_positions: np.ndarray,
    valid_track_indices: np.ndarray,
    *,
    cutoff_distance: float,
) -> np.ndarray:
    truth_array = np.asarray(truth_positions, dtype=float)
    estimate_array = np.asarray(estimated_positions, dtype=float)
    track_indices = np.asarray(valid_track_indices, dtype=int).reshape(-1)

    matches = np.full(truth_array.shape[0], -1, dtype=int)
    if truth_array.size == 0 or estimate_array.size == 0 or track_indices.size == 0:
        return matches

    pairwise_distances = np.linalg.norm(
        truth_array[:, None, :] - estimate_array[None, :, :],
        axis=-1,
    )
    gated_costs = np.where(pairwise_distances <= float(cutoff_distance), pairwise_distances, np.inf)
    assignments = _solve_exact_assignment(gated_costs)
    for truth_idx, estimate_compact_idx in enumerate(assignments):
        if estimate_compact_idx < 0:
            continue
        if not np.isfinite(gated_costs[truth_idx, estimate_compact_idx]):
            continue
        matches[truth_idx] = int(track_indices[estimate_compact_idx])
    return matches


def compute_key_target_rmse(
    truth_states: np.ndarray,
    estimated_states: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Return per-step, per-target position RMSE in meters.

    For a single rollout, the per-step RMSE reduces to the Euclidean position error
    of each target estimate relative to truth.
    """

    truth_array = np.asarray(truth_states, dtype=float)
    estimate_array = np.asarray(estimated_states, dtype=float)
    valid_array = np.asarray(valid_mask, dtype=bool)

    if truth_array.ndim != 3 or truth_array.shape[-1] < 3:
        raise ValueError(f"truth_states must have shape (T, K, >=3), got {truth_array.shape}.")
    if estimate_array.shape != truth_array.shape:
        raise ValueError("estimated_states must have the same shape as truth_states.")
    if valid_array.shape != truth_array.shape[:2]:
        raise ValueError(
            "valid_mask must have shape (T, K) matching the first two dimensions of truth_states."
        )

    rmse = np.full(truth_array.shape[:2], np.nan, dtype=float)
    for step_idx in range(truth_array.shape[0]):
        for target_idx in range(truth_array.shape[1]):
            if not bool(valid_array[step_idx, target_idx]):
                continue
            delta = estimate_array[step_idx, target_idx, :3] - truth_array[step_idx, target_idx, :3]
            rmse[step_idx, target_idx] = float(np.linalg.norm(delta))
    return rmse


def compute_key_target_ospa(
    truth_positions: np.ndarray,
    estimated_positions: np.ndarray,
    *,
    cutoff_distance: float,
    order: int = 2,
) -> float:
    """Return the OSPA distance between two key-target position sets."""

    truth_array = np.asarray(truth_positions, dtype=float)
    estimate_array = np.asarray(estimated_positions, dtype=float)
    if truth_array.ndim != 2 or truth_array.shape[1] != 3:
        raise ValueError(f"truth_positions must have shape (M, 3), got {truth_array.shape}.")
    if estimate_array.ndim != 2 or estimate_array.shape[1] != 3:
        raise ValueError(f"estimated_positions must have shape (N, 3), got {estimate_array.shape}.")
    if cutoff_distance <= 0.0:
        raise ValueError(f"cutoff_distance must be positive, got {cutoff_distance}.")
    if order <= 0:
        raise ValueError(f"order must be positive, got {order}.")

    num_truth = truth_array.shape[0]
    num_estimates = estimate_array.shape[0]
    normalizer = max(num_truth, num_estimates)
    if normalizer == 0:
        return 0.0
    if min(num_truth, num_estimates) == 0:
        return float(cutoff_distance)

    pairwise_distances = np.linalg.norm(
        truth_array[:, None, :] - estimate_array[None, :, :],
        axis=-1,
    )
    capped_costs = np.minimum(pairwise_distances, float(cutoff_distance)) ** int(order)
    assignments = _solve_exact_assignment(capped_costs)
    matched_cost = 0.0
    for truth_idx, estimate_idx in enumerate(assignments):
        if estimate_idx < 0:
            continue
        matched_cost += float(capped_costs[truth_idx, estimate_idx])

    cardinality_cost = abs(num_truth - num_estimates) * float(cutoff_distance) ** int(order)
    total_cost = (matched_cost + cardinality_cost) / float(normalizer)
    return float(total_cost ** (1.0 / int(order)))


def evaluate_key_target_tracking_episode(
    truth_states: np.ndarray,
    estimated_states: np.ndarray,
    valid_mask: np.ndarray,
    *,
    cutoff_distance: float,
    order: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-target RMSE and per-step total OSPA."""

    truth_array = np.asarray(truth_states, dtype=float)
    estimate_array = np.asarray(estimated_states, dtype=float)
    valid_array = np.asarray(valid_mask, dtype=bool)

    rmse = compute_key_target_rmse(
        truth_states=truth_array,
        estimated_states=estimate_array,
        valid_mask=valid_array,
    )
    ospa = np.zeros(truth_array.shape[0], dtype=float)
    for step_idx in range(truth_array.shape[0]):
        estimated_positions = estimate_array[step_idx, valid_array[step_idx], :3]
        ospa[step_idx] = compute_key_target_ospa(
            truth_positions=truth_array[step_idx, :, :3],
            estimated_positions=estimated_positions,
            cutoff_distance=cutoff_distance,
            order=order,
        )
    return rmse, ospa


def compute_key_target_id_switches(
    truth_states: np.ndarray,
    estimated_states: np.ndarray,
    valid_mask: np.ndarray,
    *,
    cutoff_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cumulative and per-target ID-switch counts.

    This follows the MOT / multi-target-tracking convention of counting a switch when the same
    ground-truth target is matched to a different estimated track on consecutive matched steps.
    """

    truth_array = np.asarray(truth_states, dtype=float)
    estimate_array = np.asarray(estimated_states, dtype=float)
    valid_array = np.asarray(valid_mask, dtype=bool)

    num_steps, num_targets = truth_array.shape[:2]
    per_target_counts = np.zeros(num_targets, dtype=int)
    cumulative_total = np.zeros(num_steps, dtype=int)
    previous_match = np.full(num_targets, -1, dtype=int)
    previous_step_matched = np.zeros(num_targets, dtype=bool)

    for step_idx in range(num_steps):
        valid_track_indices = np.flatnonzero(valid_array[step_idx])
        matched_track_indices = _compute_step_track_matches(
            truth_positions=truth_array[step_idx, :, :3],
            estimated_positions=estimate_array[step_idx, valid_track_indices, :3],
            valid_track_indices=valid_track_indices,
            cutoff_distance=cutoff_distance,
        )
        for target_idx in range(num_targets):
            current_match = int(matched_track_indices[target_idx])
            if current_match >= 0 and previous_step_matched[target_idx] and previous_match[target_idx] >= 0:
                if current_match != previous_match[target_idx]:
                    per_target_counts[target_idx] += 1
            previous_match[target_idx] = current_match
            previous_step_matched[target_idx] = bool(current_match >= 0)
        cumulative_total[step_idx] = int(np.sum(per_target_counts))

    return cumulative_total, per_target_counts


def compute_key_target_fragmentations(
    truth_states: np.ndarray,
    estimated_states: np.ndarray,
    valid_mask: np.ndarray,
    *,
    cutoff_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cumulative and per-target fragmentation counts.

    Fragmentation is a standard MOT diagnostic that counts how many times a target track
    changes from tracked to untracked and later becomes tracked again.
    """

    truth_array = np.asarray(truth_states, dtype=float)
    estimate_array = np.asarray(estimated_states, dtype=float)
    valid_array = np.asarray(valid_mask, dtype=bool)

    num_steps, num_targets = truth_array.shape[:2]
    per_target_counts = np.zeros(num_targets, dtype=int)
    cumulative_total = np.zeros(num_steps, dtype=int)
    has_been_tracked = np.zeros(num_targets, dtype=bool)
    open_gap = np.zeros(num_targets, dtype=bool)

    for step_idx in range(num_steps):
        valid_track_indices = np.flatnonzero(valid_array[step_idx])
        matched_track_indices = _compute_step_track_matches(
            truth_positions=truth_array[step_idx, :, :3],
            estimated_positions=estimate_array[step_idx, valid_track_indices, :3],
            valid_track_indices=valid_track_indices,
            cutoff_distance=cutoff_distance,
        )
        for target_idx in range(num_targets):
            currently_tracked = bool(matched_track_indices[target_idx] >= 0)
            if currently_tracked:
                if has_been_tracked[target_idx] and open_gap[target_idx]:
                    per_target_counts[target_idx] += 1
                has_been_tracked[target_idx] = True
                open_gap[target_idx] = False
            elif has_been_tracked[target_idx]:
                open_gap[target_idx] = True
        cumulative_total[step_idx] = int(np.sum(per_target_counts))

    return cumulative_total, per_target_counts


def export_key_target_metrics_csv(
    truth_states: np.ndarray,
    estimated_states: np.ndarray,
    valid_mask: np.ndarray,
    *,
    dt: float,
    bounds: np.ndarray,
    save_path: str | Path,
) -> Path:
    """Export per-target RMSE, key-target OSPA, and common MOT diagnostics indexed by time step."""

    truth_array = np.asarray(truth_states, dtype=float)
    estimate_array = np.asarray(estimated_states, dtype=float)
    valid_array = np.asarray(valid_mask, dtype=bool)
    bounds_array = np.asarray(bounds, dtype=float).reshape(-1)
    if bounds_array.size < 1:
        raise ValueError("bounds must contain at least one axis value.")

    cutoff_distance = float(bounds_array[0]) / 4.0
    rmse, ospa = evaluate_key_target_tracking_episode(
        truth_states=truth_array,
        estimated_states=estimate_array,
        valid_mask=valid_array,
        cutoff_distance=cutoff_distance,
        order=2,
    )
    id_switches_cumulative, _ = compute_key_target_id_switches(
        truth_states=truth_array,
        estimated_states=estimate_array,
        valid_mask=valid_array,
        cutoff_distance=cutoff_distance,
    )
    fragmentations_cumulative, _ = compute_key_target_fragmentations(
        truth_states=truth_array,
        estimated_states=estimate_array,
        valid_mask=valid_array,
        cutoff_distance=cutoff_distance,
    )

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_steps, num_targets = rmse.shape
    header = ["timestep", "time_s"]
    header.extend(f"E{target_idx + 1}_rmse_m" for target_idx in range(num_targets))
    header.append("key_target_ospa_m")
    header.append("id_switches_cumulative")
    header.append("track_fragmentations_cumulative")

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        for step_idx in range(num_steps):
            row: list[float | int | str] = [step_idx, step_idx * float(dt)]
            for target_idx in range(num_targets):
                value = rmse[step_idx, target_idx]
                row.append("" if not np.isfinite(value) else f"{value:.6f}")
            row.append(f"{ospa[step_idx]:.6f}")
            row.append(int(id_switches_cumulative[step_idx]))
            row.append(int(fragmentations_cumulative[step_idx]))
            writer.writerow(row)

    return output_path
