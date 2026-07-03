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


def compute_key_target_gospa(
    truth_positions: np.ndarray,
    estimated_positions: np.ndarray,
    *,
    cutoff_distance: float,
    order: int = 2,
    alpha: float = 2.0,
) -> float:
    """Return the GOSPA distance between two key-target position sets."""

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
    if not 0.0 < float(alpha) <= 2.0:
        raise ValueError(f"alpha must lie in (0, 2], got {alpha}.")

    num_truth = truth_array.shape[0]
    num_estimates = estimate_array.shape[0]
    if num_truth == 0 and num_estimates == 0:
        return 0.0

    if num_truth == 0 or num_estimates == 0:
        missed_or_false = num_truth + num_estimates
        total_cost = (float(cutoff_distance) ** int(order) / float(alpha)) * float(missed_or_false)
        return float(total_cost ** (1.0 / int(order)))

    pairwise_distances = np.linalg.norm(
        truth_array[:, None, :] - estimate_array[None, :, :],
        axis=-1,
    )
    gated_costs = np.where(pairwise_distances < float(cutoff_distance), pairwise_distances ** int(order), np.inf)

    if num_truth <= num_estimates:
        assignments = _solve_exact_assignment(gated_costs)
        matched_pairs = [
            (truth_idx, estimate_idx)
            for truth_idx, estimate_idx in enumerate(assignments)
            if estimate_idx >= 0 and np.isfinite(gated_costs[truth_idx, estimate_idx])
        ]
    else:
        assignments = _solve_exact_assignment(gated_costs.T)
        matched_pairs = [
            (truth_idx, estimate_idx)
            for estimate_idx, truth_idx in enumerate(assignments)
            if truth_idx >= 0 and np.isfinite(gated_costs[truth_idx, estimate_idx])
        ]

    matched_count = len(matched_pairs)
    localization_cost = 0.0
    for truth_idx, estimate_idx in matched_pairs:
        localization_cost += float(gated_costs[truth_idx, estimate_idx])

    missed_count = num_truth - matched_count
    false_count = num_estimates - matched_count
    cardinality_cost = (
        float(cutoff_distance) ** int(order) / float(alpha)
    ) * float(missed_count + false_count)
    total_cost = localization_cost + cardinality_cost
    return float(total_cost ** (1.0 / int(order)))


def evaluate_key_target_tracking_episode(
    truth_states: np.ndarray,
    estimated_states: np.ndarray,
    valid_mask: np.ndarray,
    *,
    cutoff_distance: float,
    order: int = 2,
 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-target RMSE plus per-step OSPA and GOSPA."""

    truth_array = np.asarray(truth_states, dtype=float)
    estimate_array = np.asarray(estimated_states, dtype=float)
    valid_array = np.asarray(valid_mask, dtype=bool)

    rmse = compute_key_target_rmse(
        truth_states=truth_array,
        estimated_states=estimate_array,
        valid_mask=valid_array,
    )
    ospa = np.zeros(truth_array.shape[0], dtype=float)
    gospa = np.zeros(truth_array.shape[0], dtype=float)
    for step_idx in range(truth_array.shape[0]):
        estimated_positions = estimate_array[step_idx, valid_array[step_idx], :3]
        ospa[step_idx] = compute_key_target_ospa(
            truth_positions=truth_array[step_idx, :, :3],
            estimated_positions=estimated_positions,
            cutoff_distance=cutoff_distance,
            order=order,
        )
        gospa[step_idx] = compute_key_target_gospa(
            truth_positions=truth_array[step_idx, :, :3],
            estimated_positions=estimated_positions,
            cutoff_distance=cutoff_distance,
            order=order,
            alpha=2.0,
        )
    return rmse, ospa, gospa


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


def compute_key_target_tracking_coverage(
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the per-step tracked key-target count and coverage ratio."""

    valid_array = np.asarray(valid_mask, dtype=bool)
    if valid_array.ndim != 2:
        raise ValueError(f"valid_mask must have shape (T, K), got {valid_array.shape}.")

    tracked_counts = np.sum(valid_array, axis=1, dtype=int)
    num_targets = int(valid_array.shape[1])
    if num_targets <= 0:
        tracked_ratio = np.zeros(valid_array.shape[0], dtype=float)
    else:
        tracked_ratio = tracked_counts.astype(float) / float(num_targets)
    return tracked_counts.astype(int), tracked_ratio


def summarize_key_target_tracking_episode(
    truth_states: np.ndarray,
    estimated_states: np.ndarray,
    valid_mask: np.ndarray,
    *,
    cutoff_distance: float,
    order: int = 2,
) -> dict[str, np.ndarray | float | int]:
    """Return rollout-level tracking diagnostics and summary scalars."""

    truth_array = np.asarray(truth_states, dtype=float)
    estimate_array = np.asarray(estimated_states, dtype=float)
    valid_array = np.asarray(valid_mask, dtype=bool)

    rmse, ospa, gospa = evaluate_key_target_tracking_episode(
        truth_states=truth_array,
        estimated_states=estimate_array,
        valid_mask=valid_array,
        cutoff_distance=cutoff_distance,
        order=order,
    )
    id_switches_cumulative, id_switches_per_target = compute_key_target_id_switches(
        truth_states=truth_array,
        estimated_states=estimate_array,
        valid_mask=valid_array,
        cutoff_distance=cutoff_distance,
    )
    fragmentations_cumulative, fragmentations_per_target = compute_key_target_fragmentations(
        truth_states=truth_array,
        estimated_states=estimate_array,
        valid_mask=valid_array,
        cutoff_distance=cutoff_distance,
    )
    tracked_counts, tracked_ratio = compute_key_target_tracking_coverage(valid_array)

    finite_rmse = rmse[np.isfinite(rmse)]
    mean_rmse = float(np.mean(finite_rmse)) if finite_rmse.size else np.nan
    mean_ospa = float(np.mean(ospa)) if ospa.size else np.nan
    final_ospa = float(ospa[-1]) if ospa.size else np.nan
    mean_gospa = float(np.mean(gospa)) if gospa.size else np.nan
    final_gospa = float(gospa[-1]) if gospa.size else np.nan
    mean_tracked_ratio = float(np.mean(tracked_ratio)) if tracked_ratio.size else 0.0
    final_tracked_count = int(tracked_counts[-1]) if tracked_counts.size else 0
    final_tracked_ratio = float(tracked_ratio[-1]) if tracked_ratio.size else 0.0
    total_id_switches = int(id_switches_cumulative[-1]) if id_switches_cumulative.size else 0
    total_fragmentations = int(fragmentations_cumulative[-1]) if fragmentations_cumulative.size else 0

    return {
        "rmse": rmse,
        "ospa": ospa,
        "gospa": gospa,
        "tracked_counts": tracked_counts,
        "tracked_ratio": tracked_ratio,
        "id_switches_cumulative": id_switches_cumulative,
        "id_switches_per_target": id_switches_per_target,
        "fragmentations_cumulative": fragmentations_cumulative,
        "fragmentations_per_target": fragmentations_per_target,
        "mean_rmse": mean_rmse,
        "mean_ospa": mean_ospa,
        "final_ospa": final_ospa,
        "mean_gospa": mean_gospa,
        "final_gospa": final_gospa,
        "mean_tracked_ratio": mean_tracked_ratio,
        "final_tracked_count": final_tracked_count,
        "final_tracked_ratio": final_tracked_ratio,
        "total_id_switches": total_id_switches,
        "total_fragmentations": total_fragmentations,
    }


def export_key_target_metrics_csv(
    truth_states: np.ndarray,
    estimated_states: np.ndarray,
    valid_mask: np.ndarray,
    *,
    dt: float,
    bounds: np.ndarray,
    save_path: str | Path,
    inference_time_s: np.ndarray | None = None,
) -> Path:
    """Export per-target RMSE, key-target OSPA, and common MOT diagnostics indexed by time step."""

    truth_array = np.asarray(truth_states, dtype=float)
    estimate_array = np.asarray(estimated_states, dtype=float)
    valid_array = np.asarray(valid_mask, dtype=bool)
    inference_time_array = None if inference_time_s is None else np.asarray(inference_time_s, dtype=float).reshape(-1)
    bounds_array = np.asarray(bounds, dtype=float).reshape(-1)
    if bounds_array.size < 1:
        raise ValueError("bounds must contain at least one axis value.")
    if inference_time_array is not None and inference_time_array.shape[0] != truth_array.shape[0]:
        raise ValueError(
            "inference_time_s must have one entry per timestep, "
            f"got {inference_time_array.shape[0]} and {truth_array.shape[0]}."
        )

    cutoff_distance = float(bounds_array[0]) / 4.0
    summary = summarize_key_target_tracking_episode(
        truth_states=truth_array,
        estimated_states=estimate_array,
        valid_mask=valid_array,
        cutoff_distance=cutoff_distance,
        order=2,
    )
    rmse = np.asarray(summary["rmse"], dtype=float)
    ospa = np.asarray(summary["ospa"], dtype=float)
    gospa = np.asarray(summary["gospa"], dtype=float)
    tracked_counts = np.asarray(summary["tracked_counts"], dtype=int)
    tracked_ratio = np.asarray(summary["tracked_ratio"], dtype=float)
    id_switches_cumulative = np.asarray(summary["id_switches_cumulative"], dtype=int)
    fragmentations_cumulative = np.asarray(summary["fragmentations_cumulative"], dtype=int)

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_steps, num_targets = rmse.shape
    header = ["timestep", "time_s"]
    if inference_time_array is not None:
        header.append("tracking_inference_time_s")
    header.extend(f"E{target_idx + 1}_rmse_m" for target_idx in range(num_targets))
    header.append("key_target_ospa_m")
    header.append("key_target_gospa_m")
    header.append("tracked_key_targets_count")
    header.append("tracked_key_targets_ratio")
    header.append("id_switches_cumulative")
    header.append("track_fragmentations_cumulative")

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        for step_idx in range(num_steps):
            row: list[float | int | str] = [step_idx, step_idx * float(dt)]
            if inference_time_array is not None:
                row.append(f"{inference_time_array[step_idx]:.9f}")
            for target_idx in range(num_targets):
                value = rmse[step_idx, target_idx]
                row.append("" if not np.isfinite(value) else f"{value:.6f}")
            row.append(f"{ospa[step_idx]:.6f}")
            row.append(f"{gospa[step_idx]:.6f}")
            row.append(int(tracked_counts[step_idx]))
            row.append(f"{tracked_ratio[step_idx]:.6f}")
            row.append(int(id_switches_cumulative[step_idx]))
            row.append(int(fragmentations_cumulative[step_idx]))
            writer.writerow(row)

    return output_path
