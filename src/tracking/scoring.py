"""Simple assignment helpers for collaborative key-target tracking."""

from __future__ import annotations

from itertools import permutations

import numpy as np


def _solve_exact_assignment(cost_matrix: np.ndarray) -> list[int]:
    costs = np.asarray(cost_matrix, dtype=float)
    if costs.ndim != 2:
        raise ValueError(f"cost_matrix must be 2D, got shape {costs.shape}.")

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


def compute_distance_assignment(
    pairwise_distances: np.ndarray,
) -> dict[str, np.ndarray]:
    """Assign key targets dynamically to friendly UAVs by nearest-distance coverage."""

    pairwise_distances = np.asarray(pairwise_distances, dtype=float)
    if pairwise_distances.ndim != 2:
        raise ValueError(f"pairwise_distances must be 2D, got shape {pairwise_distances.shape}.")

    num_friendlies, num_targets = pairwise_distances.shape
    if num_targets == 0:
        raise ValueError("pairwise_distances must contain at least one key target.")

    if num_friendlies == 0:
        return {
            "assigned_target": np.zeros(0, dtype=float),
            "assigned_distance": np.zeros(0, dtype=float),
        }

    pairwise_costs = pairwise_distances.copy()
    target_to_friendly = _solve_exact_assignment(pairwise_costs.T)

    assigned_targets = np.full(num_friendlies, -1, dtype=int)
    assigned_distances = np.full(num_friendlies, np.inf, dtype=float)
    for target_idx, friendly_idx in enumerate(target_to_friendly):
        if friendly_idx < 0:
            continue
        assigned_targets[friendly_idx] = int(target_idx)
        assigned_distances[friendly_idx] = float(pairwise_distances[friendly_idx, target_idx])

    unassigned_friendlies = np.flatnonzero(assigned_targets < 0)
    for friendly_idx in unassigned_friendlies:
        nearest_target_idx = int(np.argmin(pairwise_distances[friendly_idx]))
        assigned_targets[friendly_idx] = nearest_target_idx
        assigned_distances[friendly_idx] = float(pairwise_distances[friendly_idx, nearest_target_idx])

    return {
        "assigned_target": assigned_targets.astype(float),
        "assigned_distance": assigned_distances,
    }
