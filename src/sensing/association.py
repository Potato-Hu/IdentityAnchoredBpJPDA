from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Sequence

import numpy as np

from src.sensing.global_sensor import GlobalKeyTargetStatePrior
from src.sensing.local_sensor import LocalTargetObservation


STATE_DIM = 6
SUPPORTED_RECURSIVE_MOTION_MODELS = {"cv", "imm"}
SUPPORTED_IMM_MODEL_TYPES = {"cv", "ca", "ct_left", "ct_right", "ct"}
MOTION_STATE_DIM = 9


@dataclass(frozen=True)
class IMMMotionState:
    """Per-model Gaussian bank and model probabilities used by the recursive motion prior."""

    model_probabilities: np.ndarray
    model_means: np.ndarray
    model_covariances: np.ndarray


@dataclass(frozen=True)
class KeyTrackPosterior:
    """Posterior estimate for one key target in one friendly UAV's local frame."""

    global_target_index: int
    prior_relative_state: np.ndarray
    prior_covariance_diagonal: np.ndarray
    posterior_relative_state: np.ndarray
    posterior_covariance_diagonal: np.ndarray
    confidence: float
    existence_probability: float
    valid: bool
    local_candidate_weights: np.ndarray | None = None
    local_candidate_index: int = -1
    innovation_distance: float = np.inf


AssociationOutput = KeyTrackPosterior


@dataclass(frozen=True)
class MultiTargetTrackOutput:
    """All local key-target posterior tracks maintained by one friendly UAV."""

    agent_id: str
    tracks: tuple[KeyTrackPosterior, ...]
    mode: str = "local_only"


@dataclass(frozen=True)
class GMPHDComponent:
    """One Gaussian component in a GM-PHD intensity."""

    weight: float
    mean: np.ndarray
    covariance: np.ndarray
    motion_state: IMMMotionState | None = None


@dataclass(frozen=True)
class GMPHDFilterState:
    """Persistent GM-PHD mixture carried across environment steps for one UAV."""

    components: tuple[GMPHDComponent, ...]


@dataclass(frozen=True)
class BPJPDAFilterState:
    """Persistent Gaussian track state for one key target in recursive BP-JPDA."""

    mean: np.ndarray
    covariance: np.ndarray
    motion_state: IMMMotionState | None = None


@dataclass(frozen=True)
class LMBFilterState:
    """Persistent Bernoulli track state for one key target in LMB recursion."""

    existence_probability: float
    mean: np.ndarray
    covariance: np.ndarray
    motion_state: IMMMotionState | None = None


@dataclass(frozen=True)
class _BPJPDAAgentContext:
    agent_id: str
    friendly_state: np.ndarray
    prior_relative_states: np.ndarray
    prior_covariance: np.ndarray
    local_relative_states: np.ndarray
    local_covariance: np.ndarray
    valid_indices: np.ndarray
    candidate_states: np.ndarray
    predicted_means: tuple[np.ndarray, ...]
    predicted_covariances: tuple[np.ndarray, ...]
    predicted_motion_states: tuple[IMMMotionState, ...]
    track_active: np.ndarray
    likelihood_ratios: np.ndarray


def _as_state_batch(states: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(states, dtype=float)
    if array.size == 0:
        return np.zeros((0, STATE_DIM), dtype=float)
    if array.ndim != 2 or array.shape[1] != STATE_DIM:
        raise ValueError(f"{name} must have shape (N, {STATE_DIM}), got {array.shape}.")
    return array


def _as_state_vector(state: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(state, dtype=float).reshape(-1)
    if vector.shape != (STATE_DIM,):
        raise ValueError(f"{name} must have shape ({STATE_DIM},), got {vector.shape}.")
    return vector


def _as_covariance_diagonal(diagonal: Sequence[float], *, name: str) -> np.ndarray:
    vector = np.asarray(diagonal, dtype=float).reshape(-1)
    if vector.shape != (STATE_DIM,):
        raise ValueError(f"{name} must have shape ({STATE_DIM},), got {vector.shape}.")
    if np.any(vector < 0.0):
        raise ValueError(f"{name} must be >= 0 on every axis, got {vector.tolist()}.")
    return np.maximum(vector, 1.0e-6)


def _as_covariance_matrix(covariance: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(covariance, dtype=float)
    if matrix.shape == (STATE_DIM,):
        return np.diag(_as_covariance_diagonal(matrix, name=name))
    if matrix.shape != (STATE_DIM, STATE_DIM):
        raise ValueError(f"{name} must have shape ({STATE_DIM}, {STATE_DIM}), got {matrix.shape}.")
    matrix = 0.5 * (matrix + matrix.T)
    diagonal = np.maximum(np.diag(matrix), 1.0e-6)
    stabilized = matrix.copy()
    np.fill_diagonal(stabilized, diagonal)
    return stabilized


def _validate_local_observation(local_observation: LocalTargetObservation) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local_relative_states = _as_state_batch(local_observation.relative_states, name="local_observation.relative_states")
    local_mask = np.asarray(local_observation.mask, dtype=float).reshape(-1)
    if local_relative_states.shape[0] != local_mask.shape[0]:
        raise ValueError(
            "local_observation.mask length must match relative_states rows, "
            f"got {local_mask.shape[0]} and {local_relative_states.shape[0]}."
        )
    covariance_diagonal = _as_covariance_diagonal(
        local_observation.covariance_diagonal,
        name="local_observation.covariance_diagonal",
    )
    return local_relative_states, local_mask, covariance_diagonal


def _gaussian_pdf_zero_mean_diag(samples: np.ndarray, covariance_diagonal: Sequence[float]) -> np.ndarray:
    covariance_vector = _as_covariance_diagonal(covariance_diagonal, name="covariance_diagonal")
    squared_mahalanobis = np.sum(np.square(samples) / covariance_vector.reshape(1, -1), axis=1)
    normalizer = np.sqrt(np.power(2.0 * np.pi, samples.shape[1]) * np.prod(covariance_vector))
    return np.exp(-0.5 * squared_mahalanobis) / normalizer


def _gaussian_pdf(mean: np.ndarray, covariance: np.ndarray, sample: np.ndarray) -> float:
    mean_vector = _as_state_vector(mean, name="mean")
    covariance_matrix = _as_covariance_matrix(covariance, name="covariance")
    sample_vector = _as_state_vector(sample, name="sample")
    innovation = sample_vector - mean_vector
    inv_covariance = np.linalg.inv(covariance_matrix)
    mahalanobis = float(innovation.T @ inv_covariance @ innovation)
    determinant = float(np.linalg.det(covariance_matrix))
    determinant = max(determinant, 1.0e-12)
    normalizer = np.sqrt(np.power(2.0 * np.pi, STATE_DIM) * determinant)
    return float(np.exp(-0.5 * mahalanobis) / normalizer)


def _run_bp_jpda(
    likelihood_ratios: np.ndarray,
    *,
    max_iterations: int,
    convergence_tol: float,
) -> np.ndarray:
    """Approximate marginal association probabilities with bipartite BP."""

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


def _as_probability_vector(probabilities: Sequence[float], *, length: int, name: str) -> np.ndarray:
    vector = np.asarray(probabilities, dtype=float).reshape(-1)
    if vector.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {vector.shape}.")
    if np.any(vector < 0.0):
        raise ValueError(f"{name} must be >= 0 on every axis, got {vector.tolist()}.")
    total = float(np.sum(vector))
    if total <= 0.0:
        raise ValueError(f"{name} must sum to a positive value, got {vector.tolist()}.")
    return vector / total


def _as_vector_dim(value: np.ndarray, *, dim: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.shape != (dim,):
        raise ValueError(f"{name} must have shape ({dim},), got {vector.shape}.")
    return vector


def _as_covariance_matrix_dim(covariance: np.ndarray, *, dim: int, name: str) -> np.ndarray:
    matrix = np.asarray(covariance, dtype=float)
    if matrix.shape == (dim,):
        diagonal = np.maximum(np.asarray(matrix, dtype=float).reshape(-1), 1.0e-6)
        return np.diag(diagonal)
    if matrix.shape != (dim, dim):
        raise ValueError(f"{name} must have shape ({dim}, {dim}), got {matrix.shape}.")
    matrix = 0.5 * (matrix + matrix.T)
    diagonal = np.maximum(np.diag(matrix), 1.0e-6)
    stabilized = matrix.copy()
    np.fill_diagonal(stabilized, diagonal)
    return stabilized


def _motion_observation_matrix() -> np.ndarray:
    observation = np.zeros((STATE_DIM, MOTION_STATE_DIM), dtype=float)
    observation[:STATE_DIM, :STATE_DIM] = np.eye(STATE_DIM, dtype=float)
    return observation


def _expand_motion_mean(output_state_mean: np.ndarray) -> np.ndarray:
    expanded = np.zeros(MOTION_STATE_DIM, dtype=float)
    expanded[:STATE_DIM] = _as_state_vector(output_state_mean, name="output_state_mean")
    return expanded


def _expand_motion_covariance(
    output_state_covariance: np.ndarray,
    *,
    inferred_acceleration_std: np.ndarray,
) -> np.ndarray:
    covariance = np.zeros((MOTION_STATE_DIM, MOTION_STATE_DIM), dtype=float)
    covariance[:STATE_DIM, :STATE_DIM] = _as_covariance_matrix(output_state_covariance, name="output_state_covariance")
    acceleration_std = np.asarray(inferred_acceleration_std, dtype=float).reshape(3)
    covariance[STATE_DIM:, STATE_DIM:] = np.diag(np.maximum(acceleration_std, 1.0e-3) ** 2)
    return covariance


def _project_motion_mean_covariance(
    motion_mean: np.ndarray,
    motion_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = _as_vector_dim(motion_mean, dim=MOTION_STATE_DIM, name="motion_mean")
    covariance = _as_covariance_matrix_dim(motion_covariance, dim=MOTION_STATE_DIM, name="motion_covariance")
    return mean[:STATE_DIM], covariance[:STATE_DIM, :STATE_DIM]


def _gaussian_mixture_moments_generic(
    means: Sequence[np.ndarray],
    covariances: Sequence[np.ndarray],
    weights: Sequence[float],
    *,
    dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(means) == 0 or len(covariances) == 0 or len(weights) == 0:
        raise ValueError("means, covariances, and weights must all be non-empty.")
    if not (len(means) == len(covariances) == len(weights)):
        raise ValueError("means, covariances, and weights must have the same length.")

    raw_weights = np.asarray(weights, dtype=float).reshape(-1)
    total_weight = float(np.sum(raw_weights))
    if total_weight <= 0.0:
        raise ValueError("sum of Gaussian-mixture weights must be > 0.")
    normalized_weights = raw_weights / total_weight
    mean_stack = np.stack([_as_vector_dim(mean, dim=dim, name="mixture_mean") for mean in means], axis=0)
    mixture_mean = np.average(mean_stack, axis=0, weights=normalized_weights)

    mixture_covariance = np.zeros((dim, dim), dtype=float)
    for normalized_weight, component_mean, component_covariance in zip(normalized_weights, means, covariances):
        mean_vector = _as_vector_dim(component_mean, dim=dim, name="component_mean")
        covariance_matrix = _as_covariance_matrix_dim(component_covariance, dim=dim, name="component_covariance")
        delta = (mean_vector - mixture_mean).reshape(dim, 1)
        mixture_covariance += float(normalized_weight) * (covariance_matrix + delta @ delta.T)
    return mixture_mean, _as_covariance_matrix_dim(mixture_covariance, dim=dim, name="mixture_covariance")


def _kalman_predict_generic(
    mean: np.ndarray,
    covariance: np.ndarray,
    *,
    transition: np.ndarray,
    process_noise: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    transition_matrix = np.asarray(transition, dtype=float)
    dim = transition_matrix.shape[0]
    if transition_matrix.shape != (dim, dim):
        raise ValueError(f"transition must be square, got {transition_matrix.shape}.")
    state_mean = _as_vector_dim(mean, dim=dim, name="mean")
    state_covariance = _as_covariance_matrix_dim(covariance, dim=dim, name="covariance")
    process_covariance = _as_covariance_matrix_dim(process_noise, dim=dim, name="process_noise")
    predicted_mean = transition_matrix @ state_mean
    predicted_covariance = transition_matrix @ state_covariance @ transition_matrix.T + process_covariance
    return predicted_mean, _as_covariance_matrix_dim(predicted_covariance, dim=dim, name="predicted_covariance")


def _kalman_update_linear_generic(
    prior_mean: np.ndarray,
    prior_covariance: np.ndarray,
    measurement: np.ndarray,
    measurement_covariance: np.ndarray,
    *,
    observation_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    observation = np.asarray(observation_matrix, dtype=float)
    measurement_dim, state_dim = observation.shape
    predicted_mean = _as_vector_dim(prior_mean, dim=state_dim, name="prior_mean")
    predicted_covariance = _as_covariance_matrix_dim(prior_covariance, dim=state_dim, name="prior_covariance")
    measurement_vector = _as_vector_dim(measurement, dim=measurement_dim, name="measurement")
    measurement_covariance_matrix = _as_covariance_matrix_dim(
        measurement_covariance,
        dim=measurement_dim,
        name="measurement_covariance",
    )

    innovation = measurement_vector - observation @ predicted_mean
    innovation_covariance = observation @ predicted_covariance @ observation.T + measurement_covariance_matrix
    innovation_covariance = _as_covariance_matrix_dim(
        innovation_covariance,
        dim=measurement_dim,
        name="innovation_covariance",
    )
    inv_innovation_covariance = np.linalg.inv(innovation_covariance)
    kalman_gain = predicted_covariance @ observation.T @ inv_innovation_covariance
    posterior_mean = predicted_mean + kalman_gain @ innovation
    identity = np.eye(state_dim, dtype=float)
    posterior_covariance = (
        (identity - kalman_gain @ observation) @ predicted_covariance @ (identity - kalman_gain @ observation).T
        + kalman_gain @ measurement_covariance_matrix @ kalman_gain.T
    )
    posterior_covariance = _as_covariance_matrix_dim(
        posterior_covariance,
        dim=state_dim,
        name="posterior_covariance",
    )
    mahalanobis_sq = float(innovation.T @ inv_innovation_covariance @ innovation)
    return posterior_mean, posterior_covariance, mahalanobis_sq


def _default_imm_transition_matrix(model_count: int) -> np.ndarray:
    if model_count <= 0:
        raise ValueError(f"model_count must be > 0, got {model_count}.")
    if model_count == 1:
        return np.ones((1, 1), dtype=float)
    stay_probability = 0.85
    switch_probability = (1.0 - stay_probability) / float(model_count - 1)
    matrix = np.full((model_count, model_count), switch_probability, dtype=float)
    np.fill_diagonal(matrix, stay_probability)
    return matrix


def _default_imm_initial_model_probabilities(model_types: Sequence[str]) -> np.ndarray:
    resolved_types = tuple(str(model_type).lower() for model_type in model_types)
    model_count = len(resolved_types)
    if model_count == 0:
        raise ValueError("model_types must contain at least one model.")
    if model_count == 1:
        return np.ones(1, dtype=float)
    if resolved_types[0] == "cv":
        remaining_mass = 0.5
        probabilities = np.full(model_count, remaining_mass / float(model_count - 1), dtype=float)
        probabilities[0] = 0.5
        return probabilities
    return np.full(model_count, 1.0 / float(model_count), dtype=float)


def _resolve_recursive_motion_config(
    *,
    recursive_motion_model: str,
    imm_model_types: Sequence[str],
    imm_model_transition_matrix: np.ndarray | Sequence[Sequence[float]] | None,
    imm_initial_model_probabilities: np.ndarray | Sequence[float] | None,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    resolved_mode = str(recursive_motion_model).lower()
    if resolved_mode not in SUPPORTED_RECURSIVE_MOTION_MODELS:
        raise ValueError(
            "recursive_motion_model must be one of "
            f"{sorted(SUPPORTED_RECURSIVE_MOTION_MODELS)}, got {recursive_motion_model!r}."
        )
    if resolved_mode == "cv":
        return ("cv",), np.ones((1, 1), dtype=float), np.ones(1, dtype=float)

    resolved_model_types = tuple(str(model_type).lower() for model_type in imm_model_types)
    if not resolved_model_types:
        raise ValueError("imm_model_types must contain at least one model when recursive_motion_model='imm'.")
    for model_type in resolved_model_types:
        if model_type not in SUPPORTED_IMM_MODEL_TYPES:
            raise ValueError(
                "imm_model_types entries must be one of "
                f"{sorted(SUPPORTED_IMM_MODEL_TYPES)}, got {model_type!r}."
            )

    model_count = len(resolved_model_types)
    if imm_model_transition_matrix is None:
        transition_matrix = _default_imm_transition_matrix(model_count)
    else:
        transition_matrix = np.asarray(imm_model_transition_matrix, dtype=float)
        if transition_matrix.shape != (model_count, model_count):
            raise ValueError(
                "imm_model_transition_matrix must have shape "
                f"({model_count}, {model_count}), got {transition_matrix.shape}."
            )
        if np.any(transition_matrix < 0.0):
            raise ValueError("imm_model_transition_matrix must contain only non-negative entries.")
        row_sums = np.sum(transition_matrix, axis=1, keepdims=True)
        if np.any(row_sums <= 0.0):
            raise ValueError("Each row of imm_model_transition_matrix must sum to a positive value.")
        transition_matrix = transition_matrix / row_sums

    if imm_initial_model_probabilities is None:
        initial_probabilities = _default_imm_initial_model_probabilities(resolved_model_types)
    else:
        initial_probabilities = _as_probability_vector(
            imm_initial_model_probabilities,
            length=model_count,
            name="imm_initial_model_probabilities",
        )
    return resolved_model_types, transition_matrix.astype(float, copy=True), initial_probabilities.astype(float, copy=True)


def _ct_transition_matrix(dt_seconds: float, *, turn_rate_radps: float) -> np.ndarray:
    if abs(float(turn_rate_radps)) <= 1.0e-8:
        return _cv_motion_transition_matrix(dt_seconds)

    dt_value = float(dt_seconds)
    omega = float(turn_rate_radps)
    omega_dt = omega * dt_value
    sin_term = float(np.sin(omega_dt))
    cos_term = float(np.cos(omega_dt))
    transition = np.eye(MOTION_STATE_DIM, dtype=float)
    transition[0, 3] = sin_term / omega
    transition[0, 4] = -(1.0 - cos_term) / omega
    transition[1, 3] = (1.0 - cos_term) / omega
    transition[1, 4] = sin_term / omega
    transition[2, 5] = dt_value
    transition[3, 3] = cos_term
    transition[3, 4] = -sin_term
    transition[4, 3] = sin_term
    transition[4, 4] = cos_term
    transition[6:, 6:] = 0.0
    return transition


def _cv_motion_transition_matrix(dt_seconds: float) -> np.ndarray:
    transition = np.eye(MOTION_STATE_DIM, dtype=float)
    dt_value = float(dt_seconds)
    transition[0, 3] = dt_value
    transition[1, 4] = dt_value
    transition[2, 5] = dt_value
    transition[6:, 6:] = 0.0
    return transition


def _ca_motion_transition_matrix(dt_seconds: float) -> np.ndarray:
    transition = np.eye(MOTION_STATE_DIM, dtype=float)
    dt_value = float(dt_seconds)
    half_dt_sq = 0.5 * dt_value * dt_value
    transition[0, 3] = dt_value
    transition[1, 4] = dt_value
    transition[2, 5] = dt_value
    transition[0, 6] = half_dt_sq
    transition[1, 7] = half_dt_sq
    transition[2, 8] = half_dt_sq
    transition[3, 6] = dt_value
    transition[4, 7] = dt_value
    transition[5, 8] = dt_value
    return transition


def _transition_for_model(
    model_type: str,
    *,
    dt_seconds: float,
    turn_rate_radps: float,
) -> np.ndarray:
    resolved_type = str(model_type).lower()
    if resolved_type == "cv":
        return _cv_motion_transition_matrix(dt_seconds)
    if resolved_type == "ca":
        return _ca_motion_transition_matrix(dt_seconds)
    if resolved_type == "ct_left":
        return _ct_transition_matrix(dt_seconds, turn_rate_radps=abs(float(turn_rate_radps)))
    if resolved_type == "ct_right":
        return _ct_transition_matrix(dt_seconds, turn_rate_radps=-abs(float(turn_rate_radps)))
    if resolved_type == "ct":
        return _ct_transition_matrix(dt_seconds, turn_rate_radps=float(turn_rate_radps))
    raise ValueError(f"Unsupported motion-model type: {model_type!r}.")


def _process_noise_for_model(
    base_process_noise: np.ndarray,
    *,
    model_type: str,
    ct_process_noise_scale: float,
) -> np.ndarray:
    resolved_type = str(model_type).lower()
    if resolved_type in {"ct_left", "ct_right", "ct"}:
        return float(max(ct_process_noise_scale, 1.0e-6)) * np.asarray(base_process_noise, dtype=float)
    return np.asarray(base_process_noise, dtype=float)


def _default_motion_state(
    mean: np.ndarray,
    covariance: np.ndarray,
    *,
    recursive_motion_model: str,
    imm_model_types: Sequence[str],
    imm_model_transition_matrix: np.ndarray | Sequence[Sequence[float]] | None,
    imm_initial_model_probabilities: np.ndarray | Sequence[float] | None,
) -> IMMMotionState:
    model_types, _, initial_probabilities = _resolve_recursive_motion_config(
        recursive_motion_model=recursive_motion_model,
        imm_model_types=imm_model_types,
        imm_model_transition_matrix=imm_model_transition_matrix,
        imm_initial_model_probabilities=imm_initial_model_probabilities,
    )
    state_mean = _as_state_vector(mean, name="mean")
    state_covariance = _as_covariance_matrix(covariance, name="covariance")
    inferred_acceleration_std = np.maximum(np.sqrt(np.diag(state_covariance)[3:]), 1.0e-3)
    model_count = len(model_types)
    motion_mean = _expand_motion_mean(state_mean)
    motion_covariance = _expand_motion_covariance(
        state_covariance,
        inferred_acceleration_std=inferred_acceleration_std,
    )
    model_means = np.repeat(motion_mean.reshape(1, MOTION_STATE_DIM), model_count, axis=0)
    model_covariances = np.repeat(motion_covariance.reshape(1, MOTION_STATE_DIM, MOTION_STATE_DIM), model_count, axis=0)
    return IMMMotionState(
        model_probabilities=initial_probabilities.astype(float, copy=True),
        model_means=model_means.astype(float, copy=True),
        model_covariances=model_covariances.astype(float, copy=True),
    )


def _coerce_motion_state(
    motion_state: IMMMotionState | None,
    *,
    fallback_mean: np.ndarray,
    fallback_covariance: np.ndarray,
    recursive_motion_model: str,
    imm_model_types: Sequence[str],
    imm_model_transition_matrix: np.ndarray | Sequence[Sequence[float]] | None,
    imm_initial_model_probabilities: np.ndarray | Sequence[float] | None,
) -> IMMMotionState:
    model_types, _, initial_probabilities = _resolve_recursive_motion_config(
        recursive_motion_model=recursive_motion_model,
        imm_model_types=imm_model_types,
        imm_model_transition_matrix=imm_model_transition_matrix,
        imm_initial_model_probabilities=imm_initial_model_probabilities,
    )
    model_count = len(model_types)
    if motion_state is None:
        return _default_motion_state(
            fallback_mean,
            fallback_covariance,
            recursive_motion_model=recursive_motion_model,
            imm_model_types=imm_model_types,
            imm_model_transition_matrix=imm_model_transition_matrix,
            imm_initial_model_probabilities=imm_initial_model_probabilities,
        )

    probabilities = np.asarray(motion_state.model_probabilities, dtype=float).reshape(-1)
    means = np.asarray(motion_state.model_means, dtype=float)
    covariances = np.asarray(motion_state.model_covariances, dtype=float)
    if probabilities.shape != (model_count,) or means.shape != (model_count, MOTION_STATE_DIM) or covariances.shape != (
        model_count,
        MOTION_STATE_DIM,
        MOTION_STATE_DIM,
    ):
        return _default_motion_state(
            fallback_mean,
            fallback_covariance,
            recursive_motion_model=recursive_motion_model,
            imm_model_types=imm_model_types,
            imm_model_transition_matrix=imm_model_transition_matrix,
            imm_initial_model_probabilities=imm_initial_model_probabilities,
        )
    normalized_probabilities = _as_probability_vector(
        probabilities,
        length=model_count,
        name="motion_state.model_probabilities",
    )
    return IMMMotionState(
        model_probabilities=normalized_probabilities.astype(float, copy=True),
        model_means=np.stack(
            [_as_vector_dim(mean, dim=MOTION_STATE_DIM, name="motion_state.model_means") for mean in means],
            axis=0,
        ),
        model_covariances=np.stack(
            [
                _as_covariance_matrix_dim(
                    covariance,
                    dim=MOTION_STATE_DIM,
                    name="motion_state.model_covariances",
                )
                for covariance in covariances
            ],
            axis=0,
        ),
    )


def _collapse_motion_state(motion_state: IMMMotionState) -> tuple[np.ndarray, np.ndarray]:
    model_count = motion_state.model_means.shape[0]
    probabilities = _as_probability_vector(
        motion_state.model_probabilities,
        length=model_count,
        name="motion_state.model_probabilities",
    )
    projected_means: list[np.ndarray] = []
    projected_covariances: list[np.ndarray] = []
    for model_mean, model_covariance in zip(motion_state.model_means, motion_state.model_covariances):
        projected_mean, projected_covariance = _project_motion_mean_covariance(model_mean, model_covariance)
        projected_means.append(projected_mean)
        projected_covariances.append(projected_covariance)
    return _gaussian_mixture_moments(
        projected_means,
        projected_covariances,
        probabilities,
    )


def _imm_mix_model_bank(
    motion_state: IMMMotionState,
    *,
    transition_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probabilities = _as_probability_vector(
        motion_state.model_probabilities,
        length=motion_state.model_means.shape[0],
        name="motion_state.model_probabilities",
    )
    markov_matrix = np.asarray(transition_matrix, dtype=float)
    if markov_matrix.shape != (probabilities.shape[0], probabilities.shape[0]):
        raise ValueError(
            "transition_matrix must have shape "
            f"({probabilities.shape[0]}, {probabilities.shape[0]}), got {markov_matrix.shape}."
        )
    destination_probabilities = probabilities @ markov_matrix
    destination_probabilities = np.maximum(destination_probabilities, 1.0e-12)
    destination_probabilities = destination_probabilities / float(np.sum(destination_probabilities))

    mixed_means = np.zeros_like(motion_state.model_means, dtype=float)
    mixed_covariances = np.zeros_like(motion_state.model_covariances, dtype=float)
    for destination_idx in range(probabilities.shape[0]):
        numerator = probabilities * markov_matrix[:, destination_idx]
        total = float(np.sum(numerator))
        if total <= 0.0:
            mixing_weights = np.zeros_like(probabilities)
            mixing_weights[destination_idx] = 1.0
        else:
            mixing_weights = numerator / total
        mixed_means[destination_idx], mixed_covariances[destination_idx] = _gaussian_mixture_moments_generic(
            motion_state.model_means,
            motion_state.model_covariances,
            mixing_weights,
            dim=MOTION_STATE_DIM,
        )
    return destination_probabilities, mixed_means, mixed_covariances


def _predict_motion_state(
    motion_state: IMMMotionState,
    *,
    recursive_motion_model: str,
    imm_model_types: Sequence[str],
    imm_model_transition_matrix: np.ndarray | Sequence[Sequence[float]] | None,
    imm_initial_model_probabilities: np.ndarray | Sequence[float] | None,
    dt_seconds: float,
    process_position_std: np.ndarray,
    process_velocity_std: np.ndarray,
    imm_turn_rate_radps: float,
    imm_ct_process_noise_scale: float,
) -> IMMMotionState:
    model_types, transition_matrix, _ = _resolve_recursive_motion_config(
        recursive_motion_model=recursive_motion_model,
        imm_model_types=imm_model_types,
        imm_model_transition_matrix=imm_model_transition_matrix,
        imm_initial_model_probabilities=imm_initial_model_probabilities,
    )
    base_process_noise = _gm_phd_process_noise_covariance(
        dt_seconds=dt_seconds,
        process_position_std=process_position_std,
        process_velocity_std=process_velocity_std,
    )
    acceleration_std = np.maximum(np.asarray(process_velocity_std, dtype=float).reshape(3), 1.0e-3)
    motion_process_noise = np.zeros((MOTION_STATE_DIM, MOTION_STATE_DIM), dtype=float)
    motion_process_noise[:STATE_DIM, :STATE_DIM] = base_process_noise
    motion_process_noise[STATE_DIM:, STATE_DIM:] = np.diag(np.square(acceleration_std))
    predicted_probabilities, mixed_means, mixed_covariances = _imm_mix_model_bank(
        motion_state,
        transition_matrix=transition_matrix,
    )
    predicted_means = np.zeros_like(mixed_means, dtype=float)
    predicted_covariances = np.zeros_like(mixed_covariances, dtype=float)
    for model_idx, model_type in enumerate(model_types):
        transition = _transition_for_model(
            model_type,
            dt_seconds=dt_seconds,
            turn_rate_radps=imm_turn_rate_radps,
        )
        process_noise = _process_noise_for_model(
            motion_process_noise,
            model_type=model_type,
            ct_process_noise_scale=imm_ct_process_noise_scale,
        )
        predicted_means[model_idx], predicted_covariances[model_idx] = _kalman_predict_generic(
            mixed_means[model_idx],
            mixed_covariances[model_idx],
            transition=transition,
            process_noise=process_noise,
        )
    return IMMMotionState(
        model_probabilities=predicted_probabilities.astype(float, copy=True),
        model_means=predicted_means.astype(float, copy=True),
        model_covariances=predicted_covariances.astype(float, copy=True),
    )


def _anchor_motion_state(
    motion_state: IMMMotionState,
    *,
    anchor_mean: np.ndarray,
    anchor_covariance: np.ndarray,
    survival_probability: float,
    birth_weight: float,
) -> IMMMotionState:
    anchored_means = np.zeros_like(motion_state.model_means, dtype=float)
    anchored_covariances = np.zeros_like(motion_state.model_covariances, dtype=float)
    observation_matrix = _motion_observation_matrix()
    measurement_covariance = _as_covariance_matrix(anchor_covariance, name="anchor_covariance") / max(
        float(birth_weight),
        1.0e-6,
    )
    for model_idx in range(motion_state.model_means.shape[0]):
        anchored_means[model_idx], anchored_covariances[model_idx], _ = _kalman_update_linear_generic(
            motion_state.model_means[model_idx],
            motion_state.model_covariances[model_idx],
            _as_state_vector(anchor_mean, name="anchor_mean"),
            measurement_covariance,
            observation_matrix=observation_matrix,
        )
    return IMMMotionState(
        model_probabilities=motion_state.model_probabilities.astype(float, copy=True),
        model_means=anchored_means.astype(float, copy=True),
        model_covariances=anchored_covariances.astype(float, copy=True),
    )


def _update_motion_state_with_measurement(
    motion_state: IMMMotionState,
    *,
    measurement_state: np.ndarray,
    measurement_covariance_diagonal: np.ndarray,
) -> tuple[IMMMotionState, float]:
    measurement = _as_state_vector(measurement_state, name="measurement_state")
    measurement_covariance = np.diag(
        _as_covariance_diagonal(measurement_covariance_diagonal, name="measurement_covariance_diagonal")
    )
    observation_matrix = _motion_observation_matrix()
    updated_means = np.zeros_like(motion_state.model_means, dtype=float)
    updated_covariances = np.zeros_like(motion_state.model_covariances, dtype=float)
    model_likelihoods = np.zeros(motion_state.model_means.shape[0], dtype=float)
    best_innovation_distance = np.inf
    for model_idx in range(motion_state.model_means.shape[0]):
        predicted_mean = motion_state.model_means[model_idx]
        predicted_covariance = motion_state.model_covariances[model_idx]
        updated_means[model_idx], updated_covariances[model_idx], mahalanobis_sq = _kalman_update_linear_generic(
            predicted_mean,
            predicted_covariance,
            measurement,
            measurement_covariance,
            observation_matrix=observation_matrix,
        )
        projected_mean, projected_covariance = _project_motion_mean_covariance(predicted_mean, predicted_covariance)
        innovation_covariance = projected_covariance + measurement_covariance
        model_likelihoods[model_idx] = _gaussian_pdf(projected_mean, innovation_covariance, measurement)
        best_innovation_distance = min(best_innovation_distance, float(np.sqrt(max(mahalanobis_sq, 0.0))))

    posterior_weights = motion_state.model_probabilities * np.maximum(model_likelihoods, 1.0e-12)
    total_weight = float(np.sum(posterior_weights))
    if total_weight <= 0.0:
        posterior_probabilities = motion_state.model_probabilities.astype(float, copy=True)
    else:
        posterior_probabilities = posterior_weights / total_weight
    return (
        IMMMotionState(
            model_probabilities=posterior_probabilities.astype(float, copy=True),
            model_means=updated_means.astype(float, copy=True),
            model_covariances=updated_covariances.astype(float, copy=True),
        ),
        best_innovation_distance,
    )


def _merge_motion_hypotheses(
    motion_states: Sequence[IMMMotionState],
    weights: Sequence[float],
) -> IMMMotionState:
    if len(motion_states) == 0 or len(weights) == 0:
        raise ValueError("motion_states and weights must both be non-empty.")
    if len(motion_states) != len(weights):
        raise ValueError("motion_states and weights must have the same length.")

    raw_weights = np.asarray(weights, dtype=float).reshape(-1)
    total_weight = float(np.sum(raw_weights))
    if total_weight <= 0.0:
        raise ValueError("sum of hypothesis weights must be > 0.")
    normalized_weights = raw_weights / total_weight

    model_count = motion_states[0].model_means.shape[0]
    for motion_state in motion_states:
        if motion_state.model_means.shape[0] != model_count:
            raise ValueError("All motion_states must carry the same number of models.")

    merged_model_weights = np.zeros(model_count, dtype=float)
    for hypothesis_weight, motion_state in zip(normalized_weights, motion_states):
        merged_model_weights += float(hypothesis_weight) * np.asarray(motion_state.model_probabilities, dtype=float)

    merged_model_weights = np.maximum(merged_model_weights, 1.0e-12)
    merged_model_probabilities = merged_model_weights / float(np.sum(merged_model_weights))

    merged_means = np.zeros((model_count, MOTION_STATE_DIM), dtype=float)
    merged_covariances = np.zeros((model_count, MOTION_STATE_DIM, MOTION_STATE_DIM), dtype=float)
    for model_idx in range(model_count):
        per_hypothesis_weights = np.asarray(
            [
                float(hypothesis_weight) * float(motion_state.model_probabilities[model_idx])
                for hypothesis_weight, motion_state in zip(normalized_weights, motion_states)
            ],
            dtype=float,
        )
        if float(np.sum(per_hypothesis_weights)) <= 0.0:
            per_hypothesis_weights = np.full(len(motion_states), 1.0 / float(len(motion_states)), dtype=float)
        merged_means[model_idx], merged_covariances[model_idx] = _gaussian_mixture_moments_generic(
            [motion_state.model_means[model_idx] for motion_state in motion_states],
            [motion_state.model_covariances[model_idx] for motion_state in motion_states],
            per_hypothesis_weights,
            dim=MOTION_STATE_DIM,
        )
    return IMMMotionState(
        model_probabilities=merged_model_probabilities.astype(float, copy=True),
        model_means=merged_means.astype(float, copy=True),
        model_covariances=merged_covariances.astype(float, copy=True),
    )


def _kalman_predict_full(
    mean: np.ndarray,
    covariance: np.ndarray,
    *,
    transition: np.ndarray,
    process_noise: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    state_mean = _as_state_vector(mean, name="mean")
    state_covariance = _as_covariance_matrix(covariance, name="covariance")
    predicted_mean = transition @ state_mean
    predicted_covariance = transition @ state_covariance @ transition.T + process_noise
    return predicted_mean, _as_covariance_matrix(predicted_covariance, name="predicted_covariance")


def _kalman_update_full(
    prior_mean: np.ndarray,
    prior_covariance: np.ndarray,
    measurement_state: np.ndarray,
    measurement_covariance_diagonal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    predicted_mean = _as_state_vector(prior_mean, name="prior_mean")
    predicted_covariance = _as_covariance_matrix(prior_covariance, name="prior_covariance")
    measurement = _as_state_vector(measurement_state, name="measurement_state")
    measurement_covariance = np.diag(
        _as_covariance_diagonal(measurement_covariance_diagonal, name="measurement_covariance_diagonal")
    )

    innovation = measurement - predicted_mean
    innovation_covariance = predicted_covariance + measurement_covariance
    innovation_covariance = _as_covariance_matrix(innovation_covariance, name="innovation_covariance")
    inv_innovation_covariance = np.linalg.inv(innovation_covariance)
    kalman_gain = predicted_covariance @ inv_innovation_covariance
    posterior_mean = predicted_mean + kalman_gain @ innovation
    identity = np.eye(STATE_DIM, dtype=float)
    posterior_covariance = (
        (identity - kalman_gain) @ predicted_covariance @ (identity - kalman_gain).T
        + kalman_gain @ measurement_covariance @ kalman_gain.T
    )
    posterior_covariance = _as_covariance_matrix(posterior_covariance, name="posterior_covariance")
    mahalanobis_sq = float(innovation.T @ inv_innovation_covariance @ innovation)
    return posterior_mean, posterior_covariance, mahalanobis_sq


def _information_fuse_states(
    primary_mean: np.ndarray,
    primary_covariance: np.ndarray,
    secondary_mean: np.ndarray,
    secondary_covariance: np.ndarray,
    *,
    primary_weight: float,
    secondary_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    weight_primary = max(float(primary_weight), 0.0)
    weight_secondary = max(float(secondary_weight), 0.0)
    if weight_primary <= 0.0 and weight_secondary <= 0.0:
        raise ValueError("At least one information-fusion weight must be positive.")

    info_matrix = np.zeros((STATE_DIM, STATE_DIM), dtype=float)
    info_vector = np.zeros(STATE_DIM, dtype=float)
    if weight_primary > 0.0:
        primary_cov = _as_covariance_matrix(primary_covariance, name="primary_covariance")
        primary_precision = np.linalg.inv(primary_cov)
        info_matrix += weight_primary * primary_precision
        info_vector += weight_primary * (primary_precision @ _as_state_vector(primary_mean, name="primary_mean"))
    if weight_secondary > 0.0:
        secondary_cov = _as_covariance_matrix(secondary_covariance, name="secondary_covariance")
        secondary_precision = np.linalg.inv(secondary_cov)
        info_matrix += weight_secondary * secondary_precision
        info_vector += weight_secondary * (secondary_precision @ _as_state_vector(secondary_mean, name="secondary_mean"))

    fused_covariance = np.linalg.inv(info_matrix)
    fused_covariance = _as_covariance_matrix(fused_covariance, name="fused_covariance")
    fused_mean = fused_covariance @ info_vector
    return fused_mean, fused_covariance


def _position_gate_margin(covariance_diagonal: np.ndarray, gate_sigma: float) -> float:
    return float(gate_sigma) * float(np.linalg.norm(np.sqrt(covariance_diagonal[:3])))


def _gm_phd_transition_matrix(dt_seconds: float) -> np.ndarray:
    transition = np.eye(STATE_DIM, dtype=float)
    dt_value = float(dt_seconds)
    transition[0, 3] = dt_value
    transition[1, 4] = dt_value
    transition[2, 5] = dt_value
    return transition


def _gm_phd_process_noise_covariance(
    *,
    dt_seconds: float,
    process_position_std: np.ndarray,
    process_velocity_std: np.ndarray,
) -> np.ndarray:
    dt_value = max(float(dt_seconds), 1.0e-6)
    position_std = np.asarray(process_position_std, dtype=float).reshape(3)
    velocity_std = np.asarray(process_velocity_std, dtype=float).reshape(3)
    covariance = np.zeros((STATE_DIM, STATE_DIM), dtype=float)
    for axis in range(3):
        q_axis = max(float(velocity_std[axis]) ** 2, 1.0e-6)
        covariance[axis, axis] = float(position_std[axis]) ** 2 + 0.25 * dt_value * dt_value * q_axis
        covariance[axis, axis + 3] = 0.5 * dt_value * q_axis
        covariance[axis + 3, axis] = 0.5 * dt_value * q_axis
        covariance[axis + 3, axis + 3] = q_axis
    return covariance


def _gaussian_mixture_moments(
    means: Sequence[np.ndarray],
    covariances: Sequence[np.ndarray],
    weights: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    if len(means) == 0 or len(covariances) == 0 or len(weights) == 0:
        raise ValueError("means, covariances, and weights must all be non-empty.")
    if not (len(means) == len(covariances) == len(weights)):
        raise ValueError("means, covariances, and weights must have the same length.")

    raw_weights = np.asarray(weights, dtype=float).reshape(-1)
    total_weight = float(np.sum(raw_weights))
    if total_weight <= 0.0:
        raise ValueError("sum of Gaussian-mixture weights must be > 0.")
    normalized_weights = raw_weights / total_weight
    mean_stack = np.stack([_as_state_vector(mean, name="mixture_mean") for mean in means], axis=0)
    mixture_mean = np.average(mean_stack, axis=0, weights=normalized_weights)

    mixture_covariance = np.zeros((STATE_DIM, STATE_DIM), dtype=float)
    for normalized_weight, component_mean, component_covariance in zip(
        normalized_weights,
        means,
        covariances,
    ):
        mean_vector = _as_state_vector(component_mean, name="component_mean")
        covariance_matrix = _as_covariance_matrix(component_covariance, name="component_covariance")
        delta = (mean_vector - mixture_mean).reshape(STATE_DIM, 1)
        mixture_covariance += float(normalized_weight) * (covariance_matrix + delta @ delta.T)
    return mixture_mean, _as_covariance_matrix(mixture_covariance, name="mixture_covariance")


def _build_prior_relative_states(
    friendly_state: np.ndarray,
    global_target_priors: GlobalKeyTargetStatePrior,
) -> tuple[np.ndarray, np.ndarray]:
    friendly = _as_state_vector(friendly_state, name="friendly_state")
    prior_states = _as_state_batch(global_target_priors.states, name="global_target_priors.states")
    prior_covariance = _as_covariance_diagonal(
        global_target_priors.covariance_diagonal,
        name="global_target_priors.covariance_diagonal",
    )
    if prior_states.shape[0] == 0:
        return np.zeros((0, STATE_DIM), dtype=float), prior_covariance
    prior_relative_states = prior_states - friendly.reshape(1, STATE_DIM)
    return prior_relative_states, prior_covariance


def empty_key_track_posterior(
    *,
    global_target_index: int,
    prior_relative_state: np.ndarray | None = None,
    prior_covariance_diagonal: np.ndarray | None = None,
    posterior_relative_state: np.ndarray | None = None,
    posterior_covariance_diagonal: np.ndarray | None = None,
    confidence: float = 0.0,
    existence_probability: float = 0.0,
    valid: bool = False,
    local_candidate_weights: np.ndarray | None = None,
    local_candidate_index: int = -1,
    innovation_distance: float = np.inf,
) -> KeyTrackPosterior:
    prior_state = (
        np.zeros(STATE_DIM, dtype=np.float32)
        if prior_relative_state is None
        else np.asarray(prior_relative_state, dtype=np.float32).reshape(STATE_DIM)
    )
    prior_cov = (
        np.ones(STATE_DIM, dtype=np.float32)
        if prior_covariance_diagonal is None
        else np.asarray(prior_covariance_diagonal, dtype=np.float32).reshape(STATE_DIM)
    )
    posterior_state = (
        prior_state.copy()
        if posterior_relative_state is None
        else np.asarray(posterior_relative_state, dtype=np.float32).reshape(STATE_DIM)
    )
    posterior_cov = (
        prior_cov.copy()
        if posterior_covariance_diagonal is None
        else np.asarray(posterior_covariance_diagonal, dtype=np.float32).reshape(STATE_DIM)
    )
    weights = None
    if local_candidate_weights is not None:
        weights = np.asarray(local_candidate_weights, dtype=np.float32).reshape(-1)
    return KeyTrackPosterior(
        global_target_index=int(global_target_index),
        prior_relative_state=prior_state,
        prior_covariance_diagonal=prior_cov,
        posterior_relative_state=posterior_state,
        posterior_covariance_diagonal=posterior_cov,
        confidence=float(confidence),
        existence_probability=float(existence_probability),
        valid=bool(valid),
        local_candidate_weights=weights,
        local_candidate_index=int(local_candidate_index),
        innovation_distance=float(innovation_distance),
    )


def empty_association_output(
    *,
    prior_relative_state: np.ndarray | None = None,
    prior_covariance_diagonal: np.ndarray | None = None,
    global_target_index: int = -1,
) -> AssociationOutput:
    """Backward-compatible alias for an empty single-track posterior."""

    return empty_key_track_posterior(
        global_target_index=global_target_index,
        prior_relative_state=prior_relative_state,
        prior_covariance_diagonal=prior_covariance_diagonal,
    )


def empty_multi_target_track_output(
    *,
    agent_id: str,
    prior_relative_states: np.ndarray,
    prior_covariance_diagonal: np.ndarray,
    mode: str = "local_only",
) -> MultiTargetTrackOutput:
    priors = _as_state_batch(prior_relative_states, name="prior_relative_states")
    prior_covariance = _as_covariance_diagonal(prior_covariance_diagonal, name="prior_covariance_diagonal")
    tracks = tuple(
        empty_key_track_posterior(
            global_target_index=target_idx,
            prior_relative_state=priors[target_idx] if priors.shape[0] else None,
            prior_covariance_diagonal=prior_covariance,
        )
        for target_idx in range(priors.shape[0])
    )
    return MultiTargetTrackOutput(agent_id=str(agent_id), tracks=tracks, mode=str(mode))


def flatten_track_output(track_output: MultiTargetTrackOutput) -> np.ndarray:
    """Flatten a multi-target posterior set into a fixed-order feature vector."""

    ordered_tracks = sorted(track_output.tracks, key=lambda track: int(track.global_target_index))
    if not ordered_tracks:
        return np.zeros(0, dtype=np.float32)
    chunks: list[np.ndarray] = []
    for track in ordered_tracks:
        chunks.append(
            np.concatenate(
                [
                    track.prior_relative_state,
                    track.posterior_relative_state,
                    np.sqrt(track.prior_covariance_diagonal),
                    np.sqrt(track.posterior_covariance_diagonal),
                    np.asarray(
                        [1.0 if track.valid else 0.0, float(track.confidence)],
                        dtype=np.float32,
                    ),
                ]
            ).astype(np.float32)
        )
    return np.concatenate(chunks, dtype=np.float32)


def association_output_to_weight_vector(
    association_output: AssociationOutput,
    *,
    max_local_candidates: int,
) -> np.ndarray:
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


def _bp_jpda_predict_track_state(
    *,
    previous_filter_state: BPJPDAFilterState | None,
    prior_relative_state: np.ndarray,
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
) -> tuple[np.ndarray, np.ndarray, IMMMotionState]:
    anchor_mean = _as_state_vector(prior_relative_state, name="prior_relative_state")
    anchor_covariance = np.diag(_as_covariance_diagonal(prior_covariance_diagonal, name="prior_covariance_diagonal"))
    if previous_filter_state is None:
        motion_state = _default_motion_state(
            anchor_mean,
            anchor_covariance,
            recursive_motion_model=recursive_motion_model,
            imm_model_types=imm_model_types,
            imm_model_transition_matrix=imm_model_transition_matrix,
            imm_initial_model_probabilities=imm_initial_model_probabilities,
        )
        return anchor_mean, anchor_covariance, motion_state

    previous_motion_state = _coerce_motion_state(
        previous_filter_state.motion_state,
        fallback_mean=previous_filter_state.mean,
        fallback_covariance=previous_filter_state.covariance,
        recursive_motion_model=recursive_motion_model,
        imm_model_types=imm_model_types,
        imm_model_transition_matrix=imm_model_transition_matrix,
        imm_initial_model_probabilities=imm_initial_model_probabilities,
    )
    propagated_motion_state = _predict_motion_state(
        previous_motion_state,
        recursive_motion_model=recursive_motion_model,
        imm_model_types=imm_model_types,
        imm_model_transition_matrix=imm_model_transition_matrix,
        imm_initial_model_probabilities=imm_initial_model_probabilities,
        dt_seconds=dt_seconds,
        process_position_std=process_position_std,
        process_velocity_std=process_velocity_std,
        imm_turn_rate_radps=imm_turn_rate_radps,
        imm_ct_process_noise_scale=imm_ct_process_noise_scale,
    )
    anchored_motion_state = _anchor_motion_state(
        propagated_motion_state,
        anchor_mean=anchor_mean,
        anchor_covariance=anchor_covariance,
        survival_probability=survival_probability,
        birth_weight=birth_weight,
    )
    fused_mean, fused_covariance = _collapse_motion_state(anchored_motion_state)
    return fused_mean, fused_covariance, anchored_motion_state


def _gm_phd_create_component(
    weight: float,
    mean: np.ndarray,
    covariance: np.ndarray,
    *,
    motion_state: IMMMotionState | None = None,
) -> GMPHDComponent:
    return GMPHDComponent(
        weight=float(max(weight, 0.0)),
        mean=_as_state_vector(mean, name="mean").astype(float, copy=True),
        covariance=_as_covariance_matrix(covariance, name="covariance").astype(float, copy=True),
        motion_state=motion_state,
    )


def _gm_phd_predict_components(
    *,
    previous_filter_state: GMPHDFilterState | None,
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
) -> list[GMPHDComponent]:
    predicted: list[GMPHDComponent] = []

    if previous_filter_state is not None:
        for component in previous_filter_state.components:
            previous_motion_state = _coerce_motion_state(
                component.motion_state,
                fallback_mean=component.mean,
                fallback_covariance=component.covariance,
                recursive_motion_model=recursive_motion_model,
                imm_model_types=imm_model_types,
                imm_model_transition_matrix=imm_model_transition_matrix,
                imm_initial_model_probabilities=imm_initial_model_probabilities,
            )
            predicted_motion_state = _predict_motion_state(
                previous_motion_state,
                recursive_motion_model=recursive_motion_model,
                imm_model_types=imm_model_types,
                imm_model_transition_matrix=imm_model_transition_matrix,
                imm_initial_model_probabilities=imm_initial_model_probabilities,
                dt_seconds=dt_seconds,
                process_position_std=process_position_std,
                process_velocity_std=process_velocity_std,
                imm_turn_rate_radps=imm_turn_rate_radps,
                imm_ct_process_noise_scale=imm_ct_process_noise_scale,
            )
            predicted_mean, predicted_covariance = _collapse_motion_state(predicted_motion_state)
            predicted.append(
                _gm_phd_create_component(
                    float(component.weight) * float(survival_probability),
                    predicted_mean,
                    predicted_covariance,
                    motion_state=predicted_motion_state,
                )
            )

    birth_covariance = np.diag(prior_covariance_diagonal)
    for prior_relative_state in _as_state_batch(prior_relative_states, name="prior_relative_states"):
        predicted.append(
            _gm_phd_create_component(
                float(max(birth_weight, 0.0)),
                prior_relative_state,
                birth_covariance,
                motion_state=_default_motion_state(
                    prior_relative_state,
                    birth_covariance,
                    recursive_motion_model=recursive_motion_model,
                    imm_model_types=imm_model_types,
                    imm_model_transition_matrix=imm_model_transition_matrix,
                    imm_initial_model_probabilities=imm_initial_model_probabilities,
                ),
            )
        )
    return predicted


def _gm_phd_merge_components(
    components: Sequence[GMPHDComponent],
    *,
    merge_mahalanobis_threshold: float,
) -> list[GMPHDComponent]:
    if not components:
        return []

    remaining = sorted(components, key=lambda component: float(component.weight), reverse=True)
    merged: list[GMPHDComponent] = []
    threshold_sq = float(merge_mahalanobis_threshold) ** 2

    while remaining:
        anchor = remaining.pop(0)
        cluster = [anchor]
        survivors: list[GMPHDComponent] = []
        anchor_inv_covariance = np.linalg.inv(anchor.covariance)
        for component in remaining:
            delta = component.mean - anchor.mean
            mahalanobis_sq = float(delta.T @ anchor_inv_covariance @ delta)
            if mahalanobis_sq <= threshold_sq:
                cluster.append(component)
            else:
                survivors.append(component)
        remaining = survivors

        cluster_weights = np.asarray([component.weight for component in cluster], dtype=float)
        total_weight = float(np.sum(cluster_weights))
        if total_weight <= 0.0:
            continue
        cluster_means = np.stack([component.mean for component in cluster], axis=0)
        merged_mean = np.average(cluster_means, axis=0, weights=cluster_weights)

        merged_covariance = np.zeros((STATE_DIM, STATE_DIM), dtype=float)
        for component in cluster:
            delta = (component.mean - merged_mean).reshape(STATE_DIM, 1)
            merged_covariance += float(component.weight) * (component.covariance + delta @ delta.T)
        merged_covariance /= total_weight
        merged_motion_state = anchor.motion_state if len(cluster) == 1 else None
        merged.append(
            _gm_phd_create_component(
                total_weight,
                merged_mean,
                merged_covariance,
                motion_state=merged_motion_state,
            )
        )

    return merged


def _gm_phd_prune_and_cap_components(
    components: Sequence[GMPHDComponent],
    *,
    prune_threshold: float,
    merge_mahalanobis_threshold: float,
    max_components: int,
) -> tuple[GMPHDComponent, ...]:
    kept = [component for component in components if float(component.weight) >= float(prune_threshold)]
    if not kept:
        return tuple()
    merged = _gm_phd_merge_components(
        kept,
        merge_mahalanobis_threshold=merge_mahalanobis_threshold,
    )
    merged.sort(key=lambda component: float(component.weight), reverse=True)
    capped = merged[: max(int(max_components), 1)]
    return tuple(capped)


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


def _nearest_measurement_support(
    posterior_mean: np.ndarray,
    local_relative_states: np.ndarray,
    valid_indices: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    support = np.zeros(local_relative_states.shape[0], dtype=np.float32)
    if valid_indices.size == 0:
        return support, -1, np.inf
    candidate_states = local_relative_states[valid_indices]
    distances = np.linalg.norm(candidate_states[:, :3] - posterior_mean[:3].reshape(1, 3), axis=1)
    best_offset = int(np.argmin(distances))
    best_index = int(valid_indices[best_offset])
    support[best_index] = 1.0
    return support, best_index, float(distances[best_offset])


def nearest_local_candidate_association_for_friendlies(
    *,
    friendly_states: np.ndarray,
    global_target_priors: GlobalKeyTargetStatePrior,
    local_observations: dict[str, LocalTargetObservation],
    agent_ids: Sequence[str],
) -> dict[str, MultiTargetTrackOutput]:
    """Nearest-candidate baseline over all key-target priors for each friendly UAV."""

    friendly_array = _as_state_batch(friendly_states, name="friendly_states")
    if friendly_array.shape[0] != len(agent_ids):
        raise ValueError("friendly_states row count must match agent_ids length.")

    outputs: dict[str, MultiTargetTrackOutput] = {}
    for friendly_idx, agent_id in enumerate(agent_ids):
        prior_relative_states, prior_covariance = _build_prior_relative_states(
            friendly_array[friendly_idx],
            global_target_priors,
        )
        local_relative_states, local_mask, local_covariance = _validate_local_observation(local_observations[str(agent_id)])
        valid_indices = np.flatnonzero(local_mask > 0.0)
        tracks: list[KeyTrackPosterior] = []
        for target_idx in range(prior_relative_states.shape[0]):
            prior_relative_state = prior_relative_states[target_idx]
            if valid_indices.size == 0:
                tracks.append(
                    empty_key_track_posterior(
                        global_target_index=target_idx,
                        prior_relative_state=prior_relative_state,
                        prior_covariance_diagonal=prior_covariance,
                    )
                )
                continue

            candidate_states = local_relative_states[valid_indices]
            distances = np.linalg.norm(candidate_states[:, :3] - prior_relative_state[:3].reshape(1, 3), axis=1)
            best_offset = int(np.argmin(distances))
            best_index = int(valid_indices[best_offset])
            weights = np.zeros(local_relative_states.shape[0], dtype=np.float32)
            weights[best_index] = 1.0
            tracks.append(
                KeyTrackPosterior(
                    global_target_index=target_idx,
                    prior_relative_state=prior_relative_state.astype(np.float32),
                    prior_covariance_diagonal=prior_covariance.astype(np.float32),
                    posterior_relative_state=local_relative_states[best_index].astype(np.float32),
                    posterior_covariance_diagonal=local_covariance.astype(np.float32),
                    confidence=1.0,
                    existence_probability=1.0,
                    valid=True,
                    local_candidate_weights=weights,
                    local_candidate_index=best_index,
                    innovation_distance=float(distances[best_offset]),
                )
            )
        outputs[str(agent_id)] = MultiTargetTrackOutput(
            agent_id=str(agent_id),
            tracks=tuple(tracks),
            mode="local_only",
        )
    return outputs


def _build_bp_jpda_context(
    *,
    agent_id: str,
    friendly_state: np.ndarray,
    global_target_priors: GlobalKeyTargetStatePrior,
    local_observation: LocalTargetObservation,
    previous_filter_state: dict[int, BPJPDAFilterState],
    detection_radius_m: float,
    relative_velocity_limit_mps: float,
    local_detection_probability: float,
    clutter_rate_per_second: float,
    recursive_motion_model: str,
    imm_model_types: Sequence[str],
    imm_model_transition_matrix: np.ndarray | Sequence[Sequence[float]] | None,
    imm_initial_model_probabilities: np.ndarray | Sequence[float] | None,
    dt_seconds: float,
    gate_sigma: float,
    missed_detection_probability: float,
    survival_probability: float,
    birth_weight: float,
    process_position_std: np.ndarray,
    process_velocity_std: np.ndarray,
    imm_turn_rate_radps: float,
    imm_ct_process_noise_scale: float,
) -> _BPJPDAAgentContext:
    friendly = _as_state_vector(friendly_state, name="friendly_state")
    prior_relative_states, prior_covariance = _build_prior_relative_states(
        friendly,
        global_target_priors,
    )
    local_relative_states, local_mask, local_covariance = _validate_local_observation(local_observation)
    valid_indices = np.flatnonzero(local_mask > 0.0)

    predicted_means: list[np.ndarray] = []
    predicted_covariances: list[np.ndarray] = []
    predicted_motion_states: list[IMMMotionState] = []
    track_active = np.ones(prior_relative_states.shape[0], dtype=bool)
    for target_idx in range(prior_relative_states.shape[0]):
        predicted_mean, predicted_covariance, predicted_motion_state = _bp_jpda_predict_track_state(
            previous_filter_state=previous_filter_state.get(target_idx),
            prior_relative_state=prior_relative_states[target_idx],
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
        )
        predicted_means.append(predicted_mean)
        predicted_covariances.append(predicted_covariance)
        predicted_motion_states.append(predicted_motion_state)
        innovation_covariance_diagonal = np.diag(predicted_covariance) + local_covariance
        prior_position_distance = float(np.linalg.norm(predicted_mean[:3]))
        if prior_position_distance > float(detection_radius_m) + _position_gate_margin(
            innovation_covariance_diagonal,
            gate_sigma,
        ):
            track_active[target_idx] = False

    candidate_states = local_relative_states[valid_indices] if valid_indices.size else np.zeros((0, STATE_DIM), dtype=float)
    likelihood_ratios = np.zeros((prior_relative_states.shape[0], valid_indices.size), dtype=float)

    position_volume = (4.0 / 3.0) * np.pi * max(float(detection_radius_m), 1.0e-6) ** 3
    velocity_support = max(float(relative_velocity_limit_mps), 1.0e-6)
    velocity_volume = np.power(2.0 * velocity_support, 3)
    measurement_volume = max(position_volume * velocity_volume, 1.0e-9)
    clutter_rate_per_step = max(float(clutter_rate_per_second), 0.0) * max(float(dt_seconds), 1.0e-6)
    clutter_intensity = max(clutter_rate_per_step / measurement_volume, 1.0e-9)

    for target_idx in range(prior_relative_states.shape[0]):
        if not track_active[target_idx] or valid_indices.size == 0:
            continue
        predicted_mean = predicted_means[target_idx]
        predicted_covariance = predicted_covariances[target_idx]
        innovation_covariance_diagonal = np.diag(predicted_covariance) + local_covariance
        residuals = candidate_states - predicted_mean.reshape(1, STATE_DIM)
        squared_mahalanobis = np.sum(
            np.square(residuals) / innovation_covariance_diagonal.reshape(1, STATE_DIM),
            axis=1,
        )
        gated_offsets = np.flatnonzero(squared_mahalanobis <= float(gate_sigma) ** 2)
        if gated_offsets.size == 0:
            continue
        gaussian_likelihood = _gaussian_pdf_zero_mean_diag(
            residuals[gated_offsets],
            innovation_covariance_diagonal,
        )
        likelihood_ratios[target_idx, gated_offsets] = (
            local_detection_probability * gaussian_likelihood / clutter_intensity
        ) / missed_detection_probability

    return _BPJPDAAgentContext(
        agent_id=str(agent_id),
        friendly_state=friendly.astype(float, copy=True),
        prior_relative_states=prior_relative_states,
        prior_covariance=prior_covariance,
        local_relative_states=local_relative_states,
        local_covariance=local_covariance,
        valid_indices=valid_indices,
        candidate_states=candidate_states,
        predicted_means=tuple(predicted_means),
        predicted_covariances=tuple(predicted_covariances),
        predicted_motion_states=tuple(predicted_motion_states),
        track_active=track_active,
        likelihood_ratios=likelihood_ratios,
    )


def _softmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros_like(values, dtype=float)
    shifted = values - float(np.max(values))
    weights = np.exp(shifted)
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.full(values.shape, 1.0 / max(values.size, 1), dtype=float)
    return weights / total


def _cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(vector_a))
    norm_b = float(np.linalg.norm(vector_b))
    if norm_a <= 1.0e-6 or norm_b <= 1.0e-6:
        return 0.0
    return float(np.clip(float(vector_a @ vector_b) / (norm_a * norm_b), -1.0, 1.0))


def _candidate_density_scores(candidate_states: np.ndarray, *, radius_m: float) -> np.ndarray:
    if candidate_states.shape[0] == 0:
        return np.zeros(0, dtype=float)
    position_deltas = candidate_states[:, None, :3] - candidate_states[None, :, :3]
    distances = np.linalg.norm(position_deltas, axis=-1)
    neighbor_counts = np.sum((distances <= float(radius_m)) & (distances > 0.0), axis=1)
    denominator = max(candidate_states.shape[0] - 1, 1)
    return neighbor_counts.astype(float) / float(denominator)


def _normalized_entropy(probabilities: np.ndarray) -> float:
    values = np.asarray(probabilities, dtype=float).reshape(-1)
    if values.size <= 1:
        return 0.0
    clipped = np.maximum(values, 1.0e-12)
    entropy = -float(np.sum(clipped * np.log(clipped)))
    return float(np.clip(entropy / np.log(float(values.size)), 0.0, 1.0))


def _run_sinkhorn(
    log_scores: np.ndarray,
    row_marginals: np.ndarray,
    column_marginals: np.ndarray,
    *,
    iterations: int,
    temperature: float,
) -> np.ndarray:
    scores = np.asarray(log_scores, dtype=float)
    rows = np.asarray(row_marginals, dtype=float).reshape(-1)
    columns = np.asarray(column_marginals, dtype=float).reshape(-1)
    if scores.shape != (rows.shape[0], columns.shape[0]):
        raise ValueError(
            "log_scores shape must match row and column marginals, "
            f"got {scores.shape}, {rows.shape}, and {columns.shape}."
        )
    if iterations <= 0:
        raise ValueError(f"iterations must be > 0, got {iterations}.")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}.")

    row_total = float(np.sum(rows))
    column_total = float(np.sum(columns))
    if row_total <= 0.0 or column_total <= 0.0:
        return np.zeros_like(scores, dtype=float)
    columns = columns * (row_total / column_total)

    kernel = np.exp(np.clip(scores / float(temperature), -40.0, 40.0))
    kernel = np.maximum(kernel, 1.0e-12)
    for _ in range(int(iterations)):
        row_sums = np.sum(kernel, axis=1)
        kernel *= (rows / np.maximum(row_sums, 1.0e-12)).reshape(-1, 1)
        column_sums = np.sum(kernel, axis=0)
        kernel *= (columns / np.maximum(column_sums, 1.0e-12)).reshape(1, -1)
    return kernel


def _enhance_likelihoods_with_aa_bp_jpda(
    *,
    context: _BPJPDAAgentContext,
    all_contexts: Sequence[_BPJPDAAgentContext],
    gate_sigma: float,
    local_attention_temperature: float,
    cross_uav_position_scale_m: float,
    cross_uav_velocity_scale_mps: float,
    cross_uav_support_threshold: float,
    local_evidence_weight: float,
    velocity_evidence_weight: float,
    density_penalty_weight: float,
    consensus_evidence_weight: float,
    consensus_count_weight: float,
    max_log_evidence: float,
) -> np.ndarray:
    # AA-BP-JPDA keeps the baseline gated likelihoods, then reweights them with
    # local target-candidate attention and cross-UAV candidate consensus evidence.
    enhanced = np.asarray(context.likelihood_ratios, dtype=float).copy()
    if enhanced.size == 0:
        return enhanced

    candidate_density = _candidate_density_scores(context.candidate_states, radius_m=cross_uav_position_scale_m)
    agent_count = len(all_contexts)
    other_agent_denominator = max(agent_count - 1, 1)

    for target_idx in range(enhanced.shape[0]):
        active_offsets = np.flatnonzero(enhanced[target_idx] > 0.0)
        if active_offsets.size == 0:
            continue

        predicted_mean = context.predicted_means[target_idx]
        predicted_covariance = context.predicted_covariances[target_idx]
        innovation_covariance_diagonal = np.diag(predicted_covariance) + context.local_covariance
        residuals = context.candidate_states[active_offsets] - predicted_mean.reshape(1, STATE_DIM)
        mahalanobis_sq = np.sum(
            np.square(residuals) / innovation_covariance_diagonal.reshape(1, STATE_DIM),
            axis=1,
        )
        local_attention = _softmax(-mahalanobis_sq / max(float(local_attention_temperature), 1.0e-6))
        ambiguity_gate = _normalized_entropy(local_attention)

        candidate_batch = context.candidate_states[active_offsets]
        candidate_global_batch = context.friendly_state.reshape(1, STATE_DIM) + candidate_batch
        local_attention_gain = np.log(np.maximum(local_attention * float(active_offsets.size), 1.0e-6))
        velocity_alignment = np.asarray(
            [_cosine_similarity(predicted_mean[3:6], candidate_state[3:6]) for candidate_state in candidate_batch],
            dtype=float,
        )
        normalized_mahalanobis = np.sqrt(np.maximum(mahalanobis_sq, 0.0)) / max(float(gate_sigma), 1.0e-6)
        local_scores = (
            float(local_evidence_weight) * (local_attention_gain - normalized_mahalanobis)
            + float(velocity_evidence_weight) * velocity_alignment
            - float(density_penalty_weight) * candidate_density[active_offsets]
        )

        consensus_scores = np.zeros(active_offsets.size, dtype=float)
        if ambiguity_gate > 0.0:
            consensus_support = np.zeros(active_offsets.size, dtype=float)
            supporting_agents = np.zeros(active_offsets.size, dtype=float)
            for other_context in all_contexts:
                if other_context.agent_id == context.agent_id:
                    continue
                if target_idx >= other_context.likelihood_ratios.shape[0]:
                    continue
                other_offsets = np.flatnonzero(other_context.likelihood_ratios[target_idx] > 0.0)
                if other_offsets.size == 0:
                    continue
                other_global_states = other_context.friendly_state.reshape(1, STATE_DIM) + other_context.candidate_states[
                    other_offsets
                ]
                position_distances = np.linalg.norm(
                    candidate_global_batch[:, None, :3] - other_global_states[None, :, :3],
                    axis=-1,
                )
                velocity_distances = np.linalg.norm(
                    candidate_global_batch[:, None, 3:6] - other_global_states[None, :, 3:6],
                    axis=-1,
                )
                support_kernel = np.exp(
                    -0.5 * np.square(position_distances / max(float(cross_uav_position_scale_m), 1.0e-6))
                    -0.5 * np.square(velocity_distances / max(float(cross_uav_velocity_scale_mps), 1.0e-6))
                )
                best_support = np.max(support_kernel, axis=1)
                consensus_support += best_support
                supporting_agents += (best_support >= float(cross_uav_support_threshold)).astype(float)

            consensus_ratio = consensus_support / float(other_agent_denominator)
            support_count_ratio = supporting_agents / float(other_agent_denominator)
            consensus_scores = ambiguity_gate * (
                float(consensus_evidence_weight) * (consensus_ratio - 0.5)
                + float(consensus_count_weight) * support_count_ratio
            )

        evidence = np.clip(local_scores + consensus_scores, -float(max_log_evidence), float(max_log_evidence))
        enhanced[target_idx, active_offsets] *= np.exp(evidence)

    return enhanced


def _build_bp_jpda_tracks_from_marginals(
    *,
    context: _BPJPDAAgentContext,
    marginals: np.ndarray,
    confidence_threshold: float,
) -> tuple[MultiTargetTrackOutput, dict[int, BPJPDAFilterState]]:
    agent_tracks: list[KeyTrackPosterior] = []
    agent_next_states: dict[int, BPJPDAFilterState] = {}
    for target_idx in range(context.prior_relative_states.shape[0]):
        predicted_mean = context.predicted_means[target_idx]
        predicted_covariance = context.predicted_covariances[target_idx]
        predicted_motion_state = context.predicted_motion_states[target_idx]
        full_weights = np.zeros(context.local_relative_states.shape[0], dtype=np.float32)
        if context.valid_indices.size:
            full_weights[context.valid_indices] = marginals[target_idx].astype(np.float32)
        confidence = float(np.clip(np.sum(full_weights), 0.0, 1.0))
        missed_probability = float(np.clip(1.0 - confidence, 0.0, 1.0))

        posterior_motion_states: list[IMMMotionState] = [predicted_motion_state]
        posterior_weights: list[float] = [missed_probability]
        best_local_index = -1
        best_innovation_distance = np.inf
        for candidate_offset in range(context.valid_indices.size):
            candidate_weight = float(marginals[target_idx, candidate_offset])
            if candidate_weight <= 0.0:
                continue
            posterior_motion_state_j, innovation_distance = _update_motion_state_with_measurement(
                predicted_motion_state,
                measurement_state=context.candidate_states[candidate_offset],
                measurement_covariance_diagonal=context.local_covariance,
            )
            posterior_motion_states.append(posterior_motion_state_j)
            posterior_weights.append(candidate_weight)
            if innovation_distance < best_innovation_distance:
                best_innovation_distance = innovation_distance
                best_local_index = int(context.valid_indices[candidate_offset])

        if float(np.sum(posterior_weights)) <= 0.0:
            posterior_motion_states = [predicted_motion_state]
            posterior_weights = [1.0]

        posterior_motion_state = _merge_motion_hypotheses(
            posterior_motion_states,
            posterior_weights,
        )
        posterior_state, posterior_covariance = _collapse_motion_state(posterior_motion_state)
        posterior_covariance_diagonal = np.diag(posterior_covariance)
        valid = bool(context.track_active[target_idx] and confidence >= confidence_threshold and np.any(full_weights > 0.0))
        if best_local_index < 0 and context.valid_indices.size:
            weight_support, nearest_index, nearest_distance = _nearest_measurement_support(
                posterior_state,
                context.local_relative_states,
                context.valid_indices,
            )
            if not np.any(full_weights > 0.0):
                full_weights = weight_support
            if nearest_index >= 0:
                best_local_index = nearest_index
                best_innovation_distance = nearest_distance

        agent_tracks.append(
            KeyTrackPosterior(
                global_target_index=target_idx,
                prior_relative_state=context.prior_relative_states[target_idx].astype(np.float32),
                prior_covariance_diagonal=context.prior_covariance.astype(np.float32),
                posterior_relative_state=posterior_state.astype(np.float32),
                posterior_covariance_diagonal=posterior_covariance_diagonal.astype(np.float32),
                confidence=confidence,
                existence_probability=confidence,
                valid=valid,
                local_candidate_weights=full_weights,
                local_candidate_index=best_local_index,
                innovation_distance=float(best_innovation_distance),
            )
        )
        agent_next_states[target_idx] = BPJPDAFilterState(
            mean=posterior_state.astype(float, copy=True),
            covariance=posterior_covariance.astype(float, copy=True),
            motion_state=posterior_motion_state,
        )

    return (
        MultiTargetTrackOutput(
            agent_id=context.agent_id,
            tracks=tuple(agent_tracks),
            mode="local_only",
        ),
        agent_next_states,
    )


def bp_jpda_association_for_friendlies(
    *,
    friendly_states: np.ndarray,
    global_target_priors: GlobalKeyTargetStatePrior,
    local_observations: dict[str, LocalTargetObservation],
    agent_ids: Sequence[str],
    previous_filter_states: dict[str, dict[int, BPJPDAFilterState]] | None,
    detection_radius_m: float,
    relative_velocity_limit_mps: float,
    local_detection_probability: float,
    clutter_rate_per_second: float,
    recursive_motion_model: str = "imm",
    imm_model_types: Sequence[str] = ("cv", "ca", "ct_left", "ct_right"),
    imm_model_transition_matrix: np.ndarray | Sequence[Sequence[float]] | None = None,
    imm_initial_model_probabilities: np.ndarray | Sequence[float] | None = None,
    dt_seconds: float,
    gate_sigma: float = 3.0,
    max_iterations: int = 20,
    convergence_tol: float = 1.0e-3,
    min_missed_detection_probability: float = 0.05,
    survival_probability: float = 0.98,
    birth_weight: float = 0.25,
    process_position_std_m: np.ndarray | Sequence[float] = (2.0, 2.0, 1.0),
    process_velocity_std_mps: np.ndarray | Sequence[float] = (1.0, 1.0, 0.5),
    imm_turn_rate_radps: float = 0.35,
    imm_ct_process_noise_scale: float = 1.25,
    confidence_threshold: float = 0.1,
) -> tuple[dict[str, MultiTargetTrackOutput], dict[str, dict[int, BPJPDAFilterState]]]:
    """Run recursive multi-target BP-JPDA for each friendly UAV."""

    friendly_array = _as_state_batch(friendly_states, name="friendly_states")
    if friendly_array.shape[0] != len(agent_ids):
        raise ValueError("friendly_states row count must match agent_ids length.")
    if max_iterations <= 0:
        raise ValueError(f"max_iterations must be > 0, got {max_iterations}.")
    if convergence_tol <= 0.0:
        raise ValueError(f"convergence_tol must be > 0, got {convergence_tol}.")
    if gate_sigma <= 0.0:
        raise ValueError(f"gate_sigma must be > 0, got {gate_sigma}.")

    previous_states = previous_filter_states or {}
    process_position_std = np.asarray(process_position_std_m, dtype=float).reshape(3)
    process_velocity_std = np.asarray(process_velocity_std_mps, dtype=float).reshape(3)
    local_detection_probability = float(np.clip(local_detection_probability, 0.0, 1.0))
    missed_detection_probability = max(1.0 - local_detection_probability, float(min_missed_detection_probability))
    confidence_threshold = float(np.clip(confidence_threshold, 0.0, 1.0))
    position_volume = (4.0 / 3.0) * np.pi * max(float(detection_radius_m), 1.0e-6) ** 3
    velocity_support = max(float(relative_velocity_limit_mps), 1.0e-6)
    velocity_volume = np.power(2.0 * velocity_support, 3)
    measurement_volume = max(position_volume * velocity_volume, 1.0e-9)
    clutter_rate_per_step = max(float(clutter_rate_per_second), 0.0) * max(float(dt_seconds), 1.0e-6)
    clutter_intensity = max(clutter_rate_per_step / measurement_volume, 1.0e-9)

    outputs: dict[str, MultiTargetTrackOutput] = {}
    next_filter_states: dict[str, dict[int, BPJPDAFilterState]] = {}
    for friendly_idx, agent_id in enumerate(agent_ids):
        prior_relative_states, prior_covariance = _build_prior_relative_states(
            friendly_array[friendly_idx],
            global_target_priors,
        )
        local_relative_states, local_mask, local_covariance = _validate_local_observation(local_observations[str(agent_id)])
        valid_indices = np.flatnonzero(local_mask > 0.0)

        predicted_means: list[np.ndarray] = []
        predicted_covariances: list[np.ndarray] = []
        predicted_motion_states: list[IMMMotionState] = []
        track_active = np.ones(prior_relative_states.shape[0], dtype=bool)
        agent_previous_states = previous_states.get(str(agent_id), {})
        for target_idx in range(prior_relative_states.shape[0]):
            predicted_mean, predicted_covariance, predicted_motion_state = _bp_jpda_predict_track_state(
                previous_filter_state=agent_previous_states.get(target_idx),
                prior_relative_state=prior_relative_states[target_idx],
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
            )
            predicted_means.append(predicted_mean)
            predicted_covariances.append(predicted_covariance)
            predicted_motion_states.append(predicted_motion_state)
            innovation_covariance_diagonal = np.diag(predicted_covariance) + local_covariance
            prior_position_distance = float(np.linalg.norm(predicted_mean[:3]))
            if prior_position_distance > float(detection_radius_m) + _position_gate_margin(
                innovation_covariance_diagonal,
                gate_sigma,
            ):
                track_active[target_idx] = False

        candidate_states = local_relative_states[valid_indices] if valid_indices.size else np.zeros((0, STATE_DIM), dtype=float)
        likelihood_ratios = np.zeros((prior_relative_states.shape[0], valid_indices.size), dtype=float)
        for target_idx in range(prior_relative_states.shape[0]):
            if not track_active[target_idx] or valid_indices.size == 0:
                continue
            predicted_mean = predicted_means[target_idx]
            predicted_covariance = predicted_covariances[target_idx]
            innovation_covariance_diagonal = np.diag(predicted_covariance) + local_covariance
            residuals = candidate_states - predicted_mean.reshape(1, STATE_DIM)
            squared_mahalanobis = np.sum(
                np.square(residuals) / innovation_covariance_diagonal.reshape(1, STATE_DIM),
                axis=1,
            )
            gated_offsets = np.flatnonzero(squared_mahalanobis <= float(gate_sigma) ** 2)
            if gated_offsets.size == 0:
                continue
            gaussian_likelihood = _gaussian_pdf_zero_mean_diag(
                residuals[gated_offsets],
                innovation_covariance_diagonal,
            )
            likelihood_ratios[target_idx, gated_offsets] = (
                local_detection_probability * gaussian_likelihood / clutter_intensity
            ) / missed_detection_probability

        marginals = _run_bp_jpda(
            likelihood_ratios,
            max_iterations=max_iterations,
            convergence_tol=convergence_tol,
        ) if valid_indices.size and prior_relative_states.shape[0] else np.zeros_like(likelihood_ratios)

        agent_tracks: list[KeyTrackPosterior] = []
        agent_next_states: dict[int, BPJPDAFilterState] = {}
        for target_idx in range(prior_relative_states.shape[0]):
            predicted_mean = predicted_means[target_idx]
            predicted_covariance = predicted_covariances[target_idx]
            predicted_motion_state = predicted_motion_states[target_idx]
            predicted_covariance_diagonal = np.diag(predicted_covariance)
            full_weights = np.zeros(local_relative_states.shape[0], dtype=np.float32)
            if valid_indices.size:
                full_weights[valid_indices] = marginals[target_idx].astype(np.float32)
            confidence = float(np.clip(np.sum(full_weights), 0.0, 1.0))
            missed_probability = float(np.clip(1.0 - confidence, 0.0, 1.0))

            posterior_motion_states: list[IMMMotionState] = [predicted_motion_state]
            posterior_weights: list[float] = [missed_probability]
            best_local_index = -1
            best_innovation_distance = np.inf
            for candidate_offset in range(valid_indices.size):
                candidate_weight = float(marginals[target_idx, candidate_offset])
                if candidate_weight <= 0.0:
                    continue
                posterior_motion_state_j, innovation_distance = _update_motion_state_with_measurement(
                    predicted_motion_state,
                    measurement_state=candidate_states[candidate_offset],
                    measurement_covariance_diagonal=local_covariance,
                )
                posterior_motion_states.append(posterior_motion_state_j)
                posterior_weights.append(candidate_weight)
                if innovation_distance < best_innovation_distance:
                    best_innovation_distance = innovation_distance
                    best_local_index = int(valid_indices[candidate_offset])

            if float(np.sum(posterior_weights)) <= 0.0:
                posterior_motion_states = [predicted_motion_state]
                posterior_weights = [1.0]

            posterior_motion_state = _merge_motion_hypotheses(
                posterior_motion_states,
                posterior_weights,
            )
            posterior_state, posterior_covariance = _collapse_motion_state(posterior_motion_state)
            posterior_covariance_diagonal = np.diag(posterior_covariance)
            valid = bool(track_active[target_idx] and confidence >= confidence_threshold and np.any(full_weights > 0.0))
            if best_local_index < 0 and valid_indices.size:
                weight_support, nearest_index, nearest_distance = _nearest_measurement_support(
                    posterior_state,
                    local_relative_states,
                    valid_indices,
                )
                if not np.any(full_weights > 0.0):
                    full_weights = weight_support
                if nearest_index >= 0:
                    best_local_index = nearest_index
                    best_innovation_distance = nearest_distance

            agent_tracks.append(
                KeyTrackPosterior(
                    global_target_index=target_idx,
                    prior_relative_state=prior_relative_states[target_idx].astype(np.float32),
                    prior_covariance_diagonal=prior_covariance.astype(np.float32),
                    posterior_relative_state=posterior_state.astype(np.float32),
                    posterior_covariance_diagonal=posterior_covariance_diagonal.astype(np.float32),
                    confidence=confidence,
                    existence_probability=confidence,
                    valid=valid,
                    local_candidate_weights=full_weights,
                    local_candidate_index=best_local_index,
                    innovation_distance=float(best_innovation_distance),
                )
            )
            agent_next_states[target_idx] = BPJPDAFilterState(
                mean=posterior_state.astype(float, copy=True),
                covariance=posterior_covariance.astype(float, copy=True),
                motion_state=posterior_motion_state,
            )

        outputs[str(agent_id)] = MultiTargetTrackOutput(
            agent_id=str(agent_id),
            tracks=tuple(agent_tracks),
            mode="local_only",
        )
        next_filter_states[str(agent_id)] = agent_next_states
    return outputs, next_filter_states


def aa_bp_jpda_association_for_friendlies(
    *,
    friendly_states: np.ndarray,
    global_target_priors: GlobalKeyTargetStatePrior,
    local_observations: dict[str, LocalTargetObservation],
    agent_ids: Sequence[str],
    previous_filter_states: dict[str, dict[int, BPJPDAFilterState]] | None,
    detection_radius_m: float,
    relative_velocity_limit_mps: float,
    local_detection_probability: float,
    clutter_rate_per_second: float,
    recursive_motion_model: str = "imm",
    imm_model_types: Sequence[str] = ("cv", "ca", "ct_left", "ct_right"),
    imm_model_transition_matrix: np.ndarray | Sequence[Sequence[float]] | None = None,
    imm_initial_model_probabilities: np.ndarray | Sequence[float] | None = None,
    dt_seconds: float,
    gate_sigma: float = 3.0,
    max_iterations: int = 20,
    convergence_tol: float = 1.0e-3,
    min_missed_detection_probability: float = 0.05,
    survival_probability: float = 0.98,
    birth_weight: float = 0.25,
    process_position_std_m: np.ndarray | Sequence[float] = (2.0, 2.0, 1.0),
    process_velocity_std_mps: np.ndarray | Sequence[float] = (1.0, 1.0, 0.5),
    imm_turn_rate_radps: float = 0.35,
    imm_ct_process_noise_scale: float = 1.25,
    confidence_threshold: float = 0.1,
    local_attention_temperature: float = 2.0,
    cross_uav_position_scale_m: float = 8.0,
    cross_uav_velocity_scale_mps: float = 3.0,
    cross_uav_support_threshold: float = 0.35,
    local_evidence_weight: float = 0.6,
    velocity_evidence_weight: float = 0.25,
    density_penalty_weight: float = 0.35,
    consensus_evidence_weight: float = 0.9,
    consensus_count_weight: float = 0.45,
    max_log_evidence: float = 1.5,
) -> tuple[dict[str, MultiTargetTrackOutput], dict[str, dict[int, BPJPDAFilterState]]]:
    """Run BP-JPDA with local pair attention and cross-UAV consensus likelihood reweighting."""

    friendly_array = _as_state_batch(friendly_states, name="friendly_states")
    if friendly_array.shape[0] != len(agent_ids):
        raise ValueError("friendly_states row count must match agent_ids length.")
    if max_iterations <= 0:
        raise ValueError(f"max_iterations must be > 0, got {max_iterations}.")
    if convergence_tol <= 0.0:
        raise ValueError(f"convergence_tol must be > 0, got {convergence_tol}.")
    if gate_sigma <= 0.0:
        raise ValueError(f"gate_sigma must be > 0, got {gate_sigma}.")

    previous_states = previous_filter_states or {}
    process_position_std = np.asarray(process_position_std_m, dtype=float).reshape(3)
    process_velocity_std = np.asarray(process_velocity_std_mps, dtype=float).reshape(3)
    local_detection_probability = float(np.clip(local_detection_probability, 0.0, 1.0))
    missed_detection_probability = max(1.0 - local_detection_probability, float(min_missed_detection_probability))
    confidence_threshold = float(np.clip(confidence_threshold, 0.0, 1.0))

    contexts: list[_BPJPDAAgentContext] = []
    for friendly_idx, agent_id in enumerate(agent_ids):
        contexts.append(
            _build_bp_jpda_context(
                agent_id=str(agent_id),
                friendly_state=friendly_array[friendly_idx],
                global_target_priors=global_target_priors,
                local_observation=local_observations[str(agent_id)],
                previous_filter_state=previous_states.get(str(agent_id), {}),
                detection_radius_m=detection_radius_m,
                relative_velocity_limit_mps=relative_velocity_limit_mps,
                local_detection_probability=local_detection_probability,
                clutter_rate_per_second=clutter_rate_per_second,
                recursive_motion_model=recursive_motion_model,
                imm_model_types=imm_model_types,
                imm_model_transition_matrix=imm_model_transition_matrix,
                imm_initial_model_probabilities=imm_initial_model_probabilities,
                dt_seconds=dt_seconds,
                gate_sigma=gate_sigma,
                missed_detection_probability=missed_detection_probability,
                survival_probability=survival_probability,
                birth_weight=birth_weight,
                process_position_std=process_position_std,
                process_velocity_std=process_velocity_std,
                imm_turn_rate_radps=imm_turn_rate_radps,
                imm_ct_process_noise_scale=imm_ct_process_noise_scale,
            )
        )

    outputs: dict[str, MultiTargetTrackOutput] = {}
    next_filter_states: dict[str, dict[int, BPJPDAFilterState]] = {}
    for context in contexts:
        enhanced_likelihoods = _enhance_likelihoods_with_aa_bp_jpda(
            context=context,
            all_contexts=contexts,
            gate_sigma=gate_sigma,
            local_attention_temperature=local_attention_temperature,
            cross_uav_position_scale_m=cross_uav_position_scale_m,
            cross_uav_velocity_scale_mps=cross_uav_velocity_scale_mps,
            cross_uav_support_threshold=cross_uav_support_threshold,
            local_evidence_weight=local_evidence_weight,
            velocity_evidence_weight=velocity_evidence_weight,
            density_penalty_weight=density_penalty_weight,
            consensus_evidence_weight=consensus_evidence_weight,
            consensus_count_weight=consensus_count_weight,
            max_log_evidence=max_log_evidence,
        )
        marginals = _run_bp_jpda(
            enhanced_likelihoods,
            max_iterations=max_iterations,
            convergence_tol=convergence_tol,
        ) if context.valid_indices.size and context.prior_relative_states.shape[0] else np.zeros_like(enhanced_likelihoods)
        track_output, agent_next_states = _build_bp_jpda_tracks_from_marginals(
            context=context,
            marginals=marginals,
            confidence_threshold=confidence_threshold,
        )
        outputs[context.agent_id] = track_output
        next_filter_states[context.agent_id] = agent_next_states

    return outputs, next_filter_states


def _sinkhorn_attention_marginals(
    likelihood_ratios: np.ndarray,
    *,
    iterations: int,
    temperature: float,
    missed_dummy_score: float,
) -> np.ndarray:
    likelihoods = np.asarray(likelihood_ratios, dtype=float)
    if likelihoods.ndim != 2:
        raise ValueError(f"likelihood_ratios must be 2D, got {likelihoods.shape}.")
    num_targets, num_measurements = likelihoods.shape
    if num_targets == 0:
        return np.zeros_like(likelihoods, dtype=float)

    log_scores = np.full((num_targets + 1, num_measurements + 1), -40.0, dtype=float)
    valid_mask = likelihoods > 0.0
    log_scores[:num_targets, :num_measurements][valid_mask] = np.log(np.maximum(likelihoods[valid_mask], 1.0e-12))
    log_scores[:num_targets, num_measurements] = float(missed_dummy_score)
    if num_measurements > 0:
        log_scores[num_targets, :num_measurements] = 0.0
    log_scores[num_targets, num_measurements] = 0.0

    row_marginals = np.concatenate(
        [
            np.ones(num_targets, dtype=float),
            np.asarray([float(num_measurements)], dtype=float),
        ]
    )
    column_marginals = np.concatenate(
        [
            np.ones(num_measurements, dtype=float),
            np.asarray([float(num_targets)], dtype=float),
        ]
    )
    transport = _run_sinkhorn(
        log_scores,
        row_marginals,
        column_marginals,
        iterations=iterations,
        temperature=temperature,
    )
    marginals = transport[:num_targets, :num_measurements]
    return np.where(valid_mask, marginals, 0.0)


def sinkhorn_ajpda_association_for_friendlies(
    *,
    friendly_states: np.ndarray,
    global_target_priors: GlobalKeyTargetStatePrior,
    local_observations: dict[str, LocalTargetObservation],
    agent_ids: Sequence[str],
    previous_filter_states: dict[str, dict[int, BPJPDAFilterState]] | None,
    detection_radius_m: float,
    relative_velocity_limit_mps: float,
    local_detection_probability: float,
    clutter_rate_per_second: float,
    recursive_motion_model: str = "imm",
    imm_model_types: Sequence[str] = ("cv", "ca", "ct_left", "ct_right"),
    imm_model_transition_matrix: np.ndarray | Sequence[Sequence[float]] | None = None,
    imm_initial_model_probabilities: np.ndarray | Sequence[float] | None = None,
    dt_seconds: float,
    gate_sigma: float = 3.0,
    sinkhorn_iterations: int = 30,
    sinkhorn_temperature: float = 1.0,
    sinkhorn_missed_dummy_score: float = -0.5,
    min_missed_detection_probability: float = 0.05,
    survival_probability: float = 0.98,
    birth_weight: float = 0.25,
    process_position_std_m: np.ndarray | Sequence[float] = (2.0, 2.0, 1.0),
    process_velocity_std_mps: np.ndarray | Sequence[float] = (1.0, 1.0, 0.5),
    imm_turn_rate_radps: float = 0.35,
    imm_ct_process_noise_scale: float = 1.25,
    confidence_threshold: float = 0.1,
    local_attention_temperature: float = 2.0,
    cross_uav_position_scale_m: float = 8.0,
    cross_uav_velocity_scale_mps: float = 3.0,
    cross_uav_support_threshold: float = 0.35,
    local_evidence_weight: float = 0.6,
    velocity_evidence_weight: float = 0.25,
    density_penalty_weight: float = 0.35,
    consensus_evidence_weight: float = 0.9,
    consensus_count_weight: float = 0.45,
    max_log_evidence: float = 1.5,
) -> tuple[dict[str, MultiTargetTrackOutput], dict[str, dict[int, BPJPDAFilterState]]]:
    """Approximate JPDA marginals with attention evidence and Sinkhorn assignment constraints."""

    friendly_array = _as_state_batch(friendly_states, name="friendly_states")
    if friendly_array.shape[0] != len(agent_ids):
        raise ValueError("friendly_states row count must match agent_ids length.")
    if gate_sigma <= 0.0:
        raise ValueError(f"gate_sigma must be > 0, got {gate_sigma}.")

    previous_states = previous_filter_states or {}
    process_position_std = np.asarray(process_position_std_m, dtype=float).reshape(3)
    process_velocity_std = np.asarray(process_velocity_std_mps, dtype=float).reshape(3)
    local_detection_probability = float(np.clip(local_detection_probability, 0.0, 1.0))
    missed_detection_probability = max(1.0 - local_detection_probability, float(min_missed_detection_probability))
    confidence_threshold = float(np.clip(confidence_threshold, 0.0, 1.0))

    contexts: list[_BPJPDAAgentContext] = []
    for friendly_idx, agent_id in enumerate(agent_ids):
        contexts.append(
            _build_bp_jpda_context(
                agent_id=str(agent_id),
                friendly_state=friendly_array[friendly_idx],
                global_target_priors=global_target_priors,
                local_observation=local_observations[str(agent_id)],
                previous_filter_state=previous_states.get(str(agent_id), {}),
                detection_radius_m=detection_radius_m,
                relative_velocity_limit_mps=relative_velocity_limit_mps,
                local_detection_probability=local_detection_probability,
                clutter_rate_per_second=clutter_rate_per_second,
                recursive_motion_model=recursive_motion_model,
                imm_model_types=imm_model_types,
                imm_model_transition_matrix=imm_model_transition_matrix,
                imm_initial_model_probabilities=imm_initial_model_probabilities,
                dt_seconds=dt_seconds,
                gate_sigma=gate_sigma,
                missed_detection_probability=missed_detection_probability,
                survival_probability=survival_probability,
                birth_weight=birth_weight,
                process_position_std=process_position_std,
                process_velocity_std=process_velocity_std,
                imm_turn_rate_radps=imm_turn_rate_radps,
                imm_ct_process_noise_scale=imm_ct_process_noise_scale,
            )
        )

    outputs: dict[str, MultiTargetTrackOutput] = {}
    next_filter_states: dict[str, dict[int, BPJPDAFilterState]] = {}
    for context in contexts:
        enhanced_likelihoods = _enhance_likelihoods_with_aa_bp_jpda(
            context=context,
            all_contexts=contexts,
            gate_sigma=gate_sigma,
            local_attention_temperature=local_attention_temperature,
            cross_uav_position_scale_m=cross_uav_position_scale_m,
            cross_uav_velocity_scale_mps=cross_uav_velocity_scale_mps,
            cross_uav_support_threshold=cross_uav_support_threshold,
            local_evidence_weight=local_evidence_weight,
            velocity_evidence_weight=velocity_evidence_weight,
            density_penalty_weight=density_penalty_weight,
            consensus_evidence_weight=consensus_evidence_weight,
            consensus_count_weight=consensus_count_weight,
            max_log_evidence=max_log_evidence,
        )
        marginals = _sinkhorn_attention_marginals(
            enhanced_likelihoods,
            iterations=sinkhorn_iterations,
            temperature=sinkhorn_temperature,
            missed_dummy_score=sinkhorn_missed_dummy_score,
        ) if context.valid_indices.size and context.prior_relative_states.shape[0] else np.zeros_like(enhanced_likelihoods)
        track_output, agent_next_states = _build_bp_jpda_tracks_from_marginals(
            context=context,
            marginals=marginals,
            confidence_threshold=confidence_threshold,
        )
        outputs[context.agent_id] = track_output
        next_filter_states[context.agent_id] = agent_next_states

    return outputs, next_filter_states


def lmb_association_for_friendlies(
    *,
    friendly_states: np.ndarray,
    global_target_priors: GlobalKeyTargetStatePrior,
    local_observations: dict[str, LocalTargetObservation],
    agent_ids: Sequence[str],
    previous_filter_states: dict[str, dict[int, LMBFilterState]] | None,
    detection_radius_m: float,
    relative_velocity_limit_mps: float,
    local_detection_probability: float,
    clutter_rate_per_second: float,
    recursive_motion_model: str = "imm",
    imm_model_types: Sequence[str] = ("cv", "ca", "ct_left", "ct_right"),
    imm_model_transition_matrix: np.ndarray | Sequence[Sequence[float]] | None = None,
    imm_initial_model_probabilities: np.ndarray | Sequence[float] | None = None,
    dt_seconds: float,
    gate_sigma: float = 3.0,
    max_iterations: int = 20,
    convergence_tol: float = 1.0e-3,
    survival_probability: float = 0.98,
    birth_weight: float = 0.25,
    process_position_std_m: np.ndarray | Sequence[float] = (2.0, 2.0, 1.0),
    process_velocity_std_mps: np.ndarray | Sequence[float] = (1.0, 1.0, 0.5),
    imm_turn_rate_radps: float = 0.35,
    imm_ct_process_noise_scale: float = 1.25,
    confidence_threshold: float = 0.1,
) -> tuple[dict[str, MultiTargetTrackOutput], dict[str, dict[int, LMBFilterState]]]:
    """Run a single-sensor LBP-LMB baseline for each friendly UAV."""

    friendly_array = _as_state_batch(friendly_states, name="friendly_states")
    if friendly_array.shape[0] != len(agent_ids):
        raise ValueError("friendly_states row count must match agent_ids length.")
    if max_iterations <= 0:
        raise ValueError(f"max_iterations must be > 0, got {max_iterations}.")
    if convergence_tol <= 0.0:
        raise ValueError(f"convergence_tol must be > 0, got {convergence_tol}.")
    if gate_sigma <= 0.0:
        raise ValueError(f"gate_sigma must be > 0, got {gate_sigma}.")

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
    next_filter_states: dict[str, dict[int, LMBFilterState]] = {}
    for friendly_idx, agent_id in enumerate(agent_ids):
        prior_relative_states, prior_covariance = _build_prior_relative_states(
            friendly_array[friendly_idx],
            global_target_priors,
        )
        local_relative_states, local_mask, local_covariance = _validate_local_observation(local_observations[str(agent_id)])
        valid_indices = np.flatnonzero(local_mask > 0.0)

        predicted_means: list[np.ndarray] = []
        predicted_covariances: list[np.ndarray] = []
        predicted_motion_states: list[IMMMotionState] = []
        predicted_existences: list[float] = []
        track_active = np.ones(prior_relative_states.shape[0], dtype=bool)
        agent_previous_states = previous_states.get(str(agent_id)) or {}
        for target_idx in range(prior_relative_states.shape[0]):
            previous_filter_state = agent_previous_states.get(target_idx)
            predicted_mean, predicted_covariance, predicted_motion_state = _bp_jpda_predict_track_state(
                previous_filter_state=None if previous_filter_state is None else BPJPDAFilterState(
                    mean=previous_filter_state.mean,
                    covariance=previous_filter_state.covariance,
                    motion_state=previous_filter_state.motion_state,
                ),
                prior_relative_state=prior_relative_states[target_idx],
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
            )
            if previous_filter_state is None:
                predicted_existence = float(np.clip(birth_weight, 0.0, 1.0))
            else:
                predicted_existence = float(
                    np.clip(
                        survival_probability * previous_filter_state.existence_probability
                        + (1.0 - previous_filter_state.existence_probability) * float(birth_weight),
                        0.0,
                        1.0,
                    )
                )
            predicted_means.append(predicted_mean)
            predicted_covariances.append(predicted_covariance)
            predicted_motion_states.append(predicted_motion_state)
            predicted_existences.append(predicted_existence)

            innovation_covariance_diagonal = np.diag(predicted_covariance) + local_covariance
            prior_position_distance = float(np.linalg.norm(predicted_mean[:3]))
            if prior_position_distance > float(detection_radius_m) + _position_gate_margin(
                innovation_covariance_diagonal,
                gate_sigma,
            ):
                track_active[target_idx] = False

        candidate_states = local_relative_states[valid_indices] if valid_indices.size else np.zeros((0, STATE_DIM), dtype=float)
        likelihood_ratios = np.zeros((prior_relative_states.shape[0], valid_indices.size), dtype=float)
        raw_detection_scores = np.zeros((prior_relative_states.shape[0], valid_indices.size), dtype=float)
        for target_idx in range(prior_relative_states.shape[0]):
            if not track_active[target_idx] or valid_indices.size == 0:
                continue
            predicted_mean = predicted_means[target_idx]
            predicted_covariance = predicted_covariances[target_idx]
            innovation_covariance_diagonal = np.diag(predicted_covariance) + local_covariance
            residuals = candidate_states - predicted_mean.reshape(1, STATE_DIM)
            squared_mahalanobis = np.sum(
                np.square(residuals) / innovation_covariance_diagonal.reshape(1, STATE_DIM),
                axis=1,
            )
            gated_offsets = np.flatnonzero(squared_mahalanobis <= float(gate_sigma) ** 2)
            if gated_offsets.size == 0:
                continue
            gaussian_likelihood = _gaussian_pdf_zero_mean_diag(
                residuals[gated_offsets],
                innovation_covariance_diagonal,
            )
            raw_detection_scores[target_idx, gated_offsets] = (
                local_detection_probability * gaussian_likelihood / clutter_intensity
            )
            likelihood_ratios[target_idx, gated_offsets] = raw_detection_scores[target_idx, gated_offsets]

        marginals = _run_bp_jpda(
            likelihood_ratios,
            max_iterations=max_iterations,
            convergence_tol=convergence_tol,
        ) if valid_indices.size and prior_relative_states.shape[0] else np.zeros_like(likelihood_ratios)

        agent_tracks: list[KeyTrackPosterior] = []
        agent_next_states: dict[int, LMBFilterState] = {}
        for target_idx in range(prior_relative_states.shape[0]):
            predicted_mean = predicted_means[target_idx]
            predicted_covariance = predicted_covariances[target_idx]
            predicted_motion_state = predicted_motion_states[target_idx]
            predicted_covariance_diagonal = np.diag(predicted_covariance)
            predicted_existence = predicted_existences[target_idx]

            full_weights = np.zeros(local_relative_states.shape[0], dtype=np.float32)
            if valid_indices.size:
                full_weights[valid_indices] = marginals[target_idx].astype(np.float32)
            confidence = float(np.clip(np.sum(full_weights), 0.0, 1.0))

            detection_score = float(np.sum(raw_detection_scores[target_idx]))
            lmb_evidence = (1.0 - local_detection_probability) + detection_score
            posterior_existence = 0.0
            if track_active[target_idx]:
                posterior_existence = float(
                    np.clip(
                        (predicted_existence * lmb_evidence)
                        / max((1.0 - predicted_existence) + predicted_existence * lmb_evidence, 1.0e-12),
                        0.0,
                        1.0,
                    )
                )

            posterior_motion_states: list[IMMMotionState] = [predicted_motion_state]
            posterior_weights: list[float] = [max(1.0 - confidence, 1.0e-9)]
            best_local_index = -1
            best_innovation_distance = np.inf
            for candidate_offset in range(valid_indices.size):
                candidate_weight = float(marginals[target_idx, candidate_offset])
                if candidate_weight <= 0.0:
                    continue
                posterior_motion_state_j, innovation_distance = _update_motion_state_with_measurement(
                    predicted_motion_state,
                    measurement_state=candidate_states[candidate_offset],
                    measurement_covariance_diagonal=local_covariance,
                )
                posterior_motion_states.append(posterior_motion_state_j)
                posterior_weights.append(candidate_weight)
                if innovation_distance < best_innovation_distance:
                    best_innovation_distance = innovation_distance
                    best_local_index = int(valid_indices[candidate_offset])

            posterior_motion_state = _merge_motion_hypotheses(
                posterior_motion_states,
                posterior_weights,
            )
            posterior_state, posterior_covariance = _collapse_motion_state(posterior_motion_state)
            posterior_covariance_diagonal = np.diag(posterior_covariance)
            valid = bool(
                track_active[target_idx]
                and posterior_existence >= confidence_threshold
            )
            if best_local_index < 0 and valid_indices.size:
                weight_support, nearest_index, nearest_distance = _nearest_measurement_support(
                    posterior_state,
                    local_relative_states,
                    valid_indices,
                )
                if not np.any(full_weights > 0.0):
                    full_weights = weight_support
                if nearest_index >= 0:
                    best_local_index = nearest_index
                    best_innovation_distance = nearest_distance

            agent_tracks.append(
                KeyTrackPosterior(
                    global_target_index=target_idx,
                    prior_relative_state=prior_relative_states[target_idx].astype(np.float32),
                    prior_covariance_diagonal=prior_covariance.astype(np.float32),
                    posterior_relative_state=posterior_state.astype(np.float32),
                    posterior_covariance_diagonal=posterior_covariance_diagonal.astype(np.float32),
                    confidence=confidence,
                    existence_probability=posterior_existence,
                    valid=valid,
                    local_candidate_weights=full_weights,
                    local_candidate_index=best_local_index,
                    innovation_distance=float(best_innovation_distance),
                )
            )
            agent_next_states[target_idx] = LMBFilterState(
                existence_probability=posterior_existence,
                mean=posterior_state.astype(float, copy=True),
                covariance=posterior_covariance.astype(float, copy=True),
                motion_state=posterior_motion_state,
            )

        outputs[str(agent_id)] = MultiTargetTrackOutput(
            agent_id=str(agent_id),
            tracks=tuple(agent_tracks),
            mode="local_only",
        )
        next_filter_states[str(agent_id)] = agent_next_states

    return outputs, next_filter_states


def gm_phd_association_for_friendlies(
    *,
    friendly_states: np.ndarray,
    global_target_priors: GlobalKeyTargetStatePrior,
    local_observations: dict[str, LocalTargetObservation],
    agent_ids: Sequence[str],
    previous_filter_states: dict[str, GMPHDFilterState | None] | None,
    detection_radius_m: float,
    relative_velocity_limit_mps: float,
    local_detection_probability: float,
    clutter_rate_per_second: float,
    recursive_motion_model: str = "imm",
    imm_model_types: Sequence[str] = ("cv", "ca", "ct_left", "ct_right"),
    imm_model_transition_matrix: np.ndarray | Sequence[Sequence[float]] | None = None,
    imm_initial_model_probabilities: np.ndarray | Sequence[float] | None = None,
    dt_seconds: float,
    gate_sigma: float = 3.0,
    survival_probability: float = 0.98,
    birth_weight: float = 0.25,
    prune_threshold: float = 1.0e-3,
    merge_mahalanobis_threshold: float = 2.0,
    max_components: int = 20,
    process_position_std_m: np.ndarray | Sequence[float] = (2.0, 2.0, 1.0),
    process_velocity_std_mps: np.ndarray | Sequence[float] = (1.0, 1.0, 0.5),
    imm_turn_rate_radps: float = 0.35,
    imm_ct_process_noise_scale: float = 1.25,
    confidence_threshold: float = 0.1,
) -> tuple[dict[str, MultiTargetTrackOutput], dict[str, GMPHDFilterState | None]]:
    """Run a standard local GM-PHD recursion and align extracted components to key priors."""

    friendly_array = _as_state_batch(friendly_states, name="friendly_states")
    if friendly_array.shape[0] != len(agent_ids):
        raise ValueError("friendly_states row count must match agent_ids length.")
    if gate_sigma <= 0.0:
        raise ValueError(f"gate_sigma must be > 0, got {gate_sigma}.")
    if max_components <= 0:
        raise ValueError(f"max_components must be > 0, got {max_components}.")

    survival_probability = float(np.clip(survival_probability, 0.0, 1.0))
    local_detection_probability = float(np.clip(local_detection_probability, 0.0, 1.0))
    confidence_threshold = float(np.clip(confidence_threshold, 0.0, 1.0))
    process_position_std = np.asarray(process_position_std_m, dtype=float).reshape(3)
    process_velocity_std = np.asarray(process_velocity_std_mps, dtype=float).reshape(3)

    position_volume = (4.0 / 3.0) * np.pi * max(float(detection_radius_m), 1.0e-6) ** 3
    velocity_support = max(float(relative_velocity_limit_mps), 1.0e-6)
    velocity_volume = np.power(2.0 * velocity_support, 3)
    measurement_volume = max(position_volume * velocity_volume, 1.0e-9)
    clutter_rate_per_step = max(float(clutter_rate_per_second), 0.0) * max(float(dt_seconds), 1.0e-6)
    clutter_intensity = max(clutter_rate_per_step / measurement_volume, 1.0e-9)

    outputs: dict[str, MultiTargetTrackOutput] = {}
    next_filter_states: dict[str, GMPHDFilterState | None] = {}
    previous_states = previous_filter_states or {}

    for friendly_idx, agent_id in enumerate(agent_ids):
        prior_relative_states, prior_covariance = _build_prior_relative_states(
            friendly_array[friendly_idx],
            global_target_priors,
        )
        local_relative_states, local_mask, local_covariance = _validate_local_observation(local_observations[str(agent_id)])
        valid_indices = np.flatnonzero(local_mask > 0.0)
        candidate_states = local_relative_states[valid_indices] if valid_indices.size else np.zeros((0, STATE_DIM), dtype=float)

        predicted_components = _gm_phd_predict_components(
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
        )
        if not predicted_components:
            outputs[str(agent_id)] = empty_multi_target_track_output(
                agent_id=str(agent_id),
                prior_relative_states=prior_relative_states,
                prior_covariance_diagonal=prior_covariance,
                mode="local_only",
            )
            next_filter_states[str(agent_id)] = None
            continue

        missed_detection_components = [
            _gm_phd_create_component(
                (1.0 - local_detection_probability) * component.weight,
                component.mean,
                component.covariance,
                motion_state=component.motion_state,
            )
            for component in predicted_components
        ]
        updated_components: list[GMPHDComponent] = list(missed_detection_components)

        for measurement in candidate_states:
            measurement_vector = _as_state_vector(measurement, name="measurement")
            measurement_terms: list[tuple[float, np.ndarray, np.ndarray, IMMMotionState]] = []
            denominator = clutter_intensity
            for component in predicted_components:
                innovation_covariance = component.covariance + np.diag(local_covariance)
                innovation = measurement_vector - component.mean
                inv_innovation_covariance = np.linalg.inv(innovation_covariance)
                mahalanobis_sq = float(innovation.T @ inv_innovation_covariance @ innovation)
                if mahalanobis_sq > float(gate_sigma) ** 2:
                    continue

                likelihood = _gaussian_pdf(component.mean, innovation_covariance, measurement_vector)
                updated_weight = local_detection_probability * component.weight * likelihood
                component_motion_state = _coerce_motion_state(
                    component.motion_state,
                    fallback_mean=component.mean,
                    fallback_covariance=component.covariance,
                    recursive_motion_model=recursive_motion_model,
                    imm_model_types=imm_model_types,
                    imm_model_transition_matrix=imm_model_transition_matrix,
                    imm_initial_model_probabilities=imm_initial_model_probabilities,
                )
                updated_motion_state, _ = _update_motion_state_with_measurement(
                    component_motion_state,
                    measurement_state=measurement_vector,
                    measurement_covariance_diagonal=local_covariance,
                )
                posterior_mean, posterior_covariance = _collapse_motion_state(updated_motion_state)
                denominator += updated_weight
                measurement_terms.append((updated_weight, posterior_mean, posterior_covariance, updated_motion_state))

            for updated_weight, posterior_mean, posterior_covariance, updated_motion_state in measurement_terms:
                normalized_weight = updated_weight / max(denominator, 1.0e-12)
                updated_components.append(
                    _gm_phd_create_component(
                        normalized_weight,
                        posterior_mean,
                        posterior_covariance,
                        motion_state=updated_motion_state,
                    )
                )

        filtered_components = _gm_phd_prune_and_cap_components(
            updated_components,
            prune_threshold=prune_threshold,
            merge_mahalanobis_threshold=merge_mahalanobis_threshold,
            max_components=max_components,
        )
        next_filter_states[str(agent_id)] = (
            None if not filtered_components else GMPHDFilterState(components=filtered_components)
        )

        if prior_relative_states.shape[0] == 0:
            outputs[str(agent_id)] = MultiTargetTrackOutput(agent_id=str(agent_id), tracks=tuple(), mode="local_only")
            continue

        if not filtered_components:
            outputs[str(agent_id)] = empty_multi_target_track_output(
                agent_id=str(agent_id),
                prior_relative_states=prior_relative_states,
                prior_covariance_diagonal=prior_covariance,
                mode="local_only",
            )
            continue

        top_components = list(filtered_components[: max(prior_relative_states.shape[0], 1)])
        cost_matrix = np.full((prior_relative_states.shape[0], len(top_components)), np.inf, dtype=float)
        for target_idx in range(prior_relative_states.shape[0]):
            for component_idx, component in enumerate(top_components):
                innovation_covariance = component.covariance + np.diag(prior_covariance)
                innovation = component.mean - prior_relative_states[target_idx]
                inv_innovation_covariance = np.linalg.inv(innovation_covariance)
                cost_matrix[target_idx, component_idx] = float(innovation.T @ inv_innovation_covariance @ innovation)
        component_assignments = _solve_exact_assignment(cost_matrix)

        tracks: list[KeyTrackPosterior] = []
        for target_idx in range(prior_relative_states.shape[0]):
            component_idx = component_assignments[target_idx] if target_idx < len(component_assignments) else -1
            if component_idx < 0:
                tracks.append(
                    empty_key_track_posterior(
                        global_target_index=target_idx,
                        prior_relative_state=prior_relative_states[target_idx],
                        prior_covariance_diagonal=prior_covariance,
                    )
                )
                continue

            component = top_components[component_idx]
            support, nearest_index, nearest_distance = _nearest_measurement_support(
                component.mean,
                local_relative_states,
                valid_indices,
            )
            confidence = float(np.clip(component.weight, 0.0, 1.0))
            valid = bool(confidence >= confidence_threshold)
            tracks.append(
                KeyTrackPosterior(
                    global_target_index=target_idx,
                    prior_relative_state=prior_relative_states[target_idx].astype(np.float32),
                    prior_covariance_diagonal=prior_covariance.astype(np.float32),
                    posterior_relative_state=component.mean.astype(np.float32),
                    posterior_covariance_diagonal=np.diag(component.covariance).astype(np.float32),
                    confidence=confidence,
                    existence_probability=confidence,
                    valid=valid,
                    local_candidate_weights=support,
                    local_candidate_index=nearest_index,
                    innovation_distance=float(nearest_distance),
                )
            )

        outputs[str(agent_id)] = MultiTargetTrackOutput(
            agent_id=str(agent_id),
            tracks=tuple(tracks),
            mode="local_only",
        )

    return outputs, next_filter_states
