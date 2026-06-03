from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "default.yaml"
_ENV_CONFIG_PATH = _PROJECT_ROOT / "config" / "env.yaml"


def load_yaml_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config file. Empty files are treated as empty dicts."""

    config_path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return {}

    content = config_path.read_text(encoding="utf-8").strip()
    if not content:
        return {}

    data = yaml.safe_load(content)
    return data if isinstance(data, dict) else {}


def get_default_seed(path: str | Path | None = None) -> int:
    """Read the default simulation seed from global config."""

    config = load_yaml_config(path)
    seed = config.get("seed", 42)
    return int(seed)


def get_env_config(path: str | Path | None = None) -> dict[str, Any]:
    return load_yaml_config(path or _ENV_CONFIG_PATH)


def get_entity_count_config(path: str | Path | None = None) -> dict[str, int]:
    """Read default scene entity counts from env config."""

    env_config = get_env_config(path)
    count_config = env_config.get("entity_counts", {})
    if not isinstance(count_config, dict):
        count_config = {}
    return {
        "friendly_uavs": int(count_config.get("friendly_uavs", 3)),
        "key_targets": int(count_config.get("key_targets", 3)),
        "nonkey_targets": int(count_config.get("nonkey_targets", 3)),
    }


def get_num_friendly_uavs(path: str | Path | None = None) -> int:
    return int(get_entity_count_config(path).get("friendly_uavs", 3))


def get_num_key_targets(path: str | Path | None = None) -> int:
    return int(get_entity_count_config(path).get("key_targets", 3))


def get_num_nonkey_targets(path: str | Path | None = None) -> int:
    return int(get_entity_count_config(path).get("nonkey_targets", 3))


def get_initialization_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read default scene-initialization distance ranges from env config."""

    env_config = get_env_config(path)
    initialization_config = env_config.get("initialization", {})
    if not isinstance(initialization_config, dict):
        initialization_config = {}

    def _read_distance_pair(raw_value: Any, *, name: str, default: Sequence[float]) -> tuple[float, float]:
        values = np.asarray(default if raw_value is None else raw_value, dtype=float).reshape(-1)
        if values.shape != (2,):
            raise ValueError(f"{name} must be a 2D vector [min_m, max_m], got shape {values.shape}.")
        return float(values[0]), float(values[1])

    return {
        "nonkey_enemy": _read_distance_pair(
            initialization_config.get("nonkey_enemy_nearby_distance_m"),
            name="initialization.nonkey_enemy_nearby_distance_m",
            default=[2.0, 8.0],
        ),
        "friend_drone": _read_distance_pair(
            initialization_config.get("friendly_uav_nearby_distance_m"),
            name="initialization.friendly_uav_nearby_distance_m",
            default=[12.0, 28.0],
        ),
        "friend_drone_min_separation_m": float(
            initialization_config.get("friendly_uav_min_separation_m", 0.0)
        ),
    }


def get_enemy_mobility_config(path: str | Path | None = None) -> dict[str, Any]:
    env_config = get_env_config(path)
    mobility_config = env_config.get("enemy_mobility", {})
    return mobility_config if isinstance(mobility_config, dict) else {}


def _read_vector3(raw_value: Any, *, name: str, default: Sequence[float]) -> np.ndarray:
    value = default if raw_value is None else raw_value
    vector = np.asarray(value, dtype=float)
    if vector.ndim == 0:
        vector = np.full(3, float(vector), dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must be a scalar or 3D vector, got shape {vector.shape}.")
    return vector


def get_velocity_config(path: str | Path | None = None) -> dict[str, float]:
    mobility_config = get_enemy_mobility_config(path)
    velocity_config = mobility_config.get("vmax", {})
    if not isinstance(velocity_config, dict):
        velocity_config = {}

    return {
        "enemy": float(velocity_config.get("enemy", 18.0)),
        "friendly": float(velocity_config.get("friendly", 12.0)),
    }


def get_enemy_vmax(path: str | Path | None = None) -> float:
    return float(get_velocity_config(path).get("enemy", 18.0))


def get_friendly_vmax(path: str | Path | None = None) -> float:
    return float(get_velocity_config(path).get("friendly", 12.0))


def get_simulation_config(path: str | Path | None = None) -> dict[str, Any]:
    env_config = get_env_config(path)
    simulation_config = env_config.get("simulation", {})
    return simulation_config if isinstance(simulation_config, dict) else {}


def get_simulation_dt(path: str | Path | None = None) -> float:
    """Read the physical simulation time step from env config."""

    return float(get_simulation_config(path).get("dt", 1.0))


def get_num_steps(path: str | Path | None = None) -> int:
    return int(get_simulation_config(path).get("num_steps", 10))


def get_tracking_config(path: str | Path | None = None) -> dict[str, Any]:
    env_config = get_env_config(path)
    legacy_policy_input = env_config.get("policy_input", {})
    tracking_config = env_config.get("tracking", {})
    base_config = legacy_policy_input if isinstance(legacy_policy_input, dict) else {}
    override_config = tracking_config if isinstance(tracking_config, dict) else {}

    if not base_config and not override_config:
        return {}

    merged = dict(base_config)
    for key, value in override_config.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
            continue
        merged[key] = value
    return merged


def get_tracking_input_mode(path: str | Path | None = None) -> str:
    """Read the configured tracking-layer input mode."""

    tracking_config = get_tracking_config(path)
    return str(tracking_config.get("mode", "local_only")).lower()


def get_tracking_association_method(path: str | Path | None = None) -> str:
    """Read the configured local tracking association method."""

    tracking_config = get_tracking_config(path)
    return str(tracking_config.get("association_method", "none")).lower()


def get_tracking_assignment_method(path: str | Path | None = None) -> str:
    """Read the configured downstream key-track assignment method."""

    tracking_config = get_tracking_config(path)
    return str(tracking_config.get("assignment", "placeholder")).lower()


def get_policy_input_mode(path: str | Path | None = None) -> str:
    """Backward-compatible alias for the tracking-layer input mode."""

    return get_tracking_input_mode(path)


def get_policy_association_method(path: str | Path | None = None) -> str:
    """Backward-compatible alias for the tracking association method."""

    return get_tracking_association_method(path)


def get_policy_assignment_method(path: str | Path | None = None) -> str:
    """Backward-compatible alias for the downstream tracking assignment method."""

    return get_tracking_assignment_method(path)


def get_global_sensing_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read external global-sensing settings from env config."""

    env_config = get_env_config(path)
    sensing_config = env_config.get("global_sensing", {})
    if not isinstance(sensing_config, dict):
        sensing_config = {}

    return {
        "prior_position_std_m": _read_vector3(
            sensing_config.get("prior_position_std_m"),
            name="global_sensing.prior_position_std_m",
            default=[8.0, 8.0, 3.0],
        ),
        "prior_velocity_std_mps": _read_vector3(
            sensing_config.get("prior_velocity_std_mps"),
            name="global_sensing.prior_velocity_std_mps",
            default=[1.5, 1.5, 0.5],
        ),
    }


def get_local_sensing_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read local target-sensing settings from env config."""

    env_config = get_env_config(path)
    sensing_config = env_config.get("local_sensing", {})
    if not isinstance(sensing_config, dict):
        sensing_config = {}

    return {
        "detection_radius_m": float(sensing_config.get("detection_radius_m", 50.0)),
        "max_candidates": int(sensing_config.get("max_candidates", 5)),
        "measurement_position_std_m": _read_vector3(
            sensing_config.get("measurement_position_std_m"),
            name="local_sensing.measurement_position_std_m",
            default=[1.0, 1.0, 0.3],
        ),
        "measurement_velocity_std_mps": _read_vector3(
            sensing_config.get("measurement_velocity_std_mps"),
            name="local_sensing.measurement_velocity_std_mps",
            default=[0.5, 0.5, 0.2],
        ),
        "detection_probability": float(sensing_config.get("detection_probability", 0.95)),
        "clutter_rate_per_second": float(sensing_config.get("clutter_rate_per_second", 1.0)),
    }


def get_association_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read association-module settings from env config."""

    env_config = get_env_config(path)
    association_config = env_config.get("association", {})
    if not isinstance(association_config, dict):
        association_config = {}

    recursive_motion_model = str(association_config.get("recursive_motion_model", "imm")).lower()
    raw_imm_model_types = association_config.get("imm_model_types", ["cv", "ca", "ct_left", "ct_right"])
    if not isinstance(raw_imm_model_types, (list, tuple)) or len(raw_imm_model_types) == 0:
        raise ValueError("association.imm_model_types must be a non-empty list of motion-model names.")
    imm_model_types = tuple(str(model_type).lower() for model_type in raw_imm_model_types)
    model_count = len(imm_model_types)

    raw_transition_matrix = association_config.get("imm_model_transition_matrix")
    if raw_transition_matrix is None:
        imm_model_transition_matrix = None
    else:
        imm_model_transition_matrix = np.asarray(raw_transition_matrix, dtype=float)
        if imm_model_transition_matrix.shape != (model_count, model_count):
            raise ValueError(
                "association.imm_model_transition_matrix must have shape "
                f"({model_count}, {model_count}), got {imm_model_transition_matrix.shape}."
            )

    raw_initial_probabilities = association_config.get("imm_initial_model_probabilities")
    if raw_initial_probabilities is None:
        imm_initial_model_probabilities = None
    else:
        imm_initial_model_probabilities = np.asarray(raw_initial_probabilities, dtype=float).reshape(-1)
        if imm_initial_model_probabilities.shape != (model_count,):
            raise ValueError(
                "association.imm_initial_model_probabilities must have shape "
                f"({model_count},), got {imm_initial_model_probabilities.shape}."
            )

    return {
        "bp_jpda_gate_sigma": float(association_config.get("bp_jpda_gate_sigma", 3.0)),
        "bp_jpda_max_iterations": int(association_config.get("bp_jpda_max_iterations", 20)),
        "bp_jpda_convergence_tol": float(association_config.get("bp_jpda_convergence_tol", 1.0e-3)),
        "bp_jpda_min_missed_detection_probability": float(
            association_config.get("bp_jpda_min_missed_detection_probability", 0.05)
        ),
        "recursive_motion_model": recursive_motion_model,
        "imm_model_types": imm_model_types,
        "imm_model_transition_matrix": imm_model_transition_matrix,
        "imm_initial_model_probabilities": imm_initial_model_probabilities,
        "imm_turn_rate_radps": float(association_config.get("imm_turn_rate_radps", 0.35)),
        "imm_ct_process_noise_scale": float(association_config.get("imm_ct_process_noise_scale", 1.25)),
        "gm_phd_survival_probability": float(association_config.get("gm_phd_survival_probability", 0.98)),
        "gm_phd_birth_weight": float(association_config.get("gm_phd_birth_weight", 0.25)),
        "gm_phd_prune_threshold": float(association_config.get("gm_phd_prune_threshold", 1.0e-3)),
        "gm_phd_merge_mahalanobis": float(association_config.get("gm_phd_merge_mahalanobis", 2.0)),
        "gm_phd_max_components": int(association_config.get("gm_phd_max_components", 20)),
        "gm_phd_confidence_threshold": float(association_config.get("gm_phd_confidence_threshold", 0.1)),
        "delta_glmb_max_hypotheses": int(association_config.get("delta_glmb_max_hypotheses", 48)),
        "delta_glmb_hypothesis_weight_prune_threshold": float(
            association_config.get("delta_glmb_hypothesis_weight_prune_threshold", 1.0e-4)
        ),
        "gm_phd_process_position_std_m": _read_vector3(
            association_config.get("gm_phd_process_position_std_m"),
            name="association.gm_phd_process_position_std_m",
            default=[2.0, 2.0, 1.0],
        ),
        "gm_phd_process_velocity_std_mps": _read_vector3(
            association_config.get("gm_phd_process_velocity_std_mps"),
            name="association.gm_phd_process_velocity_std_mps",
            default=[1.0, 1.0, 0.5],
        ),
    }


def get_track_fusion_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read track-to-track fusion settings from env config."""

    env_config = get_env_config(path)
    fusion_config = env_config.get("track_fusion", {})
    if not isinstance(fusion_config, dict):
        fusion_config = {}

    return {
        "self_position_std_m": _read_vector3(
            fusion_config.get("self_position_std_m"),
            name="track_fusion.self_position_std_m",
            default=[1.0, 1.0, 0.5],
        ),
        "min_confidence": float(fusion_config.get("min_confidence", 0.1)),
        "mahalanobis_distance_threshold": float(
            fusion_config.get("mahalanobis_distance_threshold", 3.0)
        ),
        "ci_search_steps": int(fusion_config.get("ci_search_steps", 21)),
    }


def should_print_initial_positions(path: str | Path | None = None) -> bool:
    return bool(get_simulation_config(path).get("print_initial_positions", True))


def should_print_timestep_positions(path: str | Path | None = None) -> bool:
    return bool(get_simulation_config(path).get("print_timestep_positions", False))
