from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "default.yaml"
_ENV_CONFIG_PATH = _PROJECT_ROOT / "config" / "env.yaml"
_MAPPO_CONFIG_PATH = _PROJECT_ROOT / "config" / "mappo.yaml"


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


def get_mappo_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read MAPPO training config grouped by scenario/training/network/marl/logging sections."""

    return load_yaml_config(path or _MAPPO_CONFIG_PATH)


def get_enemy_mobility_config(path: str | Path | None = None) -> dict[str, Any]:
    env_config = get_env_config(path)
    mobility_config = env_config.get("enemy_mobility", {})
    return mobility_config if isinstance(mobility_config, dict) else {}


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


def get_policy_input_mode(path: str | Path | None = None) -> str:
    """Read the configured policy input mode."""

    env_config = get_env_config(path)
    policy_input_config = env_config.get("policy_input", {})
    if not isinstance(policy_input_config, dict):
        policy_input_config = {}
    return str(policy_input_config.get("mode", "local_only")).lower()


def get_policy_association_method(path: str | Path | None = None) -> str:
    """Read the configured policy association method."""

    env_config = get_env_config(path)
    policy_input_config = env_config.get("policy_input", {})
    if not isinstance(policy_input_config, dict):
        policy_input_config = {}
    return str(policy_input_config.get("association_method", "none")).lower()


def get_global_sensing_config(path: str | Path | None = None) -> dict[str, float]:
    """Read external global-sensing settings from env config."""

    env_config = get_env_config(path)
    sensing_config = env_config.get("global_sensing", {})
    if not isinstance(sensing_config, dict):
        sensing_config = {}

    return {
        "radar_delay_seconds": float(sensing_config.get("radar_delay_seconds", 1.0)),
        "key_enemy_position_noise_std_m": float(sensing_config.get("key_enemy_position_noise_std_m", 2.0)),
    }


def get_local_sensing_config(path: str | Path | None = None) -> dict[str, float | int]:
    """Read local target-sensing settings from env config."""

    env_config = get_env_config(path)
    sensing_config = env_config.get("local_sensing", {})
    if not isinstance(sensing_config, dict):
        sensing_config = {}

    return {
        "detection_radius_m": float(sensing_config.get("detection_radius_m", 50.0)),
        "max_candidates": int(sensing_config.get("max_candidates", 5)),
        "local_position_noise_std_m": float(sensing_config.get("local_position_noise_std_m", 1.0)),
        "detection_probability": float(sensing_config.get("detection_probability", 0.95)),
        "clutter_rate_per_second": float(sensing_config.get("clutter_rate_per_second", 1.0)),
    }


def should_print_initial_positions(path: str | Path | None = None) -> bool:
    return bool(get_simulation_config(path).get("print_initial_positions", True))


def should_print_timestep_positions(path: str | Path | None = None) -> bool:
    return bool(get_simulation_config(path).get("print_timestep_positions", False))


def get_reward_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read cooperative team-reward hyperparameters from env config."""

    env_config = get_env_config(path)
    reward_config = env_config.get("reward", {})
    if not isinstance(reward_config, dict):
        reward_config = {}

    reward_mode = str(reward_config.get("reward_mode", "sustained_tracking")).lower()
    return {
        "reward_mode": reward_mode,
        "tau_ally": float(reward_config.get("tau_ally", 1.0)),
        "lambda_safety": float(reward_config.get("lambda_safety", 0.2)),
        "J0": float(reward_config.get("J0", 1.0)),
        "power_weight": float(reward_config.get("power_weight", 1.0)),
        "progress_weight": float(reward_config.get("progress_weight", 0.4)),
        "progress_distance_scale": float(reward_config.get("progress_distance_scale", 4.0)),
        "move_penalty_weight": float(reward_config.get("move_penalty_weight", 0.02)),
    }
