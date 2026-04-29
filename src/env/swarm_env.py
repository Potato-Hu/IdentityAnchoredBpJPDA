"""Minimal multi-agent RL environment skeleton for the swarm project.

This file wraps SwarmWorld into reset/step style APIs and defines placeholder observations,
rewards, and termination logic so later modules can plug into a stable environment backbone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from src.env.spaces import (
    SwarmSpaceSpec,
    build_action_spaces,
    build_agent_ids,
    build_global_state_space,
    build_observation_spaces,
    build_space_spec,
)
from src.env.world import SwarmWorld
from src.rl.reward import compute_difference_rewards, compute_distance_assignment, compute_power_progress_rewards
from src.sensing.association import (
    AssociationOutput,
    association_output_to_weight_vector,
    bp_jpda_association_for_friendlies,
    empty_association_output,
    nearest_local_candidate_association_for_friendlies,
    nearest_neighbor_association_for_friendlies,
)
from src.sensing.global_sensor import DelayedNoisyKeyTargetSensor, GlobalKeyTargetSensingConfig
from src.sensing.local_sensor import LocalSensingConfig, LocalTargetObservation, LocalTargetSensor
from src.sensing.observation_builder import build_policy_enemy_view
from src.simulation.ground_truth import NearbyDistanceLimits, PositionDict, ScenarioGroundTruth, initialize_scene
from src.utils.config_loader import (
    get_association_config,
    get_enemy_vmax,
    get_global_sensing_config,
    get_local_sensing_config,
    get_num_steps,
    get_policy_association_method,
    get_policy_input_mode,
    get_reward_config,
)


POLICY_INPUT_MODES = {"local_only", "groundtruth"}
ASSOCIATION_METHODS = {"none", "nn", "bp_jpda"}


@dataclass
class EnvStepOutput:
    observations: dict[str, np.ndarray]
    rewards: dict[str, float]
    terminated: dict[str, bool]
    truncated: dict[str, bool]
    infos: dict[str, dict[str, Any]]


class SwarmEnv:
    """Lightweight env skeleton built on top of SwarmWorld.

    `local_only` uses legal self state plus anonymous local target candidates for
    actor/critic inputs. `groundtruth` is a debug mode that trains from full world-truth
    target features.
    """

    def __init__(
        self,
        key_num: int = 3,
        nonkey_num: int = 3,
        mydrone_num: int = 3,
        bounds: Sequence[float] | None = None,
        dt: float = 1.0,
        max_steps: int | None = None,
        seed: int | None = None,
        enemy_vmax: float | None = None,
        friendly_vmax: float | None = None,
        initial_positions: PositionDict | None = None,
        nearby_distance_limits: NearbyDistanceLimits | None = None,
        tau_ally: float | None = None,
        lambda_safety: float | None = None,
        J0: float | None = None,
        power_weight: float | None = None,
        progress_weight: float | None = None,
        progress_distance_scale: float | None = None,
        move_penalty_weight: float | None = None,
        policy_input_mode: str | None = None,
    ) -> None:
        self.key_num = int(key_num)
        self.nonkey_num = int(nonkey_num)
        self.mydrone_num = int(mydrone_num)
        self.bounds = None if bounds is None else np.asarray(bounds, dtype=float)
        self.dt = float(dt)
        self.max_steps = get_num_steps() if max_steps is None else int(max_steps)
        self.seed = seed
        self.enemy_vmax = enemy_vmax
        self.friendly_vmax = friendly_vmax
        self.initial_positions = initial_positions
        self.nearby_distance_limits = nearby_distance_limits
        resolved_policy_input_mode = get_policy_input_mode() if policy_input_mode is None else policy_input_mode
        self.policy_input_mode = str(resolved_policy_input_mode).lower()
        if self.policy_input_mode not in POLICY_INPUT_MODES:
            raise ValueError(
                "policy_input.mode must be either 'local_only' or 'groundtruth', "
                f"got {self.policy_input_mode!r}."
            )
        self.association_method = get_policy_association_method()
        if self.association_method not in ASSOCIATION_METHODS:
            raise ValueError(
                "policy_input.association_method must be one of 'none', 'nn', or 'bp_jpda', "
                f"got {self.association_method!r}."
            )
        association_config = get_association_config()
        self.bp_jpda_gate_sigma = float(association_config["bp_jpda_gate_sigma"])
        self.bp_jpda_max_iterations = int(association_config["bp_jpda_max_iterations"])
        self.bp_jpda_convergence_tol = float(association_config["bp_jpda_convergence_tol"])
        self.bp_jpda_min_missed_detection_probability = float(
            association_config["bp_jpda_min_missed_detection_probability"]
        )

        # Reward hyperparameters for the shared team reward used by all friendly UAVs.
        reward_config = get_reward_config()
        self.reward_mode = str(reward_config["reward_mode"]).lower()
        if self.reward_mode not in {"difference", "sustained_tracking"}:
            raise ValueError(
                "reward.reward_mode must be either 'difference' or 'sustained_tracking', "
                f"got {self.reward_mode!r}."
            )
        self.tau_ally = float(reward_config["tau_ally"] if tau_ally is None else tau_ally)
        self.lambda_safety = float(reward_config["lambda_safety"] if lambda_safety is None else lambda_safety)
        self.J0 = float(reward_config["J0"] if J0 is None else J0)
        self.power_weight = float(reward_config["power_weight"] if power_weight is None else power_weight)
        self.progress_weight = float(reward_config["progress_weight"] if progress_weight is None else progress_weight)
        self.progress_distance_scale = float(
            reward_config["progress_distance_scale"]
            if progress_distance_scale is None
            else progress_distance_scale
        )
        self.move_penalty_weight = float(
            reward_config["move_penalty_weight"]
            if move_penalty_weight is None
            else move_penalty_weight
        )
        sensing_config = get_global_sensing_config()
        global_sensor_config = GlobalKeyTargetSensingConfig(
            radar_delay_seconds=float(sensing_config["radar_delay_seconds"]),
            position_noise_std_m=float(sensing_config["key_enemy_position_noise_std_m"]),
        )
        self.global_key_target_sensor = DelayedNoisyKeyTargetSensor(global_sensor_config, seed=self.seed)
        local_sensing_config = get_local_sensing_config()
        local_sensor_config = LocalSensingConfig(
            detection_radius_m=float(local_sensing_config["detection_radius_m"]),
            max_candidates=int(local_sensing_config["max_candidates"]),
            position_noise_std_m=float(local_sensing_config["local_position_noise_std_m"]),
            detection_probability=float(local_sensing_config["detection_probability"]),
            clutter_rate_per_second=float(local_sensing_config["clutter_rate_per_second"]),
            dt_seconds=self.dt,
        )
        self.local_target_sensor = LocalTargetSensor(local_sensor_config, seed=self.seed)

        self.agent_ids = build_agent_ids(self.mydrone_num)
        self.scene: ScenarioGroundTruth | None = None
        self.world: SwarmWorld | None = None
        self.action_spaces: dict[str, Any] = {}
        self.observation_spaces: dict[str, Any] = {}
        self.global_state_space: Any | None = None
        self.space_spec: SwarmSpaceSpec | None = None
        self._previous_assigned_targets: np.ndarray | None = None
        self._previous_assigned_distances: np.ndarray | None = None
        self._last_observations: dict[str, np.ndarray] | None = None
        self._last_observation_time_step: int | None = None
        self._last_local_observations: dict[str, LocalTargetObservation] | None = None
        self._last_local_observation_time_step: int | None = None
        self._reset_count = 0
        self._episode_seed_rng = np.random.default_rng(seed)

    def reset(
        self,
        *,
        seed: int | None = None,
        initial_positions: PositionDict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        """Create a fresh scene/world pair and return initial observations."""

        if seed is not None:
            self.seed = seed
            self._reset_count = 0
            self._episode_seed_rng = np.random.default_rng(seed)
        if initial_positions is not None:
            self.initial_positions = initial_positions

        episode_seed = self._next_episode_seed()
        self.scene = initialize_scene(
            key_num=self.key_num,
            nonkey_num=self.nonkey_num,
            mydrone_num=self.mydrone_num,
            bounds=self.bounds,
            dt=self.dt,
            seed=episode_seed,
            enemy_vmax=self.enemy_vmax,
            friendly_vmax=self.friendly_vmax,
            initial_positions=self.initial_positions,
            nearby_distance_limits=self.nearby_distance_limits,
        )
        self.world = SwarmWorld.from_scene(self.scene)
        self.world.reset()
        self.global_key_target_sensor.reset(self.world.get_key_enemy_positions(), seed=episode_seed)
        self.local_target_sensor.reset(seed=episode_seed)
        self._last_observations = None
        self._last_observation_time_step = None
        self._last_local_observations = None
        self._last_local_observation_time_step = None
        self._build_spaces()
        self._reset_tracking_memory()

        observations = self._build_observations()
        infos = {
            agent_id: {
                "time_step": 0,
                "sim_time": 0.0,
                "friendly_interference_dbm": float(self.world.interference.friendly_received_dbm[idx]),
                "friendly_interference_watts": float(self.world.interference.friendly_received_watts[idx]),
                "key_enemy_interference_dbm": self.world.interference.key_enemy_received_dbm.copy(),
                "key_enemy_interference_watts": self.world.interference.key_enemy_received_watts.copy(),
                "policy_input_mode": self.policy_input_mode,
                "association_method": self.association_method,
                "episode_seed": episode_seed,
            }
            for idx, agent_id in enumerate(self.agent_ids)
        }
        return observations, infos

    def step(self, actions: Mapping[str, int]) -> EnvStepOutput:
        """Advance the world by one step using per-agent friendly actions."""

        if self.world is None:
            raise RuntimeError("reset() must be called before step().")

        ordered_actions = np.asarray([int(actions[agent_id]) for agent_id in self.agent_ids], dtype=int)
        result = self.world.step(friendly_actions=ordered_actions, dt=self.dt)
        self.global_key_target_sensor.record_truth(self.world.get_key_enemy_positions())

        observations = self._build_observations()

        # Strict interference reward: each friendly UAV receives its counterfactual
        # marginal contribution to received jamming power at key targets.
        # Non-key enemies remain in the scene as clutter, but they do not enter this simple
        # baseline reward directly.
        key_enemy_positions = self.world.get_key_enemy_positions()
        pairwise_key_distances = self._pairwise_distances(result.friendly_positions, key_enemy_positions)
        if self.reward_mode == "sustained_tracking":
            assignment_components = compute_distance_assignment(pairwise_distances=pairwise_key_distances)
            assigned_targets = assignment_components["assigned_target"].astype(int)
            assigned_distances = assignment_components["assigned_distance"]
            if self._previous_assigned_targets is None or self._previous_assigned_distances is None:
                self._reset_tracking_memory()
            assert self._previous_assigned_targets is not None
            assert self._previous_assigned_distances is not None
            pairwise_key_powers = self.world.interference.key_enemy_pairwise_watts
            ally_interference_powers = result.friendly_interference_watts
            tracking_rewards, assignment_components = compute_power_progress_rewards(
                assigned_targets=assigned_targets,
                assigned_distances=assigned_distances,
                previous_assigned_targets=self._previous_assigned_targets,
                previous_assigned_distances=self._previous_assigned_distances,
                pairwise_jamming_powers=pairwise_key_powers,
                ally_interference_powers=ally_interference_powers,
                actions=ordered_actions,
                J0=self.J0,
                tau_ally=self.tau_ally,
                lambda_safety=self.lambda_safety,
                power_weight=self.power_weight,
                progress_weight=self.progress_weight,
                progress_distance_scale=self.progress_distance_scale,
                move_penalty_weight=self.move_penalty_weight,
            )
            self._previous_assigned_targets = assigned_targets.copy()
            self._previous_assigned_distances = assigned_distances.copy()
            difference_rewards = np.zeros_like(tracking_rewards)
        else:
            difference_rewards = compute_difference_rewards(
                pairwise_jamming_powers=self.world.interference.key_enemy_pairwise_watts,
                ally_interference_powers=result.friendly_interference_watts,
                tau_ally=self.tau_ally,
                J0=self.J0,
                lambda_safety=self.lambda_safety,
            )
            tracking_rewards = np.zeros_like(difference_rewards)
            assignment_components = self._empty_assignment_components(
                num_friendlies=len(self.agent_ids),
                pairwise_key_distances=pairwise_key_distances,
            )

        total_rewards = difference_rewards + tracking_rewards
        rewards = {agent_id: float(total_rewards[idx]) for idx, agent_id in enumerate(self.agent_ids)}

        done = result.time_step >= self.max_steps
        terminated = {agent_id: done for agent_id in self.agent_ids}
        truncated = {agent_id: False for agent_id in self.agent_ids}
        infos = {
            agent_id: {
                "time_step": result.time_step,
                "sim_time": result.sim_time,
                "action": int(result.friendly_actions[idx]),
                "friendly_interference_dbm": float(result.friendly_interference_dbm[idx]),
                "friendly_interference_watts": float(result.friendly_interference_watts[idx]),
                "key_enemy_interference_dbm": result.key_enemy_interference_dbm.copy(),
                "key_enemy_interference_watts": result.key_enemy_interference_watts.copy(),
                "policy_input_mode": self.policy_input_mode,
                "association_method": self.association_method,
                "reward_mode": self.reward_mode,
                "difference_reward": float(difference_rewards[idx]),
                "tracking_reward": float(tracking_rewards[idx]),
                "assigned_key_target": int(assignment_components["assigned_target"][idx]),
                "assigned_key_distance": float(assignment_components["assigned_distance"][idx]),
                "assigned_key_power": float(assignment_components["assigned_power"][idx]),
                "power_reward": float(assignment_components["power_reward"][idx]),
                "progress_reward": float(assignment_components["progress_reward"][idx]),
                "move_penalty": float(assignment_components["move_penalty"][idx]),
                "safety_penalty": float(assignment_components["safety_penalty"][idx]),
                "team_reward": float(np.sum(total_rewards)),
            }
            for idx, agent_id in enumerate(self.agent_ids)
        }
        return EnvStepOutput(
            observations=observations,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            infos=infos,
        )

    @staticmethod
    def _pairwise_distances(friendly_positions: np.ndarray, key_enemy_positions: np.ndarray) -> np.ndarray:
        friendly_array = np.asarray(friendly_positions, dtype=float)
        key_array = np.asarray(key_enemy_positions, dtype=float)
        if friendly_array.shape[0] == 0 or key_array.shape[0] == 0:
            return np.zeros((friendly_array.shape[0], key_array.shape[0]), dtype=float)
        return np.linalg.norm(friendly_array[:, None, :] - key_array[None, :, :], axis=-1)

    def _reset_tracking_memory(self) -> None:
        self._previous_assigned_targets = np.full(len(self.agent_ids), -1, dtype=int)
        self._previous_assigned_distances = np.full(len(self.agent_ids), np.nan, dtype=float)

    def _next_episode_seed(self) -> int | None:
        if self.seed is None:
            return int(self._episode_seed_rng.integers(0, np.iinfo(np.int32).max))
        episode_seed = int(self.seed) + self._reset_count
        self._reset_count += 1
        return episode_seed

    @staticmethod
    def _empty_assignment_components(
        num_friendlies: int,
        pairwise_key_distances: np.ndarray,
    ) -> dict[str, np.ndarray]:
        nearest_distances = (
            np.min(pairwise_key_distances, axis=1)
            if pairwise_key_distances.shape[1]
            else np.zeros(num_friendlies, dtype=float)
        )
        return {
            "assigned_target": np.full(num_friendlies, -1.0, dtype=float),
            "assigned_distance": nearest_distances.astype(float),
            "assigned_power": np.zeros(num_friendlies, dtype=float),
            "power_reward": np.zeros(num_friendlies, dtype=float),
            "progress_reward": np.zeros(num_friendlies, dtype=float),
            "move_penalty": np.zeros(num_friendlies, dtype=float),
            "safety_penalty": np.zeros(num_friendlies, dtype=float),
        }

    def get_global_key_priors(self) -> np.ndarray:
        """Return delayed/noisy key-target priors for the future association module."""

        if self.world is None:
            raise RuntimeError("reset() must be called before reading global key priors.")
        return self.global_key_target_sensor.observe(
            time_step=self.world.time_step,
            dt=self.dt,
            fallback_key_positions=self.world.get_key_enemy_positions(),
        )

    def get_association_outputs(self) -> dict[str, AssociationOutput]:
        """Return association outputs without feeding them to actor/critic."""

        if self.world is None:
            raise RuntimeError("reset() must be called before reading association outputs.")
        if self.policy_input_mode != "local_only":
            return {agent_id: empty_association_output() for agent_id in self.agent_ids}

        local_observations = self._get_current_local_observations()
        if self.association_method == "none":
            return nearest_local_candidate_association_for_friendlies(
                local_observations=local_observations,
                agent_ids=self.agent_ids,
            )

        global_key_priors = self.get_global_key_priors()
        if global_key_priors.size == 0:
            return {agent_id: empty_association_output() for agent_id in self.agent_ids}

        friendly_positions = self.world.get_friendly_positions()
        if self.association_method == "bp_jpda":
            return bp_jpda_association_for_friendlies(
                friendly_positions=friendly_positions,
                global_target_positions=global_key_priors,
                local_observations=local_observations,
                agent_ids=self.agent_ids,
                detection_radius_m=float(self.local_target_sensor.config.detection_radius_m),
                local_position_noise_std_m=float(self.local_target_sensor.config.position_noise_std_m),
                global_position_noise_std_m=float(self.global_key_target_sensor.config.position_noise_std_m),
                prior_delay_seconds=float(self.global_key_target_sensor.config.radar_delay_seconds),
                target_speed_std_mps=float(
                    self.enemy_vmax if self.enemy_vmax is not None else get_enemy_vmax()
                ),
                local_detection_probability=float(self.local_target_sensor.config.detection_probability),
                clutter_rate_per_second=float(self.local_target_sensor.config.clutter_rate_per_second),
                dt_seconds=float(self.local_target_sensor.config.dt_seconds),
                gate_sigma=self.bp_jpda_gate_sigma,
                max_iterations=self.bp_jpda_max_iterations,
                convergence_tol=self.bp_jpda_convergence_tol,
                min_missed_detection_probability=self.bp_jpda_min_missed_detection_probability,
            )

        return nearest_neighbor_association_for_friendlies(
            friendly_positions=friendly_positions,
            global_target_positions=global_key_priors,
            local_observations=local_observations,
            agent_ids=self.agent_ids,
            detection_radius_m=float(self.local_target_sensor.config.detection_radius_m),
        )

    def get_global_state(self) -> np.ndarray:
        """Return the critic state for the configured policy-input mode."""

        if self.world is None:
            raise RuntimeError("reset() must be called before get_global_state().")

        if self.policy_input_mode == "groundtruth":
            bounds = np.asarray(self.world.bounds, dtype=float)
            enemy_positions = (self.world.get_key_enemy_positions() / bounds).reshape(-1)
            friendly_positions = (self.world.get_friendly_positions() / bounds).reshape(-1)
            key_enemy_interference_dbm = self.world.interference.key_enemy_received_dbm.reshape(-1)
            friendly_interference_dbm = self.world.interference.friendly_received_dbm.reshape(-1)
            return np.concatenate(
                [enemy_positions, friendly_positions, key_enemy_interference_dbm, friendly_interference_dbm],
                dtype=np.float32,
            )

        if self._last_observation_time_step != self.world.time_step or self._last_observations is None:
            self._build_observations()
        assert self._last_observations is not None
        ordered_obs = [self._last_observations[agent_id] for agent_id in self.agent_ids]
        return np.concatenate(ordered_obs, dtype=np.float32)

    def get_local_observations(self) -> dict[str, LocalTargetObservation]:
        """Return fixed-length local target observations for each friendly UAV.

        These anonymous local candidates are the only target observations currently fed
        to actor observations and critic state.
        """

        if self.world is None:
            raise RuntimeError("reset() must be called before get_local_observations().")

        enemy_positions = self.world.get_enemy_positions()
        friendly_positions = self.world.get_friendly_positions()
        return {
            agent_id: self.local_target_sensor.observe(
                friendly_position=friendly_positions[idx],
                target_positions=enemy_positions,
            )
            for idx, agent_id in enumerate(self.agent_ids)
        }

    def _get_current_local_observations(self) -> dict[str, LocalTargetObservation]:
        if self.world is None:
            raise RuntimeError("reset() must be called before get_local_observations().")
        if (
            self._last_local_observation_time_step == self.world.time_step
            and self._last_local_observations is not None
        ):
            return self._last_local_observations
        self._last_local_observations = self.get_local_observations()
        self._last_local_observation_time_step = self.world.time_step
        return self._last_local_observations

    def _build_spaces(self) -> None:
        if self.world is None or not self.world.friendly_uavs:
            return

        action_dim = self.world.friendly_uavs[0].action_dim
        num_enemies = len(self.world.enemy_nodes)
        num_friendlies = len(self.world.friendly_uavs)
        num_key_enemies = len(self.world.get_key_enemy_nodes())
        max_local_candidates = int(self.local_target_sensor.config.max_candidates)
        self.space_spec = build_space_spec(
            num_friendlies,
            num_enemies,
            num_key_enemies,
            max_local_candidates,
            action_dim,
            self.policy_input_mode,
        )
        self.action_spaces = build_action_spaces(num_friendlies, action_dim)
        self.observation_spaces = build_observation_spaces(
            num_friendlies,
            num_enemies,
            num_key_enemies,
            max_local_candidates,
            self.world.bounds,
            self.policy_input_mode,
        )
        self.global_state_space = build_global_state_space(
            num_friendlies,
            num_enemies,
            num_key_enemies,
            max_local_candidates,
            self.world.bounds,
            self.policy_input_mode,
        )

    def _build_observations(self) -> dict[str, np.ndarray]:
        if self.world is None:
            raise RuntimeError("reset() must be called before building observations.")

        if self.policy_input_mode == "groundtruth":
            return self._build_groundtruth_observations()
        return self._build_local_only_observations()

    def _build_local_only_observations(self) -> dict[str, np.ndarray]:
        if self.world is None:
            raise RuntimeError("reset() must be called before building observations.")

        local_observations = self._get_current_local_observations()
        association_outputs = self.get_association_outputs()
        friendly_positions = self.world.get_friendly_positions()
        bounds = np.asarray(self.world.bounds, dtype=float)
        max_local_candidates = int(self.local_target_sensor.config.max_candidates)
        observations: dict[str, np.ndarray] = {}
        for idx, agent_id in enumerate(self.agent_ids):
            self_position = friendly_positions[idx]
            self_position_normalized = self_position / bounds
            local_observation = local_observations[agent_id]
            local_relative = local_observation.relative_positions / bounds
            association_weights = association_output_to_weight_vector(
                association_outputs[agent_id],
                max_local_candidates=max_local_candidates,
            )
            observation = np.concatenate(
                [
                    self_position_normalized,
                    local_relative.reshape(-1),
                    local_observation.mask,
                    association_weights,
                ]
            ).astype(np.float32)
            observations[agent_id] = observation
        self._last_observations = {agent_id: obs.copy() for agent_id, obs in observations.items()}
        self._last_observation_time_step = self.world.time_step
        return observations

    def _build_groundtruth_observations(self) -> dict[str, np.ndarray]:
        if self.world is None:
            raise RuntimeError("reset() must be called before building observations.")

        enemy_positions, enemy_is_key = build_policy_enemy_view(
            enemy_positions=self.world.get_enemy_positions(),
            enemy_roles=[node.role for node in self.world.enemy_nodes],
            key_target_positions=self.world.get_key_enemy_positions(),
            include_non_key=False,
        )
        key_enemy_positions = self.world.get_key_enemy_positions()
        friendly_positions = self.world.get_friendly_positions()
        bounds = np.asarray(self.world.bounds, dtype=float)
        distance_norm = float(np.linalg.norm(bounds))
        pairwise_key_distances = self._pairwise_distances(friendly_positions, key_enemy_positions)
        assigned_key_targets = (
            compute_distance_assignment(pairwise_distances=pairwise_key_distances)["assigned_target"].astype(int)
            if key_enemy_positions.size
            else np.zeros(0, dtype=int)
        )
        observations: dict[str, np.ndarray] = {}
        for idx, agent_id in enumerate(self.agent_ids):
            self_position = friendly_positions[idx]
            self_position_normalized = self_position / bounds
            enemy_position_features = enemy_positions / bounds
            enemy_features = np.column_stack([enemy_position_features, enemy_is_key]).reshape(-1)
            key_distances = (
                np.linalg.norm(key_enemy_positions - self_position, axis=1) / distance_norm
                if key_enemy_positions.size
                else np.zeros(0, dtype=float)
            )
            if key_enemy_positions.size:
                assigned_key_idx = int(assigned_key_targets[idx])
                assigned_key_relative = (key_enemy_positions[assigned_key_idx] - self_position) / bounds
                assigned_key_distance = np.asarray([key_distances[assigned_key_idx]], dtype=float)
                assigned_key_one_hot = np.zeros(key_enemy_positions.shape[0], dtype=float)
                assigned_key_one_hot[assigned_key_idx] = 1.0
                assigned_key_features = np.concatenate(
                    [assigned_key_relative, assigned_key_distance, assigned_key_one_hot]
                )
            else:
                assigned_key_features = np.zeros(0, dtype=float)
            friendly_relative = ((friendly_positions - self_position) / bounds).reshape(-1)
            observation = np.concatenate(
                [
                    self_position_normalized,
                    enemy_features,
                    key_distances,
                    assigned_key_features,
                    friendly_relative,
                ]
            ).astype(np.float32)
            observations[agent_id] = observation
        self._last_observations = {agent_id: obs.copy() for agent_id, obs in observations.items()}
        self._last_observation_time_step = self.world.time_step
        return observations
