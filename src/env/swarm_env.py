"""Tracking-oriented environment wrapper for the swarm project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from src.env.world import SwarmWorld
from src.sensing.association import (
    BPJPDAFilterState,
    GMPHDFilterState,
    KeyTrackPosterior,
    LMBFilterState,
    MultiTargetTrackOutput,
    STATE_DIM,
    bp_jpda_association_for_friendlies,
    covariance_anchor_bp_jpda_association_for_friendlies,
    empty_multi_target_track_output,
    flatten_track_output,
    gm_phd_association_for_friendlies,
    hard_anchor_bp_jpda_association_for_friendlies,
    labeled_anchor_bp_jpda_association_for_friendlies,
    lmb_association_for_friendlies,
    nearest_local_candidate_association_for_friendlies,
    prior_only_association_for_friendlies,
)
from src.sensing.delta_glmb import DeltaGLMBFilterState, delta_glmb_association_for_friendlies
from src.sensing.global_sensor import (
    GlobalKeyTargetStatePrior,
    GlobalKeyTargetStatePriorSensor,
    GlobalKeyTargetSensingConfig,
)
from src.sensing.local_sensor import LocalSensingConfig, LocalTargetObservation, LocalTargetSensor
from src.simulation.ground_truth import NearbyDistanceLimits, PositionDict, ScenarioGroundTruth, initialize_scene
from src.tracking.controller import (
    ControlCommand,
    build_placeholder_control_commands,
    build_tracking_control_commands,
    extract_actions,
)
from src.tracking.fusion import FusedTrackSet
from src.tracking.fusion_center import FusionCenter, LocalTrackSet, build_local_track_sets
from src.tracking.task_assignment import (
    TaskAssignmentResult,
    build_empty_task_assignment,
    build_hungarian_task_assignment,
)
from src.utils.config_loader import (
    get_association_config,
    get_enemy_vmax,
    get_friendly_vmax,
    get_global_sensing_config,
    get_local_sensing_config,
    get_num_friendly_uavs,
    get_num_key_targets,
    get_num_nonkey_targets,
    get_num_steps,
    get_tracking_association_method,
    get_tracking_assignment_method,
    get_tracking_input_mode,
    get_track_fusion_config,
)

TRACKING_INPUT_MODES = {"local_only", "groundtruth"}
ASSOCIATION_METHODS = {
    "none",
    "prior_only",
    "bp_jpda",
    "hard_anchor_bp_jpda",
    "covariance_anchor_bp_jpda",
    "labeled_anchor_bp_jpda",
    "gm_phd",
    "lmb",
    "delta_glmb",
}
ASSIGNMENT_METHODS = {"placeholder", "hungarian"}


@dataclass
class EnvStepOutput:
    observations: dict[str, np.ndarray]
    terminated: dict[str, bool]
    truncated: dict[str, bool]
    infos: dict[str, dict[str, Any]]


class SwarmEnv:
    """Tracking-oriented env wrapper built on top of SwarmWorld."""

    def __init__(
        self,
        key_num: int | None = None,
        nonkey_num: int | None = None,
        mydrone_num: int | None = None,
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
        distance_scale_m: float | None = None,
        tracking_input_mode: str | None = None,
        policy_input_mode: str | None = None,
    ) -> None:
        self.key_num = int(get_num_key_targets() if key_num is None else key_num)
        self.nonkey_num = int(get_num_nonkey_targets() if nonkey_num is None else nonkey_num)
        self.mydrone_num = int(get_num_friendly_uavs() if mydrone_num is None else mydrone_num)
        self.bounds = None if bounds is None else np.asarray(bounds, dtype=float)
        self.dt = float(dt)
        self.max_steps = get_num_steps() if max_steps is None else int(max_steps)
        self.seed = seed
        self.enemy_vmax = float(get_enemy_vmax() if enemy_vmax is None else enemy_vmax)
        self.friendly_vmax = float(get_friendly_vmax() if friendly_vmax is None else friendly_vmax)
        self.initial_positions = initial_positions
        self.nearby_distance_limits = nearby_distance_limits
        self.agent_ids = tuple(f"friendly_{idx}" for idx in range(self.mydrone_num))

        raw_tracking_input_mode = tracking_input_mode if tracking_input_mode is not None else policy_input_mode
        resolved_tracking_input_mode = (
            get_tracking_input_mode() if raw_tracking_input_mode is None else raw_tracking_input_mode
        )
        self.tracking_input_mode = str(resolved_tracking_input_mode).lower()
        if self.tracking_input_mode not in TRACKING_INPUT_MODES:
            raise ValueError(
                "tracking.mode must be either 'local_only' or 'groundtruth', "
                f"got {self.tracking_input_mode!r}."
            )

        self.association_method = get_tracking_association_method()
        if self.association_method not in ASSOCIATION_METHODS:
            raise ValueError(
                "tracking.association_method must be one of 'none', 'bp_jpda', "
                "'prior_only', 'bp_jpda', 'hard_anchor_bp_jpda', 'covariance_anchor_bp_jpda', "
                "'labeled_anchor_bp_jpda', 'gm_phd', 'lmb', or 'delta_glmb', "
                f"got {self.association_method!r}."
            )
        self.assignment_method = get_tracking_assignment_method()
        if self.assignment_method not in ASSIGNMENT_METHODS:
            raise ValueError(
                "tracking.assignment must be one of 'placeholder' or 'hungarian', "
                f"got {self.assignment_method!r}."
            )

        association_config = get_association_config()
        self.nominal_clutter_rate_per_second = float(association_config["nominal_clutter_rate_per_second"])
        self.bp_jpda_gate_sigma = float(association_config["bp_jpda_gate_sigma"])
        self.bp_jpda_max_iterations = int(association_config["bp_jpda_max_iterations"])
        self.bp_jpda_convergence_tol = float(association_config["bp_jpda_convergence_tol"])
        self.bp_jpda_min_missed_detection_probability = float(
            association_config["bp_jpda_min_missed_detection_probability"]
        )
        self.association_track_max_missed_steps = int(association_config["association_track_max_missed_steps"])
        self.aa_bp_jpda_local_attention_temperature = float(
            association_config["aa_bp_jpda_local_attention_temperature"]
        )
        self.aa_bp_jpda_cross_uav_position_scale_m = float(
            association_config["aa_bp_jpda_cross_uav_position_scale_m"]
        )
        self.aa_bp_jpda_cross_uav_velocity_scale_mps = float(
            association_config["aa_bp_jpda_cross_uav_velocity_scale_mps"]
        )
        self.aa_bp_jpda_cross_uav_support_threshold = float(
            association_config["aa_bp_jpda_cross_uav_support_threshold"]
        )
        self.aa_bp_jpda_local_evidence_weight = float(association_config["aa_bp_jpda_local_evidence_weight"])
        self.aa_bp_jpda_velocity_evidence_weight = float(association_config["aa_bp_jpda_velocity_evidence_weight"])
        self.aa_bp_jpda_density_penalty_weight = float(association_config["aa_bp_jpda_density_penalty_weight"])
        self.aa_bp_jpda_consensus_evidence_weight = float(
            association_config["aa_bp_jpda_consensus_evidence_weight"]
        )
        self.aa_bp_jpda_consensus_count_weight = float(association_config["aa_bp_jpda_consensus_count_weight"])
        self.aa_bp_jpda_max_log_evidence = float(association_config["aa_bp_jpda_max_log_evidence"])
        self.recursive_motion_model = str(association_config["recursive_motion_model"]).lower()
        self.imm_model_types = tuple(str(model_type) for model_type in association_config["imm_model_types"])
        self.imm_model_transition_matrix = association_config["imm_model_transition_matrix"]
        self.imm_initial_model_probabilities = association_config["imm_initial_model_probabilities"]
        self.imm_turn_rate_radps = float(association_config["imm_turn_rate_radps"])
        self.imm_ct_process_noise_scale = float(association_config["imm_ct_process_noise_scale"])
        self.gm_phd_survival_probability = float(association_config["gm_phd_survival_probability"])
        self.gm_phd_birth_weight = float(association_config["gm_phd_birth_weight"])
        self.gm_phd_prune_threshold = float(association_config["gm_phd_prune_threshold"])
        self.gm_phd_merge_mahalanobis = float(association_config["gm_phd_merge_mahalanobis"])
        self.gm_phd_max_components = int(association_config["gm_phd_max_components"])
        self.gm_phd_confidence_threshold = float(association_config["gm_phd_confidence_threshold"])
        self.delta_glmb_hypothesis_weight_prune_threshold = float(
            association_config["delta_glmb_hypothesis_weight_prune_threshold"]
        )
        self.gm_phd_process_position_std_m = np.asarray(
            association_config["gm_phd_process_position_std_m"], dtype=float
        )
        self.gm_phd_process_velocity_std_mps = np.asarray(
            association_config["gm_phd_process_velocity_std_mps"], dtype=float
        )
        fusion_config = get_track_fusion_config()
        self.fusion_self_position_std_m = np.asarray(fusion_config["self_position_std_m"], dtype=float)
        self.fusion_min_confidence = float(fusion_config["min_confidence"])
        self.fusion_mahalanobis_distance_threshold = float(
            fusion_config["mahalanobis_distance_threshold"]
        )
        self.fusion_ci_search_steps = int(fusion_config["ci_search_steps"])
        self.fusion_center = FusionCenter(
            self_position_std_m=self.fusion_self_position_std_m,
            min_confidence=self.fusion_min_confidence,
            mahalanobis_distance_threshold=self.fusion_mahalanobis_distance_threshold,
            ci_search_steps=self.fusion_ci_search_steps,
        )

        sensing_config = get_global_sensing_config()
        global_sensor_config = GlobalKeyTargetSensingConfig(
            update_interval_seconds=float(sensing_config["update_interval_seconds"]),
            latency_mean_seconds=float(sensing_config["latency_mean_seconds"]),
            latency_std_seconds=float(sensing_config["latency_std_seconds"]),
            dt_seconds=self.dt,
            prior_position_std_m=np.asarray(sensing_config["prior_position_std_m"], dtype=float),
            prior_velocity_std_mps=np.asarray(sensing_config["prior_velocity_std_mps"], dtype=float),
            prior_bias_position_std_m=np.asarray(sensing_config["prior_bias_position_std_m"], dtype=float),
            prior_bias_velocity_std_mps=np.asarray(sensing_config["prior_bias_velocity_std_mps"], dtype=float),
            prior_bias_correlation_steps=float(sensing_config["prior_bias_correlation_steps"]),
            prior_missed_probability=float(sensing_config["prior_missed_probability"]),
        )
        self.global_key_target_sensor = GlobalKeyTargetStatePriorSensor(global_sensor_config, seed=self.seed)

        local_sensing_config = get_local_sensing_config()
        self.relative_velocity_limit_mps = float(self.enemy_vmax + self.friendly_vmax)
        local_sensor_config = LocalSensingConfig(
            detection_radius_m=float(local_sensing_config["detection_radius_m"]),
            measurement_position_std_m=np.asarray(local_sensing_config["measurement_position_std_m"], dtype=float),
            measurement_velocity_std_mps=np.asarray(local_sensing_config["measurement_velocity_std_mps"], dtype=float),
            detection_probability=float(local_sensing_config["detection_probability"]),
            detection_probability_min=float(local_sensing_config["detection_probability_min"]),
            detection_full_probability_range_m=float(local_sensing_config["detection_full_probability_range_m"]),
            detection_density_penalty_alpha=float(local_sensing_config["detection_density_penalty_alpha"]),
            clutter_rate_per_second=float(local_sensing_config["clutter_rate_per_second"]),
            clutter_hotspot_probability=float(local_sensing_config["clutter_hotspot_probability"]),
            clutter_hotspot_position_std_m=np.asarray(local_sensing_config["clutter_hotspot_position_std_m"], dtype=float),
            clutter_hotspot_velocity_std_mps=np.asarray(
                local_sensing_config["clutter_hotspot_velocity_std_mps"], dtype=float
            ),
            dt_seconds=self.dt,
            relative_velocity_limit_mps=self.relative_velocity_limit_mps,
        )
        self.local_target_sensor = LocalTargetSensor(local_sensor_config, seed=self.seed)

        self.scene: ScenarioGroundTruth | None = None
        self.world: SwarmWorld | None = None
        self._last_observations: dict[str, np.ndarray] | None = None
        self._last_observation_time_step: int | None = None
        self._last_local_observations: dict[str, LocalTargetObservation] | None = None
        self._last_local_observation_time_step: int | None = None
        self._last_association_outputs: dict[str, MultiTargetTrackOutput] | None = None
        self._last_association_time_step: int | None = None
        self._last_local_track_sets: dict[str, LocalTrackSet] | None = None
        self._last_local_track_set_time_step: int | None = None
        self._last_fused_track_set: FusedTrackSet | None = None
        self._last_fused_time_step: int | None = None
        self._last_task_assignment: TaskAssignmentResult | None = None
        self._last_task_assignment_time_step: int | None = None
        self._previous_task_assignment: TaskAssignmentResult | None = None
        self._bp_jpda_filter_states: dict[str, dict[int, BPJPDAFilterState]] = {}
        self._gm_phd_filter_states: dict[str, GMPHDFilterState | None] = {}
        self._lmb_filter_states: dict[str, dict[int, LMBFilterState]] = {}
        self._delta_glmb_filter_states: dict[str, DeltaGLMBFilterState] = {}
        self._prior_only_missed_counts = np.zeros(self.key_num, dtype=int)
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
        self.global_key_target_sensor.reset(seed=episode_seed)
        self.local_target_sensor.reset(seed=episode_seed)
        self._last_observations = None
        self._last_observation_time_step = None
        self._last_local_observations = None
        self._last_local_observation_time_step = None
        self._last_association_outputs = None
        self._last_association_time_step = None
        self._last_local_track_sets = None
        self._last_local_track_set_time_step = None
        self._last_fused_track_set = None
        self._last_fused_time_step = None
        self._reset_task_assignment_cache()
        self._previous_task_assignment = None
        self._bp_jpda_filter_states = {agent_id: {} for agent_id in self.agent_ids}
        self._gm_phd_filter_states = {agent_id: None for agent_id in self.agent_ids}
        self._lmb_filter_states = {agent_id: {} for agent_id in self.agent_ids}
        self._delta_glmb_filter_states = {
            agent_id: DeltaGLMBFilterState(hypotheses=tuple()) for agent_id in self.agent_ids
        }
        self._prior_only_missed_counts = np.zeros(self.key_num, dtype=int)

        observations = self._build_observations()
        infos = {
            agent_id: {
                "time_step": 0,
                "sim_time": 0.0,
                "friendly_interference_dbm": float(self.world.interference.friendly_received_dbm[idx]),
                "friendly_interference_watts": float(self.world.interference.friendly_received_watts[idx]),
                "key_enemy_interference_dbm": self.world.interference.key_enemy_received_dbm.copy(),
                "key_enemy_interference_watts": self.world.interference.key_enemy_received_watts.copy(),
                "tracking_input_mode": self.tracking_input_mode,
                "association_method": self.association_method,
                "assignment_method": self.assignment_method,
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

        observations = self._build_observations()
        if self._last_task_assignment is not None:
            self._previous_task_assignment = self._last_task_assignment
        self._reset_task_assignment_cache()

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
                "tracking_input_mode": self.tracking_input_mode,
                "association_method": self.association_method,
                "assignment_method": self.assignment_method,
            }
            for idx, agent_id in enumerate(self.agent_ids)
        }
        return EnvStepOutput(
            observations=observations,
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

    def _next_episode_seed(self) -> int | None:
        if self.seed is None:
            return int(self._episode_seed_rng.integers(0, np.iinfo(np.int32).max))
        episode_seed = int(self.seed) + self._reset_count
        self._reset_count += 1
        return episode_seed

    def get_global_key_priors(self) -> GlobalKeyTargetStatePrior:
        """Return current noisy external key-target priors for local association."""

        if self.world is None:
            raise RuntimeError("reset() must be called before reading global key priors.")
        return self.global_key_target_sensor.observe(
            time_step=self.world.time_step,
            key_target_states=self.world.get_key_enemy_states(),
        )

    def _previous_fused_anchor_arrays(
        self,
        *,
        num_key_targets: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        if self.world is None or self._last_fused_track_set is None or self._last_fused_time_step is None:
            return None, None, None
        if int(self._last_fused_time_step) >= int(self.world.time_step):
            return None, None, None

        states = np.zeros((int(num_key_targets), STATE_DIM), dtype=float)
        covariances = np.zeros((int(num_key_targets), STATE_DIM, STATE_DIM), dtype=float)
        valid_mask = np.zeros(int(num_key_targets), dtype=bool)
        elapsed_steps = max(int(self.world.time_step) - int(self._last_fused_time_step), 1)
        prediction_dt = float(elapsed_steps) * float(self.dt)
        transition = np.eye(STATE_DIM, dtype=float)
        transition[:3, 3:6] = np.eye(3, dtype=float) * prediction_dt
        process_diagonal = np.concatenate(
            [
                np.square(self.gm_phd_process_position_std_m),
                np.square(self.gm_phd_process_velocity_std_mps),
            ]
        ) * float(elapsed_steps)

        for track in self._last_fused_track_set.tracks:
            target_idx = int(track.global_target_index)
            if target_idx < 0 or target_idx >= int(num_key_targets) or not bool(track.valid):
                continue
            state = np.asarray(track.fused_state_mean, dtype=float).reshape(STATE_DIM).copy()
            state[:3] = state[:3] + state[3:6] * prediction_dt
            covariance = np.asarray(track.fused_state_covariance, dtype=float)
            if covariance.shape == (STATE_DIM,):
                covariance = np.diag(covariance)
            covariance = 0.5 * (covariance + covariance.T)
            covariance = transition @ covariance @ transition.T + np.diag(process_diagonal)
            states[target_idx] = state
            covariances[target_idx] = covariance
            valid_mask[target_idx] = True

        if not np.any(valid_mask):
            return None, None, None
        return states, covariances, valid_mask

    def get_association_outputs(self) -> dict[str, MultiTargetTrackOutput]:
        """Return per-friendly multi-target local track posteriors for the current time step."""

        if self.world is None:
            raise RuntimeError("reset() must be called before reading association outputs.")
        if self._last_association_time_step == self.world.time_step and self._last_association_outputs is not None:
            return {agent_id: output for agent_id, output in self._last_association_outputs.items()}

        if self.tracking_input_mode == "groundtruth":
            outputs = self._build_groundtruth_track_outputs()
            self._last_association_outputs = outputs
            self._last_association_time_step = self.world.time_step
            return {agent_id: output for agent_id, output in outputs.items()}

        global_key_priors = self.get_global_key_priors()
        friendly_states = self.world.get_friendly_states()

        if self.association_method == "prior_only":
            previous_states, previous_covariances, previous_valid_mask = self._previous_fused_anchor_arrays(
                num_key_targets=global_key_priors.states.shape[0]
            )
            outputs, self._prior_only_missed_counts = prior_only_association_for_friendlies(
                friendly_states=friendly_states,
                global_target_priors=global_key_priors,
                agent_ids=self.agent_ids,
                previous_global_states=previous_states,
                previous_global_covariances=previous_covariances,
                previous_global_valid_mask=previous_valid_mask,
                previous_missed_counts=self._prior_only_missed_counts,
                max_missed_steps=self.association_track_max_missed_steps,
                prediction_confidence=self.gm_phd_confidence_threshold,
            )
            self._last_association_outputs = outputs
            self._last_association_time_step = self.world.time_step
            return {agent_id: output for agent_id, output in outputs.items()}

        local_observations = self._get_current_local_observations()

        if self.association_method == "none":
            outputs = nearest_local_candidate_association_for_friendlies(
                friendly_states=friendly_states,
                global_target_priors=global_key_priors,
                local_observations=local_observations,
                agent_ids=self.agent_ids,
            )
            self._last_association_outputs = outputs
            self._last_association_time_step = self.world.time_step
            return {agent_id: output for agent_id, output in outputs.items()}

        if global_key_priors.states.size == 0:
            empty_outputs = {
                agent_id: empty_multi_target_track_output(
                    agent_id=agent_id,
                    prior_relative_states=np.zeros((0, 6), dtype=float),
                    prior_covariance_diagonal=global_key_priors.covariance_diagonal,
                    mode=self.tracking_input_mode,
                )
                for agent_id in self.agent_ids
            }
            self._bp_jpda_filter_states = {agent_id: {} for agent_id in self.agent_ids}
            self._gm_phd_filter_states = {agent_id: None for agent_id in self.agent_ids}
            self._lmb_filter_states = {agent_id: {} for agent_id in self.agent_ids}
            self._delta_glmb_filter_states = {
                agent_id: DeltaGLMBFilterState(hypotheses=tuple()) for agent_id in self.agent_ids
            }
            self._last_association_outputs = empty_outputs
            self._last_association_time_step = self.world.time_step
            return {agent_id: output for agent_id, output in empty_outputs.items()}

        if self.association_method == "bp_jpda":
            outputs, next_filter_states = bp_jpda_association_for_friendlies(
                friendly_states=friendly_states,
                global_target_priors=global_key_priors,
                local_observations=local_observations,
                agent_ids=self.agent_ids,
                previous_filter_states=self._bp_jpda_filter_states,
                detection_radius_m=float(self.local_target_sensor.config.detection_radius_m),
                relative_velocity_limit_mps=self.relative_velocity_limit_mps,
                local_detection_probability=float(self.local_target_sensor.config.detection_probability),
                local_detection_probability_min=float(
                    self.local_target_sensor.config.detection_probability_min
                ),
                detection_full_probability_range_m=float(
                    self.local_target_sensor.config.detection_full_probability_range_m
                ),
                clutter_rate_per_second=self.nominal_clutter_rate_per_second,
                recursive_motion_model=self.recursive_motion_model,
                imm_model_types=self.imm_model_types,
                imm_model_transition_matrix=self.imm_model_transition_matrix,
                imm_initial_model_probabilities=self.imm_initial_model_probabilities,
                dt_seconds=float(self.local_target_sensor.config.dt_seconds),
                gate_sigma=self.bp_jpda_gate_sigma,
                max_iterations=self.bp_jpda_max_iterations,
                convergence_tol=self.bp_jpda_convergence_tol,
                min_missed_detection_probability=self.bp_jpda_min_missed_detection_probability,
                max_missed_steps=self.association_track_max_missed_steps,
                survival_probability=self.gm_phd_survival_probability,
                birth_weight=self.gm_phd_birth_weight,
                process_position_std_m=self.gm_phd_process_position_std_m,
                process_velocity_std_mps=self.gm_phd_process_velocity_std_mps,
                imm_turn_rate_radps=self.imm_turn_rate_radps,
                imm_ct_process_noise_scale=self.imm_ct_process_noise_scale,
                confidence_threshold=self.gm_phd_confidence_threshold,
            )
            self._bp_jpda_filter_states = next_filter_states
            self._last_association_outputs = outputs
            self._last_association_time_step = self.world.time_step
            return {agent_id: output for agent_id, output in outputs.items()}

        anchor_methods = {
            "hard_anchor_bp_jpda": hard_anchor_bp_jpda_association_for_friendlies,
            "covariance_anchor_bp_jpda": covariance_anchor_bp_jpda_association_for_friendlies,
            "labeled_anchor_bp_jpda": labeled_anchor_bp_jpda_association_for_friendlies,
        }
        if self.association_method in anchor_methods:
            outputs, next_filter_states = anchor_methods[self.association_method](
                friendly_states=friendly_states,
                global_target_priors=global_key_priors,
                local_observations=local_observations,
                agent_ids=self.agent_ids,
                previous_filter_states=self._bp_jpda_filter_states,
                detection_radius_m=float(self.local_target_sensor.config.detection_radius_m),
                relative_velocity_limit_mps=self.relative_velocity_limit_mps,
                local_detection_probability=float(self.local_target_sensor.config.detection_probability),
                local_detection_probability_min=float(
                    self.local_target_sensor.config.detection_probability_min
                ),
                detection_full_probability_range_m=float(
                    self.local_target_sensor.config.detection_full_probability_range_m
                ),
                clutter_rate_per_second=self.nominal_clutter_rate_per_second,
                recursive_motion_model=self.recursive_motion_model,
                imm_model_types=self.imm_model_types,
                imm_model_transition_matrix=self.imm_model_transition_matrix,
                imm_initial_model_probabilities=self.imm_initial_model_probabilities,
                dt_seconds=float(self.local_target_sensor.config.dt_seconds),
                gate_sigma=self.bp_jpda_gate_sigma,
                max_iterations=self.bp_jpda_max_iterations,
                convergence_tol=self.bp_jpda_convergence_tol,
                min_missed_detection_probability=self.bp_jpda_min_missed_detection_probability,
                max_missed_steps=self.association_track_max_missed_steps,
                survival_probability=self.gm_phd_survival_probability,
                birth_weight=self.gm_phd_birth_weight,
                process_position_std_m=self.gm_phd_process_position_std_m,
                process_velocity_std_mps=self.gm_phd_process_velocity_std_mps,
                imm_turn_rate_radps=self.imm_turn_rate_radps,
                imm_ct_process_noise_scale=self.imm_ct_process_noise_scale,
                confidence_threshold=self.gm_phd_confidence_threshold,
            )
            self._bp_jpda_filter_states = next_filter_states
            self._last_association_outputs = outputs
            self._last_association_time_step = self.world.time_step
            return {agent_id: output for agent_id, output in outputs.items()}

        if self.association_method == "gm_phd":
            outputs, next_filter_states = gm_phd_association_for_friendlies(
                friendly_states=friendly_states,
                global_target_priors=global_key_priors,
                local_observations=local_observations,
                agent_ids=self.agent_ids,
                previous_filter_states=self._gm_phd_filter_states,
                detection_radius_m=float(self.local_target_sensor.config.detection_radius_m),
                relative_velocity_limit_mps=self.relative_velocity_limit_mps,
                local_detection_probability=float(self.local_target_sensor.config.detection_probability),
                local_detection_probability_min=float(
                    self.local_target_sensor.config.detection_probability_min
                ),
                detection_full_probability_range_m=float(
                    self.local_target_sensor.config.detection_full_probability_range_m
                ),
                clutter_rate_per_second=self.nominal_clutter_rate_per_second,
                recursive_motion_model=self.recursive_motion_model,
                imm_model_types=self.imm_model_types,
                imm_model_transition_matrix=self.imm_model_transition_matrix,
                imm_initial_model_probabilities=self.imm_initial_model_probabilities,
                dt_seconds=float(self.local_target_sensor.config.dt_seconds),
                gate_sigma=self.bp_jpda_gate_sigma,
                survival_probability=self.gm_phd_survival_probability,
                birth_weight=self.gm_phd_birth_weight,
                prune_threshold=self.gm_phd_prune_threshold,
                merge_mahalanobis_threshold=self.gm_phd_merge_mahalanobis,
                max_components=self.gm_phd_max_components,
                process_position_std_m=self.gm_phd_process_position_std_m,
                process_velocity_std_mps=self.gm_phd_process_velocity_std_mps,
                imm_turn_rate_radps=self.imm_turn_rate_radps,
                imm_ct_process_noise_scale=self.imm_ct_process_noise_scale,
                confidence_threshold=self.gm_phd_confidence_threshold,
                max_missed_steps=self.association_track_max_missed_steps,
            )
            self._gm_phd_filter_states = next_filter_states
            self._last_association_outputs = outputs
            self._last_association_time_step = self.world.time_step
            return {agent_id: output for agent_id, output in outputs.items()}

        if self.association_method == "lmb":
            outputs, next_filter_states = lmb_association_for_friendlies(
                friendly_states=friendly_states,
                global_target_priors=global_key_priors,
                local_observations=local_observations,
                agent_ids=self.agent_ids,
                previous_filter_states=self._lmb_filter_states,
                detection_radius_m=float(self.local_target_sensor.config.detection_radius_m),
                relative_velocity_limit_mps=self.relative_velocity_limit_mps,
                local_detection_probability=float(self.local_target_sensor.config.detection_probability),
                local_detection_probability_min=float(
                    self.local_target_sensor.config.detection_probability_min
                ),
                detection_full_probability_range_m=float(
                    self.local_target_sensor.config.detection_full_probability_range_m
                ),
                clutter_rate_per_second=self.nominal_clutter_rate_per_second,
                recursive_motion_model=self.recursive_motion_model,
                imm_model_types=self.imm_model_types,
                imm_model_transition_matrix=self.imm_model_transition_matrix,
                imm_initial_model_probabilities=self.imm_initial_model_probabilities,
                dt_seconds=float(self.local_target_sensor.config.dt_seconds),
                gate_sigma=self.bp_jpda_gate_sigma,
                max_iterations=self.bp_jpda_max_iterations,
                convergence_tol=self.bp_jpda_convergence_tol,
                survival_probability=self.gm_phd_survival_probability,
                birth_weight=self.gm_phd_birth_weight,
                process_position_std_m=self.gm_phd_process_position_std_m,
                process_velocity_std_mps=self.gm_phd_process_velocity_std_mps,
                imm_turn_rate_radps=self.imm_turn_rate_radps,
                imm_ct_process_noise_scale=self.imm_ct_process_noise_scale,
                confidence_threshold=self.gm_phd_confidence_threshold,
                max_missed_steps=self.association_track_max_missed_steps,
            )
            self._lmb_filter_states = next_filter_states
            self._last_association_outputs = outputs
            self._last_association_time_step = self.world.time_step
            return {agent_id: output for agent_id, output in outputs.items()}

        if self.association_method == "delta_glmb":
            outputs, next_filter_states = delta_glmb_association_for_friendlies(
                friendly_states=friendly_states,
                global_target_priors=global_key_priors,
                local_observations=local_observations,
                agent_ids=self.agent_ids,
                previous_filter_states=self._delta_glmb_filter_states,
                detection_radius_m=float(self.local_target_sensor.config.detection_radius_m),
                relative_velocity_limit_mps=self.relative_velocity_limit_mps,
                local_detection_probability=float(self.local_target_sensor.config.detection_probability),
                local_detection_probability_min=float(
                    self.local_target_sensor.config.detection_probability_min
                ),
                detection_full_probability_range_m=float(
                    self.local_target_sensor.config.detection_full_probability_range_m
                ),
                clutter_rate_per_second=self.nominal_clutter_rate_per_second,
                recursive_motion_model=self.recursive_motion_model,
                imm_model_types=self.imm_model_types,
                imm_model_transition_matrix=self.imm_model_transition_matrix,
                imm_initial_model_probabilities=self.imm_initial_model_probabilities,
                dt_seconds=float(self.local_target_sensor.config.dt_seconds),
                gate_sigma=self.bp_jpda_gate_sigma,
                survival_probability=self.gm_phd_survival_probability,
                birth_weight=self.gm_phd_birth_weight,
                process_position_std_m=self.gm_phd_process_position_std_m,
                process_velocity_std_mps=self.gm_phd_process_velocity_std_mps,
                imm_turn_rate_radps=self.imm_turn_rate_radps,
                imm_ct_process_noise_scale=self.imm_ct_process_noise_scale,
                confidence_threshold=self.gm_phd_confidence_threshold,
                hypothesis_weight_prune_threshold=self.delta_glmb_hypothesis_weight_prune_threshold,
                max_missed_steps=self.association_track_max_missed_steps,
            )
            self._delta_glmb_filter_states = next_filter_states
            self._last_association_outputs = outputs
            self._last_association_time_step = self.world.time_step
            return {agent_id: output for agent_id, output in outputs.items()}

        raise RuntimeError(
            "Unsupported association_method reached after validation: "
            f"{self.association_method!r}."
        )

    def get_track_posterior_outputs(self) -> dict[str, MultiTargetTrackOutput]:
        """Alias exposing the current per-agent key-target posterior tracks."""

        return self.get_association_outputs()

    def get_local_track_sets(self) -> dict[str, LocalTrackSet]:
        """Return per-agent local track summaries reported to the centralized fusion layer."""

        if self.world is None:
            raise RuntimeError("reset() must be called before reading local track sets.")
        if self._last_local_track_set_time_step == self.world.time_step and self._last_local_track_sets is not None:
            return {agent_id: track_set for agent_id, track_set in self._last_local_track_sets.items()}

        local_track_sets = build_local_track_sets(
            timestamp=self.world.time_step,
            track_outputs=self.get_association_outputs(),
            agent_ids=self.agent_ids,
        )
        self._last_local_track_sets = local_track_sets
        self._last_local_track_set_time_step = self.world.time_step
        return {agent_id: track_set for agent_id, track_set in local_track_sets.items()}

    def get_fused_track_set(self) -> FusedTrackSet:
        """Return the centralized fused key-track set in the global frame."""

        if self.world is None:
            raise RuntimeError("reset() must be called before reading fused tracks.")
        if self._last_fused_time_step == self.world.time_step and self._last_fused_track_set is not None:
            return self._last_fused_track_set

        fused_track_set = self.fusion_center.fuse(
            timestamp=self.world.time_step,
            local_track_sets=self.get_local_track_sets(),
            friendly_states=self.world.get_friendly_states(),
            agent_ids=self.agent_ids,
        )
        self._last_fused_track_set = fused_track_set
        self._last_fused_time_step = self.world.time_step
        return fused_track_set

    def get_local_observations(self) -> dict[str, LocalTargetObservation]:
        """Return fixed-length local measurement sets for each friendly UAV."""

        if self.world is None:
            raise RuntimeError("reset() must be called before get_local_observations().")

        enemy_states = self.world.get_enemy_states()
        friendly_states = self.world.get_friendly_states()
        return {
            agent_id: self.local_target_sensor.observe(
                friendly_state=friendly_states[idx],
                target_states=enemy_states,
            )
            for idx, agent_id in enumerate(self.agent_ids)
        }

    def build_task_assignment(self) -> TaskAssignmentResult:
        """Build downstream key-track assignments using the configured assignment method."""

        if self.world is None:
            raise RuntimeError("reset() must be called before building task assignments.")
        if (
            self._last_task_assignment is not None
            and self._last_task_assignment_time_step == self.world.time_step
        ):
            return self._last_task_assignment
        if self.assignment_method == "placeholder":
            assignment = build_empty_task_assignment(self.agent_ids, timestamp=self.world.time_step)
            self._last_task_assignment = assignment
            self._last_task_assignment_time_step = self.world.time_step
            return assignment
        if self.assignment_method == "hungarian":
            assignment = build_hungarian_task_assignment(
                self.agent_ids,
                timestamp=self.world.time_step,
                friendly_states=self.world.get_friendly_states(),
                fused_track_set=self.get_fused_track_set(),
                prediction_dt=self.dt,
                switch_penalty=self.friendly_vmax * self.dt,
                previous_assignment=self._previous_task_assignment,
            )
            self._last_task_assignment = assignment
            self._last_task_assignment_time_step = self.world.time_step
            return assignment
        raise RuntimeError(
            "Unsupported assignment_method reached after validation: "
            f"{self.assignment_method!r}."
        )

    def build_task_assignment_placeholder(self) -> TaskAssignmentResult:
        """Backward-compatible alias that now respects the configured assignment method."""

        return self.build_task_assignment()

    def build_control_command_placeholder(
        self,
        assignment: TaskAssignmentResult | None = None,
    ) -> dict[str, ControlCommand]:
        """Backward-compatible alias that now performs simple pursuit control when assigned."""

        if self.world is None:
            raise RuntimeError("reset() must be called before building control commands.")
        if assignment is not None:
            return self.build_control_commands(assignment=assignment)
        return build_placeholder_control_commands(self.agent_ids, stop_action=0)

    def build_control_commands(
        self,
        assignment: TaskAssignmentResult | None = None,
    ) -> dict[str, ControlCommand]:
        """Build simple control commands that pursue the assigned fused target state."""

        if self.world is None:
            raise RuntimeError("reset() must be called before building control commands.")
        if assignment is None:
            assignment = self.build_task_assignment()
        commands = build_tracking_control_commands(
            self.agent_ids,
            friendly_states=self.world.get_friendly_states(),
            assignment=assignment,
            fused_track_set=self.get_fused_track_set(),
            stop_action=0,
        )
        return commands

    def plan_tracking_actions_placeholder(
        self,
        assignment: TaskAssignmentResult | None = None,
    ) -> dict[str, int]:
        """Backward-compatible alias returning pursuit actions from the configured assignment."""

        commands = self.build_control_commands(assignment=assignment)
        return extract_actions(commands)

    def plan_tracking_actions(
        self,
        assignment: TaskAssignmentResult | None = None,
    ) -> dict[str, int]:
        """Return discrete actions that pursue the assigned fused key-track."""

        commands = self.build_control_commands(assignment=assignment)
        return extract_actions(commands)

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

    def _build_groundtruth_track_outputs(self) -> dict[str, MultiTargetTrackOutput]:
        if self.world is None:
            raise RuntimeError("reset() must be called before building groundtruth track outputs.")

        friendly_states = self.world.get_friendly_states()
        key_enemy_states = self.world.get_key_enemy_states()
        prior_covariance = np.full(STATE_DIM, 1.0e-6, dtype=np.float32)
        outputs: dict[str, MultiTargetTrackOutput] = {}
        for friendly_idx, agent_id in enumerate(self.agent_ids):
            if key_enemy_states.size == 0:
                outputs[agent_id] = empty_multi_target_track_output(
                    agent_id=agent_id,
                    prior_relative_states=np.zeros((0, STATE_DIM), dtype=float),
                    prior_covariance_diagonal=prior_covariance,
                    mode="groundtruth",
                )
                continue
            relative_truth = key_enemy_states - friendly_states[friendly_idx].reshape(1, STATE_DIM)
            tracks = tuple(
                KeyTrackPosterior(
                    global_target_index=target_idx,
                    prior_relative_state=relative_truth[target_idx].astype(np.float32),
                    prior_covariance_diagonal=prior_covariance.copy(),
                    posterior_relative_state=relative_truth[target_idx].astype(np.float32),
                    posterior_covariance_diagonal=prior_covariance.copy(),
                    confidence=1.0,
                    existence_probability=1.0,
                    valid=True,
                    local_candidate_weights=None,
                    local_candidate_index=-1,
                    innovation_distance=0.0,
                )
                for target_idx in range(relative_truth.shape[0])
            )
            outputs[agent_id] = MultiTargetTrackOutput(
                agent_id=agent_id,
                tracks=tracks,
                mode="groundtruth",
            )
        return outputs

    def _build_observations(self) -> dict[str, np.ndarray]:
        if self.world is None:
            raise RuntimeError("reset() must be called before building observations.")
        association_outputs = self.get_association_outputs()
        friendly_states = self.world.get_friendly_states()
        bounds = np.asarray(self.world.bounds, dtype=float)
        self_state_scale = np.concatenate(
            [bounds, np.full(3, max(self.friendly_vmax, 1.0e-6), dtype=float)]
        )

        observations: dict[str, np.ndarray] = {}
        for idx, agent_id in enumerate(self.agent_ids):
            self_state = friendly_states[idx]
            track_features = flatten_track_output(association_outputs[agent_id])
            observation = np.concatenate([self_state / self_state_scale, track_features]).astype(np.float32)
            observations[agent_id] = observation
        self._last_observations = {agent_id: obs.copy() for agent_id, obs in observations.items()}
        self._last_observation_time_step = self.world.time_step
        return observations

    def _reset_task_assignment_cache(self) -> None:
        self._last_task_assignment = None
        self._last_task_assignment_time_step = None
