from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src.entities.friendly_uav import FriendlyUAV
from src.env.swarm_env import SwarmEnv
from src.simulation.trajectory_generator import TrajectoryBundle
from src.tracking.metrics import export_key_target_metrics_csv, summarize_key_target_tracking_episode
from src.utils.config_loader import (
    get_default_seed,
    get_num_friendly_uavs,
    get_num_key_targets,
    get_num_nonkey_targets,
    get_num_steps,
    get_selected_scenario_name,
    get_tracking_association_method,
    get_simulation_dt,
)
from src.visualization.plot_scene import (
    plot_key_enemy_nearest_friendly_distance_curves,
    plot_scene_snapshot,
    plot_tracking_rollout_trajectories,
)


def _association_method_suffix(association_method: str) -> str:
    suffix_by_method = {
        "none": "none",
        "prior_only": "prioronly",
        "bp_jpda": "bpjpda",
        "hard_anchor_bp_jpda": "hardanchorbpjpda",
        "covariance_anchor_bp_jpda": "covarianceanchorbpjpda",
        "labeled_anchor_bp_jpda": "labeledanchorbpjpda",
        "gm_phd": "gmphd",
        "lmb": "lmb",
        "delta_glmb": "dglmb",
    }
    return suffix_by_method.get(str(association_method).lower(), str(association_method).lower())


def _scenario_output_name() -> str:
    selected_scenario = get_selected_scenario_name()
    if selected_scenario is None:
        return "default_scenario"
    scenario_name = str(selected_scenario).strip()
    return scenario_name or "default_scenario"


def export_position_timeseries_csv(scene, trajectories, save_path: str | Path) -> Path:
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    enemy_labels = [f"E{idx + 1}_{node.role}" for idx, node in enumerate(scene.enemy_nodes)]
    friendly_labels = [f"F{idx + 1}" for idx in range(trajectories.friendly_positions.shape[1])]

    header = ["timestep", "time_s"]
    for label in enemy_labels:
        header.extend([f"{label}_x", f"{label}_y", f"{label}_z"])
    for label in friendly_labels:
        header.extend([f"{label}_x", f"{label}_y", f"{label}_z", f"{label}_action", f"{label}_action_name"])

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        for step_idx in range(trajectories.enemy_positions.shape[0]):
            row: list[float | int | str] = [step_idx, step_idx * float(trajectories.dt)]
            for coords in trajectories.enemy_positions[step_idx]:
                row.extend(f"{value:.6f}" for value in coords)
            for friendly_idx, coords in enumerate(trajectories.friendly_positions[step_idx]):
                row.extend(f"{value:.6f}" for value in coords)
                if step_idx < trajectories.friendly_actions.shape[0]:
                    action = int(trajectories.friendly_actions[step_idx, friendly_idx])
                    action_name = FriendlyUAV.action_name(action)
                else:
                    action = ""
                    action_name = ""
                row.extend([action, action_name])
            writer.writerow(row)

    return output_path


def _capture_key_target_tracking_state(
    env: SwarmEnv,
    *,
    num_key_targets: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if env.world is None:
        raise RuntimeError("reset() must be called before capturing key-target tracking state.")

    truth_states = np.asarray(env.world.get_key_enemy_states(), dtype=float)
    if truth_states.shape[0] != num_key_targets:
        raise RuntimeError(
            "Key-target truth count changed during rollout: "
            f"expected {num_key_targets}, got {truth_states.shape[0]}."
        )

    state_dim = int(truth_states.shape[1]) if truth_states.ndim == 2 and truth_states.size > 0 else 6
    fused_state_means = np.full((num_key_targets, state_dim), np.nan, dtype=float)
    fused_valid_mask = np.zeros(num_key_targets, dtype=bool)
    fused_track_set = env.get_fused_track_set()
    inference_time_s = env.get_tracking_runtime_s()
    for track in fused_track_set.tracks:
        target_idx = int(track.global_target_index)
        if target_idx < 0 or target_idx >= num_key_targets or not bool(track.valid):
            continue
        track_state = np.asarray(track.fused_state_mean, dtype=float).reshape(-1)
        fused_state_means[target_idx, : min(state_dim, track_state.shape[0])] = track_state[:state_dim]
        fused_valid_mask[target_idx] = True
    return truth_states.copy(), fused_state_means, fused_valid_mask, float(inference_time_s)


def rollout_tracking_episode(
    *,
    key_num: int | None = None,
    nonkey_num: int | None = None,
    mydrone_num: int | None = None,
    seed: int | None = None,
    association_method: str | None = None,
) -> tuple[object, TrajectoryBundle, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    resolved_seed = get_default_seed() if seed is None else int(seed)
    resolved_key_num = get_num_key_targets() if key_num is None else int(key_num)
    resolved_nonkey_num = get_num_nonkey_targets() if nonkey_num is None else int(nonkey_num)
    resolved_mydrone_num = get_num_friendly_uavs() if mydrone_num is None else int(mydrone_num)
    env = SwarmEnv(
        key_num=resolved_key_num,
        nonkey_num=resolved_nonkey_num,
        mydrone_num=resolved_mydrone_num,
        dt=get_simulation_dt(),
        max_steps=get_num_steps(),
        seed=resolved_seed,
        association_method=association_method,
    )
    env.reset(seed=resolved_seed)
    if env.world is None or env.scene is None:
        raise RuntimeError("Tracking rollout failed to initialize the scene and world.")

    num_steps = int(env.max_steps)
    num_enemies = len(env.world.enemy_nodes)
    num_friendlies = len(env.world.friendly_uavs)
    num_key_enemies = len(env.world.get_key_enemy_nodes())
    state_dim = int(env.world.get_key_enemy_states().shape[1]) if num_key_enemies > 0 else 6

    enemy_positions = np.zeros((num_steps + 1, num_enemies, 3), dtype=float)
    friendly_positions = np.zeros((num_steps + 1, num_friendlies, 3), dtype=float)
    friendly_actions = np.zeros((num_steps, num_friendlies), dtype=int)
    key_enemy_interference_watts = np.zeros((num_steps + 1, num_key_enemies), dtype=float)
    key_enemy_interference_dbm = np.zeros((num_steps + 1, num_key_enemies), dtype=float)
    friendly_interference_watts = np.zeros((num_steps + 1, num_friendlies), dtype=float)
    friendly_interference_dbm = np.zeros((num_steps + 1, num_friendlies), dtype=float)
    key_truth_states = np.zeros((num_steps + 1, num_key_enemies, state_dim), dtype=float)
    fused_key_state_means = np.full((num_steps + 1, num_key_enemies, state_dim), np.nan, dtype=float)
    fused_key_valid_mask = np.zeros((num_steps + 1, num_key_enemies), dtype=bool)
    tracking_inference_time_s = np.zeros(num_steps + 1, dtype=float)

    enemy_positions[0] = env.world.get_enemy_positions()
    friendly_positions[0] = env.world.get_friendly_positions()
    key_enemy_interference_watts[0] = env.world.interference.key_enemy_received_watts
    key_enemy_interference_dbm[0] = env.world.interference.key_enemy_received_dbm
    friendly_interference_watts[0] = env.world.interference.friendly_received_watts
    friendly_interference_dbm[0] = env.world.interference.friendly_received_dbm
    (
        key_truth_states[0],
        fused_key_state_means[0],
        fused_key_valid_mask[0],
        tracking_inference_time_s[0],
    ) = _capture_key_target_tracking_state(
        env,
        num_key_targets=num_key_enemies,
    )

    for step_idx in range(1, num_steps + 1):
        assignment = env.build_task_assignment()
        actions = env.plan_tracking_actions(assignment=assignment)
        env.step(actions)
        assert env.world is not None
        enemy_positions[step_idx] = env.world.get_enemy_positions()
        friendly_positions[step_idx] = env.world.get_friendly_positions()
        friendly_actions[step_idx - 1] = np.asarray(
            [int(actions[agent_id]) for agent_id in env.agent_ids],
            dtype=int,
        )
        key_enemy_interference_watts[step_idx] = env.world.interference.key_enemy_received_watts
        key_enemy_interference_dbm[step_idx] = env.world.interference.key_enemy_received_dbm
        friendly_interference_watts[step_idx] = env.world.interference.friendly_received_watts
        friendly_interference_dbm[step_idx] = env.world.interference.friendly_received_dbm
        (
            key_truth_states[step_idx],
            fused_key_state_means[step_idx],
            fused_key_valid_mask[step_idx],
            tracking_inference_time_s[step_idx],
        ) = _capture_key_target_tracking_state(
            env,
            num_key_targets=num_key_enemies,
        )

    bundle = TrajectoryBundle(
        enemy_positions=enemy_positions,
        friendly_positions=friendly_positions,
        friendly_actions=friendly_actions,
        key_enemy_interference_watts=key_enemy_interference_watts,
        key_enemy_interference_dbm=key_enemy_interference_dbm,
        friendly_interference_watts=friendly_interference_watts,
        friendly_interference_dbm=friendly_interference_dbm,
        dt=float(env.dt),
        bounds=env.world.bounds.copy(),
    )
    return env.scene, bundle, key_truth_states, fused_key_state_means, fused_key_valid_mask, tracking_inference_time_s


def export_tracking_rollout(
    *,
    output_dir: str | Path = "outputs/tracking_rollouts",
    show: bool = False,
    seed: int | None = None,
    key_num: int | None = None,
    nonkey_num: int | None = None,
    mydrone_num: int | None = None,
) -> dict[str, Path | list[Path]]:
    (
        scene,
        trajectories,
        key_truth_states,
        fused_key_state_means,
        fused_key_valid_mask,
        tracking_inference_time_s,
    ) = rollout_tracking_episode(
        key_num=key_num,
        nonkey_num=nonkey_num,
        mydrone_num=mydrone_num,
        seed=seed,
    )
    association_method = str(get_tracking_association_method()).lower()
    association_suffix = _association_method_suffix(get_tracking_association_method())
    scenario_name = _scenario_output_name()

    output_root = Path(output_dir) / scenario_name / association_method
    output_root.mkdir(parents=True, exist_ok=True)

    trajectory_npz = output_root / "tracking_rollout_episode.npz"
    snapshot_pdf = output_root / "tracking_rollout_initial_scene.pdf"
    trajectory_pdf = output_root / "tracking_rollout_all_trajectories.pdf"
    positions_csv = output_root / f"tracking_rollout_positions_{association_suffix}.csv"
    metrics_csv = output_root / f"tracking_rollout_key_target_metrics_{association_suffix}.csv"
    key_distance_pdf = output_root / "tracking_rollout_key_enemy_nearest_friendly_distances_combined.pdf"

    np.savez(
        trajectory_npz,
        enemy_positions=trajectories.enemy_positions,
        friendly_positions=trajectories.friendly_positions,
        friendly_actions=trajectories.friendly_actions,
        key_enemy_interference_watts=trajectories.key_enemy_interference_watts,
        key_enemy_interference_dbm=trajectories.key_enemy_interference_dbm,
        friendly_interference_watts=trajectories.friendly_interference_watts,
        friendly_interference_dbm=trajectories.friendly_interference_dbm,
        key_truth_states=key_truth_states,
        fused_key_state_means=fused_key_state_means,
        fused_key_valid_mask=fused_key_valid_mask,
        tracking_inference_time_s=tracking_inference_time_s,
        dt=np.asarray(trajectories.dt),
        bounds=trajectories.bounds,
    )

    plot_scene_snapshot(
        scene=scene,
        trajectories=trajectories,
        timestep=0,
        save_path=snapshot_pdf,
        title="",
        show=show,
        include_trajectory_legend=False,
        legend_fontsize=12.0,
    )
    plot_tracking_rollout_trajectories(
        scene=scene,
        trajectories=trajectories,
        save_path=trajectory_pdf,
        title="",
        show=show,
        legend_fontsize=12.0,
    )
    export_position_timeseries_csv(
        scene=scene,
        trajectories=trajectories,
        save_path=positions_csv,
    )
    export_key_target_metrics_csv(
        truth_states=key_truth_states,
        estimated_states=fused_key_state_means,
        valid_mask=fused_key_valid_mask,
        dt=float(trajectories.dt),
        bounds=trajectories.bounds,
        save_path=metrics_csv,
        inference_time_s=tracking_inference_time_s,
    )
    cutoff_distance = float(np.asarray(trajectories.bounds, dtype=float).reshape(-1)[0]) / 4.0
    tracking_summary = summarize_key_target_tracking_episode(
        truth_states=key_truth_states,
        estimated_states=fused_key_state_means,
        valid_mask=fused_key_valid_mask,
        cutoff_distance=cutoff_distance,
        order=2,
    )
    plot_key_enemy_nearest_friendly_distance_curves(
        scene=scene,
        trajectories=trajectories,
        save_path=key_distance_pdf,
        show=show,
        legend_fontsize=12.0,
    )

    print(f"Saved trajectory data: {trajectory_npz}")
    print(f"Saved initial snapshot: {snapshot_pdf}")
    print(f"Saved combined trajectory figure: {trajectory_pdf}")
    print(f"Saved position/action CSV: {positions_csv}")
    print(f"Saved key-target metrics CSV: {metrics_csv}")
    print(
        "Key-target tracking coverage by step: "
        + np.array2string(
            np.asarray(tracking_summary["tracked_counts"], dtype=int),
            separator=", ",
        )
        + " / "
        + str(key_truth_states.shape[1])
    )
    print(
        "Tracking summary: "
        f"mean_ospa={float(tracking_summary['mean_ospa']):.3f} m, "
        f"final_ospa={float(tracking_summary['final_ospa']):.3f} m, "
        f"mean_gospa={float(tracking_summary['mean_gospa']):.3f} m, "
        f"final_gospa={float(tracking_summary['final_gospa']):.3f} m, "
        f"valid_position_rmse={float(tracking_summary['position_rmse_valid']):.3f} m, "
        f"coverage_penalized_rmse={float(tracking_summary['coverage_penalized_rmse']):.3f} m, "
        f"mean_coverage={100.0 * float(tracking_summary['mean_tracked_ratio']):.1f}%, "
        f"final_coverage={int(tracking_summary['final_tracked_count'])}/{key_truth_states.shape[1]}, "
        f"id_switches={int(tracking_summary['total_id_switches'])}, "
        f"fragmentations={int(tracking_summary['total_fragmentations'])}"
    )
    print(
        "Tracking inference time: "
        f"mean={float(np.mean(tracking_inference_time_s)):.6f} s/step, "
        f"total={float(np.sum(tracking_inference_time_s)):.6f} s"
    )
    print(f"Saved key enemy nearest-friendly distance figure: {key_distance_pdf}")

    return {
        "trajectory_npz": trajectory_npz,
        "snapshot_pdf": snapshot_pdf,
        "trajectory_pdf": trajectory_pdf,
        "positions_csv": positions_csv,
        "metrics_csv": metrics_csv,
        "key_distance_pdf": key_distance_pdf,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export and visualize a tracking rollout.")
    parser.add_argument("--output-dir", type=str, default="outputs/tracking_rollouts", help="Directory for exported tracking files.")
    parser.add_argument("--seed", type=int, default=None, help="Optional environment seed. Defaults to config/default.yaml.")
    parser.add_argument("--key-num", type=int, default=None, help="Number of key targets. Defaults to env.yaml.")
    parser.add_argument("--nonkey-num", type=int, default=None, help="Number of non-key targets. Defaults to env.yaml.")
    parser.add_argument("--mydrone-num", type=int, default=None, help="Number of friendly UAVs. Defaults to env.yaml.")
    parser.add_argument("--show", dest="show", action="store_true", default=False, help="Display figures interactively in addition to saving them.")
    parser.add_argument("--no-show", dest="show", action="store_false", help="Save figures without opening interactive windows.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_tracking_rollout(
        output_dir=args.output_dir,
        show=args.show,
        seed=args.seed,
        key_num=args.key_num,
        nonkey_num=args.nonkey_num,
        mydrone_num=args.mydrone_num,
    )
