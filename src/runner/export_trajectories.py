from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src.entities.friendly_uav import FriendlyUAV
from src.rl.mappo import MAPPO
from src.runner.evaluate import rollout_policy_episode
from src.runner.train import initialize_simple_baseline_env
from src.utils.checkpoint import find_best_checkpoint, restore_training_state
from src.utils.config_loader import get_mappo_config
from src.visualization.plot_scene import (
    plot_key_enemy_interference_curves,
    plot_scene_snapshot,
    plot_trained_policy_trajectories,
)


def _key_interference_figure_paths(scene, base_path: Path) -> list[Path]:
    key_labels = [f"E{idx + 1}" for idx, node in enumerate(scene.enemy_nodes) if node.role == "key"]
    if not key_labels:
        return []
    return [base_path.with_name(f"{base_path.stem}_{label}{base_path.suffix}") for label in key_labels]


def resolve_export_checkpoint(checkpoint_path: str | Path | None, config_path: str | Path | None = None) -> Path:
    if checkpoint_path is not None:
        return Path(checkpoint_path)

    config = get_mappo_config(config_path)
    logging_config = config.get("logging", {})
    checkpoint_dir = Path(logging_config.get("checkpoint_dir", "outputs/checkpoints"))
    best_checkpoint = find_best_checkpoint(checkpoint_dir)
    if best_checkpoint is None:
        raise FileNotFoundError(f"No checkpoint available in {checkpoint_dir}")
    return best_checkpoint


def export_key_enemy_interference_csv(scene, trajectories, save_path: str | Path) -> Path:
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    key_labels = [f"E{idx + 1}" for idx, node in enumerate(scene.enemy_nodes) if node.role == "key"]
    powers_watts = np.asarray(trajectories.key_enemy_interference_watts, dtype=float)
    powers_dbm = np.asarray(trajectories.key_enemy_interference_dbm, dtype=float)
    if len(key_labels) != powers_watts.shape[1]:
        key_labels = [f"key_enemy_{idx + 1}" for idx in range(powers_watts.shape[1])]

    header = ["timestep", "time_s"]
    header.extend(f"{label}_watts" for label in key_labels)
    header.extend(f"{label}_dbm" for label in key_labels)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        for step_idx in range(powers_watts.shape[0]):
            row = [step_idx, step_idx * float(trajectories.dt)]
            row.extend(f"{value:.12e}" for value in powers_watts[step_idx])
            row.extend(f"{value:.6f}" for value in powers_dbm[step_idx])
            writer.writerow(row)

    return output_path


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


def export_trained_policy_trajectory(
    checkpoint_path: str | Path | None = None,
    config_path: str | Path | None = None,
    device: str | None = None,
    output_dir: str | Path = "outputs/policy_rollouts",
    show: bool = True,
    seed: int | None = None,
) -> dict[str, Path | list[Path]]:
    resolved_checkpoint = resolve_export_checkpoint(checkpoint_path, config_path)
    env, _ = initialize_simple_baseline_env(config_path)
    config = get_mappo_config(config_path)
    trainer = MAPPO(env=env, config=config, device=device)
    payload = restore_training_state(
        path=resolved_checkpoint,
        actor=trainer.actor,
        critic=trainer.critic,
        actor_optimizer=trainer.actor_optimizer,
        critic_optimizer=trainer.critic_optimizer,
        map_location=trainer.device,
    )
    trainer.load_value_normalizer_state(payload.get("value_normalizer_state"))

    scene, trajectories, stats = rollout_policy_episode(trainer, deterministic=True, seed=seed)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    trajectory_npz = output_root / "trained_policy_episode.npz"
    snapshot_pdf = output_root / "trained_policy_initial_scene.pdf"
    trajectory_pdf = output_root / "trained_policy_all_trajectories.pdf"
    positions_csv = output_root / "trained_policy_positions.csv"
    key_interference_csv = output_root / "trained_policy_key_enemy_interference.csv"
    key_interference_pdf = output_root / "trained_policy_key_enemy_interference.pdf"
    key_interference_pdfs = _key_interference_figure_paths(scene, key_interference_pdf)

    np.savez(
        trajectory_npz,
        enemy_positions=trajectories.enemy_positions,
        friendly_positions=trajectories.friendly_positions,
        friendly_actions=trajectories.friendly_actions,
        key_enemy_interference_watts=trajectories.key_enemy_interference_watts,
        key_enemy_interference_dbm=trajectories.key_enemy_interference_dbm,
        friendly_interference_watts=trajectories.friendly_interference_watts,
        friendly_interference_dbm=trajectories.friendly_interference_dbm,
        dt=np.asarray(trajectories.dt),
        bounds=trajectories.bounds,
        episode_return=np.asarray(stats["episode_return"]),
        episode_length=np.asarray(stats["episode_length"]),
    )

    plot_scene_snapshot(
        scene=scene,
        trajectories=trajectories,
        timestep=0,
        save_path=snapshot_pdf,
        title="Initial Scene for Trained MAPPO Rollout",
        show=show,
    )
    plot_trained_policy_trajectories(
        scene=scene,
        trajectories=trajectories,
        save_path=trajectory_pdf,
        title="Trained MAPPO Enemy and Friendly Trajectories",
        show=show,
    )
    export_key_enemy_interference_csv(
        scene=scene,
        trajectories=trajectories,
        save_path=key_interference_csv,
    )
    export_position_timeseries_csv(
        scene=scene,
        trajectories=trajectories,
        save_path=positions_csv,
    )
    plot_key_enemy_interference_curves(
        scene=scene,
        trajectories=trajectories,
        save_path=key_interference_pdf,
        title="Trained MAPPO Key Enemy Received Interference Power",
        show=show,
    )

    print(f"Loaded checkpoint: {resolved_checkpoint}")
    if seed is not None:
        print(f"Evaluation seed: {int(seed)}")
    print(f"Episode return: {stats['episode_return']:.6f}")
    print(f"Episode length: {int(stats['episode_length'])}")
    print(f"Saved trajectory data: {trajectory_npz}")
    print(f"Saved initial snapshot: {snapshot_pdf}")
    print(f"Saved combined trajectory figure: {trajectory_pdf}")
    print(f"Saved position/action CSV: {positions_csv}")
    print(f"Saved key enemy interference CSV: {key_interference_csv}")
    for figure_path in key_interference_pdfs:
        print(f"Saved key enemy interference figure: {figure_path}")

    return {
        "checkpoint": resolved_checkpoint,
        "trajectory_npz": trajectory_npz,
        "snapshot_pdf": snapshot_pdf,
        "trajectory_pdf": trajectory_pdf,
        "positions_csv": positions_csv,
        "key_interference_csv": key_interference_csv,
        "key_interference_pdfs": key_interference_pdfs,
    }



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export and visualize trajectories from a trained MAPPO checkpoint.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint path. If omitted, export uses the best checkpoint in checkpoint_dir by default.")
    parser.add_argument("--config", type=str, default=None, help="Optional config YAML path.")
    parser.add_argument("--device", type=str, default="cuda", help="Optional torch device. Default is cuda.")
    parser.add_argument("--output-dir", type=str, default="outputs/policy_rollouts", help="Directory for exported trajectory files.")
    parser.add_argument("--seed", type=int, default=None, help="Optional environment seed used only for this exported rollout.")
    parser.add_argument("--show", dest="show", action="store_true", default=True, help="Display figures interactively in addition to saving them.")
    parser.add_argument("--no-show", dest="show", action="store_false", help="Save figures without opening interactive windows.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_trained_policy_trajectory(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
        output_dir=args.output_dir,
        show=args.show,
        seed=args.seed,
    )
