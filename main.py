from __future__ import annotations

from pathlib import Path

from src.simulation.trajectory_generator import generate_scene_trajectories
from src.utils.config_loader import (
    get_enemy_vmax,
    get_friendly_vmax,
    get_num_friendly_uavs,
    get_num_key_targets,
    get_num_nonkey_targets,
    get_num_steps,
)
from src.visualization.plot_scene import plot_scene_snapshot, plot_trajectories_until_timestep


def main() -> None:
    output_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    enemy_vmax = get_enemy_vmax()
    friendly_vmax = get_friendly_vmax()
    num_steps = get_num_steps()
    key_num = get_num_key_targets()
    nonkey_num = get_num_nonkey_targets()
    mydrone_num = get_num_friendly_uavs()

    scene, trajectories = generate_scene_trajectories(
        num_steps=num_steps,
        key_num=key_num,
        nonkey_num=nonkey_num,
        mydrone_num=mydrone_num,
    )

    print("============Configured vmax============ ")
    print(f" enemy: {enemy_vmax}, friendly: {friendly_vmax} ")
    print()
    print("============Configured num_steps============ ")
    print(f" {num_steps} ")
    print()
    print("============Configured entity counts============ ")
    print(f" key: {key_num}, non-key: {nonkey_num}, friendly: {mydrone_num} ")
    print()
    print("============Initialized scene============ ")
    print(f" enemies: {len(scene.enemy_nodes)}, friendly: {len(scene.friendly_uavs)} ")
    print()

    plot_scene_snapshot(
        scene,
        trajectories,
        timestep=0,
        save_path=output_dir / "main_initial_scene.pdf",
        title="Initial 3D Scene",
        show=False,
    )
    plot_trajectories_until_timestep(
        scene,
        trajectories,
        timestep=trajectories.num_steps,
        target="enemy",
        save_path=output_dir / "main_enemy_trajectories.pdf",
        title="Enemy UAV Trajectories",
        show=False,
    )
    plot_trajectories_until_timestep(
        scene,
        trajectories,
        timestep=trajectories.num_steps,
        target="friendly",
        save_path=output_dir / "main_friendly_trajectories.pdf",
        title="Friendly UAV Trajectories",
        show=False,
    )

    print("Generated figures:")
    print(output_dir / "main_initial_scene.pdf")
    print(output_dir / "main_enemy_trajectories.pdf")
    print(output_dir / "main_friendly_trajectories.pdf")


if __name__ == "__main__":
    main()
