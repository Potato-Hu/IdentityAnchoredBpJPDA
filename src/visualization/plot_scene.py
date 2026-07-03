from __future__ import annotations

from pathlib import Path
from typing import Literal

from src.visualization.matplotlib_style import configure_matplotlib_for_ieee_pdf

configure_matplotlib_for_ieee_pdf()

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import LinearLocator

from src.simulation.ground_truth import ScenarioGroundTruth
from src.simulation.trajectory_generator import TrajectoryBundle, generate_demo_trajectories
from src.utils.config_loader import get_num_steps

_VIEW = (26, -58)
_COLORS = {
    "key_entity": "#B42318",
    "key_trajectory": "#6D0F1B",
    "non_key_entity": "#6B7280",
    "non_key_trajectory": "#6B7280",
    "friendly_entity": "#2F6BFF",
    "friendly_trajectory": "#1C9A96",
}
_MARKERS = {
    "key": "D",
    "non_key": "^",
    "friendly": "o",
}
_MARKER_SIZES = {
    "snapshot": 42,
    "trajectory_end": 22,
    "legend": 7,
}
_LINE_WIDTHS = {
    "key": 1.35,
    "non_key": 1.0,
    "friendly": 1.25,
}
_TEXT_SIZE = 7.5


def _style_3d_axes(ax, bounds: np.ndarray, view: tuple[float, float] = _VIEW) -> None:
    bounds_array = np.asarray(bounds, dtype=float).copy()
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.93, 0.96, 1.0, 0.78))
        axis.pane.set_edgecolor((0.83, 0.89, 0.97, 0.95))

    ax.set_xlim(0.0, float(bounds_array[0]))
    ax.set_ylim(0.0, float(bounds_array[1]))
    ax.set_zlim(0.0, float(bounds_array[2]))
    ax.set_box_aspect(tuple(bounds_array.tolist()))
    ax.view_init(elev=view[0], azim=view[1])
    ax.tick_params(pad=2, labelsize=9)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    ax.zaxis.set_major_locator(LinearLocator(numticks=5))
    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass


def _ensure_output_dir(save_path: str | Path | None) -> Path | None:
    if save_path is None:
        return None
    output_path = Path(save_path)
    if output_path.suffix.lower() != ".pdf":
        output_path = output_path.with_suffix(".pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _per_key_output_path(base_path: Path, key_label: str) -> Path:
    return base_path.with_name(f"{base_path.stem}_{key_label}{base_path.suffix}")


def _build_legend() -> list[Line2D]:
    return [
        Line2D([0], [0], marker=_MARKERS["key"], color="w", markerfacecolor=_COLORS["key_entity"], markeredgecolor=_COLORS["key_entity"], markersize=_MARKER_SIZES["legend"], label="Key enemy UAV"),
        Line2D([0], [0], marker=_MARKERS["non_key"], color="w", markerfacecolor=_COLORS["non_key_entity"], markeredgecolor=_COLORS["non_key_entity"], markersize=_MARKER_SIZES["legend"], label="Non-key enemy UAV"),
        Line2D([0], [0], marker=_MARKERS["friendly"], color="w", markerfacecolor=_COLORS["friendly_entity"], markeredgecolor=_COLORS["friendly_entity"], markersize=_MARKER_SIZES["legend"], label="Friendly UAV"),
        Line2D([0], [0], color=_COLORS["key_trajectory"], linewidth=_LINE_WIDTHS["key"], linestyle="-", label="Key enemy trajectory"),
        Line2D([0], [0], color=_COLORS["non_key_trajectory"], linewidth=_LINE_WIDTHS["non_key"], linestyle="--", label="Non-key enemy trajectory"),
        Line2D([0], [0], color=_COLORS["friendly_trajectory"], linewidth=_LINE_WIDTHS["friendly"], linestyle="-", label="Friendly trajectory"),
    ]


def _label_offset(bounds: np.ndarray) -> np.ndarray:
    bounds_array = np.asarray(bounds, dtype=float)
    return np.maximum(bounds_array * np.array([0.012, 0.012, 0.02]), np.array([1.5, 1.5, 1.5]))


def _annotate_entity(ax, coord: np.ndarray, label: str, color: str, bounds: np.ndarray) -> None:
    offset = _label_offset(bounds)
    ax.text(
        coord[0] + offset[0],
        coord[1] + offset[1],
        coord[2] + offset[2],
        label,
        color=color,
        fontsize=_TEXT_SIZE,
        zorder=12,
    )


def _enemy_entity_color(role: str) -> str:
    return _COLORS["key_entity"] if role == "key" else _COLORS["non_key_entity"]


def _enemy_trajectory_color(role: str) -> str:
    return _COLORS["key_trajectory"] if role == "key" else _COLORS["non_key_trajectory"]


def plot_scene_snapshot(
    scene: ScenarioGroundTruth,
    trajectories: TrajectoryBundle,
    timestep: int,
    save_path: str | Path | None = None,
    title: str | None = None,
    show: bool = False,
):
    max_step = trajectories.num_steps
    if timestep < 0 or timestep > max_step:
        raise ValueError(f"timestep must be in [0, {max_step}], got {timestep}.")

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    key_indices = [idx for idx, node in enumerate(scene.enemy_nodes) if node.role == "key"]
    non_key_indices = [idx for idx, node in enumerate(scene.enemy_nodes) if node.role != "key"]

    if key_indices:
        coords = trajectories.enemy_positions[timestep, key_indices]
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=_COLORS["key_entity"], marker=_MARKERS["key"], s=_MARKER_SIZES["snapshot"], depthshade=False)
    if non_key_indices:
        coords = trajectories.enemy_positions[timestep, non_key_indices]
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=_COLORS["non_key_entity"], marker=_MARKERS["non_key"], s=_MARKER_SIZES["snapshot"], depthshade=False)

    friendly_coords = trajectories.friendly_positions[timestep]
    ax.scatter(friendly_coords[:, 0], friendly_coords[:, 1], friendly_coords[:, 2], c=_COLORS["friendly_entity"], marker=_MARKERS["friendly"], s=_MARKER_SIZES["snapshot"], depthshade=False)

    for idx, node in enumerate(scene.enemy_nodes):
        coord = trajectories.enemy_positions[timestep, idx]
        _annotate_entity(ax, coord, f"E{idx + 1}", _enemy_entity_color(node.role), scene.bounds)

    for idx in range(len(scene.friendly_uavs)):
        coord = trajectories.friendly_positions[timestep, idx]
        _annotate_entity(ax, coord, f"F{idx + 1}", _COLORS["friendly_entity"], scene.bounds)

    _style_3d_axes(ax, scene.bounds)
    ax.legend(handles=_build_legend(), loc="upper left", frameon=False)
    ax.set_title(title or f"Scene Snapshot at t = {timestep * trajectories.dt:.1f}s")

    output_path = _ensure_output_dir(save_path)
    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def plot_trajectories_until_timestep(
    scene: ScenarioGroundTruth,
    trajectories: TrajectoryBundle,
    timestep: int,
    target: Literal["all", "enemy", "friendly"] = "all",
    save_path: str | Path | None = None,
    title: str | None = None,
    show: bool = False,
):
    max_step = trajectories.num_steps
    if timestep < 0 or timestep > max_step:
        raise ValueError(f"timestep must be in [0, {max_step}], got {timestep}.")

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    end = timestep + 1

    if target in {"all", "enemy"}:
        for idx, node in enumerate(scene.enemy_nodes):
            coords = trajectories.enemy_positions[:end, idx, :]
            entity_color = _enemy_entity_color(node.role)
            trajectory_color = _enemy_trajectory_color(node.role)
            marker = _MARKERS["key"] if node.role == "key" else _MARKERS["non_key"]
            linestyle = "-" if node.role == "key" else "--"
            alpha = 0.9 if node.role == "key" else 0.72
            linewidth = _LINE_WIDTHS["key"] if node.role == "key" else _LINE_WIDTHS["non_key"]
            ax.plot(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                color=trajectory_color,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=alpha,
            )
            ax.scatter(coords[-1, 0], coords[-1, 1], coords[-1, 2], c=entity_color, marker=marker, s=_MARKER_SIZES["trajectory_end"], depthshade=False)
            _annotate_entity(ax, coords[-1], f"E{idx + 1}", entity_color, scene.bounds)

    if target in {"all", "friendly"}:
        for idx in range(len(scene.friendly_uavs)):
            coords = trajectories.friendly_positions[:end, idx, :]
            ax.plot(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                color=_COLORS["friendly_trajectory"],
                linewidth=_LINE_WIDTHS["friendly"],
                linestyle="-",
                alpha=0.88,
            )
            ax.scatter(coords[-1, 0], coords[-1, 1], coords[-1, 2], c=_COLORS["friendly_entity"], marker=_MARKERS["friendly"], s=_MARKER_SIZES["trajectory_end"], depthshade=False)
            _annotate_entity(ax, coords[-1], f"F{idx + 1}", _COLORS["friendly_entity"], scene.bounds)

    _style_3d_axes(ax, scene.bounds)
    ax.legend(handles=_build_legend(), loc="upper left", frameon=False)

    default_titles = {
        "all": f"All UAV Trajectories up to t = {timestep * trajectories.dt:.1f}s",
        "enemy": f"Enemy Trajectories up to t = {timestep * trajectories.dt:.1f}s",
        "friendly": f"Friendly Trajectories up to t = {timestep * trajectories.dt:.1f}s",
    }
    ax.set_title(title or default_titles[target])

    output_path = _ensure_output_dir(save_path)
    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def plot_tracking_rollout_trajectories(
    scene: ScenarioGroundTruth,
    trajectories: TrajectoryBundle,
    save_path: str | Path | None = None,
    title: str | None = None,
    show: bool = False,
):
    return plot_trajectories_until_timestep(
        scene=scene,
        trajectories=trajectories,
        timestep=trajectories.num_steps,
        target="all",
        save_path=save_path,
        title=title or "Tracking Rollout Enemy and Friendly Trajectories",
        show=show,
    )


def plot_trained_policy_trajectories(
    scene: ScenarioGroundTruth,
    trajectories: TrajectoryBundle,
    save_path: str | Path | None = None,
    title: str | None = None,
    show: bool = False,
):
    """Backward-compatible alias for the rollout trajectory plot."""

    return plot_tracking_rollout_trajectories(
        scene=scene,
        trajectories=trajectories,
        save_path=save_path,
        title=title,
        show=show,
    )


def plot_key_enemy_interference_curves(
    scene: ScenarioGroundTruth,
    trajectories: TrajectoryBundle,
    save_path: str | Path | None = None,
    title: str | None = None,
    show: bool = False,
    scale: Literal["dbm", "watts", "log_watts"] = "dbm",
):
    """Plot one received-power curve per key enemy.

    Separate figures avoid a close-range FSPL spike on one key target visually flattening
    the other key targets on a shared linear axis.
    """

    key_labels = [f"E{idx + 1}" for idx, node in enumerate(scene.enemy_nodes) if node.role == "key"]
    powers_watts = np.asarray(trajectories.key_enemy_interference_watts, dtype=float)
    powers_dbm = np.asarray(trajectories.key_enemy_interference_dbm, dtype=float)
    powers = powers_dbm if scale == "dbm" else powers_watts
    if powers.ndim != 2:
        raise ValueError(f"key enemy interference powers must be 2D, got shape {powers.shape}.")
    if len(key_labels) != powers.shape[1]:
        key_labels = [f"Key enemy {idx + 1}" for idx in range(powers.shape[1])]

    time_seconds = np.arange(powers.shape[0], dtype=float) * float(trajectories.dt)
    cmap = plt.get_cmap("tab10" if powers.shape[1] <= 10 else "tab20")
    output_path = _ensure_output_dir(save_path)
    figures = []

    for idx in range(powers.shape[1]):
        fig, ax = plt.subplots(figsize=(10, 5.6))
        with sns.axes_style("whitegrid"), sns.plotting_context("notebook"):
            sns.lineplot(
                x=time_seconds,
                y=powers[:, idx],
                color=cmap(idx % cmap.N),
                linewidth=2.2,
                label=f"{key_labels[idx]} received power",
                ax=ax,
            )
        ax.set_facecolor("#F8FAFC")
        if scale == "log_watts":
            ax.set_yscale("log")
        figure_title = f"{title} - {key_labels[idx]}" if title is not None else f"{key_labels[idx]} Received Interference Power"
        ax.set_title(figure_title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Received interference power (dBm)" if scale == "dbm" else "Received interference power (W)")
        ax.grid(True, which="major", alpha=0.28, color="#94A3B8", linewidth=0.8)
        ax.grid(False, which="minor")
        ax.legend(frameon=False)
        sns.despine(ax=ax, offset=4)
        if scale == "watts":
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        fig.tight_layout()

        if output_path is not None:
            plt.savefig(_per_key_output_path(output_path, key_labels[idx]), bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        figures.append((fig, ax))

    return figures


def plot_key_enemy_nearest_friendly_distance_curves(
    scene: ScenarioGroundTruth,
    trajectories: TrajectoryBundle,
    save_path: str | Path | None = None,
    title: str | None = None,
    show: bool = False,
):
    """Plot all key-enemy nearest-friendly distance curves in one figure."""

    key_indices = [idx for idx, node in enumerate(scene.enemy_nodes) if node.role == "key"]
    key_labels = [f"E{idx + 1}" for idx in key_indices]
    if not key_indices:
        return []

    enemy_positions = np.asarray(trajectories.enemy_positions[:, key_indices, :], dtype=float)
    friendly_positions = np.asarray(trajectories.friendly_positions, dtype=float)
    if enemy_positions.ndim != 3 or friendly_positions.ndim != 3:
        raise ValueError("enemy_positions and friendly_positions must be 3D arrays over time.")

    pairwise_distances = np.linalg.norm(
        enemy_positions[:, :, None, :] - friendly_positions[:, None, :, :],
        axis=-1,
    )
    nearest_distances = np.min(pairwise_distances, axis=2) if friendly_positions.shape[1] else np.zeros(
        (enemy_positions.shape[0], enemy_positions.shape[1]),
        dtype=float,
    )

    time_seconds = np.arange(nearest_distances.shape[0], dtype=float) * float(trajectories.dt)
    cmap = plt.get_cmap("tab10" if nearest_distances.shape[1] <= 10 else "tab20")
    output_path = _ensure_output_dir(save_path)

    fig, ax = plt.subplots(figsize=(10, 5.6))
    with sns.axes_style("whitegrid"), sns.plotting_context("notebook"):
        for idx in range(nearest_distances.shape[1]):
            sns.lineplot(
                x=time_seconds,
                y=nearest_distances[:, idx],
                color=cmap(idx % cmap.N),
                linewidth=2.2,
                label=key_labels[idx],
                ax=ax,
            )
    ax.set_facecolor("#F8FAFC")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Nearest friendly distance (m)")
    ax.grid(True, which="major", alpha=0.28, color="#94A3B8", linewidth=0.8)
    ax.grid(False, which="minor")
    ax.legend(frameon=False)
    sns.despine(ax=ax, offset=4)
    fig.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return [(fig, ax)]


def generate_demo_figures(
    num_steps: int | None = None,
    dt: float = 1.0,
    output_dir: str | Path = "outputs/figures",
    show: bool = False,
) -> dict[str, Path]:
    resolved_steps = get_num_steps() if num_steps is None else int(num_steps)
    scene, trajectories = generate_demo_trajectories(num_steps=resolved_steps, dt=dt)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    initial_path = output_root / "initial_scene.pdf"
    enemy_path = output_root / "enemy_trajectories.pdf"
    friendly_path = output_root / "friendly_trajectories.pdf"

    plot_scene_snapshot(
        scene,
        trajectories,
        timestep=0,
        save_path=initial_path,
        title="Initial 3D Scene",
        show=show,
    )
    plot_trajectories_until_timestep(
        scene,
        trajectories,
        timestep=resolved_steps,
        target="enemy",
        save_path=enemy_path,
        title="Enemy UAV Trajectories",
        show=show,
    )
    plot_trajectories_until_timestep(
        scene,
        trajectories,
        timestep=resolved_steps,
        target="friendly",
        save_path=friendly_path,
        title="Friendly UAV Trajectories",
        show=show,
    )

    return {
        "initial": initial_path,
        "enemy": enemy_path,
        "friendly": friendly_path,
    }


if __name__ == "__main__":
    generate_demo_figures(show=False)
