from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import numpy as np

from src.env.swarm_env import ASSOCIATION_METHODS
from src.runner.export_tracking_rollout import rollout_tracking_episode
from src.tracking.metrics import summarize_key_target_tracking_episode
from src.utils.config_loader import get_selected_scenario_name, get_tracking_association_method
from src.visualization.matplotlib_style import configure_matplotlib_for_ieee_pdf

configure_matplotlib_for_ieee_pdf()

import matplotlib.pyplot as plt
import seaborn as sns


_BOOTSTRAP_RESAMPLES = 2000
_BOOTSTRAP_SEED = 20260813
_FIGURE_SIZE = (7.2, 4.6)
_AXIS_LABEL_SIZE = 17.0
_TICK_LABEL_SIZE = 15.0
_LEGEND_SIZE = 15.0
_LINE_WIDTH = 2.6
_MARKER_SIZE = 5
_MARKER_EDGE_WIDTH = 1
_CI_ALPHA = 0.08
_DEFAULT_METHODS = (
    "bp_jpda",
    "gm_phd",
    "lmb",
    "delta_glmb",
    "labeled_anchor_bp_jpda",
)
_METHOD_LABELS = {
    "none": "Nearest neighbor",
    "prior_only": "Surveillance-only",
    "bp_jpda": "BP-JPDA",
    "hard_anchor_bp_jpda": "Hard-anchor",
    "covariance_anchor_bp_jpda": "Covariance-anchor BP-JPDA",
    "labeled_anchor_bp_jpda": "IA-BP-JPDA",
    "gm_phd": "GM-PHD",
    "lmb": "LMB",
    "delta_glmb": r"$\delta$-GLMB",
}
_METHOD_COLORS = {
    "bp_jpda": "#F77A1A",
    "gm_phd": "#F8B931",
    "lmb": "#07C08F",
    "delta_glmb": "#FC68B9",
    "labeled_anchor_bp_jpda": "#0072B2",
    "prior_only": "#9545D6",
    "hard_anchor_bp_jpda": "#34AF04",
    
}
_METHOD_LINESTYLES = {
    "bp_jpda": "-",
    "gm_phd": "-",
    "lmb": "--",
    "delta_glmb": "--",
    "labeled_anchor_bp_jpda": "-",
    "prior_only": "-",
    "hard_anchor_bp_jpda": "-",
}
_METHOD_MARKERS = {
    "bp_jpda": "D",
    "gm_phd": "v",
    "lmb": "o",
    "delta_glmb": "D",
    "labeled_anchor_bp_jpda": "o",
    "prior_only": "s",
    "hard_anchor_bp_jpda": "v",

}


def _scenario_output_name() -> str:
    selected_scenario = get_selected_scenario_name()
    if selected_scenario is None:
        return "default_scenario"
    return str(selected_scenario).strip() or "default_scenario"


def _comparison_output_root(output_dir: str | Path) -> Path:
    return Path(output_dir) / _scenario_output_name() / "method_comparison"


def _mean_nearest_uav_distance(
    key_truth_states: np.ndarray,
    friendly_positions: np.ndarray,
) -> np.ndarray:
    key_positions = np.asarray(key_truth_states, dtype=float)[..., :3]
    uav_positions = np.asarray(friendly_positions, dtype=float)
    if key_positions.ndim != 3 or uav_positions.ndim != 3:
        raise ValueError("Key-target and friendly-UAV trajectories must both be 3D arrays over time.")
    if key_positions.shape[0] != uav_positions.shape[0]:
        raise ValueError("Key-target and friendly-UAV trajectories must have the same number of time steps.")
    if key_positions.shape[1] == 0 or uav_positions.shape[1] == 0:
        raise ValueError("Nearest-UAV distance requires at least one key target and one friendly UAV.")

    pairwise_distances = np.linalg.norm(
        key_positions[:, :, None, :] - uav_positions[:, None, :, :],
        axis=-1,
    )
    return np.mean(np.min(pairwise_distances, axis=2), axis=1)


def _bootstrap_mean_confidence_interval(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples = np.asarray(values, dtype=float)
    if samples.ndim != 2 or samples.shape[0] < 2:
        raise ValueError("At least two Monte Carlo runs are required to estimate a confidence interval.")

    mean = np.mean(samples, axis=0)
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    resample_indices = rng.integers(
        0,
        samples.shape[0],
        size=(_BOOTSTRAP_RESAMPLES, samples.shape[0]),
    )
    bootstrap_means = np.mean(samples[resample_indices], axis=1)
    lower, upper = np.percentile(bootstrap_means, [2.5, 97.5], axis=0)
    return mean, lower, upper


def _plot_metric_curves(
    *,
    time_seconds: np.ndarray,
    values_by_method: dict[str, np.ndarray],
    ylabel: str,
    save_path: Path,
    show_confidence_interval: bool,
    show: bool,
) -> None:
    with sns.axes_style("whitegrid"), sns.plotting_context("notebook"):
        fig, ax = plt.subplots(figsize=_FIGURE_SIZE)
        for method_idx, (method, values) in enumerate(values_by_method.items()):
            mean, lower, upper = _bootstrap_mean_confidence_interval(values)
            color = _METHOD_COLORS.get(method, plt.get_cmap("tab10")(method_idx % 10))
            if show_confidence_interval:
                ax.fill_between(
                    time_seconds,
                    lower,
                    upper,
                    color=color,
                    alpha=_CI_ALPHA,
                    linewidth=0.0,
                    zorder=1,
                )
            ax.plot(
                time_seconds,
                mean,
                color=color,
                linestyle=_METHOD_LINESTYLES.get(method, "-"),
                linewidth=_LINE_WIDTH,
                marker=_METHOD_MARKERS.get(method, "o"),
                markersize=_MARKER_SIZE,
                markerfacecolor="none",
                markeredgecolor=color,
                markeredgewidth=_MARKER_EDGE_WIDTH,
                zorder=3,
                label=_METHOD_LABELS.get(method, method),
            )

    ax.set_xlim(float(time_seconds[0]), float(time_seconds[-1]))
    ax.margins(x=0.0)
    ax.set_xlabel("Time (s)", fontsize=_AXIS_LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=_AXIS_LABEL_SIZE)
    ax.tick_params(
    axis="both",
    which="major",          
    direction="in",         
    length=6,
    width=1.0,
    colors="black",         
    labelsize=_TICK_LABEL_SIZE,
    bottom=True,
    left=True,
    top=False,               
    right=True,             
    labeltop=False,         
    labelright=False,
)
    ax.grid(True, which="major", alpha=0.3, color="#88888A", linewidth=0.8)
    ax.grid(False, which="minor")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        fontsize=_LEGEND_SIZE,
        ncol=2,
        columnspacing=1.1,
        handlelength=2.4,
    )
    sns.despine(ax=ax, offset=4)
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _write_metric_rows(
    csv_path: Path,
    *,
    method: str,
    seed: int,
    time_seconds: np.ndarray,
    gospa: np.ndarray,
    nearest_uav: np.ndarray,
    id_switches_cumulative: np.ndarray,
    tracking_runtime_ms: np.ndarray,
    write_header: bool,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w" if write_header else "a", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "method",
            "seed",
            "timestep",
            "time_s",
            "gospa_m",
            "nearest_uav_distance_m",
            "id_switches_cumulative",
            "tracking_runtime_ms",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for timestep, (time_s, gospa_value, nearest_value, id_switches, runtime_ms) in enumerate(
            zip(
                time_seconds,
                gospa,
                nearest_uav,
                id_switches_cumulative,
                tracking_runtime_ms,
            )
        ):
            writer.writerow(
                {
                    "method": method,
                    "seed": int(seed),
                    "timestep": int(timestep),
                    "time_s": f"{float(time_s):.6f}",
                    "gospa_m": f"{float(gospa_value):.9f}",
                    "nearest_uav_distance_m": f"{float(nearest_value):.9f}",
                    "id_switches_cumulative": int(id_switches),
                    "tracking_runtime_ms": f"{float(runtime_ms):.9f}",
                }
            )


def _load_metrics_csv(
    csv_path: str | Path,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    input_path = Path(csv_path)
    rows_by_method: dict[str, dict[int, list[tuple[int, float, float, float]]]] = {}
    with input_path.open("r", newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            method = str(row["method"])
            seed = int(row["seed"])
            rows_by_method.setdefault(method, {}).setdefault(seed, []).append(
                (
                    int(row["timestep"]),
                    float(row["time_s"]),
                    float(row["gospa_m"]),
                    float(row["nearest_uav_distance_m"]),
                )
            )
    if not rows_by_method:
        raise ValueError(f"No Monte Carlo metric rows found in {input_path}.")

    reference_time: np.ndarray | None = None
    gospa_by_method: dict[str, np.ndarray] = {}
    nearest_uav_by_method: dict[str, np.ndarray] = {}
    for method, runs_by_seed in rows_by_method.items():
        gospa_runs: list[np.ndarray] = []
        nearest_uav_runs: list[np.ndarray] = []
        for seed in sorted(runs_by_seed):
            run_rows = sorted(runs_by_seed[seed], key=lambda item: item[0])
            time_seconds = np.asarray([item[1] for item in run_rows], dtype=float)
            if reference_time is None:
                reference_time = time_seconds
            elif time_seconds.shape != reference_time.shape or not np.allclose(time_seconds, reference_time):
                raise ValueError("All methods and seeds in the CSV must use the same time grid.")
            gospa_runs.append(np.asarray([item[2] for item in run_rows], dtype=float))
            nearest_uav_runs.append(np.asarray([item[3] for item in run_rows], dtype=float))
        gospa_by_method[method] = np.stack(gospa_runs, axis=0)
        nearest_uav_by_method[method] = np.stack(nearest_uav_runs, axis=0)

    assert reference_time is not None
    return reference_time, gospa_by_method, nearest_uav_by_method


def plot_tracking_monte_carlo_csv(
    *,
    csv_path: str | Path,
    output_dir: str | Path = "outputs/figure",
    show_confidence_interval: bool = True,
    show: bool = False,
) -> dict[str, Path]:
    time_seconds, gospa_by_method, nearest_uav_by_method = _load_metrics_csv(csv_path)
    output_root = _comparison_output_root(output_dir)
    gospa_path = output_root / "monte_carlo_gospa_comparison.pdf"
    nearest_uav_path = output_root / "monte_carlo_nearest_uav_distance_comparison.pdf"

    _plot_metric_curves(
        time_seconds=time_seconds,
        values_by_method=gospa_by_method,
        ylabel="GOSPA distance (m)",
        save_path=gospa_path,
        show_confidence_interval=show_confidence_interval,
        show=show,
    )
    _plot_metric_curves(
        time_seconds=time_seconds,
        values_by_method=nearest_uav_by_method,
        ylabel="Mean nearest-UAV distance (m)",
        save_path=nearest_uav_path,
        show_confidence_interval=show_confidence_interval,
        show=show,
    )

    print(f"Loaded Monte Carlo metrics: {Path(csv_path)}")
    print(f"Saved Monte Carlo GOSPA figure: {gospa_path}")
    print(f"Saved Monte Carlo nearest-UAV figure: {nearest_uav_path}")
    return {
        "metrics_csv": Path(csv_path),
        "gospa_pdf": gospa_path,
        "nearest_uav_pdf": nearest_uav_path,
    }


def run_tracking_monte_carlo(
    *,
    num_seeds: int,
    start_seed: int,
    association_methods: Sequence[str] | None = None,
    output_dir: str | Path = "outputs/figure",
    show_confidence_interval: bool = True,
    show: bool = False,
) -> dict[str, Path]:
    if num_seeds < 2:
        raise ValueError(f"num_seeds must be at least 2, got {num_seeds}.")

    if association_methods is None:
        resolved_methods = _DEFAULT_METHODS
    else:
        resolved_methods = tuple(str(method).lower() for method in association_methods)
    if not resolved_methods:
        resolved_methods = (str(get_tracking_association_method()).lower(),)
    if len(set(resolved_methods)) != len(resolved_methods):
        raise ValueError("association_methods must not contain duplicates.")
    unknown_methods = sorted(set(resolved_methods) - ASSOCIATION_METHODS)
    if unknown_methods:
        raise ValueError(
            f"Unknown association methods {unknown_methods}; expected values from {sorted(ASSOCIATION_METHODS)}."
        )

    output_root = _comparison_output_root(output_dir)
    metrics_csv_path = output_root / "monte_carlo_step_metrics.csv"
    summary_csv_path = output_root / "monte_carlo_summary_metrics.csv"
    reference_dt: float | None = None
    reference_num_steps: int | None = None
    write_csv_header = True
    summary_values_by_method: dict[str, list[tuple[float, int, float]]] = {
        method: [] for method in resolved_methods
    }

    total_runs = len(resolved_methods) * num_seeds
    completed_runs = 0
    for method in resolved_methods:
        for run_idx in range(num_seeds):
            seed = int(start_seed) + run_idx
            completed_runs += 1
            print(f"[{completed_runs}/{total_runs}] Running {method} with seed {seed}...")
            (
                _,
                trajectories,
                key_truth_states,
                fused_key_state_means,
                fused_key_valid_mask,
                tracking_inference_time_s,
            ) = rollout_tracking_episode(
                seed=seed,
                association_method=method,
            )

            cutoff_distance = float(np.asarray(trajectories.bounds, dtype=float).reshape(-1)[0]) / 4.0
            summary = summarize_key_target_tracking_episode(
                truth_states=key_truth_states,
                estimated_states=fused_key_state_means,
                valid_mask=fused_key_valid_mask,
                cutoff_distance=cutoff_distance,
                order=2,
            )
            gospa = np.asarray(summary["gospa"], dtype=float)
            id_switches_cumulative = np.asarray(summary["id_switches_cumulative"], dtype=int)
            tracking_runtime_ms = np.asarray(tracking_inference_time_s, dtype=float) * 1e3
            nearest_uav = _mean_nearest_uav_distance(
                key_truth_states,
                trajectories.friendly_positions,
            )
            if not (
                gospa.shape
                == nearest_uav.shape
                == id_switches_cumulative.shape
                == tracking_runtime_ms.shape
            ):
                raise RuntimeError("All per-step metrics must have the same number of time steps.")

            summary_values_by_method[method].append(
                (
                    float(summary["mean_gospa"]),
                    int(summary["total_id_switches"]),
                    float(np.mean(tracking_runtime_ms)),
                )
            )

            if reference_num_steps is None:
                reference_num_steps = int(gospa.shape[0])
                reference_dt = float(trajectories.dt)
            elif gospa.shape[0] != reference_num_steps or not np.isclose(float(trajectories.dt), reference_dt):
                raise RuntimeError("All Monte Carlo runs must use the same duration and simulation interval.")

            time_seconds = np.arange(gospa.shape[0], dtype=float) * float(trajectories.dt)
            _write_metric_rows(
                metrics_csv_path,
                method=method,
                seed=seed,
                time_seconds=time_seconds,
                gospa=gospa,
                nearest_uav=nearest_uav,
                id_switches_cumulative=id_switches_cumulative,
                tracking_runtime_ms=tracking_runtime_ms,
                write_header=write_csv_header,
            )
            write_csv_header = False

    assert reference_dt is not None and reference_num_steps is not None
    with summary_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "method",
            "num_runs",
            "mean_gospa_m",
            "mean_id_switches_per_run",
            "mean_tracking_runtime_ms_per_step",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for method in resolved_methods:
            values = np.asarray(summary_values_by_method[method], dtype=float)
            writer.writerow(
                {
                    "method": method,
                    "num_runs": int(values.shape[0]),
                    "mean_gospa_m": f"{float(np.mean(values[:, 0])):.9f}",
                    "mean_id_switches_per_run": f"{float(np.mean(values[:, 1])):.9f}",
                    "mean_tracking_runtime_ms_per_step": f"{float(np.mean(values[:, 2])):.9f}",
                }
            )

    print(f"Saved Monte Carlo step metrics: {metrics_csv_path}")
    print(f"Saved Monte Carlo summary metrics: {summary_csv_path}")
    outputs = plot_tracking_monte_carlo_csv(
        csv_path=metrics_csv_path,
        output_dir=output_dir,
        show_confidence_interval=show_confidence_interval,
        show=show,
    )
    outputs["summary_csv"] = summary_csv_path
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired-seed tracking Monte Carlo experiments and plot aggregate curves."
    )
    parser.add_argument("--num-seeds", type=int, default=None, help="Number of consecutive seeds to run (at least 2).")
    parser.add_argument("--start-seed", type=int, default=0, help="First seed in the consecutive seed range.")
    parser.add_argument(
        "--methods",
        type=str,
        choices=sorted(ASSOCIATION_METHODS),
        nargs="+",
        default=list(_DEFAULT_METHODS),
        help="Association methods plotted together. Defaults to the four baselines and the proposed method.",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/figure", help="Root directory for PDF figures.")
    parser.add_argument("--plot-only", action="store_true", help="Regenerate figures from the saved metrics CSV.")
    parser.add_argument("--csv-path", type=str, default=None, help="Metrics CSV used with --plot-only.")
    parser.add_argument("--no-ci", action="store_true", help="Plot Monte Carlo means without confidence bands.")
    parser.add_argument("--show", action="store_true", help="Display figures interactively after saving them.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.plot_only:
        resolved_csv_path = args.csv_path or (
            _comparison_output_root(args.output_dir) / "monte_carlo_step_metrics.csv"
        )
        plot_tracking_monte_carlo_csv(
            csv_path=resolved_csv_path,
            output_dir=args.output_dir,
            show_confidence_interval=not args.no_ci,
            show=args.show,
        )
    else:
        if args.num_seeds is None:
            raise SystemExit("--num-seeds is required unless --plot-only is specified.")
        run_tracking_monte_carlo(
            num_seeds=args.num_seeds,
            start_seed=args.start_seed,
            association_methods=args.methods,
            output_dir=args.output_dir,
            show_confidence_interval=not args.no_ci,
            show=args.show,
        )
