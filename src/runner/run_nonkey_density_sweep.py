from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import numpy as np

from src.runner.export_tracking_rollout import rollout_tracking_episode
from src.tracking.metrics import summarize_key_target_tracking_episode
from src.utils.config_loader import get_selected_scenario_name
from src.visualization.matplotlib_style import configure_matplotlib_for_ieee_pdf

configure_matplotlib_for_ieee_pdf()

import matplotlib.pyplot as plt
import seaborn as sns


_ASSOCIATION_METHOD = "labeled_anchor_bp_jpda"
_METHOD_LABEL = "IA-BP-JPDA"
_ASSOCIATION_METHODS = (_ASSOCIATION_METHOD, "bp_jpda")
_METHOD_LABELS = {_ASSOCIATION_METHOD: _METHOD_LABEL, "bp_jpda": "BP-JPDA"}
_RUN_METRIC_FIELDS = (
    "method", "seed", "key_targets", "nonkey_targets", "nonkey_to_key_ratio",
    "mean_gospa_m", "total_id_switches",
)
_DEFAULT_RATIOS = tuple(range(3, 21))
_BOOTSTRAP_RESAMPLES = 2000
_BOOTSTRAP_SEED = 20260813
_FIGURE_SIZE = (7.2, 4.6)
_AXIS_LABEL_SIZE = 18.0
_TICK_LABEL_SIZE = 16.0
_LEGEND_SIZE = 16.0
_LINE_WIDTH = 2.6
_MARKER_SIZE = 5
_MARKER_EDGE_WIDTH = 1
_CI_ALPHA = 0.08
_METHOD_COLOR = "#0072B2"
_IDSW_COLOR = "#D55E00"
_METHOD_STYLES = {
    _ASSOCIATION_METHOD: (_METHOD_COLOR, "-", "o"),
    "bp_jpda": ("#F77A1A", "--", "s"),
}


def _scenario_output_name() -> str:
    selected_scenario = get_selected_scenario_name()
    if selected_scenario is None:
        return "default_scenario"
    return str(selected_scenario).strip() or "default_scenario"


def _output_root(output_dir: str | Path) -> Path:
    return (
        Path(output_dir)
        / _scenario_output_name()
        / "nonkey_density_sweep"
        / _ASSOCIATION_METHOD
    )


def _bootstrap_mean_confidence_interval(values: np.ndarray) -> tuple[float, float, float]:
    samples = np.asarray(values, dtype=float).reshape(-1)
    if samples.size < 2:
        raise ValueError("At least two Monte Carlo runs are required to estimate a confidence interval.")

    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    resample_indices = rng.integers(0, samples.size, size=(_BOOTSTRAP_RESAMPLES, samples.size))
    bootstrap_means = np.mean(samples[resample_indices], axis=1)
    lower, upper = np.percentile(bootstrap_means, [2.5, 97.5])
    return float(np.mean(samples)), float(lower), float(upper)


def _read_run_metrics(
    csv_path: str | Path,
) -> tuple[dict[tuple[str, int, int], tuple[float, float]], int | None]:
    input_path = Path(csv_path)
    records: dict[tuple[str, int, int], tuple[float, float]] = {}
    key_counts: set[int] = set()

    with input_path.open("r", newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            method = str(row["method"]).strip().lower()
            if method not in _ASSOCIATION_METHODS:
                raise ValueError(f"Unsupported density-sweep method: {method}.")
            ratio = int(row["nonkey_to_key_ratio"])
            seed = int(row["seed"])
            key_num = int(row["key_targets"])
            if key_num <= 0 or ratio <= 0 or int(row["nonkey_targets"]) != key_num * ratio:
                raise ValueError("Density-sweep target counts must agree with the density ratio.")
            key_counts.add(key_num)
            record_key = (method, ratio, seed)
            if record_key in records:
                raise ValueError(f"Duplicate density-sweep record: {record_key}.")
            records[record_key] = (
                float(row["mean_gospa_m"]), float(row["total_id_switches"]),
            )

    if len(key_counts) > 1:
        raise ValueError("All density-sweep rows must use the same number of key targets.")
    return records, next(iter(key_counts), None)


def _load_run_metrics(
    csv_path: str | Path,
) -> tuple[np.ndarray, dict[str, dict[int, np.ndarray]], dict[str, dict[int, np.ndarray]], int]:
    records, key_num = _read_run_metrics(csv_path)
    if not records or key_num is None:
        raise ValueError(f"No density-sweep rows found in {Path(csv_path)}.")

    ratios = np.asarray(sorted({ratio for _, ratio, _ in records}), dtype=int)
    reference_seeds = {seed for _, _, seed in records}
    methods = [method for method in _ASSOCIATION_METHODS if any(key[0] == method for key in records)]
    gospa_by_method: dict[str, dict[int, np.ndarray]] = {}
    idsw_by_method: dict[str, dict[int, np.ndarray]] = {}
    for method in methods:
        gospa_by_method[method] = {}
        idsw_by_method[method] = {}
        for ratio in ratios:
            seeds = sorted(seed for m, r, seed in records if m == method and r == int(ratio))
            if set(seeds) != reference_seeds:
                raise ValueError("Every method and density ratio must use the same Monte Carlo seed set.")
            gospa_by_method[method][int(ratio)] = np.asarray(
                [records[method, int(ratio), seed][0] for seed in seeds], dtype=float,
            )
            idsw_by_method[method][int(ratio)] = np.asarray(
                [records[method, int(ratio), seed][1] for seed in seeds], dtype=float,
            )
    return ratios, gospa_by_method, idsw_by_method, key_num


def _summarize_metric_by_ratio(
    ratios: np.ndarray,
    values_by_ratio: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means: list[float] = []
    lowers: list[float] = []
    uppers: list[float] = []
    for ratio in ratios:
        mean, lower, upper = _bootstrap_mean_confidence_interval(values_by_ratio[int(ratio)])
        means.append(mean)
        lowers.append(lower)
        uppers.append(upper)
    return np.asarray(means), np.asarray(lowers), np.asarray(uppers)


def _plot_density_metric(
    *,
    ratios: np.ndarray,
    summaries_by_method: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    ylabel: str,
    save_path: Path,
    show_confidence_interval: bool,
    show: bool,
) -> None:
    with sns.axes_style("whitegrid"), sns.plotting_context("notebook"):
        fig, ax = plt.subplots(figsize=_FIGURE_SIZE)
        for method, (mean, lower, upper) in summaries_by_method.items():
            color, linestyle, marker = _METHOD_STYLES[method]
            if show_confidence_interval:
                ax.fill_between(
                    ratios,
                    lower,
                    upper,
                    color=color,
                    alpha=_CI_ALPHA,
                    linewidth=0.0,
                    zorder=1,
                )
            ax.plot(
                ratios,
                mean,
                color=color,
                linestyle=linestyle,
                linewidth=_LINE_WIDTH,
                marker=marker,
                markersize=_MARKER_SIZE,
                markerfacecolor="none",
                markeredgecolor=color,
                markeredgewidth=_MARKER_EDGE_WIDTH,
                zorder=3,
                label=_METHOD_LABELS[method],
            )

    if ratios.size == 1:
        ax.set_xlim(float(ratios[0]) - 0.5, float(ratios[0]) + 0.5)
    else:
        ax.set_xlim(float(ratios[0]), float(ratios[-1]))
    ax.margins(x=0.0)
    ax.set_xticks(ratios)
    ax.set_xlabel(r"Non-key/key-target ratio $\rho_{\mathrm{N/K}}$", fontsize=_AXIS_LABEL_SIZE)
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
        ncol=min(2, len(summaries_by_method)),
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


def _plot_density_metrics_combined(
    *,
    ratios: np.ndarray,
    gospa_summaries: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    idsw_summaries: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    save_path: Path,
    show_confidence_interval: bool,
    show: bool,
) -> None:
    max_value = max(
        float(np.max(summary[2 if show_confidence_interval else 0]))
        for summaries in (gospa_summaries, idsw_summaries)
        for summary in summaries.values()
    )
    y_upper = max(10.0, 5.0 * np.ceil(max_value / 5.0))
    with sns.axes_style("whitegrid"), sns.plotting_context("notebook"):
        fig, gospa_ax = plt.subplots(figsize=_FIGURE_SIZE)
        idsw_ax = gospa_ax.twinx()
        lines = []
        for ax, summaries, color, metric, ylabel in (
            (gospa_ax, gospa_summaries, _METHOD_COLOR, "GOSPA", "Time-averaged GOSPA (m)"),
            (idsw_ax, idsw_summaries, _IDSW_COLOR, "IDSW", "Mean IDSW per run"),
        ):
            for method, (mean, lower, upper) in summaries.items():
                _, linestyle, marker = _METHOD_STYLES[method]
                if show_confidence_interval:
                    ax.fill_between(
                        ratios, lower, upper, color=color, alpha=_CI_ALPHA,
                        linewidth=0.0, zorder=1,
                    )
                line, = ax.plot(
                    ratios, mean, color=color, linestyle=linestyle,
                    linewidth=_LINE_WIDTH, marker=marker, markersize=_MARKER_SIZE,
                    markerfacecolor="none", markeredgecolor=color,
                    markeredgewidth=_MARKER_EDGE_WIDTH, zorder=3,
                    label=f"{_METHOD_LABELS[method]}: {metric}",
                )
                lines.append(line)
            ax.set_ylabel(ylabel, fontsize=_AXIS_LABEL_SIZE, color=color)
            # Leave a small gap below the first labeled tick at 5.
            ax.set_ylim(4.5, y_upper)
            ax.set_yticks(np.arange(5.0, y_upper + 0.5, 5.0))
            is_left = ax is gospa_ax
            ax.tick_params(
                axis="y", which="major", direction="in", length=6, width=1.0,
                colors=color, labelsize=_TICK_LABEL_SIZE,
                left=is_left, right=not is_left,
                labelleft=is_left, labelright=not is_left,
            )

    if ratios.size == 1:
        gospa_ax.set_xlim(float(ratios[0]) - 0.5, float(ratios[0]) + 0.5)
    else:
        gospa_ax.set_xlim(float(ratios[0]), float(ratios[-1]))
    gospa_ax.margins(x=0.0)
    gospa_ax.set_xticks(ratios)
    gospa_ax.set_xlabel(r"Non-key/key-target ratio $\rho_{\mathrm{N/K}}$", fontsize=_AXIS_LABEL_SIZE)
    gospa_ax.tick_params(
        axis="x", which="major", direction="in", length=6, width=1.0,
        colors="black", labelsize=_TICK_LABEL_SIZE, bottom=True, top=False,
    )
    gospa_ax.grid(True, which="major", alpha=0.3, color="#88888A", linewidth=0.8)
    gospa_ax.grid(False, which="minor")
    idsw_ax.grid(False)
    gospa_ax.legend(
        handles=lines, loc="lower center", bbox_to_anchor=(0.5, 1.02),
        frameon=False, fontsize=_LEGEND_SIZE, ncol=2, handlelength=2.4,
    )
    sns.despine(ax=gospa_ax, offset=0)
    sns.despine(ax=idsw_ax, left=True, right=False, bottom=True, offset=0)
    gospa_ax.spines["left"].set_color("black")
    gospa_ax.spines["bottom"].set_color("black")
    idsw_ax.spines["right"].set_color("black")
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_nonkey_density_sweep_csv(
    *,
    csv_path: str | Path,
    output_dir: str | Path = "outputs/figure",
    show_confidence_interval: bool = True,
    show: bool = False,
) -> dict[str, Path]:
    ratios, gospa_by_method, idsw_by_method, key_num = _load_run_metrics(csv_path)
    gospa_summaries = {
        method: _summarize_metric_by_ratio(ratios, values)
        for method, values in gospa_by_method.items()
    }
    idsw_summaries = {
        method: _summarize_metric_by_ratio(ratios, values)
        for method, values in idsw_by_method.items()
    }

    output_root = _output_root(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_csv_path = output_root / "nonkey_density_summary_metrics.csv"
    with summary_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "method",
            "key_targets",
            "nonkey_targets",
            "nonkey_to_key_ratio",
            "num_runs",
            "mean_gospa_m",
            "gospa_ci95_lower_m",
            "gospa_ci95_upper_m",
            "mean_id_switches_per_run",
            "idsw_ci95_lower",
            "idsw_ci95_upper",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for method in gospa_by_method:
            gospa_mean, gospa_lower, gospa_upper = gospa_summaries[method]
            idsw_mean, idsw_lower, idsw_upper = idsw_summaries[method]
            for idx, ratio in enumerate(ratios):
                writer.writerow(
                    {
                        "method": method,
                        "key_targets": key_num,
                        "nonkey_targets": key_num * int(ratio),
                        "nonkey_to_key_ratio": int(ratio),
                        "num_runs": int(gospa_by_method[method][int(ratio)].size),
                        "mean_gospa_m": f"{gospa_mean[idx]:.9f}",
                        "gospa_ci95_lower_m": f"{gospa_lower[idx]:.9f}",
                        "gospa_ci95_upper_m": f"{gospa_upper[idx]:.9f}",
                        "mean_id_switches_per_run": f"{idsw_mean[idx]:.9f}",
                        "idsw_ci95_lower": f"{idsw_lower[idx]:.9f}",
                        "idsw_ci95_upper": f"{idsw_upper[idx]:.9f}",
                    }
                )

    gospa_pdf_path = output_root / "nonkey_density_mean_gospa.pdf"
    idsw_pdf_path = output_root / "nonkey_density_idsw.pdf"
    combined_pdf_path = output_root / "nonkey_density_gospa_idsw_combined.pdf"
    _plot_density_metric(
        ratios=ratios,
        summaries_by_method=gospa_summaries,
        ylabel="Time-averaged GOSPA (m)",
        save_path=gospa_pdf_path,
        show_confidence_interval=show_confidence_interval,
        show=show,
    )
    _plot_density_metric(
        ratios=ratios,
        summaries_by_method=idsw_summaries,
        ylabel="Mean IDSW per run",
        save_path=idsw_pdf_path,
        show_confidence_interval=show_confidence_interval,
        show=show,
    )
    print(f"Loaded density-sweep metrics: {Path(csv_path)}")
    print(f"Saved density-sweep summary: {summary_csv_path}")
    print(f"Saved density-sweep GOSPA figure: {gospa_pdf_path}")
    print(f"Saved density-sweep IDSW figure: {idsw_pdf_path}")
    outputs = {
        "run_metrics_csv": Path(csv_path),
        "summary_csv": summary_csv_path,
        "gospa_pdf": gospa_pdf_path,
        "idsw_pdf": idsw_pdf_path,
    }
    _plot_density_metrics_combined(
        ratios=ratios,
        gospa_summaries=gospa_summaries,
        idsw_summaries=idsw_summaries,
        save_path=combined_pdf_path,
        show_confidence_interval=show_confidence_interval,
        show=show,
    )
    print(f"Saved density-sweep combined figure: {combined_pdf_path}")
    outputs["combined_pdf"] = combined_pdf_path
    return outputs


def run_nonkey_density_sweep(
    *,
    num_seeds: int,
    start_seed: int,
    key_num: int = 3,
    ratios: Sequence[int] = _DEFAULT_RATIOS,
    output_dir: str | Path = "outputs/figure",
    show_confidence_interval: bool = True,
    show: bool = False,
    association_methods: Sequence[str] = _ASSOCIATION_METHODS,
    resume: bool = False,
) -> dict[str, Path]:
    if num_seeds < 2:
        raise ValueError(f"num_seeds must be at least 2, got {num_seeds}.")
    if key_num <= 0:
        raise ValueError(f"key_num must be positive, got {key_num}.")

    resolved_ratios = tuple(int(ratio) for ratio in ratios)
    if not resolved_ratios or any(ratio <= 0 for ratio in resolved_ratios):
        raise ValueError("ratios must contain positive integers.")
    if len(set(resolved_ratios)) != len(resolved_ratios):
        raise ValueError("ratios must not contain duplicates.")
    resolved_ratios = tuple(sorted(resolved_ratios))
    methods = tuple(str(method).lower() for method in association_methods)
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("association_methods must be nonempty and contain no duplicates.")
    if set(methods) - set(_ASSOCIATION_METHODS):
        raise ValueError(f"association_methods must be selected from {_ASSOCIATION_METHODS}.")
    seeds = range(int(start_seed), int(start_seed) + int(num_seeds))

    output_root = _output_root(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    run_metrics_csv_path = output_root / "nonkey_density_run_metrics.csv"
    existing: dict[tuple[str, int, int], tuple[float, float]] = {}
    if resume and run_metrics_csv_path.exists():
        with run_metrics_csv_path.open("r", newline="", encoding="utf-8") as csv_file:
            header = csv.DictReader(csv_file).fieldnames
        if header is not None and tuple(header) != _RUN_METRIC_FIELDS:
            raise ValueError("Existing CSV columns do not match the density-sweep run format.")
        existing, existing_key_num = _read_run_metrics(run_metrics_csv_path)
        if existing_key_num is not None and existing_key_num != key_num:
            raise ValueError("Existing CSV key-target count does not match --key-num.")
        if any(ratio not in resolved_ratios or seed not in seeds for _, ratio, seed in existing):
            raise ValueError(
                "Existing CSV ratios/seeds fall outside the requested sweep; "
                "use matching options or a new output directory."
            )

    pending_runs = [
        (method, ratio, seed)
        for method in methods
        for ratio in resolved_ratios
        for seed in seeds
        if (method, ratio, seed) not in existing
    ]
    write_header = not existing
    total_runs = len(pending_runs)
    if resume:
        reused = len(methods) * len(resolved_ratios) * num_seeds - total_runs
        print(f"Reusing {reused} saved runs; {total_runs} runs remain.")

    for completed_runs, (method, ratio, seed) in enumerate(pending_runs, start=1):
        nonkey_num = key_num * ratio
        print(
            f"[{completed_runs}/{total_runs}] Running {method} with "
            f"seed {seed}, key targets {key_num}, non-key targets {nonkey_num} "
            f"(ratio {ratio})..."
        )
        (
            _,
            trajectories,
            key_truth_states,
            fused_key_state_means,
            fused_key_valid_mask,
            _,
        ) = rollout_tracking_episode(
            key_num=key_num,
            nonkey_num=nonkey_num,
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
        with run_metrics_csv_path.open(
            "w" if write_header else "a", newline="", encoding="utf-8",
        ) as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=_RUN_METRIC_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "method": method,
                    "seed": seed,
                    "key_targets": key_num,
                    "nonkey_targets": nonkey_num,
                    "nonkey_to_key_ratio": ratio,
                    "mean_gospa_m": f"{float(summary['mean_gospa']):.9f}",
                    "total_id_switches": int(summary["total_id_switches"]),
                }
            )
        write_header = False

    print(f"Saved density-sweep run metrics: {run_metrics_csv_path}")
    return plot_nonkey_density_sweep_csv(
        csv_path=run_metrics_csv_path,
        output_dir=output_dir,
        show_confidence_interval=show_confidence_interval,
        show=show,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare IA-BP-JPDA and BP-JPDA across non-key/key-target ratios using a common "
            "set of Monte Carlo seeds."
        )
    )
    parser.add_argument("--num-seeds", type=int, default=None, help="Number of Monte Carlo seeds (at least 2).")
    parser.add_argument("--start-seed", type=int, default=0, help="First seed in the consecutive seed range.")
    parser.add_argument("--key-num", type=int, default=3, help="Fixed number of key targets.")
    parser.add_argument(
        "--methods", nargs="+", choices=_ASSOCIATION_METHODS, default=list(_ASSOCIATION_METHODS),
        help="Methods to run. Defaults to IA-BP-JPDA and BP-JPDA with paired seeds.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Keep existing CSV rows and run only missing method/ratio/seed combinations. Use the same scenario and sensing configuration.",
    )
    parser.add_argument(
        "--ratios",
        type=int,
        nargs="+",
        default=list(_DEFAULT_RATIOS),
        help="Non-key/key-target ratios. Defaults to all integers from 3 to 20.",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/figure", help="Root directory for outputs.")
    parser.add_argument("--plot-only", action="store_true", help="Regenerate figures from a saved run-level CSV.")
    parser.add_argument("--csv-path", type=str, default=None, help="Run-level CSV used with --plot-only.")
    parser.add_argument("--no-ci", action="store_true", help="Plot Monte Carlo means without confidence bands.")
    parser.add_argument("--show", action="store_true", help="Display figures interactively after saving them.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.plot_only:
        resolved_csv_path = args.csv_path or (
            _output_root(args.output_dir) / "nonkey_density_run_metrics.csv"
        )
        plot_nonkey_density_sweep_csv(
            csv_path=resolved_csv_path,
            output_dir=args.output_dir,
            show_confidence_interval=not args.no_ci,
            show=args.show,
        )
    else:
        if args.num_seeds is None:
            raise SystemExit("--num-seeds is required unless --plot-only is specified.")
        run_nonkey_density_sweep(
            num_seeds=args.num_seeds,
            start_seed=args.start_seed,
            key_num=args.key_num,
            ratios=args.ratios,
            output_dir=args.output_dir,
            show_confidence_interval=not args.no_ci,
            show=args.show,
            association_methods=args.methods,
            resume=args.resume,
        )
