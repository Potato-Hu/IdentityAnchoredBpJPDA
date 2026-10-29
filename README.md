# SwarmJam

English | [简体中文](README_CN.md)

SwarmJam is a **key-target-prioritized multi-target tracking and data-association research platform** for cooperative multi-UAV scenarios. It simulates friendly UAVs, key targets, and non-key distractors in dynamic 3D environments, combining weak global priors, anonymous local observations, multiple association algorithms, and centralized track fusion to evaluate tracking accuracy, coverage, and identity continuity for key targets.

> The current version focuses on multi-target tracking, data association, and cooperative tracking evaluation. It does not include a runnable reinforcement-learning training entry point.

## Features

- **3D scenario simulation:** key and non-key targets, friendly UAVs, and Gauss-Markov or random-walk motion models.
- **Multi-source sensing:** delayed, noisy, and occasionally missed global key-target priors, plus noisy local anonymous observations with missed detections and clutter.
- **Multi-target data association:** nearest-neighbor, surveillance-only, BP-JPDA, GM-PHD, LMB, δ-GLMB, and several prior-anchored BP-JPDA variants.
- **Centralized track fusion:** consistency checking and covariance-intersection fusion of local UAV tracks into global key-target tracks.
- **Cooperative tracking loop:** distance-based assignment (`hungarian` configuration mode) and discrete 3D motion control.
- **Evaluation and visualization:** RMSE, OSPA, GOSPA, tracking coverage, ID switches, fragmentations, runtime, trajectory plots, and CSV/NPZ exports.

System flow:

```text
3D scene and target motion
        │
        ├── Weak global key-target priors
        └── Anonymous observations from each UAV
                        │
                        ▼
            Local association and tracking
                        │
                        ▼
          Centralized association and fusion
                        │
                        ▼
             Assignment → UAV control
                        │
                        ▼
              Metrics and result export
```

## Requirements

- Python 3.10–3.12
- Windows, Linux, or macOS
- Main dependencies: NumPy, Matplotlib, Seaborn, and PyYAML

Create an isolated virtual environment from the project root.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick Start

Run all commands from the project root.

### 1. Verify scene generation and basic visualization

```bash
python main.py
```

This creates PDF figures for the initial 3D scene, enemy-target trajectories, and friendly-UAV trajectories under `outputs/figures/`.

### 2. Run one complete tracking experiment

```bash
python -m src.runner.export_tracking_rollout --no-show
```

Default values come from `config/default.yaml` and `config/env.yaml`. Common parameters can be overridden from the command line:

```bash
python -m src.runner.export_tracking_rollout \
  --seed 10 \
  --key-num 3 \
  --nonkey-num 30 \
  --mydrone-num 5 \
  --no-show
```

The same command on one line, which is convenient in Windows PowerShell:

```powershell
python -m src.runner.export_tracking_rollout --seed 10 --key-num 3 --nonkey-num 30 --mydrone-num 5 --no-show
```

Results are saved to:

```text
outputs/tracking_rollouts/<scenario>/<association_method>/
```

The directory contains trajectory data in NPZ format, position and action CSV files, a key-target metrics CSV, and scene and trajectory PDFs.

### 3. Run a Monte Carlo method comparison

```bash
python -m src.runner.run_tracking_monte_carlo --num-seeds 10 --start-seed 0
```

By default, this compares `bp_jpda`, `gm_phd`, `lmb`, `delta_glmb`, and `labeled_anchor_bp_jpda`. To compare only selected methods:

```bash
python -m src.runner.run_tracking_monte_carlo \
  --num-seeds 10 \
  --methods bp_jpda labeled_anchor_bp_jpda
```

`--num-seeds` must be at least 2. Summary CSV files and comparison figures with 95% confidence intervals are saved to:

```text
outputs/figure/<scenario>/method_comparison/
```

To regenerate figures from an existing metrics file without rerunning the simulations:

```bash
python -m src.runner.run_tracking_monte_carlo --plot-only --csv-path <metrics-csv-path>
```

### 4. Run a non-key-target density sweep

The following command evaluates tracking performance at different non-key/key-target count ratios:

```bash
python -m src.runner.run_nonkey_density_sweep \
  --num-seeds 10 \
  --key-num 3 \
  --ratios 3 5 10 15 20
```

This experiment always uses `labeled_anchor_bp_jpda`. The total number of runs is the number of seeds multiplied by the number of ratios, so large experiments may take a while.

Each completed sweep generates separate GOSPA and IDSW figures plus `nonkey_density_gospa_idsw_combined.pdf` in the same output directory. The combined figure uses a solid blue GOSPA curve on the left axis and a dashed orange IDSW curve on the right axis.
Regenerate all three figures directly from the saved run-level CSV with:

```bash
python -m src.runner.run_nonkey_density_sweep --plot-only --csv-path <nonkey_density_run_metrics.csv>
```

## Configuration

### Global and environment configuration

| File | Purpose |
| --- | --- |
| `config/default.yaml` | Default random seed |
| `config/env.yaml` | Simulation step and duration, tracking mode, association method, assignment, recursive motion model, and fusion parameters |
| `config/scenarios/*.yaml` | Entity counts, initialization regions, map bounds, motion, sensing noise, missed detections, and clutter |

Select a scenario in `config/env.yaml`:

```yaml
selected_scenario: scenerios_many_nonkey_partitioned
```

Included scenarios:

- `scenerios_base`: base scenario;
- `scenerios_many_nonkey_partitioned`: partitioned motion with many non-key targets;
- `scenario_high_noise`: stronger global-prior noise.

The `scenerios_*` spelling is retained from the current filenames and must be used exactly as shown in the configuration.

### Tracking and association

A single tracking rollout uses the following section of `config/env.yaml`:

```yaml
tracking:
  mode: local_only
  association_method: labeled_anchor_bp_jpda
  assignment: hungarian
```

- `tracking.mode: local_only` uses weak global priors, anonymous local observations, and an association algorithm. This is the default research mode.
- `tracking.mode: groundtruth` exposes key-target ground-truth tracks directly and is intended only for debugging or reference comparisons.
- `tracking.assignment` supports `hungarian` and `placeholder`.

The current `hungarian` mode uses exact permutation search for small assignments and a greedy fallback for larger cases; it does not require SciPy.

Available `association_method` values:

| Method | Description |
| --- | --- |
| `none` | Nearest-local-candidate baseline |
| `prior_only` | External key-target priors and motion prediction only |
| `bp_jpda` | BP-optimized JPDA baseline |
| `hard_anchor_bp_jpda` | Current labeled prior used as a hard anchor |
| `covariance_anchor_bp_jpda` | Covariance-fused anchored BP-JPDA |
| `labeled_anchor_bp_jpda` | Fixed key labels anchored to global priors (IA-BP-JPDA) |
| `gm_phd` | GM-PHD recursive baseline |
| `lmb` | Labeled Multi-Bernoulli-style baseline |
| `delta_glmb` | Compact δ-GLMB multi-hypothesis baseline |

Association and recursive-estimation parameters are under `association` in `config/env.yaml`; local-track fusion parameters are under `track_fusion`. The recursive motion prior supports either a single `cv` model or an `imm` bank containing `cv`, `ca`, `ct_left`, and `ct_right`.

## Project Structure

```text
SwarmJam/
├── config/                 # Global, environment, and scenario configuration
├── docs/                   # System model, problem formulation, and algorithm notes
├── outputs/                # Simulation data, metrics, and figures
├── src/
│   ├── entities/           # Friendly-UAV and enemy-target entities
│   ├── env/                # Environment wrapper and world-state updates
│   ├── interference/       # Channel and interference calculations
│   ├── runner/             # Single-run, Monte Carlo, and density-sweep entry points
│   ├── sensing/            # Global/local sensing and association algorithms
│   ├── simulation/         # Initialization, motion, and trajectory generation
│   ├── tracking/           # Fusion, metrics, assignment, and control
│   ├── utils/              # Configuration and random-seed utilities
│   └── visualization/      # Plotting and animation
├── main.py                 # Scene-generation and basic-visualization entry point
└── requirements.txt        # Python dependencies
```

## Further Reading

- [System model](docs/system_model.md)
- [Problem formulation](docs/problem_formulation.md)
- [Project summary](docs/project_summary.md)
- [Labeled-anchor BP-JPDA notes](docs/labeled_anchor_bp_jpda.md)

SwarmJam is currently a research prototype. For reproducible experiments, fix the random seed and record the selected scenario, association method, and complete configuration.
