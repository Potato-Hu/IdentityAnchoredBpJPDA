# System Model

## 1. Scope and Implementation Status

SwarmJam implements a **centralized multi-UAV key-target tracking loop**. The current executable pipeline is:

1. initialize and update a 3D ground-truth scene;
2. generate delayed and noisy labeled priors for key targets;
3. generate anonymous local measurements for every friendly UAV;
4. run per-UAV association and recursive key-track estimation;
5. transform local tracks to the global frame and fuse them centrally;
6. assign fused key tracks to friendly UAVs;
7. select discrete pursuit actions and advance the world;
8. export trajectories, tracking metrics, runtime data, and figures.

The implementation is deterministic/statistical tracking code. It contains no reinforcement-learning policy, neural network, training loop, replay buffer, or learned association scorer.

Main implementation entry points:

- environment: `src/env/swarm_env.py`
- scene and motion: `src/simulation/`
- sensing and association: `src/sensing/`
- fusion, assignment, control, and metrics: `src/tracking/`
- executable experiments: `src/runner/`

## 2. Entities, State, and Dynamics

### 2.1 State

Every friendly UAV and enemy target uses a 6D kinematic state:

```text
x = [px, py, pz, vx, vy, vz]
```

Enemy targets are divided into `key` and `non-key` roles. Only key targets receive persistent global labels and are evaluated as desired tracks. Non-key targets remain observable by local sensors and create association ambiguity.

### 2.2 Motion

Enemy motion is configured per scenario and currently supports Gauss-Markov and random-walk models. Some scenarios additionally correlate or spatially partition non-key motion.

Friendly UAVs use 27 discrete actions:

- action `0`: stop;
- 26 normalized directions from the Cartesian product of `{-1, 0, 1}` on three axes, excluding the all-zero direction.

A selected direction is scaled by the configured friendly-UAV maximum speed. World updates clip positions to the configured 3D bounds.

### 2.3 Scene Initialization

Scene initialization is scenario-dependent. The base scenario places non-key targets near key targets and friendlies near the key-target region. Other included scenarios can use explicit sampling boxes, uniformly distributed non-key targets, or partitioned motion. Therefore, proximity-based initialization is not a universal system assumption.

Relevant configuration is selected through `config/env.yaml` and resolved from `config/scenarios/*.yaml`.

## 3. Sensing Model

### 3.1 Weak Global Key-Target Priors

`GlobalKeyTargetStatePriorSensor` in `src/sensing/global_sensor.py` produces labeled 6D reports for key targets only. The model includes:

- a configurable reporting interval;
- non-negative Gaussian delivery latency;
- independent position and velocity noise;
- a temporally correlated common bias;
- per-key missed reports;
- a validity mask;
- diagonal reported covariance.

When a delayed report arrives, its reported position is propagated to the current time with its reported velocity. The position covariance is enlarged using the report age. If no report arrives at a step, the returned prior rows are invalid rather than being treated as current truth.

The row index is the persistent key identity used by the labeled tracking methods.

### 3.2 Local Anonymous Measurements

`LocalTargetSensor` in `src/sensing/local_sensor.py` produces a variable-length set of relative 6D candidates for each friendly UAV:

```text
z_ij = x_target_j - x_friendly_i + measurement noise
```

The local sensor:

- observes both key and non-key targets inside its detection radius;
- removes target IDs and key/non-key roles;
- applies range- and local-density-dependent detection probability;
- adds position and velocity measurement noise;
- draws the clutter count from a Poisson distribution;
- mixes uniform clutter with target-proximal hotspot clutter.

`LocalTargetObservation` contains `relative_states`, a validity `mask`, and the diagonal measurement covariance. The current sensor returns only valid rows, so the array length may differ across UAVs and time steps.

## 4. Local Association and Recursive Tracking

`SwarmEnv` supports two tracking input modes:

- `local_only`: run the configured local association method using weak priors and local measurements;
- `groundtruth`: construct key tracks from truth for debugging and reference comparisons.

Available `local_only` association methods are:

| Configuration value | Implemented behavior |
| --- | --- |
| `none` | nearest-local-candidate baseline |
| `prior_only` | labeled external reports plus motion prediction, without local candidates |
| `bp_jpda` | recursive BP-JPDA |
| `hard_anchor_bp_jpda` | BP-JPDA centered directly on the current labeled prior |
| `covariance_anchor_bp_jpda` | covariance-fused labeled anchor without fallback validity |
| `labeled_anchor_bp_jpda` | covariance-fused labeled anchor with fallback validity |
| `gm_phd` | per-UAV GM-PHD recursive baseline |
| `lmb` | labeled multi-Bernoulli-style recursive baseline |
| `delta_glmb` | compact labeled multi-hypothesis baseline |

Recursive estimators support either:

- `cv`: one constant-velocity model;
- `imm`: an interacting multiple-model bank. The default bank is `cv`, `ca`, `ct_left`, and `ct_right`.

The estimator motion model is independent of the ground-truth mobility model, allowing deliberate model mismatch. Tracks are removed after the configured number of consecutive unsupported steps.

Each per-UAV output is a `MultiTargetTrackOutput` containing one `KeyTrackPosterior` per key label, with prior and posterior relative state, covariance diagonal, confidence/existence, validity, and optional local-candidate association information.

## 5. Centralized Fusion

The fusion pipeline is implemented in `src/tracking/fusion_center.py` and `src/tracking/fusion.py`.

For every valid local posterior:

1. convert the relative state to the global frame by adding the friendly-UAV state;
2. add configured friendly-UAV localization uncertainty to position covariance;
3. group candidates by `global_target_index`;
4. discard candidates below `track_fusion.min_confidence`;
5. reject inconsistent candidates with a Mahalanobis-distance gate;
6. combine accepted estimates sequentially with covariance intersection.

Fusion is label-guided rather than a general unlabeled cross-UAV assignment: local tracks are first grouped by their persistent key index. If no valid candidate survives for a label, the fusion output retains an invalid prior-based fallback state.

The resulting `FusedTrack` stores the label, global mean and covariance, contributing UAVs, fused confidence/existence, and validity.

The fusion interface assumes synchronous, lossless availability of all local track summaries. It does not model packets, bandwidth, link delay, or distributed consensus.

## 6. Assignment and Motion Control

Task assignment and motion control are implemented, not placeholders in the default configuration.

- `tracking.assignment: hungarian` builds a distance cost matrix using one-step predicted fused target positions and adds a switching penalty based on the previous assignment.
- The implementation solves small assignments by exact permutation search and falls back to a greedy solver for larger or rectangular cases; `hungarian` is the configuration name, not a dependency on SciPy's Hungarian implementation.
- When there are more UAVs than tracks, remaining UAVs are assigned to their nearest track.
- `tracking.assignment: placeholder` returns no assignments.

`build_tracking_control_commands` chooses the discrete direction with the largest dot product toward the assigned valid fused track. An unassigned UAV, or a UAV whose assigned track is invalid/missing, receives the stop action.

## 7. Interference Quantities

`src/interference/channel.py` computes free-space-path-loss received-power snapshots at key targets and friendly UAVs. These values are stored in rollouts and exposed in environment information, but they are not currently part of the association likelihood, assignment cost, or control action.

## 8. Evaluation and Experiment Outputs

`src/tracking/metrics.py` implements and exports:

- per-key position error and valid-track position RMSE;
- coverage-penalized RMSE;
- OSPA and GOSPA;
- tracked-target count and ratio;
- cumulative and per-target ID switches;
- cumulative and per-target fragmentations.

Tracking runtime is measured as association time plus fusion time for each step.

Executable workflows:

- `python main.py`: scene and trajectory visualization only;
- `python -m src.runner.export_tracking_rollout --no-show`: one closed-loop tracking rollout;
- `python -m src.runner.run_tracking_monte_carlo --num-seeds N`: paired-seed method comparison;
- `python -m src.runner.run_nonkey_density_sweep --num-seeds N`: non-key/key density sweep for `labeled_anchor_bp_jpda`.

## 9. Current Modeling Boundaries

The current repository does not implement:

- reinforcement learning or any learned model;
- raw-measurement sharing between UAVs;
- a communication protocol or bandwidth model;
- continuous UAV control or transmit-power actions;
- a fully unlabeled cross-UAV track-association solver.

Within those boundaries, scene generation, sensing, local association, recursive tracking, centralized fusion, assignment, pursuit control, metrics, and experiment export are all executable.
