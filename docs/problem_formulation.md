# Problem Formulation

## 1. Task Definition

The project is formulated as a **centralized-fusion key-target multi-target tracking** problem rather than a multi-agent reinforcement learning task.

The practical objective is:

> Maintain accurate, continuous, and identity-consistent estimates of multiple key targets when each friendly UAV only observes noisy local anonymous candidates and non-key targets create clutter and association ambiguity.

The intended formulation emphasizes:

- weak external priors on key targets
- local anonymous observations at each friendly UAV
- key-target-priority-aware association
- learned scoring for classical association backends
- centralized fusion of per-UAV local track summaries
- key-target-oriented evaluation metrics

---

## 2. Core Assumptions

### 2.1 Target Types

The environment contains:

- `key targets`: the targets we care about maintaining
- `non-key targets`: distractors that may move near key targets and generate association confusion

The task is not to estimate all targets equally well. It is to **preferentially protect the quality of the key-target tracks**.

### 2.2 Weak External Key Priors

An external sensing source provides weak 6D priors for key targets:

$$
\hat{x}^{prior}_k(t) = x_k(t) + \epsilon_k(t)
$$

where the covariance of `\epsilon_k(t)` is intentionally large enough that the prior serves only as coarse guidance.

These priors are used to:

- initialize key-track hypotheses
- support local association
- stabilize key-target identity over time

They are not treated as exact ground truth and should not directly solve target assignment for each UAV.

### 2.3 Local Anonymous Observations

Each friendly UAV `i` receives only a local observation set

$$
Z_i(t) = \{z_i^1(t), z_i^2(t), \dots, z_i^{m_i}(t)\}
$$

with the following properties:

- candidates are anonymous
- observations are noisy
- detections may be missed
- clutter / false alarms may appear
- candidates may come from either key or non-key targets

Therefore, the task is partially observed at the sensor level even though the final objective is estimation rather than policy learning.

### 2.4 Central Fusion Assumption

Friendly UAVs do not exchange raw local measurements in the target formulation. Instead, each UAV exposes only its **local key-track summary** to an ideal centralized controller.

The centralized fusion layer therefore solves:

- which local tracks from different UAVs correspond to the same key target
- how to fuse inconsistent local estimates
- how to maintain key-target identity consistency across UAVs

The repository intentionally does **not** model a communication stack here. There is no packet protocol, bandwidth constraint, asynchronous delivery, or wireless-link simulation in the current scope. The abstraction is simply that all local track summaries are available to the fusion center each step.

### 2.5 Motion and Action Space

Friendly UAVs still use the existing discrete 3D motion action space:

- one `stop` action
- 26 normalized 3D movement directions

This action space is retained for later task-assignment and tracking-control modules, but policy learning is not the primary focus of the current formulation.

---

## 3. Estimation and Fusion Structure

The problem is decomposed into three layers.

### 3.1 Local Association and State Estimation

Each UAV maintains local key-track hypotheses and updates them from:

- weak external key priors
- previous local track predictions
- current local anonymous candidate observations

The main baseline algorithms belong here, such as:

- nearest-neighbor style association
- JPDA-style association
- GM-PHD-style recursive estimation

For recursive local-track maintenance, the repository now supports two motion-prior settings:

- `cv`: a single constant-velocity model
- `imm`: an interacting multiple-model predictor that defaults to a bank of `cv`, `ca`, `ct_left`, and `ct_right`

This local recursive motion prior is an estimator-side approximation. It should be interpreted separately from the
ground-truth scene mobility, which may remain more maneuvering or partially mismatched.

The research direction of interest is **priority-aware association**, where key-target continuity and identity preservation influence association strength.

More specifically, the intended novelty is not an end-to-end replacement of Bayesian tracking. The target direction is a
**learned association-scoring layer** that consumes weak priors, local anonymous candidates, previous track state, and
uncertainty, then outputs:

- track-candidate association scores
- missed-detection probability corrections
- existence-probability corrections

These learned quantities are then used inside a classical recursive backend such as `JPDA` or `LMB`.

### 3.2 Centralized Track Fusion

Each UAV exports local key-track summaries instead of raw measurements. A central controller then associates and fuses these summaries across UAVs.

The research direction of interest is **collaborative key-target tracking with centralized fusion**, where multiple UAVs cooperate through:

- local track maintenance
- per-UAV summary reporting to the fusion center
- track-to-track association
- centralized fusion

### 3.3 Task Assignment and Control

After fused key-target tracks are available, a downstream module can:

- assign key targets to UAVs
- choose motion actions that preserve or improve future tracking quality

These modules are currently placeholders in the revised architecture and are not yet the source of algorithmic novelty.

---

## 4. Outputs of Interest

The estimation stack should output a **set of key-target posterior tracks** rather than a single per-agent target estimate.

Each key-target track should provide at least:

- state mean
- state covariance
- existence / confidence score
- key label or key-target identity hypothesis
- continuity / identity-consistency metadata

These outputs are the natural inputs to:

- centralized fusion
- downstream task assignment
- downstream motion control

---

## 5. Evaluation Metrics

The primary evaluation metrics are:

- `key-target OSPA`
- `key-target RMSE`

Recommended secondary diagnostics are:

- key-target track continuity
- key-target identity-switch count
- key-target fragmentation count

The current repository already supports `key-target RMSE` and `key-target OSPA`. For the proposed learned-association
direction, the identity-switch and fragmentation diagnostics should be elevated from optional reporting to primary
evaluation quantities.

The metrics are intentionally key-target-centric rather than symmetric over all targets.

---

## 6. Summary

The project is best described as:

- a **key-target-oriented multi-target tracking problem with centralized fusion**
- with **weak external key priors**
- under **local anonymous partial observations**
- in the presence of **non-key distractor targets**
- solved through **local association, local track maintenance, and centralized track fusion**
- and improved through **learned scoring layered onto classical association backends**

This differs from the earlier MAPPO-centered formulation in two important ways:

1. the core problem is estimation and fusion, not policy optimization
2. the main outputs are multi-key-target track posteriors, not per-agent RL observations
