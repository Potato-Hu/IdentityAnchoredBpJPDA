# System Model

## 1. Scope

The current target architecture is a **centralized multi-UAV key-target tracking system** with the following end-to-end flow:

1. Ground-truth scene generation and motion update
2. Weak external sensing of key-target priors
3. Per-UAV local sensing of anonymous target candidates
4. Local association and local key-track maintenance
5. Central collection of per-UAV local key-track summaries
6. Centralized track-to-track association and fusion
7. Downstream task assignment and motion-control placeholders

The system is no longer defined around MAPPO as the main modeling choice.

---

## 2. Entities

### 2.1 Friendly UAVs

- Count: `N_f`
- State: 3D position and 3D velocity
- Control: discrete motion action

The retained action space is:

- one `stop` action
- 26 normalized 3D movement directions

This action space is preserved for later task-assignment and tracking-control work.

### 2.2 Enemy Targets

Targets are divided into:

- `key`
- `non-key`

Key targets are the objects of interest for tracking quality. Non-key targets exist mainly to increase ambiguity, clutter pressure, and identity confusion near key targets.

---

## 3. Scene Initialization

Implemented in `src/simulation/ground_truth.py`.

The intended initialization logic is:

- key targets are placed within the scene bounds
- non-key targets are generated near key targets so they act as realistic distractors
- friendly UAVs are generated near the overall key-target region so they begin close enough to observe targets while still having overlapping fields of view

This is intentionally different from a strict one-UAV-per-key-target initialization. The overlap is meant to support collaborative tracking rather than independent single-target pursuit.

---

## 4. Sensing Model

### 4.1 External Key Priors

Implemented in `src/sensing/global_sensor.py`.

The external sensing source provides weak 6D priors for key targets:

- position mean
- velocity mean
- diagonal covariance

These priors are noisy and should be interpreted as coarse key-target hypotheses rather than exact global truth.

Their role is to:

- seed or stabilize key-track hypotheses
- preserve key-target semantics
- support local association under ambiguity

### 4.2 Local Sensor

Implemented in `src/sensing/local_sensor.py`.

Each UAV receives a fixed-length local candidate set containing:

- relative 6D candidate states
- a validity mask
- local measurement covariance

Properties of the local sensing model:

- limited detection radius
- probabilistic missed detections
- additive measurement noise
- clutter generation
- no target id
- no explicit key / non-key label

Therefore, each UAV observes only anonymous local candidates.

---

## 5. Local Association and Local Tracking

Implemented mainly in `src/sensing/association.py`.

The local association layer is intended to consume:

- weak key-target priors
- current local candidate observations
- previous local track hypotheses

and produce:

- local key-target posterior track summaries

Supported baseline families include:

- nearest-candidate style matching
- JPDA-style soft association
- GM-PHD-style recursive estimation

For the recursive baselines, the local motion prior is now configurable:

- `cv`: single constant-velocity prediction
- `imm`: an interacting multiple-model predictor with a default bank of `cv`, `ca`, `ct_left`, and `ct_right`

This is intentionally separate from the ground-truth enemy mobility. The environment may evolve targets with a
Gauss-Markov motion process while the local recursive estimators propagate key-track hypotheses with an approximate
CV or IMM kinematic prior before association and measurement update.

The target research direction is not symmetric target association. It is **priority-aware association**, where:

- key-target continuity matters more than target symmetry
- identity consistency for key tracks is emphasized
- non-key targets are treated primarily as confusing distractors

The intended algorithmic extension is a **learned scoring front-end** for the classical recursive estimators already
present in the repository. Rather than replacing the full filter, the learned module should use temporal context, weak
global priors, local multi-candidate observations, and uncertainty to adjust:

- track-candidate association strengths
- missed-detection branches
- existence-confidence updates

The corrected scores then flow into a classical backend such as `JPDA` or `LMB`.

---

## 6. Central Fusion Layer

This layer is the core centralized extension of the local estimators.

Each UAV exports a local key-track summary to an ideal central fusion controller instead of broadcasting raw measurements to peers. The central layer then performs:

1. cross-UAV track-to-track association
2. track consistency checking
3. centralized fusion of matching key tracks

Expected outputs:

- fused key-target state means
- fused covariances
- confidence / existence scores
- identity-consistency metadata

This keeps the architecture compatible with standard tracking baselines while keeping the repository focused on estimation, fusion, and downstream assignment rather than communication-protocol implementation.

The current code models this as an **ideal centralized interface**:

- every friendly UAV's local track set is synchronously available to the fusion center
- no explicit packet format, bandwidth budget, or wireless-link simulator is modeled
- the fusion center returns one `FusedTrackSet` back to the rest of the system

---

## 7. Task Assignment and Motion Control

Task assignment and control are downstream modules and are not yet algorithmically filled in.

The intended interfaces are:

- **task assignment**: map fused key tracks to friendly UAV responsibilities
- **motion control**: map assigned key tracks to discrete UAV actions

The placeholders should preserve the current UAV action space and allow future replacement by:

- rule-based tracking control
- optimization-based assignment
- learning-based control if needed later

---

## 8. Evaluation Quantities

The primary quantities of interest are key-target-centric:

- key-target OSPA
- key-target RMSE

Recommended diagnostic quantities:

- key-track continuity
- identity-switch count
- missed-track duration

At present, the repository directly exports `key-target RMSE` and `key-target OSPA`. For the priority-aware learned
association direction, the continuity and identity diagnostics should become first-class reporting metrics rather than
only secondary summaries.

The system is therefore evaluated as a key-target tracking system, not primarily as a reward-optimization environment.

---

## 9. Repository Orientation

After the modeling shift, the repository should be interpreted as containing three categories of code:

1. **core tracking code**
   - scene generation
   - sensing
   - association
   - centralized fusion

2. **downstream placeholders**
   - task assignment
   - motion control

3. **tracking rollout and visualization utilities**
   - export scripts
   - plotting
   - diagnostic summaries
