# Problem Formulation

## 1. Task Definition

SwarmJam formulates a **centralized-fusion, key-target multi-target tracking problem**. It is not formulated as a multi-agent reinforcement-learning problem.

Let:

- `F = {1, ..., N_f}` be the friendly UAVs;
- `K = {1, ..., N_k}` be the labeled key targets;
- `D = {1, ..., N_d}` be the non-key distractor targets.

Every object has a 6D state

$$
x(t) = [p_x, p_y, p_z, v_x, v_y, v_z]^T.
$$

The objective is to maintain accurate, available, and identity-consistent posterior tracks for the labels in `K`, even though each UAV observes only noisy anonymous local candidates and targets in `D` create ambiguity.

The current code evaluates this objective with tracking metrics; it does not optimize a learned policy or a trainable loss.

## 2. Ground-Truth Process

Enemy motion is generated independently of the tracker. Scenarios select Gauss-Markov or random-walk motion and may add companion-heading or spatial-partition constraints for non-key targets.

This separation is important: the recursive tracker may use a CV or IMM kinematic approximation even when the truth process follows Gauss-Markov motion. Tracking therefore includes configurable model mismatch.

Friendly UAVs evolve inside the same 3D bounds using one stop action and 26 discrete movement directions.

## 3. Observation Model

### 3.1 Delayed Labeled Global Reports

At a scheduled sampling time `s`, the external source forms a report for key target `k`:

$$
r_k(s) = x_k(s) + b(s) + \epsilon_k(s),
$$

where `b(s)` is a temporally correlated common bias and `epsilon_k(s)` is independent Gaussian position/velocity noise. A Bernoulli missed-report event determines the report validity for each key label.

The report arrives at

$$
a = s + \max(\mathcal{N}(\mu_L, \sigma_L^2), 0).
$$

When the newest arrived report is read at time `t`, the implementation propagates its position with the reported velocity:

$$
\hat p_k(t) = r_{p,k}(s) + r_{v,k}(s)(t-s).
$$

The diagonal position covariance is enlarged as

$$
P_p(t) = P_p(s) + P_v(s)(t-s)^2.
$$

These reports preserve a stable row-to-label mapping, but they are weak priors rather than ground truth.

### 3.2 Per-UAV Anonymous Local Candidate Set

Friendly UAV `i` receives a variable-length set

$$
Z_i(t) = \{z_i^1(t), \ldots, z_i^{m_i(t)}(t)\}.
$$

A detected target candidate follows

$$
z_i^j(t) = x_j(t) - x_i^F(t) + \nu_i^j(t),
$$

with diagonal position/velocity measurement covariance. Candidate rows contain neither target IDs nor key/non-key roles.

The local sensor model includes:

- a finite omnidirectional detection radius;
- detection probability that decreases with range and local target density;
- missed detections;
- Poisson clutter per time step;
- a mixture of uniform-volume and target-proximal hotspot clutter.

Consequently, observations from key targets, non-key targets, and false alarms compete in the same local candidate set.

## 4. Local Key-Track Estimation

For each friendly UAV, the local layer estimates a posterior for every persistent key index:

$$
\mathcal{T}_i(t) = \{(\hat x_{i,k}, P_{i,k}, c_{i,k}, e_{i,k}, v_{i,k}) : k \in K\},
$$

where `c` is confidence, `e` is existence probability, and `v` is the valid flag.

The current implementation provides the following alternatives:

- nearest-local-candidate (`none`);
- global reports and motion prediction only (`prior_only`);
- recursive `bp_jpda`;
- `hard_anchor_bp_jpda`, `covariance_anchor_bp_jpda`, and `labeled_anchor_bp_jpda`;
- `gm_phd`;
- `lmb`;
- `delta_glmb`.

Recursive methods use either one CV model or an IMM bank. The default IMM contains CV, constant acceleration, and fixed-rate left/right coordinated-turn branches. Filter state expires after the configured number of consecutive unsupported steps.

`labeled_anchor_bp_jpda` is the project's implemented key-priority method. It uses the global-prior row index as identity, fuses a current labeled prior with the recursive motion prediction, and performs BP-JPDA over anonymous local candidates. It is a deterministic probabilistic tracker, not a learned scoring model. See `docs/labeled_anchor_bp_jpda.md`.

## 5. Centralized Fusion

Each UAV exports local key-track summaries rather than raw measurements. A local relative posterior is lifted into the global frame:

$$
\hat x^G_{i,k} = x_i^F + \hat x_{i,k}.
$$

Configured friendly-UAV localization variance is added to its position covariance. Candidates are then:

1. grouped by their existing key index `k`;
2. filtered by confidence;
3. checked against the current fused state with a Mahalanobis gate;
4. fused sequentially by covariance intersection.

For two accepted estimates, the implementation searches a finite grid of `w` values and uses

$$
P^{-1}(w) = wP_1^{-1} + (1-w)P_2^{-1},
$$

$$
\hat x(w) = P(w)\left[wP_1^{-1}\hat x_1 + (1-w)P_2^{-1}\hat x_2\right],
$$

choosing the candidate with minimum covariance trace.

This is label-guided fusion. The current code does not solve a fully unlabeled track-to-track assignment problem across UAVs.

## 6. Assignment and Closed-Loop Motion

After fusion, the default `hungarian` assignment mode constructs a cost from the distance between each UAV and a one-step predicted fused target position. A switch penalty discourages changing the previous target assignment.

Despite the configuration name, the current solver uses exact permutation search for small cases and a greedy fallback for larger or rectangular cases. Surplus UAVs are assigned to their nearest target.

For an assigned valid track, the controller chooses the discrete action direction with the greatest alignment to the target displacement. Missing or invalid assignments produce the stop action.

Thus the executable experiment is a closed tracking-and-pursuit loop, but it contains no learned controller.

## 7. Outputs and Evaluation

The main output is a `FusedTrackSet` with one labeled `FusedTrack` per available key index. Each fused track contains:

- global state mean and covariance;
- stable key label/index;
- contributing UAV IDs;
- fused confidence and existence;
- validity.

Key-target evaluation includes:

- per-target position error;
- valid-only position RMSE;
- coverage-penalized RMSE;
- OSPA;
- GOSPA;
- tracked-target count and ratio;
- ID switches;
- track fragmentations;
- association-plus-fusion runtime.

Monte Carlo comparisons use common seeds across association methods. The non-key-density sweep reports GOSPA and ID-switch behavior as the distractor/key ratio changes.

## 8. Scope Boundaries

The implemented problem assumes:

- an ideal synchronous central fusion interface;
- persistent key identities supplied by the ordered global-report rows;
- known friendly-UAV states with configurable localization uncertainty during fusion;
- discrete kinematic pursuit control.

The current scope excludes:

- reinforcement learning, supervised learning, and neural association scoring;
- communication protocols, packet loss, bandwidth limits, and asynchronous fusion;
- continuous control or power-control actions;
- equal-priority estimation of all non-key targets.

In concise form, SwarmJam studies **labeled key-target tracking from weak global priors and anonymous local measurements, with classical recursive association, centralized covariance-intersection fusion, and rule-based cooperative pursuit**.
