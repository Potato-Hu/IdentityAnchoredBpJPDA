# System Model

## 1. Current Scope

The current project implements a **multi-UAV cooperative jamming and tracking training environment** with the following end-to-end pipeline:

1. Ground-truth scene generation and motion update
2. Delayed/noisy global sensing of **key enemies only**
3. Per-UAV local sensing of nearby **anonymous target candidates**
4. Local association between global key priors and local candidates
5. MAPPO training on policy observations built from local sensing
6. Interference computation and reward feedback

The code is no longer based on the earlier design where global observations were fed directly into policy training. In the current implementation:

- **Global sensing is used only as an association prior**
- **Local observations are the main policy input in the default training mode**
- **Non-key enemies remain in the scene mainly as clutter / distractors**

---

## 2. Entities

### 2.1 Friendly UAVs

- Count: `N_f`
- State: 3D position and internally maintained velocity direction
- Control: **discrete motion action**

The implemented action space is not continuous velocity control. Each UAV selects one action from a fixed discrete direction set:

- one `stop` action
- 26 normalized 3D movement directions

The chosen direction is scaled by the UAV maximum speed `vmax`.

### 2.2 Enemy Nodes

Enemy nodes are divided into:

- **key enemies**
- **non-key enemies**

Total count:

$$
N_e = N_k + N_c
$$

In the current code:

- key enemies are the targets relevant to global sensing, association, interference statistics, and rewards
- non-key enemies are present in local sensing and scene dynamics, but are **not directly optimized in the current reward function**

---

## 3. Dynamics and World State

The runtime world state is maintained by `src/env/world.py`.

- Enemy nodes move according to the configured mobility model in `config/env.yaml`
- Friendly UAVs move according to discrete actions and fixed speed
- Positions are clipped to 3D map bounds
- One environment step advances both enemy motion and friendly motion, then recomputes interference

The physical step size is controlled by:

$$
\Delta t = \texttt{simulation.dt}
$$

---

## 4. Sensing Model

### 4.1 Global Sensor

Implemented in `src/sensing/global_sensor.py`.

Its role is:

- observe **key enemy positions only**
- apply configurable radar delay
- add Gaussian position noise

So the returned global prior is better understood as:

$$
z_k^{global}(t) = x_k(t-\tau) + \epsilon_k
$$

where:

- `\tau` is the configured radar delay
- `\epsilon_k` is Gaussian noise

Important implementation note:

- these global observations are **not directly concatenated into actor observations**
- they are used by the association module as prior information

### 4.2 Local Sensor

Implemented in `src/sensing/local_sensor.py`.

Each friendly UAV receives a fixed-length local observation containing:

- relative positions of nearby detected candidates
- a binary mask indicating which candidate slots are valid

Current local sensing properties:

- limited detection radius
- probabilistic missed detections
- additive local position noise
- Poisson clutter generation
- no target id
- no key/non-key label

This means the policy sees only **anonymous local candidates**, not labeled enemy tracks.

---

## 5. Association Model

Implemented in `src/sensing/association.py`.

The project currently supports three association modes:

- `none`
- `nn`
- `bp_jpda`

### 5.1 `none`

No global prior is used. Each UAV simply selects its nearest valid local candidate.

### 5.2 `nn`

Nearest-neighbor matching is performed between delayed/noisy global key priors and local candidates.

### 5.3 `bp_jpda`

This is the current approximate JPDA-style implementation:

- each UAV solves its own local association problem
- global key priors act as hypothesized targets
- local anonymous candidates act as measurements
- gating uses fused uncertainty from local noise, global noise, and delay-induced motion uncertainty
- loopy belief propagation approximates marginal association probabilities

The key output used by policy construction is **not a global target state estimate**, but a fixed-length weight vector over local candidates.

That is:

- local candidate positions come from the local sensor
- association contributes a per-candidate confidence / weight
- the policy still consumes a local-candidate representation

---

## 6. Policy Observation Construction

Implemented mainly in `src/env/swarm_env.py`.

### 6.1 Default Mode: `local_only`

This is the current training baseline.

Per-agent observation consists of:

- normalized self position
- normalized relative positions of local candidates
- local candidate validity mask
- association weight vector aligned with the candidate slots

Therefore, the default policy input is:

$$
o_i = [\text{self},\ \text{local candidates},\ \text{mask},\ \text{association weights}]
$$

This is the most important design change relative to the early documents:

- **global sensing is not directly used for training inputs**
- **local candidate observations are directly used for training**
- **association only injects soft guidance into those local candidate slots**

### 6.2 Debug Mode: `groundtruth`

The project also keeps a `groundtruth` mode for debugging.

In that mode, actor/critic inputs can use world-truth key-target features and related auxiliary information. This mode is for engineering comparison and debugging, not the main intended partial-observation setting.

---

## 7. Critic State

The project uses MAPPO with a centralized critic.

In the current implementation:

- under `local_only`, the critic state is the concatenation of all agents' local policy observations
- under `groundtruth`, the critic state contains normalized key-target positions, friendly positions, and interference-related quantities

So the critic is centralized, but in the default mode it is still built from **stacked local observations**, not full omniscient world truth.

---

## 8. Interference Model

Interference is computed in `src/interference/channel.py` and used through the world snapshot.

The current environment explicitly computes:

- received jamming power at **key enemies**
- received interference at **friendly UAVs**

These quantities are used by reward computation and logging.

At the current stage, non-key enemy interference is not the main optimization target in reward design, even though non-key enemies remain part of the sensed scene.

---

## 9. Reward-Relevant Quantities

The current implementation supports two reward modes:

- `difference`
- `sustained_tracking`

### 9.1 Difference Reward

This reward is based on:

- marginal contribution of each friendly UAV to key-target jamming utility
- friendly safety penalty

The target utility uses a logarithmic saturation form:

$$
U = \frac{1}{N_k}\sum_k \log\left(1 + \frac{J_k}{J_0}\right)
$$

Each agent receives its counterfactual difference reward minus ally interference penalty.

### 9.2 Sustained Tracking Reward

This mode adds:

- dynamic key-target assignment
- assigned-target received power reward
- distance progress reward
- movement penalty
- friendly safety penalty

This mode is more aligned with "keep tracking and suppressing assigned key targets" than with the original coarse reward sketch.

---

## 10. Summary of What Changed from the Original Concept

Compared with the original design notes, the current codebase has the following important differences:

1. Global observations are **not** used directly as policy inputs in the default setting.
2. Local anonymous observations are the core actor inputs.
3. Association outputs are encoded as **weights over local candidate slots**, not as direct fused target states for policy consumption.
4. The action space is **discrete 3D motion**, not continuous control.
5. The implemented reward currently focuses on **key-target jamming utility and friendly safety**, rather than an explicit `key - nonkey - friendly` full decomposition.
6. The centralized critic in default mode uses concatenated local observations rather than full ground-truth state.
