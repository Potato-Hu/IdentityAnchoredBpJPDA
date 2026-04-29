# Problem Formulation

## 1. Task Definition

The current project formulates the problem as a **cooperative multi-agent reinforcement learning** task for multiple friendly UAVs.

The practical objective of the current implementation is:

> Learn cooperative motion policies that improve effective jamming on key enemies while maintaining safe behavior under partial local observations and uncertain target association.

This wording matches the current code better than the earlier formulation, because the present baseline is centered on:

- partial observability
- local candidate sensing
- key-target-focused reward
- optional global-to-local association assistance

---

## 2. Observation, State, and Action

### 2.1 Per-Agent Observation

In the default `local_only` mode, each UAV observes:

- its own normalized position
- relative positions of a fixed number of local target candidates
- a binary mask for valid candidate slots
- association weights aligned with those local candidate slots

So the observation is not:

- direct global radar observation
- explicit enemy identity
- explicit key/non-key labels
- clean fused target tracks

Instead, the policy must act from a partially observed local candidate set.

### 2.2 Global Information Usage

Delayed noisy global sensing of key enemies is available in the environment, but in the current implementation it is used only to support association.

That means:

- global sensing helps infer which local candidates may correspond to key targets
- global sensing is **not** directly fed into the actor in the default training mode

### 2.3 Critic State

The project uses a centralized critic.

Current critic-state construction depends on `policy_input.mode`:

- `local_only`: critic state is the concatenation of all agents' policy observations
- `groundtruth`: critic state uses debug-friendly world-truth target/friendly features and interference statistics

Therefore, the critic is centralized, but the main baseline does **not** assume full global truth during normal training.

### 2.4 Action Space

The implemented action for UAV `i` is a discrete motion command:

$$
a_i \in \mathcal{A}_{move}
$$

where `\mathcal{A}_{move}` contains:

- one stop action
- 26 normalized 3D movement directions

The action updates the UAV velocity direction at fixed magnitude determined by friendly `vmax`.

The current training pipeline does **not** optimize transmit power allocation as a separate action.

---

## 3. Association-Aware Partial Observation Formulation

The current training formulation is best described as:

1. the world contains key and non-key enemies
2. each UAV receives anonymous local candidate measurements
3. delayed/noisy global key priors may be matched to those local candidates
4. the policy consumes the local candidates plus an association-derived weight vector

This can be written conceptually as:

$$
o_i(t) = \phi\Big(x_i(t),\ \mathcal{Z}^{local}_i(t),\ m_i(t),\ w_i(t)\Big)
$$

where:

- `x_i(t)` is friendly UAV self state used in the observation
- `\mathcal{Z}^{local}_i(t)` is the fixed-length local candidate set
- `m_i(t)` is the candidate validity mask
- `w_i(t)` is the association weight vector

This is the key modeling choice in the current codebase.

---

## 4. Reward Function

The earlier simplified reward description

$$
r = R_{key} - R_{nonkey} - R_{friendly}
$$

is no longer an accurate description of the implemented baseline.

The current code supports two concrete reward modes.

### 4.1 Difference Reward

For each agent, the environment computes:

- global key-target jamming utility
- the counterfactual utility with that agent removed
- a friendly safety penalty

Conceptually:

$$
r_i^{diff} = \Big(U(J) - U(J^{-i})\Big) - \lambda_{safe} \cdot \max(0, I_i^{ally} - \tau_{ally})
$$

with:

$$
U(J) = \frac{1}{N_k}\sum_k \log\left(1 + \frac{J_k}{J_0}\right)
$$

This reward directly emphasizes marginal contribution to key-target suppression.

### 4.2 Sustained Tracking Reward

The alternative reward mode first assigns each friendly UAV to a key target, then combines:

- assigned-target power reward
- distance progress reward toward the assigned target
- movement penalty
- friendly safety penalty

Conceptually:

$$
r_i^{track} =
\alpha \log\left(1 + \frac{J_{i,\pi(i)}}{J_0}\right)
+ \beta \cdot \text{progress}_i
- \lambda_{move}\cdot \text{move}_i
- \lambda_{safe}\cdot \text{safety}_i
$$

where `\pi(i)` is the current assigned key target.

### 4.3 What Is Not Yet in the Reward

In the current implementation:

- non-key enemies are not directly penalized in the reward
- the reward is mainly built around key-target utility and ally safety

So non-key enemies currently matter more through **observation clutter and association difficulty** than through an explicit reward penalty term.

---

## 5. Policy Optimization

The project uses MAPPO with:

- decentralized actor
- centralized critic
- shared policy across agents

The training objective remains the standard discounted return maximization:

$$
\max_\theta \mathbb{E}\left[\sum_t \gamma^t r_t\right]
$$

But the practical implementation is specifically designed around:

- partial local observations
- centralized training with stacked agent observations
- optional association guidance

---

## 6. Main Challenges Reflected by the Current Code

1. **Partial observability**: each UAV only sees local anonymous candidates.
2. **Association uncertainty**: key-target priors are delayed/noisy and must be matched to local candidates.
3. **Clutter robustness**: non-key enemies and false alarms contaminate local sensing.
4. **Cooperative coordination**: multiple UAVs must avoid redundant behavior and improve total key-target suppression.
5. **Safety trade-off**: stronger jamming contribution must be balanced against friendly interference penalties.

---

## 7. Current Formulation Summary

The current project is more accurately summarized as:

- a **key-target-oriented cooperative MAPPO baseline**
- trained primarily from **local anonymous observations**
- optionally assisted by **association between global key priors and local candidates**
- evaluated with rewards based on **key-target jamming utility** and **friendly safety**

This is different from the original conceptual formulation in two important ways:

1. training does not directly rely on global observations in the default setting
2. reward design is currently focused on key-target effectiveness rather than a full explicit key/non-key/friendly decomposition
