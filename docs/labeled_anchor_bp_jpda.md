# Labeled-Anchor BP-JPDA Design Notes

## Motivation

The baseline BP-JPDA keeps one recursive local track per key target and computes a
target-candidate likelihood matrix from Mahalanobis-gated Gaussian likelihoods. This works
well when the local recursive track remains close to the correct key target, but in clutter-heavy
scenes a key label can be pulled toward non-key candidates. Once that happens, the next BP-JPDA
step gates around the wrong local posterior and the error can persist.

The delta-GLMB baseline is better at preserving target identity because each target is explicitly
labeled, but it is more complex and produced weaker behavior for some other key targets in the
current scenarios. `labeled_anchor_bp_jpda` keeps the BP-JPDA association machinery and adds only
a light label anchor:

- key target `E1` keeps label `l1` through global target index `0`;
- key target `E2` keeps label `l2` through global target index `1`;
- key target `E3` keeps label `l3` through global target index `2`;
- BP-JPDA still estimates target-candidate association marginals, but each target's prediction is
  anchored to the current global prior for that same label.

The method does not solve a delta-GLMB-style label-to-measurement hypothesis problem. It relies on
the global prior's key-target order as the persistent identity source.

## Method Summary

For every friendly UAV and key target label, the method builds a labeled anchor prediction before
running BP-JPDA:

```text
previous local posterior -> IMM/CV motion prediction -> x_motion, P_motion
current global prior for the same label -> x_prior, P_prior
covariance-aware anchor fusion -> x_anchor, P_anchor
BP-JPDA gating/likelihood around x_anchor, P_anchor
```

If no previous local state exists, the current global prior initializes the local recursive state.
If the current global prior is unavailable, the method falls back to the recursive motion
prediction. This keeps the tracker usable under occasional global-prior misses.

## Covariance-Aware Anchor Fusion

When both the recursive motion prediction and the current global prior are available, the anchor is
formed by a covariance-aware Kalman/information update. Conceptually:

```text
P_anchor^-1 = P_motion^-1 + P_prior^-1
x_anchor = P_anchor * (P_motion^-1 x_motion + P_prior^-1 x_prior)
```

Equivalently, the implementation applies a linear Kalman update to each IMM motion-model branch,
using the global prior as a full-state measurement with covariance `P_prior`.

This is different from a fixed weighted average:

```text
x_anchor = alpha * x_prior + (1 - alpha) * x_motion
```

There is no manually tuned `alpha`. The covariance matrices determine the relative influence:

- smaller covariance means higher confidence and stronger influence;
- larger covariance means lower confidence and weaker influence;
- the fused covariance becomes the uncertainty used by the later likelihood and gate.

Compared with the earlier hard-anchor version, this avoids discarding temporal continuity. The
global prior still pulls each label back to its intended target, but the previous local trajectory
continues to contribute when it is statistically reliable.

## BP-JPDA Association Stage

After the labeled anchor is built, the method uses the existing BP-JPDA graph:

```text
residual_ij = z_j - x_anchor_i
S_ij = P_anchor_i + R_local
d2_ij = residual_ij^T S_ij^-1 residual_ij
```

Edges are created only for candidates inside the Mahalanobis gate:

```text
d2_ij <= gate_sigma^2
```

For gated edges, the likelihood ratio is the same BP-JPDA form used by the baseline:

```text
psi_ij = P_D * N(z_j ; x_anchor_i, S_ij) / clutter_intensity / P_missed
```

The loopy BP solver then computes approximate JPDA association marginals over the labeled
target-candidate graph. The downstream Gaussian mixture update is reused unchanged.

The key difference from baseline BP-JPDA is the prediction center. Baseline BP-JPDA gates around
the recursive local posterior, which can drift. `labeled_anchor_bp_jpda` gates around the
covariance-fused labeled anchor.

## Anchor Fallback Validity

In clutter-heavy scenes, requiring a local measurement association at every step creates
fragmented tracks. A key target can have a reasonable labeled anchor prediction but no confident
local candidate in one UAV's view. If that track is marked invalid, CI fusion and Hungarian
assignment may temporarily lose the target, causing the controller to stop pursuing it.

For this reason, `labeled_anchor_bp_jpda` enables an anchor fallback validity rule:

```text
if BP has measurement support:
    output the normal BP-JPDA posterior
else if the labeled anchor is active:
    output the anchor prediction as a low-confidence valid track
```

The fallback confidence is clamped to the configured local confidence threshold, so it can pass the
fusion `min_confidence` check without pretending to be high-confidence measurement evidence.

This rule is intentionally enabled only for `labeled_anchor_bp_jpda`. The ordinary `bp_jpda`,
`hard_anchor_bp_jpda`, and `covariance_anchor_bp_jpda` paths require local measurement support.

## Why Gating Still Remains

Covariance-aware fusion answers the question:

```text
Where should label i be now, and with what uncertainty?
```

Gating answers a different question:

```text
Which local candidates are statistically plausible for label i?
```

Without gating, all clutter candidates would still enter the association graph and dilute BP-JPDA
association probability. The fused anchor defines the center and covariance; gating uses that
distribution to keep the target-candidate graph sparse and physically plausible.

## Relationship to Other Association Methods

### Baseline BP-JPDA

Baseline BP-JPDA:

```text
previous posterior -> motion/prior prediction -> BP-JPDA
```

The global prior acts as a weak recursive correction through the existing anchor/birth mechanism.
If the previous posterior is already biased by clutter, the next gate can remain biased.

### Labeled-Anchor BP-JPDA

Labeled-anchor BP-JPDA:

```text
previous posterior -> motion prediction
current global prior for same key label
motion/prior covariance fusion -> BP-JPDA
anchor fallback valid output
```

The label identity comes from the global prior row index, while BP-JPDA still handles local
candidate competition.

### Delta-GLMB

Delta-GLMB explicitly maintains labeled multi-target hypotheses and measurement assignment
hypotheses. `labeled_anchor_bp_jpda` deliberately avoids that complexity. It borrows only the
practical idea that each key target should have a stable label identity.

## Implementation Points

The implementation lives in `src/sensing/association.py`:

- `_covariance_fuse_motion_state_with_anchor(...)` performs the covariance-aware anchor update for
  each IMM branch.
- `_bp_jpda_predict_track_state(..., covariance_anchor_prior=True)` selects covariance-aware
  anchoring for labeled-anchor mode.
- `_build_bp_jpda_context(..., anchor_to_global_prior=True)` builds BP-JPDA likelihoods around the
  labeled anchor.
- `_build_bp_jpda_tracks_from_marginals(..., allow_anchor_fallback_valid=True)` enables the
  anchor fallback validity rule.
- `labeled_anchor_bp_jpda_association_for_friendlies(...)` wires the above pieces into a separate
  association method.

The environment entry is registered as:

```yaml
tracking:
  association_method: labeled_anchor_bp_jpda
```

No neural network, supervised training, learned embedding, or large set of manual weights is
introduced.

## Observed Effect in the Clutter-Heavy Scenario

In `scenerios_many_nonkey_partitioned`, the earlier hard-anchor version reduced identity drift but
created intermittent invalid tracks. That produced high fragmentation and allowed the controller to
temporarily stop pursuing a key target.

After covariance-aware fusion and anchor fallback validity, the labeled-anchor method produced:

```text
mean_coverage = 100.0%
final_coverage = 3/3
id_switches = 0
fragmentations = 0
```

The E3 nearest-friendly-distance spike was also reduced in the tested rollout:

```text
hard anchor peak:              about 40.55 m
covariance-aware + fallback:   about 15.66 m
```

The tradeoff is that fallback-valid tracks may slightly increase RMSE when no local measurement
supports the anchor in that step. This is intentional: the method favors continuous labeled target
availability for downstream CI fusion, Hungarian assignment, and controller stability.
