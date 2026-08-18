# Labeled-Anchor BP-JPDA Design Notes

## 1. Implementation Status

`labeled_anchor_bp_jpda` is an implemented deterministic association and recursive-tracking method in `src/sensing/association.py`. It combines:

- persistent key identities from ordered global-prior rows;
- CV or IMM motion prediction;
- covariance-aware fusion of the motion prediction and current labeled prior;
- Mahalanobis-gated BP-JPDA over anonymous local candidates;
- an anchor-based fallback-validity rule.

It contains no neural network, learned embedding, supervised training, reinforcement learning, or trainable parameter.

The plotting code labels this method `IA-BP-JPDA`.

## 2. Motivation

The baseline `bp_jpda` maintains one recursive local state per key index. In clutter-heavy scenes, a local posterior can be pulled toward a non-key target or false alarm. Because the next prediction is based mainly on that recursive state, later gates may remain centered on the wrong object.

The external global sensor already preserves the semantic order of key targets. Labeled-anchor BP-JPDA uses that ordered prior as a persistent identity reference while retaining soft local candidate competition through BP-JPDA.

The method does not maintain delta-GLMB-style joint label-assignment hypotheses. Label identity is inherited from `global_target_index`, i.e. global-prior row `k` corresponds to local key track `k` and fused label `E{k+1}`.

## 3. Inputs and Per-UAV State

For every friendly UAV, the method consumes:

- the UAV's 6D global state;
- the current `GlobalKeyTargetStatePrior`;
- the current variable-length `LocalTargetObservation`;
- the previous `BPJPDAFilterState` for each key index;
- sensing, clutter, gating, recursion, and IMM parameters.

A local recursive state contains a mean, full covariance, optional per-model IMM bank, and consecutive missed-step count. The exported `KeyTrackPosterior` contains the prior/posterior relative states, covariance diagonals, confidence/existence, validity, and candidate-association information.

## 4. Labeled Prediction

For key index `k`, the global prior is first converted to the UAV-relative frame. The prediction logic is:

```text
previous local state
    → CV/IMM propagation
    → motion prediction (x_motion, P_motion)

current valid global-prior row k
    → relative labeled anchor (x_prior, P_prior)

motion prediction + labeled anchor
    → covariance-aware anchor (x_anchor, P_anchor)
```

The edge cases implemented by `_bp_jpda_predict_track_state` are:

- no previous state + valid prior: initialize the recursive state from the prior;
- previous state + missing prior: use motion prediction only;
- no previous state + missing prior: mark the track inactive in the BP context;
- expired previous state: treat it as absent.

### 4.1 Covariance-Aware Fusion

For each IMM branch, the global prior is treated as a full-state identity observation. Conceptually, for independent Gaussian inputs:

$$
P_a^{-1} = P_m^{-1} + P_p^{-1},
$$

$$
x_a = P_a(P_m^{-1}x_m + P_p^{-1}x_p).
$$

The branch probabilities and Gaussian bank are then collapsed into the anchor mean and covariance used by BP-JPDA.

There is no fixed interpolation coefficient. A lower covariance gives the corresponding input more influence.

## 5. Gating and BP-JPDA Likelihood

For target `k` and local candidate `j`, the implementation uses the anchor-centered residual

$$
r_{kj} = z_j - x_{a,k}.
$$

The code forms a diagonal innovation covariance from the anchor covariance diagonal and local measurement covariance:

$$
S_{k} = \operatorname{diag}(P_{a,k}) + R.
$$

A candidate edge is retained when

$$
d^2_{kj} = r_{kj}^T S_k^{-1}r_{kj} \leq \gamma^2,
$$

where `gamma` is `association.bp_jpda_gate_sigma`.

For retained edges, the likelihood ratio has the implemented form

$$
\psi_{kj} = \frac{P_D\,\mathcal{N}(r_{kj};0,S_k)}{\lambda_c(z_j)\,P_{miss}}.
$$

`P_D` depends on range and local target density. `P_miss` is lower-bounded by the configured minimum missed-detection probability. The clutter intensity `lambda_c` uses the same uniform/hotspot mixture family as the local sensor rather than a single uniform constant.

The loopy BP routine `_run_bp_jpda` iterates until the configured tolerance or iteration limit and returns approximate target-candidate marginals.

## 6. Posterior Update

For each target, `_build_bp_jpda_tracks_from_marginals` builds a Gaussian mixture containing:

- a missed-detection branch based on the predicted anchor state;
- one measurement-updated branch for every candidate with positive BP marginal.

The branches are moment-matched back into the recursive CV/IMM state. Local confidence is the clipped sum of candidate marginals, and the current implementation sets existence probability to the same output confidence.

A normal BP-supported track is valid only when:

- the track is active;
- association confidence reaches the configured confidence threshold;
- at least one candidate has positive association support.

## 7. Anchor Fallback Validity

`labeled_anchor_bp_jpda` additionally calls the posterior builder with `allow_anchor_fallback_valid=True`. A track can therefore remain valid without local measurement support when either:

- its predicted track remains active; or
- the current labeled global report is valid.

For a fallback-valid output:

- posterior state remains the anchor/motion prediction when no candidate contributes;
- output confidence is floored at the association confidence threshold;
- a valid current global report resets the missed-step counter;
- otherwise the missed-step counter advances and the recursive state is eventually removed at `association_track_max_missed_steps`.

The fusion layer applies its own independent `track_fusion.min_confidence` threshold, so fallback validity does not bypass fusion configuration.

This rule improves labeled-track availability for fusion, assignment, and pursuit control, but it can retain a biased prior-supported state when local evidence is absent. Coverage and position accuracy must therefore be reported together.

## 8. Relationship to the Other BP-JPDA Modes

| Method | Prediction center | Requires local measurement support | Anchor fallback |
| --- | --- | --- | --- |
| `bp_jpda` | recursive prediction with the baseline anchor/birth correction | yes | no |
| `hard_anchor_bp_jpda` | current valid labeled prior replaces the recursive prediction | yes | no |
| `covariance_anchor_bp_jpda` | covariance-fused motion prediction and labeled prior | yes | no |
| `labeled_anchor_bp_jpda` | covariance-fused motion prediction and labeled prior | no, when fallback conditions hold | yes |

All four modes reuse the BP graph, gating, clutter model, marginal solver, and Gaussian-mixture posterior machinery. Their main differences are how the prediction is anchored and whether an unsupported anchor may still be exported as valid.

## 9. Central Fusion Interaction

Each local labeled posterior is transformed to the global frame by adding the friendly-UAV state and localization covariance. Fusion then:

1. groups tracks by the same `global_target_index`;
2. filters by confidence;
3. rejects Mahalanobis-inconsistent contributors;
4. combines accepted estimates by covariance intersection.

Stable label grouping is therefore supplied before CI fusion; the fusion center does not rediscover labels from unlabeled local tracks.

## 10. Implementation Map

The relevant functions in `src/sensing/association.py` are:

- `_covariance_fuse_motion_state_with_anchor`: fuses the prior into each motion-model branch;
- `_bp_jpda_predict_track_state`: handles initialization, propagation, hard anchoring, and covariance anchoring;
- `_build_bp_jpda_context`: builds active tracks, gates, clutter intensities, and likelihood ratios;
- `_run_bp_jpda`: computes approximate association marginals;
- `_build_bp_jpda_tracks_from_marginals`: performs mixture update, validity, confidence, and missed-count handling;
- `_anchored_bp_jpda_association_for_friendlies`: common driver for the three anchored variants;
- `labeled_anchor_bp_jpda_association_for_friendlies`: enables covariance anchoring and fallback validity.

Enable the method in `config/env.yaml`:

```yaml
tracking:
  mode: local_only
  association_method: labeled_anchor_bp_jpda
```

Key parameters are under `association` and `track_fusion` in the same file.

## 11. Evaluation Guidance and Limitations

Compare association methods with the same scene configuration and paired random seeds. Report at least:

- GOSPA or OSPA;
- valid-only and coverage-penalized position error;
- tracked-target coverage;
- ID switches;
- fragmentations;
- association-plus-fusion runtime.

Do not interpret fallback coverage alone as evidence of accurate local measurement tracking.

Current limitations:

- correctness of key identity depends on stable global-prior row ordering;
- anchor fusion assumes the motion prediction and global prior can be combined as modeled;
- local gating uses diagonal innovation covariance;
- fallback validity can trade position accuracy for continuity;
- the method is not learned and does not adapt parameters from data.

No fixed performance numbers are part of this design specification; results depend on scenario, seed, and configuration and should be generated through the repository's rollout or Monte Carlo runners.
