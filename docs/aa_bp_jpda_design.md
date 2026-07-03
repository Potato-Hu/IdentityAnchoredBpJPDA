# AA-BP-JPDA Design Notes

## Motivation

The baseline BP-JPDA builds a target-candidate likelihood matrix from gated Gaussian
measurement likelihoods, then uses belief propagation to approximate the JPDA marginal
association probabilities. This is efficient compared with exact JPDA, but in clutter-heavy
multi-UAV scenes the local Gaussian likelihood can still be ambiguous: multiple non-key
targets may be close to a key-target prediction in one UAV's local view.

AA-BP-JPDA keeps the BP-JPDA recursion unchanged and modifies only the association evidence
before BP message passing. The method reweights the baseline likelihood matrix by combining
local target-candidate attention with cross-UAV consensus evidence:

```text
psi_ij' = psi_ij * exp(clip(s_local_ij + s_consensus_ij))
```

The original gated BP-JPDA likelihood `psi_ij` remains the support of the sparse graph. If
`psi_ij = 0`, AA-BP-JPDA does not create a new association edge.

## Stage 1: Target-Candidate Local Attention

For each UAV and each key target, AA-BP-JPDA gathers the candidates that passed the baseline
Mahalanobis gate. These candidates form a small target-specific set. The local attention score
is computed from physical pair features:

- Mahalanobis residual between the predicted target state and local candidate.
- Candidate competition under a softmax over the gated candidates of the same target.
- Velocity direction consistency between the target prediction and candidate velocity.
- Local candidate density penalty, used to downweight crowded clutter regions.

This stage uses the useful property of attention, namely parallel competition over a candidate
set, without importing the unnecessary parts of a standard Transformer. There is no positional
encoding, no encoder stack, no generated hidden state, and no autoregressive computation.

## Stage 2: Cross-UAV Consensus Attention

Each local candidate is lifted into the global frame by adding the observing UAV state. For a
given key target, candidates from other UAVs that also passed the baseline gate provide support
when they are consistent in global position and velocity.

The cross-UAV support uses a kernel over pairwise candidate distances:

```text
support = exp(-0.5 * position_distance^2 / sigma_pos^2
              -0.5 * velocity_distance^2 / sigma_vel^2)
```

For each candidate, the method keeps the strongest support from every other UAV and also counts
how many UAVs provide support above the configured support threshold. The consensus term rewards
candidates that are independently supported by other local views.

## Ambiguity-Gated Consensus

Cross-UAV consensus is only useful when a target has ambiguous local candidates. AA-BP-JPDA
therefore computes the normalized entropy of the local attention distribution:

```text
H_i = -sum_j a_ij log(a_ij) / log(|G_i|)
```

where `G_i` is the gated candidate set for target `i`. A single-candidate target has zero
ambiguity, so cross-UAV consensus is skipped. For multi-candidate targets, the consensus score is
scaled continuously by `H_i`. This avoids adding another hard ambiguity threshold while still
allocating cross-UAV evidence to the cases where the local association is uncertain.

## Computational Mechanism

AA-BP-JPDA is designed to keep the sparse BP-JPDA graph and add attention-style parallel evidence
computation on top of it:

- The baseline Mahalanobis gate defines sparse target-candidate edges.
- Local attention is computed per target over the gated candidate set as a vector operation.
- Cross-UAV consensus is computed as pairwise support matrices between candidate batches, rather
  than as a sequential candidate loop.
- Consensus is skipped for non-ambiguous single-candidate target sets and continuously reduced for
  low-entropy target sets.

This is analogous in spirit to the efficiency advantage of attention over recurrent sequence
processing: the method evaluates candidate-set compatibility in parallel, while BP still provides
the approximate marginal inference that makes JPDA tractable.

## Current Parameters

```yaml
aa_bp_jpda_local_attention_temperature: 2.0
aa_bp_jpda_cross_uav_position_scale_m: 8.0
aa_bp_jpda_cross_uav_velocity_scale_mps: 3.0
aa_bp_jpda_cross_uav_support_threshold: 0.35
aa_bp_jpda_local_evidence_weight: 0.6
aa_bp_jpda_velocity_evidence_weight: 0.25
aa_bp_jpda_density_penalty_weight: 0.35
aa_bp_jpda_consensus_evidence_weight: 0.9
aa_bp_jpda_consensus_count_weight: 0.45
aa_bp_jpda_max_log_evidence: 1.5
```

These parameters control evidence scaling, not the existence of the sparse association graph. The
graph itself still comes from the baseline BP-JPDA gate.

## Sinkhorn Attention JPDA Approximation

`sinkhorn_ajpda` keeps the same local attention and cross-UAV consensus evidence module, but
replaces BP message passing with Sinkhorn normalization on an augmented assignment matrix. It is
intended as a JPDA-style approximation, not as an exact JPDA marginalization.

For each UAV, the method first builds the same enhanced likelihood matrix used by AA-BP-JPDA:

```text
psi_ij' = psi_ij * exp(clip(s_local_ij + s_consensus_ij))
```

It then builds an augmented matrix:

```text
rows    = key targets + clutter dummy row
columns = local candidates + missed-detection dummy column
```

The real target-candidate block uses `log(psi_ij')`. The target-to-missed-detection dummy and
clutter-to-candidate dummy entries use a neutral score of zero because `psi_ij` is already a
likelihood ratio relative to missed detection and clutter. Invalid gated-out target-candidate
edges remain near zero support and are zeroed after normalization.

The row and column masses are:

```text
target row mass = 1 for each target
clutter row mass = number of candidates
candidate column mass = 1 for each candidate
missed-detection column mass = number of targets
```

This lets each target softly choose one candidate or the missed-detection dummy, and lets each
candidate softly choose one target or the clutter dummy. The dustbin row/column absorb unmatched
mass, making the assignment constraints closer to multi-target tracking than a plain row-wise
attention softmax.

The resulting top-left block is used as the approximate target-candidate marginal matrix for the
same Gaussian mixture update used by BP-JPDA:

```text
beta_ij ~= Sinkhorn(log(psi_ij'))_ij
```

Compared with BP message passing, Sinkhorn is a matrix-scaling approximation to the assignment
constraints. Its main computational appeal is that the normalization is parallel over rows and
columns and can be implemented as repeated matrix operations, while still enforcing soft
target-measurement competition.

Current extra parameters:

```yaml
sinkhorn_ajpda_iterations: 30
sinkhorn_ajpda_temperature: 1.0
sinkhorn_ajpda_missed_dummy_score: -0.5
```

`sinkhorn_ajpda_missed_dummy_score` controls the target-to-missed-detection dummy score.
Negative values make missed detection less attractive, which is useful when detection
probability is high and the tracker should prefer valid key-target candidates over the missed
branch.
