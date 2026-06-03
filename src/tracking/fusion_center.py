from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from src.sensing.association import KeyTrackPosterior, MultiTargetTrackOutput
from src.tracking.fusion import FusedTrackSet, build_fused_track_set


@dataclass(frozen=True)
class LocalTrackSummary:
    """One friendly UAV's local key-track posterior exported to the fusion center."""

    agent_id: str
    timestamp: int
    global_target_index: int
    key_label: str
    prior_relative_state: np.ndarray
    prior_covariance_diagonal: np.ndarray
    posterior_relative_state: np.ndarray
    posterior_covariance_diagonal: np.ndarray
    confidence: float
    existence_probability: float
    valid: bool
    local_candidate_index: int
    innovation_distance: float
    local_candidate_weights: np.ndarray | None = None


@dataclass(frozen=True)
class LocalTrackSet:
    """All local key-track summaries reported by one friendly UAV at one step."""

    agent_id: str
    timestamp: int
    tracks: tuple[LocalTrackSummary, ...]


def build_local_track_set(
    *,
    agent_id: str,
    timestamp: int,
    track_output: MultiTargetTrackOutput,
) -> LocalTrackSet:
    """Convert one local multi-target posterior into an explicit fusion-center report."""

    return LocalTrackSet(
        agent_id=str(agent_id),
        timestamp=int(timestamp),
        tracks=tuple(
            LocalTrackSummary(
                agent_id=str(agent_id),
                timestamp=int(timestamp),
                global_target_index=int(track.global_target_index),
                key_label=f"E{int(track.global_target_index) + 1}",
                prior_relative_state=np.asarray(track.prior_relative_state, dtype=np.float32).copy(),
                prior_covariance_diagonal=np.asarray(track.prior_covariance_diagonal, dtype=np.float32).copy(),
                posterior_relative_state=np.asarray(track.posterior_relative_state, dtype=np.float32).copy(),
                posterior_covariance_diagonal=np.asarray(
                    track.posterior_covariance_diagonal,
                    dtype=np.float32,
                ).copy(),
                confidence=float(track.confidence),
                existence_probability=float(track.existence_probability),
                valid=bool(track.valid),
                local_candidate_index=int(track.local_candidate_index),
                innovation_distance=float(track.innovation_distance),
                local_candidate_weights=(
                    None
                    if track.local_candidate_weights is None
                    else np.asarray(track.local_candidate_weights, dtype=np.float32).copy()
                ),
            )
            for track in track_output.tracks
        ),
    )


def build_local_track_sets(
    *,
    timestamp: int,
    track_outputs: Mapping[str, MultiTargetTrackOutput],
    agent_ids: Sequence[str],
) -> dict[str, LocalTrackSet]:
    """Build one explicit local-track report per friendly UAV."""

    local_track_sets: dict[str, LocalTrackSet] = {}
    for agent_id in agent_ids:
        track_output = track_outputs.get(str(agent_id))
        if track_output is None:
            track_output = MultiTargetTrackOutput(agent_id=str(agent_id), tracks=tuple())
        local_track_sets[str(agent_id)] = build_local_track_set(
            agent_id=str(agent_id),
            timestamp=timestamp,
            track_output=track_output,
        )
    return local_track_sets


class FusionCenter:
    """Ideal centralized controller that collects local tracks and returns fused tracks."""

    def __init__(
        self,
        *,
        self_position_std_m: np.ndarray | Sequence[float],
        min_confidence: float,
        mahalanobis_distance_threshold: float,
        ci_search_steps: int,
    ) -> None:
        self.self_position_std_m = np.asarray(self_position_std_m, dtype=float).reshape(3)
        self.min_confidence = float(min_confidence)
        self.mahalanobis_distance_threshold = float(mahalanobis_distance_threshold)
        self.ci_search_steps = int(ci_search_steps)

    @staticmethod
    def _to_track_output(local_track_set: LocalTrackSet) -> MultiTargetTrackOutput:
        return MultiTargetTrackOutput(
            agent_id=str(local_track_set.agent_id),
            tracks=tuple(
                KeyTrackPosterior(
                    global_target_index=int(track.global_target_index),
                    prior_relative_state=np.asarray(track.prior_relative_state, dtype=np.float32).copy(),
                    prior_covariance_diagonal=np.asarray(
                        track.prior_covariance_diagonal,
                        dtype=np.float32,
                    ).copy(),
                    posterior_relative_state=np.asarray(track.posterior_relative_state, dtype=np.float32).copy(),
                    posterior_covariance_diagonal=np.asarray(
                        track.posterior_covariance_diagonal,
                        dtype=np.float32,
                    ).copy(),
                    confidence=float(track.confidence),
                    existence_probability=float(track.existence_probability),
                    valid=bool(track.valid),
                    local_candidate_weights=(
                        None
                        if track.local_candidate_weights is None
                        else np.asarray(track.local_candidate_weights, dtype=np.float32).copy()
                    ),
                    local_candidate_index=int(track.local_candidate_index),
                    innovation_distance=float(track.innovation_distance),
                )
                for track in local_track_set.tracks
            ),
        )

    def fuse(
        self,
        *,
        timestamp: int,
        local_track_sets: Mapping[str, LocalTrackSet],
        friendly_states: np.ndarray,
        agent_ids: Sequence[str],
    ) -> FusedTrackSet:
        """Fuse all friendly UAV local track sets under an ideal centralized assumption."""

        track_outputs = {
            str(agent_id): self._to_track_output(local_track_set)
            for agent_id, local_track_set in local_track_sets.items()
        }
        return build_fused_track_set(
            timestamp=int(timestamp),
            track_outputs=track_outputs,
            friendly_states=friendly_states,
            agent_ids=agent_ids,
            self_position_std_m=self.self_position_std_m,
            min_confidence=self.min_confidence,
            mahalanobis_distance_threshold=self.mahalanobis_distance_threshold,
            ci_search_steps=self.ci_search_steps,
        )
