from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from src.entities.friendly_uav import FriendlyUAV
from src.tracking.fusion import FusedTrackSet
from src.tracking.task_assignment import TaskAssignmentResult


@dataclass(frozen=True)
class ControlCommand:
    agent_id: str
    action: int
    target_track_id: str | None = None
    note: str = "placeholder_stop"


def build_placeholder_control_commands(
    agent_ids: Sequence[str],
    *,
    stop_action: int = 0,
) -> dict[str, ControlCommand]:
    """Return placeholder control commands that keep the current discrete action space intact."""

    return {
        str(agent_id): ControlCommand(
            agent_id=str(agent_id),
            action=int(stop_action),
        )
        for agent_id in agent_ids
    }


def _select_action_toward_target(relative_position: np.ndarray, *, stop_action: int) -> int:
    target_delta = np.asarray(relative_position, dtype=float).reshape(3)
    distance = float(np.linalg.norm(target_delta))
    if distance <= 1.0e-6:
        return int(stop_action)

    desired_direction = target_delta / distance
    best_action = int(stop_action)
    best_score = -np.inf
    for action_idx, direction in enumerate(FriendlyUAV.ACTION_DIRECTIONS):
        score = float(np.dot(desired_direction, np.asarray(direction, dtype=float).reshape(3)))
        if score > best_score:
            best_score = score
            best_action = int(action_idx)
    return best_action


def build_tracking_control_commands(
    agent_ids: Sequence[str],
    *,
    friendly_states: np.ndarray,
    assignment: TaskAssignmentResult,
    fused_track_set: FusedTrackSet,
    stop_action: int = 0,
) -> dict[str, ControlCommand]:
    """Build simple pursuit commands toward the assigned fused target state."""

    friendly_array = np.asarray(friendly_states, dtype=float)
    if friendly_array.ndim != 2 or friendly_array.shape[1] < 3:
        raise ValueError(f"friendly_states must have shape (N, >=3), got {friendly_array.shape}.")
    if friendly_array.shape[0] != len(agent_ids):
        raise ValueError("friendly_states row count must match agent_ids length.")

    fused_track_by_label = {str(track.key_label): track for track in fused_track_set.tracks}
    assignment_by_agent = {str(item.agent_id): item for item in assignment.assignments}

    commands: dict[str, ControlCommand] = {}
    for friendly_idx, agent_id in enumerate(agent_ids):
        assigned_track = assignment_by_agent.get(str(agent_id))
        if assigned_track is None or not bool(assigned_track.assigned) or assigned_track.key_track_id is None:
            commands[str(agent_id)] = ControlCommand(
                agent_id=str(agent_id),
                action=int(stop_action),
                target_track_id=None,
                note="unassigned_stop",
            )
            continue

        fused_track = fused_track_by_label.get(str(assigned_track.key_track_id))
        if fused_track is None or not bool(fused_track.valid):
            commands[str(agent_id)] = ControlCommand(
                agent_id=str(agent_id),
                action=int(stop_action),
                target_track_id=str(assigned_track.key_track_id),
                note="missing_track_stop",
            )
            continue

        relative_position = (
            np.asarray(fused_track.fused_state_mean, dtype=float).reshape(-1)[:3]
            - friendly_array[friendly_idx, :3]
        )
        chosen_action = _select_action_toward_target(relative_position, stop_action=stop_action)
        commands[str(agent_id)] = ControlCommand(
            agent_id=str(agent_id),
            action=int(chosen_action),
            target_track_id=str(fused_track.key_label),
            note="pursue_assigned_track",
        )
    return commands


def extract_actions(commands: Mapping[str, ControlCommand]) -> dict[str, int]:
    return {str(agent_id): int(command.action) for agent_id, command in commands.items()}
