from __future__ import annotations

import numpy as np

from src.env.swarm_env import SwarmEnv
from src.rl.mappo import MAPPO
from src.simulation.ground_truth import ScenarioGroundTruth
from src.simulation.trajectory_generator import TrajectoryBundle


def _clone_env_for_evaluation(env: SwarmEnv, seed: int | None) -> SwarmEnv:
    bounds = None if env.bounds is None else env.bounds.copy()
    return SwarmEnv(
        key_num=env.key_num,
        nonkey_num=env.nonkey_num,
        mydrone_num=env.mydrone_num,
        bounds=bounds,
        dt=env.dt,
        max_steps=env.max_steps,
        seed=seed,
        enemy_vmax=env.enemy_vmax,
        friendly_vmax=env.friendly_vmax,
        initial_positions=env.initial_positions,
        nearby_distance_limits=env.nearby_distance_limits,
        tau_ally=env.tau_ally,
        lambda_safety=env.lambda_safety,
        J0=env.J0,
        power_weight=env.power_weight,
        progress_weight=env.progress_weight,
        progress_distance_scale=env.progress_distance_scale,
        move_penalty_weight=env.move_penalty_weight,
        policy_input_mode=env.policy_input_mode,
    )


def evaluate_policy(trainer: MAPPO, num_episodes: int = 3) -> dict[str, float]:
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    eval_seed = None if trainer.env.seed is None else int(trainer.env.seed) + 1_000_000
    eval_env = _clone_env_for_evaluation(trainer.env, seed=eval_seed)

    for _ in range(num_episodes):
        _, _, episode_stats = rollout_policy_episode(trainer, deterministic=True, env=eval_env)
        episode_returns.append(float(episode_stats["episode_return"]))
        episode_lengths.append(int(episode_stats["episode_length"]))

    return {
        "eval_avg_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "eval_std_return": float(np.std(episode_returns)) if episode_returns else 0.0,
        "eval_avg_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
    }



def rollout_policy_episode(
    trainer: MAPPO,
    deterministic: bool = True,
    env: SwarmEnv | None = None,
    seed: int | None = None,
) -> tuple[ScenarioGroundTruth, TrajectoryBundle, dict[str, float]]:
    active_env = trainer.env if env is None else env
    if env is None:
        if seed is None:
            trainer.reset_env()
        else:
            observations, _ = trainer.env.reset(seed=int(seed))
            trainer._current_observations = observations
            trainer._current_state = trainer.env.get_global_state()
        assert trainer._current_observations is not None
        assert trainer._current_state is not None
        observations = trainer._current_observations
        state = trainer._current_state
    else:
        observations, _ = active_env.reset(seed=int(seed)) if seed is not None else active_env.reset()
        state = active_env.get_global_state()
    assert active_env.world is not None
    assert active_env.scene is not None

    world = active_env.world
    scene = active_env.scene
    max_steps = active_env.max_steps
    num_enemies = len(world.enemy_nodes)
    num_friendlies = len(world.friendly_uavs)
    num_key_enemies = len(world.get_key_enemy_nodes())

    enemy_positions = np.zeros((max_steps + 1, num_enemies, 3), dtype=float)
    friendly_positions = np.zeros((max_steps + 1, num_friendlies, 3), dtype=float)
    friendly_actions = np.zeros((max_steps, num_friendlies), dtype=int)
    key_enemy_interference_watts = np.zeros((max_steps + 1, num_key_enemies), dtype=float)
    key_enemy_interference_dbm = np.zeros((max_steps + 1, num_key_enemies), dtype=float)
    friendly_interference_watts = np.zeros((max_steps + 1, num_friendlies), dtype=float)
    friendly_interference_dbm = np.zeros((max_steps + 1, num_friendlies), dtype=float)

    enemy_positions[0] = world.get_enemy_positions()
    friendly_positions[0] = world.get_friendly_positions()
    key_enemy_interference_watts[0] = world.interference.key_enemy_received_watts
    key_enemy_interference_dbm[0] = world.interference.key_enemy_received_dbm
    friendly_interference_watts[0] = world.interference.friendly_received_watts
    friendly_interference_dbm[0] = world.interference.friendly_received_dbm

    episode_return = 0.0
    episode_length = 0
    done = False

    while not done and episode_length < max_steps:
        actions, _, _, _ = trainer.select_actions(
            observations,
            state,
            deterministic=deterministic,
        )
        step_output = active_env.step(actions)
        episode_length += 1

        assert active_env.world is not None
        world = active_env.world
        enemy_positions[episode_length] = world.get_enemy_positions()
        friendly_positions[episode_length] = world.get_friendly_positions()
        friendly_actions[episode_length - 1] = np.asarray(
            [actions[agent_id] for agent_id in trainer.agent_ids],
            dtype=int,
        )
        key_enemy_interference_watts[episode_length] = world.interference.key_enemy_received_watts
        key_enemy_interference_dbm[episode_length] = world.interference.key_enemy_received_dbm
        friendly_interference_watts[episode_length] = world.interference.friendly_received_watts
        friendly_interference_dbm[episode_length] = world.interference.friendly_received_dbm

        rewards = np.asarray([step_output.rewards[agent_id] for agent_id in trainer.agent_ids], dtype=float)
        episode_return += float(np.mean(rewards))
        done = bool(
            any(step_output.terminated[agent_id] for agent_id in trainer.agent_ids)
            or any(step_output.truncated[agent_id] for agent_id in trainer.agent_ids)
        )

        if not done:
            observations = step_output.observations
            state = active_env.get_global_state()
            if env is None:
                trainer._current_observations = observations
                trainer._current_state = state

    bundle = TrajectoryBundle(
        enemy_positions=enemy_positions[: episode_length + 1],
        friendly_positions=friendly_positions[: episode_length + 1],
        friendly_actions=friendly_actions[:episode_length],
        key_enemy_interference_watts=key_enemy_interference_watts[: episode_length + 1],
        key_enemy_interference_dbm=key_enemy_interference_dbm[: episode_length + 1],
        friendly_interference_watts=friendly_interference_watts[: episode_length + 1],
        friendly_interference_dbm=friendly_interference_dbm[: episode_length + 1],
        dt=active_env.dt,
        bounds=world.bounds.copy(),
    )
    stats = {
        "episode_return": float(episode_return),
        "episode_length": float(episode_length),
    }
    return scene, bundle, stats
