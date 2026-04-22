"""Training entry for the TEST-ONLY simple MAPPO baseline.

This path intentionally keeps the temporary engineering shortcuts:
- no association module output is used by actor/critic
- actor/critic inputs follow config/env.yaml policy_input.mode
- rewards and evaluation logs still use world truth

The goal is only to prove the simplified cooperative MAPPO chain can train end-to-end.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from src.env.swarm_env import SwarmEnv
from src.rl.mappo import MAPPO
from src.runner.evaluate import evaluate_policy
from src.utils.checkpoint import find_latest_checkpoint, restore_training_state, save_checkpoint
from src.utils.config_loader import get_mappo_config, get_num_steps, get_simulation_dt
from src.utils.logger import TrainingLogger
from src.visualization.plot_training import plot_training_history


def build_simple_baseline_env(config_path: str | Path | None = None) -> SwarmEnv:
    """Create the TEST-ONLY simple training environment for MAPPO."""

    config = get_mappo_config(config_path)
    scenario_config = config.get("scenario", {})
    training_config = config.get("training", {})
    baseline_assumptions = config.get("baseline_assumptions", {})
    raw_seed = training_config.get("seed", 42)
    seed = None if raw_seed is None else int(raw_seed)

    if not bool(baseline_assumptions.get("use_ground_truth_scene_rules", True)):
        raise ValueError(
            "The simple baseline requires baseline_assumptions.use_ground_truth_scene_rules=true "
            "so scene generation stays consistent with ground_truth.py."
        )

    if not bool(baseline_assumptions.get("no_association", True)):
        raise ValueError("The simple baseline environment expects no_association=true.")

    return SwarmEnv(
        key_num=int(scenario_config.get("key_enemy_count", 3)),
        nonkey_num=int(scenario_config.get("nonkey_enemy_count", 3)),
        mydrone_num=int(scenario_config.get("friendly_uav_count", 3)),
        dt=get_simulation_dt(),
        max_steps=get_num_steps(),
        seed=seed,
    )


def initialize_simple_baseline_env(config_path: str | Path | None = None) -> tuple[SwarmEnv, dict[str, Any]]:
    """Create and reset the simple baseline SwarmEnv."""

    env = build_simple_baseline_env(config_path)
    observations, infos = env.reset()
    global_state = env.get_global_state()

    if env.space_spec is None:
        raise RuntimeError("SwarmEnv space specification was not built during reset().")

    summary = {
        "num_agents": len(env.agent_ids),
        "agent_ids": env.agent_ids.copy(),
        "num_key_enemies": len(env.scene.key_enemy_nodes) if env.scene is not None else 0,
        "num_nonkey_enemies": len(env.scene.non_key_enemy_nodes) if env.scene is not None else 0,
        "observation_dim": env.space_spec.per_agent_obs_dim,
        "global_state_dim": env.space_spec.global_state_dim,
        "action_dim": env.space_spec.per_agent_action_dim,
        "policy_input_mode": env.policy_input_mode,
        "dt": env.dt,
        "max_steps": env.max_steps,
        "initial_observation_shapes": {agent_id: obs.shape for agent_id, obs in observations.items()},
        "initial_info_keys": {agent_id: sorted(agent_info.keys()) for agent_id, agent_info in infos.items()},
        "global_state_shape": global_state.shape,
    }
    return env, summary


def resolve_resume_checkpoint(resume: str | Path | None, checkpoint_dir: Path) -> Path | None:
    if resume is None:
        return None
    if str(resume).lower() == "latest":
        latest = find_latest_checkpoint(checkpoint_dir)
        if latest is None:
            raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
        return latest
    return Path(resume)


def train_simple_baseline(
    config_path: str | Path | None = None,
    device: str | torch.device | None = "cuda",
    resume: str | Path | None = None,
) -> tuple[MAPPO, list[dict[str, float]]]:
    env, summary = initialize_simple_baseline_env(config_path)
    config = get_mappo_config(config_path)
    training_config = config.get("training", {})
    logging_config = config.get("logging", {})

    checkpoint_dir = Path(logging_config.get("checkpoint_dir", "outputs/checkpoints"))
    log_dir = Path(logging_config.get("log_dir", "outputs/logs"))
    logger = TrainingLogger(log_dir)

    trainer = MAPPO(env=env, config=config, device=device)
    num_updates = int(training_config.get("num_updates", 200))
    log_interval = max(1, int(logging_config.get("log_interval", 10)))
    checkpoint_interval = max(1, int(logging_config.get("checkpoint_interval", 50)))
    eval_interval = max(1, int(logging_config.get("eval_interval", 50)))

    start_update = 1
    history: list[dict[str, float]] = []
    latest_eval_stats = {
        "eval_avg_return": 0.0,
        "eval_std_return": 0.0,
        "eval_avg_length": 0.0,
    }
    best_eval_return = float("-inf")

    resume_checkpoint = resolve_resume_checkpoint(resume, checkpoint_dir)
    if resume_checkpoint is not None:
        payload = restore_training_state(
            path=resume_checkpoint,
            actor=trainer.actor,
            critic=trainer.critic,
            actor_optimizer=trainer.actor_optimizer,
            critic_optimizer=trainer.critic_optimizer,
            map_location=trainer.device,
        )
        trainer.load_value_normalizer_state(payload.get("value_normalizer_state"))
        start_update = int(payload["update"]) + 1
        extra = payload.get("extra", {}) if isinstance(payload.get("extra", {}), dict) else {}
        raw_history = extra.get("history")
        if isinstance(raw_history, list):
            history = raw_history
        else:
            history = logger.read_history()
        if history:
            last_entry = history[-1]
            latest_eval_stats = {
                "eval_avg_return": float(last_entry.get("eval_avg_return", 0.0)),
                "eval_std_return": float(last_entry.get("eval_std_return", 0.0)),
                "eval_avg_length": float(last_entry.get("eval_avg_length", 0.0)),
            }
            best_eval_return = max(float(entry.get("eval_avg_return", float("-inf"))) for entry in history)
        print(f"Resuming training from checkpoint: {resume_checkpoint}")
        print(f"Resume starts at update {start_update} / target {num_updates}")

    print("------")
    if env.policy_input_mode == "local_only":
        print(f"policy_input.mode={env.policy_input_mode}, association_method={env.association_method}")
    else:
        print(f"policy_input.mode={env.policy_input_mode}")
    print("------")
    print("Simple MAPPO debug training started.")
    print(
        " | ".join(
            [
                f"agents={summary['num_agents']}",
                f"obs_dim={summary['observation_dim']}",
                f"state_dim={summary['global_state_dim']}",
                f"action_dim={summary['action_dim']}",
                f"batch_size={trainer.config.batch_size}",
                f"minibatch_size={trainer.config.minibatch_size}",
                f"device={trainer.device}",
            ]
        )
    )

    interrupted = False
    try:
        for update_idx in range(start_update, num_updates + 1):
            rollout_stats = trainer.collect_rollout()
            optimization_stats = trainer.update()

            if update_idx == 1 or update_idx % eval_interval == 0 or update_idx == num_updates:
                latest_eval_stats = evaluate_policy(trainer, num_episodes=3)

            stats = {
                "update": float(update_idx),
                **rollout_stats,
                **optimization_stats,
                **latest_eval_stats,
            }
            history.append(stats)

            if stats["eval_avg_return"] >= best_eval_return:
                best_eval_return = float(stats["eval_avg_return"])
                save_checkpoint(
                    path=checkpoint_dir / "mappo_best.pt",
                    actor=trainer.actor,
                    critic=trainer.critic,
                    actor_optimizer=trainer.actor_optimizer,
                    critic_optimizer=trainer.critic_optimizer,
                    update=update_idx,
                    config=config,
                    extra={"latest_stats": stats, "history": history, "is_best": True},
                    value_normalizer_state=trainer.get_value_normalizer_state(),
                )

            if update_idx == 1 or update_idx % log_interval == 0 or update_idx == num_updates:
                print(
                    " | ".join(
                        [
                            f"update={update_idx}/{num_updates}",
                            f"steps={int(stats['rollout_steps'])}",
                            f"batch_size={int(stats['batch_size'])}",
                            f"episodes={int(stats['completed_episodes'])}",
                            f"avg_step_reward={stats['average_step_reward']:.6f}",
                            f"eval_avg_return={stats['eval_avg_return']:.6f}",
                            f"actor_loss={stats['actor_loss']:.4f}",
                            f"critic_loss={stats['critic_loss']:.4f}",
                            f"entropy={stats['entropy']:.4f}",
                        ]
                    )
                )

            if update_idx % checkpoint_interval == 0 or update_idx == num_updates:
                save_checkpoint(
                    path=checkpoint_dir / f"mappo_update_{update_idx:04d}.pt",
                    actor=trainer.actor,
                    critic=trainer.critic,
                    actor_optimizer=trainer.actor_optimizer,
                    critic_optimizer=trainer.critic_optimizer,
                    update=update_idx,
                    config=config,
                    extra={"latest_stats": stats, "history": history},
                    value_normalizer_state=trainer.get_value_normalizer_state(),
                )
    except KeyboardInterrupt:
        interrupted = True
        last_update = int(history[-1]["update"]) if history else start_update - 1
        interrupt_path = save_checkpoint(
            path=checkpoint_dir / "mappo_interrupt_latest.pt",
            actor=trainer.actor,
            critic=trainer.critic,
            actor_optimizer=trainer.actor_optimizer,
            critic_optimizer=trainer.critic_optimizer,
            update=last_update,
            config=config,
            extra={"history": history, "interrupted": True},
            value_normalizer_state=trainer.get_value_normalizer_state(),
        )
        print(f"KeyboardInterrupt received. Interrupt checkpoint saved to: {interrupt_path}")

    logger.write_history(history)
    plot_path = plot_training_history(history, log_dir / "training_curve.pdf")

    if interrupted:
        print("Training stopped early and can be resumed from the interrupt checkpoint.")
    else:
        print("Training finished.")
    print(f"History saved to: {logger.csv_path}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    if plot_path is not None:
        print(f"Training curve: {plot_path}")
    return trainer, history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the debug MAPPO baseline.")
    parser.add_argument("--config", type=str, default=None, help="Optional path to config YAML.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device. Default is cuda.")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from a checkpoint path, or pass 'latest' to resume from the newest file in checkpoint_dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, history = train_simple_baseline(config_path=args.config, device=args.device, resume=args.resume)
    if history:
        final_stats = history[-1]
        print(
            "Final stats: "
            f"avg_step_reward={final_stats['average_step_reward']:.6f}, "
            f"eval_avg_return={final_stats['eval_avg_return']:.6f}, "
            f"actor_loss={final_stats['actor_loss']:.4f}, "
            f"critic_loss={final_stats['critic_loss']:.4f}"
        )


if __name__ == "__main__":
    main()
