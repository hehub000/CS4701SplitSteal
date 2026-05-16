from __future__ import annotations

"""
PPO trainer for Split-or-Steal.

The trainer steps the game turn by turn, updates after every PPO-controlled
dialogue move, and applies a terminal update after final actions are resolved.
"""

import argparse
import csv
import random
from pathlib import Path
from statistics import mean
import json

import numpy as np
import torch
import torch.nn.functional as F

from agentClass import AgentController, AgentPersonality
from dialogueClass import FinalAction
from gameClass import Game
from playerClass import Player

from ppo_buffer import RolloutBuffer
from ppo_env import PPOController
from personalityClass import PersonalityVector
from ppo_model import ActorCritic


TARGETS: dict[str, AgentPersonality] = {
    "honest": AgentPersonality.HONEST,
    "deceptive_threaten": AgentPersonality.DECEPTIVE_THREATEN,
    "tit_for_tat": AgentPersonality.TIT_FOR_TAT,
    "always_split": AgentPersonality.ALWAYS_SPLIT,
    "always_steal": AgentPersonality.ALWAYS_STEAL,
}


def resolve_actions(game: Game) -> tuple[dict[int, FinalAction], dict[int, float]]:
    actions: dict[int, FinalAction] = {}
    for player in game.players:
        if player.controller is None:
            raise ValueError(f"Player {player.name} has no controller")
        actions[player.player_id] = player.controller.choose_final_action(game)

    p1, p2 = game.players
    a1 = actions[p1.player_id]
    a2 = actions[p2.player_id]

    payouts = {p1.player_id: 0.0, p2.player_id: 0.0}
    jackpot = game.jackpot

    if a1 == FinalAction.SPLIT and a2 == FinalAction.SPLIT:
        payouts[p1.player_id] = jackpot / 2
        payouts[p2.player_id] = jackpot / 2
    elif a1 == FinalAction.STEAL and a2 == FinalAction.SPLIT:
        payouts[p1.player_id] = jackpot
    elif a1 == FinalAction.SPLIT and a2 == FinalAction.STEAL:
        payouts[p2.player_id] = jackpot

    p1.money += payouts[p1.player_id]
    p2.money += payouts[p2.player_id]
    return actions, payouts


def write_csv(path: Path, rows: list[dict[str, float | str | int]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def update_policy(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    *,
    clip_range: float,
    value_coef: float,
    entropy_coef: float,
    epochs: int,
    minibatch_size: int,
    gamma: float,
    gae_lambda: float,
    device: torch.device,
) -> dict[str, float]:
    if len(buffer) == 0:
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "loss": 0.0}

    buffer.finish_trajectory(last_value=0.0, gamma=gamma, gae_lambda=gae_lambda)

    advantages = np.asarray(buffer.advantages, dtype=np.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std(ddof=0) + 1e-8)

    indices = np.arange(len(buffer.transitions))
    total_policy = 0.0
    total_value = 0.0
    total_entropy = 0.0
    total_loss = 0.0
    steps = 0

    for _ in range(epochs):
        np.random.shuffle(indices)
        for start in range(0, len(indices), minibatch_size):
            batch_indices = indices[start : start + minibatch_size]

            policy_losses = []
            value_losses = []
            entropies = []

            for index in batch_indices:
                transition = buffer.transitions[index]
                obs = torch.tensor(transition.observation, dtype=torch.float32, device=device)
                action = torch.tensor(transition.action, dtype=torch.int64, device=device)
                old_logprob = torch.tensor(transition.logprob, dtype=torch.float32, device=device)
                advantage = torch.tensor(float(advantages[index]), dtype=torch.float32, device=device)
                target_return = torch.tensor(float(buffer.returns[index]), dtype=torch.float32, device=device)

                logprob, entropy, value = model.evaluate_action(obs, action, head=transition.head)
                ratio = torch.exp(logprob - old_logprob)
                clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)

                policy_loss = -torch.min(ratio * advantage, clipped * advantage)
                value_loss = F.mse_loss(value.squeeze(-1), target_return)

                policy_losses.append(policy_loss)
                value_losses.append(value_loss)
                entropies.append(entropy.mean())

            loss = torch.stack(policy_losses).mean() + value_coef * torch.stack(value_losses).mean() - entropy_coef * torch.stack(entropies).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_policy += float(torch.stack(policy_losses).mean().item())
            total_value += float(torch.stack(value_losses).mean().item())
            total_entropy += float(torch.stack(entropies).mean().item())
            total_loss += float(loss.item())
            steps += 1

    buffer.clear()
    steps = max(steps, 1)
    return {
        "policy_loss": total_policy / steps,
        "value_loss": total_value / steps,
        "entropy": total_entropy / steps,
        "loss": total_loss / steps,
    }


def play_dialogue_phase(game: Game, ppo_player_id: int, *, update_after_step) -> None:
    for turn in range(game.max_turns):
        game.current_turn = turn
        current_player = game.players[turn % len(game.players)]

        if current_player.controller is None:
            raise ValueError(f"Player {current_player.name} has no controller")

        move = current_player.controller.choose_dialogue_move(game)
        game.record_dialogue_move(move)

        if current_player.player_id == ppo_player_id:
            update_after_step(turn)


def run_episode(
    *,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    target: AgentPersonality,
    args: argparse.Namespace,
    device: torch.device,
    ai_starts: bool,
    deterministic: bool = False,
    seed_personality: PersonalityVector | None = None,
) -> dict[str, float | str | int]:
    ai_player = Player(player_id=1 if ai_starts else 2, name="PPOAI")
    target_player = Player(player_id=2 if ai_starts else 1, name=f"Target:{target.value}")

    ppo_controller = PPOController(
        ai_player,
        model,
        buffer,
        obs_dim=args.obs_dim,
        device=str(device),
        seed_personality=seed_personality,
    )
    if deterministic:
        ppo_controller.model.eval()
    else:
        ppo_controller.model.train()

    ai_player.controller = ppo_controller
    target_player.controller = AgentController(target_player, target)
    players = [ai_player, target_player] if ai_starts else [target_player, ai_player]
    game = Game(players=players, jackpot=args.jackpot, max_turns=args.max_turns)
    ppo_controller.begin_episode(game)
    updates: list[dict[str, float | int]] = []
    ppo_step_counter = 0

    def step_update(turn: int) -> None:
        nonlocal ppo_step_counter, updates
        ppo_step_counter += 1
        if args.rollout_length > 1 and (ppo_step_counter % args.rollout_length) != 0:
            return

        metrics = update_policy(
            model,
            optimizer,
            buffer,
            clip_range=args.clip_range,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            epochs=args.ppo_epochs,
            minibatch_size=max(1, args.minibatch_size),
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            device=device,
        )
        if metrics["loss"] != metrics["loss"]:
            raise RuntimeError("PPO update produced NaN loss")

        updates.append({"update_idx": ppo_step_counter, "turn": turn, **metrics})

    play_dialogue_phase(game, ai_player.player_id, update_after_step=step_update if not deterministic else (lambda turn: None))

    actions, payouts = resolve_actions(game)
    ai_action = actions[ai_player.player_id]
    target_action = actions[target_player.player_id]
    ai_payout = payouts[ai_player.player_id]

    both_stole = ai_action == FinalAction.STEAL and target_action == FinalAction.STEAL
    terminal_reward = ai_payout / max(args.jackpot, 1e-6)
    if both_stole:
        terminal_reward -= args.both_steal_penalty

    ppo_controller.apply_terminal_reward(terminal_reward)

    metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "loss": 0.0}
    if not deterministic:
        metrics = update_policy(
            model,
            optimizer,
            buffer,
            clip_range=args.clip_range,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            epochs=args.ppo_epochs,
            minibatch_size=max(1, args.minibatch_size),
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            device=device,
        )
        updates.append({"update_idx": ppo_step_counter + 1, "turn": game.current_turn, **metrics})
    else:
        buffer.clear()

    return {
        "target": target.value,
        "ai_payout": float(ai_payout),
        "score": float(terminal_reward),
        "both_steal": int(both_stole),
        "ai_action": ai_action.name,
        "target_action": target_action.name,
        **metrics,
        "updates": updates,
    }


def evaluate_policy(model: ActorCritic, args: argparse.Namespace, *, device: torch.device) -> dict[str, float]:
    random_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    scores = []
    payouts = []

    eval_buffer = RolloutBuffer()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    try:
        model.eval()
        for episode in range(args.eval_episodes):
            target_name = args.targets[episode % len(args.targets)]
            result = run_episode(
                model=model,
                optimizer=optimizer,
                buffer=eval_buffer,
                target=TARGETS[target_name],
                args=args,
                device=device,
                ai_starts=bool(episode % 2 == 0) if args.alternate_starts else args.ai_starts,
                deterministic=True,
                seed_personality=(PersonalityVector.from_csv(args.seed_personality) if args.seed_personality else None),
            )
            scores.append(float(result["score"]))
            payouts.append(float(result["ai_payout"]))
    finally:
        random.setstate(random_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)

    return {"eval_score": mean(scores) if scores else 0.0, "eval_payout": mean(payouts) if payouts else 0.0}


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    learn_personality = args.seed_personality is not None
    model = ActorCritic(obs_dim=args.obs_dim, hidden_size=args.hidden_size, learn_personality=learn_personality).to(device)
    
    # Initialize personality parameters from seed if provided
    if args.seed_personality:
        seed_pv = PersonalityVector.from_csv(args.seed_personality)
        model.set_personality_from_vector(seed_pv)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    buffer = RolloutBuffer()

    log_rows: list[dict[str, float | str | int]] = []
    update_rows: list[dict[str, float | str | int]] = []
    score_history: list[float] = []

    for episode in range(args.episodes):
        ai_starts = bool(episode % 2 == 0) if args.alternate_starts else args.ai_starts

        target_name = args.targets[episode % len(args.targets)]
        result = run_episode(
            model=model,
            optimizer=optimizer,
            buffer=buffer,
            target=TARGETS[target_name],
            args=args,
            device=device,
            ai_starts=ai_starts,
            deterministic=False,
            seed_personality=(PersonalityVector.from_csv(args.seed_personality) if args.seed_personality else None),
        )

        score_history.append(float(result["score"]))
        row = {
            "episode": episode,
            "target": target_name,
            "ai_starts": int(ai_starts),
            **result,
        }
        # remove updates from the per-episode log row to keep that CSV compact
        updates = row.pop("updates", []) if "updates" in row else []
        log_rows.append(row)

        for u in updates:
            update_rows.append({"episode": episode, "target": target_name, "ai_starts": int(ai_starts), **u})

        print(
            f"episode={episode:04d} target={target_name:18s} score={result['score']:6.3f} "
            f"payout={result['ai_payout']:7.2f} both_steal={result['both_steal']} "
            f"avg_score={mean(score_history[-50:]):6.3f}"
        )

        if args.eval_interval > 0 and (episode + 1) % args.eval_interval == 0:
            eval_metrics = evaluate_policy(model, args, device=device)
            print(f"  eval_score={eval_metrics['eval_score']:.3f} eval_payout={eval_metrics['eval_payout']:.2f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "obs_dim": args.obs_dim,
            "hidden_size": args.hidden_size,
        },
        args.out_model,
    )
    write_csv(args.log, log_rows)
    write_csv(args.update_log, update_rows)
    # Build a compact JSON summary with per-target and overall statistics
    summary: dict = {}
    scores = [float(r["score"]) for r in log_rows] if log_rows else []
    payouts = [float(r["ai_payout"]) for r in log_rows] if log_rows else []

    summary["episodes"] = len(log_rows)
    summary["mean_score"] = mean(scores) if scores else 0.0
    summary["mean_payout"] = mean(payouts) if payouts else 0.0

    per_target: dict = {}
    for r in log_rows:
        t = r["target"]
        per_target.setdefault(t, {"count": 0, "scores": [], "payouts": [], "both_steal": 0})
        per_target[t]["count"] += 1
        per_target[t]["scores"].append(float(r["score"]))
        per_target[t]["payouts"].append(float(r["ai_payout"]))
        per_target[t]["both_steal"] += int(r.get("both_steal", 0))

    summary["per_target"] = {}
    for t, data in per_target.items():
        cnt = data["count"]
        summary["per_target"][t] = {
            "count": cnt,
            "mean_score": mean(data["scores"]) if data["scores"] else 0.0,
            "mean_payout": mean(data["payouts"]) if data["payouts"] else 0.0,
            "both_steal_rate": data["both_steal"] / cnt if cnt else 0.0,
        }

    # Update-level aggregates
    if update_rows:
        pols = [float(u["policy_loss"]) for u in update_rows]
        vals = [float(u["value_loss"]) for u in update_rows]
        ents = [float(u["entropy"]) for u in update_rows]
        losses = [float(u["loss"]) for u in update_rows]
        summary["updates"] = {
            "count": len(update_rows),
            "policy_loss_mean": mean(pols),
            "value_loss_mean": mean(vals),
            "entropy_mean": mean(ents),
            "loss_mean": mean(losses),
        }
    else:
        summary["updates"] = {"count": 0}

    try:
        with args.summary_file.open("w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved PPO summary to: {args.summary_file}")
    except Exception:
        print("Warning: failed to write summary file")
    
    # Save learned personality if learning was enabled
    if learn_personality:
        personality_dict = model.get_personality_dict()
        personality_path = Path(str(args.out_model).replace("_model.pt", "_learned_personality.csv"))
        with personality_path.open("w") as f:
            f.write("aggression,trust_baseline,lie_propensity,evasiveness,cooperativeness\n")
            f.write(",".join(str(personality_dict[k]) for k in ["aggression", "trust_baseline", "lie_propensity", "evasiveness", "cooperativeness"]))
        print(f"Saved learned personality to: {personality_path}")
    
    print(f"\nSaved PPO model to: {args.out_model}")
    print(f"Saved PPO log to: {args.log}")
    print(f"Saved PPO update log to: {args.update_log}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO trainer for Split-or-Steal.")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--jackpot", type=float, default=100.0)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--both-steal-penalty", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--targets", nargs="+", choices=sorted(TARGETS), default=sorted(TARGETS))
    parser.add_argument("--ai-starts", action="store_true", default=True)
    parser.add_argument("--alternate-starts", action="store_true", default=True)
    parser.add_argument("--obs-dim", type=int, default=48)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=1)
    parser.add_argument("--rollout-length", type=int, default=4)
    parser.add_argument("--update-log", type=Path, default=Path("ppo_updates.csv"))
    parser.add_argument("--summary-file", type=Path, default=Path("ppo_summary.json"))
    parser.add_argument("--seed-personality", type=Path, default=None)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-model", type=Path, default=Path("ppo_policy.pt"))
    parser.add_argument("--log", type=Path, default=Path("ppo_trials.csv"))
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
