from __future__ import annotations

"""
Bayesian optimization trainer for Split-or-Steal personality vectors.

This trains all current PersonalityVector traits:
    - aggression
    - trust_baseline
    - lie_propensity
    - evasiveness
    - cooperativeness

It treats each batch of games as one noisy evaluation, then uses a Gaussian
Process surrogate model + Expected Improvement to choose the next vector.

Dependencies:
    pip install scikit-learn scipy numpy

Run example:
    python train_bayesian.py --trials 50 --games-per-target 40 --turn-counts 6 12

Profile-constrained examples:
    python train_bayesian.py --profile trusting --out best_trusting.csv
    python train_bayesian.py --profile aggressive --out best_aggressive.csv
    python train_bayesian.py --profile deceiver --out best_deceiver.csv
"""

import argparse
import csv
import random
from dataclasses import asdict, fields
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from agentClass import AgentController, AgentPersonality
from aiController import AIController
from dialogueClass import FinalAction, PROMPT_ACTS, SpeechAct
from gameClass import Game
from personalityClass import PersonalityVector
from playerClass import Player


TARGETS: dict[str, AgentPersonality] = {
    "honest": AgentPersonality.HONEST,
    "deceptive_threaten": AgentPersonality.DECEPTIVE_THREATEN,
    "tit_for_tat": AgentPersonality.TIT_FOR_TAT,
    "always_split": AgentPersonality.ALWAYS_SPLIT,
    "always_steal": AgentPersonality.ALWAYS_STEAL,
}

TRAIT_NAMES: tuple[str, ...] = tuple(field.name for field in fields(PersonalityVector))
TRAIT_DIMENSIONS = len(TRAIT_NAMES)
PROFILE_PRESETS: dict[str, dict[str, tuple[float, float, float]]] = {
    "trusting": {
        "aggression": (0.10, 0.00, 0.35),
        "trust_baseline": (0.90, 0.65, 1.00),
        "lie_propensity": (0.10, 0.00, 0.35),
        "evasiveness": (0.10, 0.00, 0.35),
        "cooperativeness": (0.90, 0.65, 1.00),
    },
    "aggressive": {
        "aggression": (0.90, 0.65, 1.00),
        "trust_baseline": (0.20, 0.00, 0.45),
        "lie_propensity": (0.50, 0.25, 0.75),
        "evasiveness": (0.60, 0.35, 0.90),
        "cooperativeness": (0.20, 0.00, 0.45),
    },
    "deceiver": {
        "aggression": (0.40, 0.20, 0.65),
        "trust_baseline": (0.30, 0.00, 0.50),
        "lie_propensity": (0.95, 0.75, 1.00),
        "evasiveness": (0.80, 0.60, 1.00),
        "cooperativeness": (0.20, 0.00, 0.45),
    },
}


def vector_from_x(x: np.ndarray | list[float] | tuple[float, ...]) -> PersonalityVector:
    values = [float(value) for value in x]
    if len(values) != TRAIT_DIMENSIONS:
        raise ValueError(f"Expected {TRAIT_DIMENSIONS} trait values, got {len(values)}.")

    p = PersonalityVector(**dict(zip(TRAIT_NAMES, values)))
    p.clamp()
    return p


def trait_row_from_x(x: np.ndarray | list[float] | tuple[float, ...]) -> dict[str, float]:
    values = [float(value) for value in x]
    return {name: values[index] for index, name in enumerate(TRAIT_NAMES)}


def profile_base_x(profile: str) -> np.ndarray:
    return np.asarray(
        [PROFILE_PRESETS[profile][name][0] for name in TRAIT_NAMES],
        dtype=float,
    )


def profile_bounds(profile: str) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray([PROFILE_PRESETS[profile][name][1] for name in TRAIT_NAMES], dtype=float)
    upper = np.asarray([PROFILE_PRESETS[profile][name][2] for name in TRAIT_NAMES], dtype=float)
    return lower, upper


def sample_x(rng: np.random.Generator, profile: str) -> np.ndarray:
    if profile == "free":
        return rng.random(TRAIT_DIMENSIONS)

    lower, upper = profile_bounds(profile)
    return lower + rng.random(TRAIT_DIMENSIONS) * (upper - lower)


def profile_distance_metrics(x: np.ndarray | list[float], profile: str) -> dict[str, float | str]:
    if profile == "free":
        return {
            "profile": profile,
            "profile_l1_distance": "",
            "profile_l2_distance": "",
            "profile_max_trait_delta": "",
        }

    values = np.asarray(x, dtype=float)
    base = profile_base_x(profile)
    deltas = np.abs(values - base)
    return {
        "profile": profile,
        "profile_l1_distance": float(np.sum(deltas)),
        "profile_l2_distance": float(np.linalg.norm(deltas)),
        "profile_max_trait_delta": float(np.max(deltas)),
    }


def resolve_actions(game: Game) -> tuple[dict[int, FinalAction], dict[int, float]]:
    """Resolve final actions without mutating Player.money."""
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

    return actions, payouts


def dialogue_metrics(game: Game, ai_player_id: int, target_player_id: int) -> dict[str, int | float | str]:
    moves = game.dialogue_history.moves
    ai_moves = [move for move in moves if move.speaker_id == ai_player_id]
    target_moves = [move for move in moves if move.speaker_id == target_player_id]

    metrics: dict[str, int | float | str] = {
        "dialogue_moves": len(moves),
        "ai_prompt_count": sum(1 for move in ai_moves if move.act in PROMPT_ACTS),
        "target_prompt_count": sum(1 for move in target_moves if move.act in PROMPT_ACTS),
        "ai_response_count": sum(1 for move in ai_moves if move.ref_move_id is not None),
        "target_response_count": sum(1 for move in target_moves if move.ref_move_id is not None),
    }

    for act in SpeechAct:
        key = act.name.lower()
        metrics[f"ai_{key}_count"] = sum(1 for move in ai_moves if move.act == act)
        metrics[f"target_{key}_count"] = sum(1 for move in target_moves if move.act == act)

    return metrics


def run_one_game(
    personality: PersonalityVector,
    target: AgentPersonality,
    *,
    ai_starts: bool,
    jackpot: float,
    max_turns: int,
    seed: int,
    both_steal_penalty: float,
    ai_steal_penalty: float,
    ai_exploit_penalty: float,
    target_exploit_penalty: float,
    both_split_bonus: float,
    ai_response_bonus: float,
) -> dict[str, float | str | int]:
    """Run one AI-vs-baseline game and return scoring plus visualization metrics."""
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))

    ai_player = Player(player_id=1 if ai_starts else 2, name="BayesAI")
    target_player = Player(player_id=2 if ai_starts else 1, name=f"Target:{target.value}")

    ai_controller = AIController(ai_player, personality)
    ai_player.controller = ai_controller
    target_player.controller = AgentController(target_player, target)

    players = [ai_player, target_player] if ai_starts else [target_player, ai_player]
    game = Game(players=players, jackpot=jackpot, max_turns=max_turns)
    game.run_dialogue_phase()

    actions, payouts = resolve_actions(game)
    ai_action = actions[ai_player.player_id]
    target_action = actions[target_player.player_id]
    ai_payout = payouts[ai_player.player_id]

    both_split = ai_action == FinalAction.SPLIT and target_action == FinalAction.SPLIT
    both_stole = ai_action == FinalAction.STEAL and target_action == FinalAction.STEAL
    ai_exploited_target = ai_action == FinalAction.STEAL and target_action == FinalAction.SPLIT
    target_exploited_ai = ai_action == FinalAction.SPLIT and target_action == FinalAction.STEAL
    dialogue = dialogue_metrics(game, ai_player.player_id, target_player.player_id)
    score = (
        ai_payout
        - (both_steal_penalty if both_stole else 0.0)
        - (ai_steal_penalty if ai_action == FinalAction.STEAL else 0.0)
        - (ai_exploit_penalty if ai_exploited_target else 0.0)
        - (target_exploit_penalty if target_exploited_ai else 0.0)
        + (both_split_bonus if both_split else 0.0)
        + ai_response_bonus * float(dialogue["ai_response_count"])
    )

    beliefs = ai_controller.beliefs

    return {
        "target": target.value,
        "max_turns": max_turns,
        "ai_starts": int(ai_starts),
        "ai_payout": float(ai_payout),
        "score": float(score),
        "both_split": int(both_split),
        "both_stole": int(both_stole),
        "ai_exploited_target": int(ai_exploited_target),
        "target_exploited_ai": int(target_exploited_ai),
        "ai_split": int(ai_action == FinalAction.SPLIT),
        "ai_steal": int(ai_action == FinalAction.STEAL),
        "target_split": int(target_action == FinalAction.SPLIT),
        "target_steal": int(target_action == FinalAction.STEAL),
        "ai_action": ai_action.name,
        "target_action": target_action.name,
        "final_ai_trust": beliefs.trust_in_opponent,
        "final_ai_suspicion": beliefs.suspicion_level,
        "final_ai_intention": beliefs.intention.name,
        "final_ai_own_open_prompts": len(beliefs.own_open_prompts),
        "final_ai_opponent_open_prompts": len(beliefs.opponent_open_prompts),
        **dialogue,
    }


def mean_field(rows: list[dict[str, object]], field_name: str) -> float:
    return mean(float(row[field_name]) for row in rows)


def add_subset_metrics(
    result: dict[str, float],
    rows: list[dict[str, object]],
    *,
    prefix: str,
) -> None:
    result[f"score_{prefix}"] = mean_field(rows, "score")
    result[f"payout_{prefix}"] = mean_field(rows, "ai_payout")
    result[f"ai_steal_rate_{prefix}"] = mean_field(rows, "ai_steal")
    result[f"both_steal_rate_{prefix}"] = mean_field(rows, "both_stole")


def evaluate_personality(
    x: np.ndarray,
    *,
    target_names: list[str],
    games_per_target: int,
    jackpot: float,
    turn_counts: list[int],
    seed: int,
    both_steal_penalty: float,
    ai_steal_penalty: float,
    ai_exploit_penalty: float,
    target_exploit_penalty: float,
    both_split_bonus: float,
    ai_response_bonus: float,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    """Evaluate one vector over diverse games and return aggregate plus per-game rows."""
    personality = vector_from_x(x)
    rows: list[dict[str, object]] = []

    game_index = 0
    for max_turns in turn_counts:
        for target_name in target_names:
            target = TARGETS[target_name]
            for repeat in range(games_per_target):
                for ai_starts in (True, False):
                    row = run_one_game(
                        personality,
                        target,
                        ai_starts=ai_starts,
                        jackpot=jackpot,
                        max_turns=max_turns,
                        seed=seed + 10_000 * game_index + repeat,
                        both_steal_penalty=both_steal_penalty,
                        ai_steal_penalty=ai_steal_penalty,
                        ai_exploit_penalty=ai_exploit_penalty,
                        target_exploit_penalty=target_exploit_penalty,
                        both_split_bonus=both_split_bonus,
                        ai_response_bonus=ai_response_bonus,
                    )
                    row["game_index"] = game_index
                    row["repeat"] = repeat
                    rows.append(row)
                    game_index += 1

    result = {
        "score": mean_field(rows, "score"),
        "avg_payout": mean_field(rows, "ai_payout"),
        "ai_split_rate": mean_field(rows, "ai_split"),
        "ai_steal_rate": mean_field(rows, "ai_steal"),
        "target_split_rate": mean_field(rows, "target_split"),
        "target_steal_rate": mean_field(rows, "target_steal"),
        "both_split_rate": mean_field(rows, "both_split"),
        "both_steal_rate": mean_field(rows, "both_stole"),
        "ai_exploit_rate": mean_field(rows, "ai_exploited_target"),
        "target_exploit_rate": mean_field(rows, "target_exploited_ai"),
        "avg_final_ai_trust": mean_field(rows, "final_ai_trust"),
        "avg_final_ai_suspicion": mean_field(rows, "final_ai_suspicion"),
        "avg_ai_prompt_count": mean_field(rows, "ai_prompt_count"),
        "avg_target_prompt_count": mean_field(rows, "target_prompt_count"),
        "avg_ai_response_count": mean_field(rows, "ai_response_count"),
        "avg_target_response_count": mean_field(rows, "target_response_count"),
        "avg_final_ai_own_open_prompts": mean_field(rows, "final_ai_own_open_prompts"),
        "avg_final_ai_opponent_open_prompts": mean_field(rows, "final_ai_opponent_open_prompts"),
    }

    for act in SpeechAct:
        key = act.name.lower()
        result[f"avg_ai_{key}_count"] = mean_field(rows, f"ai_{key}_count")
        result[f"avg_target_{key}_count"] = mean_field(rows, f"target_{key}_count")

    for target_name in target_names:
        target_rows = [row for row in rows if row["target"] == TARGETS[target_name].value]
        add_subset_metrics(result, target_rows, prefix=f"vs_{target_name}")

    for max_turns in turn_counts:
        turn_rows = [row for row in rows if int(row["max_turns"]) == max_turns]
        add_subset_metrics(result, turn_rows, prefix=f"turns_{max_turns}")

    return result, rows


def expected_improvement(
    candidate_x: np.ndarray,
    gp: GaussianProcessRegressor,
    best_y: float,
    xi: float,
) -> np.ndarray:
    """Expected improvement for a maximization objective."""
    mu, sigma = gp.predict(candidate_x, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    improvement = mu - best_y - xi
    z = improvement / sigma
    return improvement * norm.cdf(z) + sigma * norm.pdf(z)


def propose_next_x(
    x_observed: list[list[float]],
    y_observed: list[float],
    *,
    rng: np.random.Generator,
    candidate_pool_size: int,
    xi: float,
    profile: str,
) -> np.ndarray:
    """Fit GP surrogate and choose the next vector by Expected Improvement."""
    x_train = np.asarray(x_observed, dtype=float)
    y_train = np.asarray(y_observed, dtype=float)

    kernel = (
        ConstantKernel(1.0, (0.1, 10.0))
        * Matern(
            length_scale=np.ones(TRAIT_DIMENSIONS),
            length_scale_bounds=(0.05, 10.0),
            nu=2.5,
        )
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-8, 100.0))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=4,
        random_state=int(rng.integers(0, 2**31 - 1)),
    )
    gp.fit(x_train, y_train)

    if profile == "free":
        candidates = rng.random((candidate_pool_size, TRAIT_DIMENSIONS))
    else:
        lower, upper = profile_bounds(profile)
        candidates = lower + rng.random((candidate_pool_size, TRAIT_DIMENSIONS)) * (upper - lower)

    ei = expected_improvement(candidates, gp, best_y=max(y_observed), xi=xi)
    return candidates[int(np.argmax(ei))]


def write_results_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalized_turn_counts(args: argparse.Namespace) -> list[int]:
    counts = args.turn_counts if args.turn_counts is not None else [args.max_turns]
    unique_counts = sorted(set(counts))
    if any(count <= 0 for count in unique_counts):
        raise ValueError("Turn counts must be positive integers.")
    return unique_counts


def train(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    target_names = args.targets
    turn_counts = normalized_turn_counts(args)

    x_observed: list[list[float]] = []
    y_observed: list[float] = []
    log_rows: list[dict[str, object]] = []
    game_log_rows: list[dict[str, object]] = []
    first_score: float | None = None
    previous_trial_score: float | None = None

    for trial in range(args.trials):
        if trial == 0 and args.profile != "free":
            x = profile_base_x(args.profile)
            source = "profile_seed"
        elif trial < args.initial_random or not x_observed:
            x = sample_x(rng, args.profile)
            source = "random"
        else:
            x = propose_next_x(
                x_observed,
                y_observed,
                rng=rng,
                candidate_pool_size=args.candidate_pool_size,
                xi=args.xi,
                profile=args.profile,
            )
            source = "bayesian_ei"

        metrics, game_rows = evaluate_personality(
            x,
            target_names=target_names,
            games_per_target=args.games_per_target,
            jackpot=args.jackpot,
            turn_counts=turn_counts,
            seed=args.seed + trial * 1_000_000,
            both_steal_penalty=args.both_steal_penalty,
            ai_steal_penalty=args.ai_steal_penalty,
            ai_exploit_penalty=args.ai_exploit_penalty,
            target_exploit_penalty=args.target_exploit_penalty,
            both_split_bonus=args.both_split_bonus,
            ai_response_bonus=args.ai_response_bonus,
        )

        x_list = [float(v) for v in x]
        trait_row = trait_row_from_x(x_list)
        profile_row = profile_distance_metrics(x_list, args.profile)
        previous_best_score = max(y_observed) if y_observed else None

        x_observed.append(x_list)
        y_observed.append(metrics["score"])

        if first_score is None:
            first_score = metrics["score"]

        best_idx = int(np.argmax(y_observed))
        best_x = x_observed[best_idx]
        best_score = y_observed[best_idx]

        score_delta_from_previous = (
            ""
            if previous_trial_score is None
            else metrics["score"] - previous_trial_score
        )
        improvement_over_previous_best = (
            0.0
            if previous_best_score is None
            else metrics["score"] - previous_best_score
        )

        row: dict[str, object] = {
            "trial": trial,
            "source": source,
            **trait_row,
            **profile_row,
            **metrics,
            "score_delta_from_previous_trial": score_delta_from_previous,
            "improvement_over_previous_best": improvement_over_previous_best,
            "improved_best": int(previous_best_score is None or metrics["score"] > previous_best_score),
            "best_score_so_far": best_score,
            "best_trial_so_far": best_idx,
            "best_score_improvement_from_initial": best_score - first_score,
        }
        log_rows.append(row)

        for game_row in game_rows:
            game_log_rows.append({
                "trial": trial,
                "source": source,
                **trait_row,
                **profile_row,
                **game_row,
            })

        previous_trial_score = metrics["score"]

        best_traits = ", ".join(
            f"{name}={best_x[index]:.3f}" for index, name in enumerate(TRAIT_NAMES)
        )
        print(
            f"trial={trial:03d} source={source:11s} "
            f"score={metrics['score']:7.2f} payout={metrics['avg_payout']:7.2f} "
            f"ai_steal={metrics['ai_steal_rate']:.2f} both_steal={metrics['both_steal_rate']:.2f} | "
            f"best={best_score:7.2f} improvement={best_score - first_score:7.2f} "
            f"x=({best_traits})"
        )

    best_idx = int(np.argmax(y_observed))
    best_personality = vector_from_x(x_observed[best_idx])
    best_personality.to_csv(args.out)
    write_results_csv(args.log, log_rows)
    write_results_csv(args.game_log, game_log_rows)

    print("\nBest personality:")
    for key, value in asdict(best_personality).items():
        print(f"  {key}: {value:.4f}")
    print(f"\nSaved best vector to: {args.out}")
    print(f"Saved trial log to: {args.log}")
    print(f"Saved per-game log to: {args.game_log}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bayesian optimizer for PersonalityVector.")
    parser.add_argument(
        "--profile",
        choices=["free", *sorted(PROFILE_PRESETS)],
        default="free",
        help="Constrain search to a personality archetype while improving it.",
    )
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--initial-random", type=int, default=10)
    parser.add_argument("--games-per-target", type=int, default=25)
    parser.add_argument("--candidate-pool-size", type=int, default=2500)
    parser.add_argument("--xi", type=float, default=1.0, help="Exploration term for Expected Improvement.")
    parser.add_argument("--jackpot", type=float, default=100.0)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument(
        "--turn-counts",
        nargs="+",
        type=int,
        default=None,
        help="Optional dialogue lengths to evaluate each vector on, e.g. --turn-counts 6 12.",
    )
    parser.add_argument("--both-steal-penalty", type=float, default=0.0)
    parser.add_argument("--ai-steal-penalty", type=float, default=0.0)
    parser.add_argument("--ai-exploit-penalty", type=float, default=0.0)
    parser.add_argument("--target-exploit-penalty", type=float, default=0.0)
    parser.add_argument("--both-split-bonus", type=float, default=0.0)
    parser.add_argument("--ai-response-bonus", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--targets", nargs="+", choices=sorted(TARGETS), default=sorted(TARGETS))
    parser.add_argument("--out", type=Path, default=Path("best_bayesian.csv"))
    parser.add_argument("--log", type=Path, default=Path("bayesian_trials.csv"))
    parser.add_argument("--game-log", type=Path, default=Path("bayesian_games.csv"))
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
