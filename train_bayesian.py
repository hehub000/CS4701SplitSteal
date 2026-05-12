from __future__ import annotations

"""
Bayesian optimization trainer for Split-or-Steal personality vectors.

This trains the three current PersonalityVector traits:
    - aggression
    - trust_baseline
    - lie_propensity

It treats each full game as one noisy evaluation, then uses a Gaussian Process
surrogate model + Expected Improvement to choose the next vector to test.

Dependencies:
    pip install scikit-learn scipy numpy

Run example:
    python train_bayesian.py --trials 50 --games-per-target 40 --out best_bayesian.csv
"""

import argparse
import csv
import math
import random
from dataclasses import asdict
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from agentClass import AgentController, AgentPersonality
from aiController import AIController
from dialogueClass import FinalAction
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


def vector_from_x(x: np.ndarray | list[float] | tuple[float, float, float]) -> PersonalityVector:
    p = PersonalityVector(
        aggression=float(x[0]),
        trust_baseline=float(x[1]),
        lie_propensity=float(x[2]),
    )
    p.clamp()
    return p


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


def run_one_game(
    personality: PersonalityVector,
    target: AgentPersonality,
    *,
    ai_starts: bool,
    jackpot: float,
    max_turns: int,
    seed: int,
    both_steal_penalty: float,
) -> dict[str, float | str | int]:
    """Run one AI-vs-baseline game and return scoring details."""
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))

    ai_player = Player(player_id=1 if ai_starts else 2, name="BayesAI")
    target_player = Player(player_id=2 if ai_starts else 1, name=f"Target:{target.value}")

    ai_player.controller = AIController(ai_player, personality)
    target_player.controller = AgentController(target_player, target)

    players = [ai_player, target_player] if ai_starts else [target_player, ai_player]
    game = Game(players=players, jackpot=jackpot, max_turns=max_turns)
    game.run_dialogue_phase()

    actions, payouts = resolve_actions(game)
    ai_action = actions[ai_player.player_id]
    target_action = actions[target_player.player_id]
    ai_payout = payouts[ai_player.player_id]

    both_stole = ai_action == FinalAction.STEAL and target_action == FinalAction.STEAL
    score = ai_payout - (both_steal_penalty if both_stole else 0.0)

    return {
        "target": target.value,
        "ai_starts": int(ai_starts),
        "ai_payout": float(ai_payout),
        "score": float(score),
        "both_stole": int(both_stole),
        "ai_action": ai_action.name,
        "target_action": target_action.name,
    }


def evaluate_personality(
    x: np.ndarray,
    *,
    target_names: list[str],
    games_per_target: int,
    jackpot: float,
    max_turns: int,
    seed: int,
    both_steal_penalty: float,
) -> dict[str, float]:
    """Evaluate one vector over many games and return aggregate metrics."""
    personality = vector_from_x(x)
    rows = []

    game_index = 0
    for target_name in target_names:
        target = TARGETS[target_name]
        for repeat in range(games_per_target):
            # Alternate starting positions to avoid overfitting to turn order.
            for ai_starts in (True, False):
                rows.append(
                    run_one_game(
                        personality,
                        target,
                        ai_starts=ai_starts,
                        jackpot=jackpot,
                        max_turns=max_turns,
                        seed=seed + 10_000 * game_index + repeat,
                        both_steal_penalty=both_steal_penalty,
                    )
                )
                game_index += 1

    scores = [float(r["score"]) for r in rows]
    payouts = [float(r["ai_payout"]) for r in rows]
    both_steal_rate = mean(float(r["both_stole"]) for r in rows)

    result = {
        "score": mean(scores),
        "avg_payout": mean(payouts),
        "both_steal_rate": both_steal_rate,
    }

    for target_name in target_names:
        target_rows = [r for r in rows if r["target"] == TARGETS[target_name].value]
        result[f"score_vs_{target_name}"] = mean(float(r["score"]) for r in target_rows)
        result[f"payout_vs_{target_name}"] = mean(float(r["ai_payout"]) for r in target_rows)

    return result


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
) -> np.ndarray:
    """Fit GP surrogate and choose the next vector by Expected Improvement."""
    x_train = np.asarray(x_observed, dtype=float)
    y_train = np.asarray(y_observed, dtype=float)

    kernel = (
        ConstantKernel(1.0, (0.1, 10.0))
        * Matern(length_scale=np.ones(3), length_scale_bounds=(0.05, 2.0), nu=2.5)
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 100.0))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=4,
        random_state=int(rng.integers(0, 2**31 - 1)),
    )
    gp.fit(x_train, y_train)

    candidates = rng.random((candidate_pool_size, 3))
    ei = expected_improvement(candidates, gp, best_y=max(y_observed), xi=xi)
    return candidates[int(np.argmax(ei))]


def write_results_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    target_names = args.targets

    x_observed: list[list[float]] = []
    y_observed: list[float] = []
    log_rows: list[dict[str, float]] = []

    for trial in range(args.trials):
        if trial < args.initial_random:
            x = rng.random(3)
            source = "random"
        else:
            x = propose_next_x(
                x_observed,
                y_observed,
                rng=rng,
                candidate_pool_size=args.candidate_pool_size,
                xi=args.xi,
            )
            source = "bayesian_ei"

        metrics = evaluate_personality(
            x,
            target_names=target_names,
            games_per_target=args.games_per_target,
            jackpot=args.jackpot,
            max_turns=args.max_turns,
            seed=args.seed + trial * 1_000_000,
            both_steal_penalty=args.both_steal_penalty,
        )

        x_list = [float(v) for v in x]
        x_observed.append(x_list)
        y_observed.append(metrics["score"])

        row = {
            "trial": trial,
            "source": source,
            "aggression": x_list[0],
            "trust_baseline": x_list[1],
            "lie_propensity": x_list[2],
            **metrics,
        }
        log_rows.append(row)

        best_idx = int(np.argmax(y_observed))
        best_x = x_observed[best_idx]
        best_score = y_observed[best_idx]
        print(
            f"trial={trial:03d} source={source:11s} "
            f"score={metrics['score']:7.2f} payout={metrics['avg_payout']:7.2f} "
            f"both_steal={metrics['both_steal_rate']:.2f} | "
            f"best={best_score:7.2f} "
            f"x=({best_x[0]:.3f}, {best_x[1]:.3f}, {best_x[2]:.3f})"
        )

    best_idx = int(np.argmax(y_observed))
    best_personality = vector_from_x(x_observed[best_idx])
    best_personality.to_csv(args.out)
    write_results_csv(args.log, log_rows)

    print("\nBest personality:")
    for key, value in asdict(best_personality).items():
        print(f"  {key}: {value:.4f}")
    print(f"\nSaved best vector to: {args.out}")
    print(f"Saved trial log to: {args.log}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bayesian optimizer for PersonalityVector.")
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--initial-random", type=int, default=10)
    parser.add_argument("--games-per-target", type=int, default=25)
    parser.add_argument("--candidate-pool-size", type=int, default=2500)
    parser.add_argument("--xi", type=float, default=1.0, help="Exploration term for Expected Improvement.")
    parser.add_argument("--jackpot", type=float, default=100.0)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--both-steal-penalty", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--targets", nargs="+", choices=sorted(TARGETS), default=sorted(TARGETS))
    parser.add_argument("--out", type=Path, default=Path("best_bayesian.csv"))
    parser.add_argument("--log", type=Path, default=Path("bayesian_trials.csv"))
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
