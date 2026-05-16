#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import subprocess
from pathlib import Path
from statistics import mean
import sys
import torch

# ensure repo root on sys.path when run from scripts/
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from personalityClass import PersonalityVector
import train_ppo as tp
from ppo_model import ActorCritic
from ppo_buffer import RolloutBuffer


def write_rows(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_training(seed_csv: Path, train_episodes: int, device: str, extra_args: list[str], out_prefix: str):
    cmd = [
        sys.executable, "train_ppo.py",
        "--seed-personality", str(seed_csv),
        "--episodes", str(train_episodes),
        "--device", device,
        "--out-model", f"{out_prefix}_model.pt",
        "--log", f"{out_prefix}_trials.csv",
        "--update-log", f"{out_prefix}_updates.csv",
        "--summary-file", f"{out_prefix}_summary.json",
    ] + extra_args
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def eval_model(model_path: Path, seed_personality: PersonalityVector, eval_episodes: int, device: str, max_turns: int):
    device_t = torch.device(device)
    model = ActorCritic(obs_dim=48, hidden_size=128).to(device_t)
    ckpt = torch.load(model_path, map_location=device_t)
    model.load_state_dict(ckpt["model_state_dict"])
    eval_buffer = RolloutBuffer()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    rows = []
    targets = sorted(tp.TARGETS)
    for ep in range(eval_episodes):
        target_name = targets[ep % len(targets)]
        target = tp.TARGETS[target_name]
        result = tp.run_episode(
            model=model,
            optimizer=optimizer,
            buffer=eval_buffer,
            target=target,
            args=argparse.Namespace(
                obs_dim=48,
                jackpot=100.0,
                max_turns=max_turns,
                both_steal_penalty=0.0,
                clip_range=0.2,
                value_coef=0.5,
                entropy_coef=0.01,
                ppo_epochs=4,
                minibatch_size=1,
                gamma=0.99,
                gae_lambda=0.95,
                rollout_length=4,
                eval_interval=0,
                eval_episodes=0,
                seed=7,
                targets=targets,
                ai_starts=True,
                alternate_starts=True,
                hidden_size=128,
                learning_rate=3e-4,
                update_log=Path("ppo_eval_updates.csv"),
                summary_file=Path("ppo_eval_summary.json"),
                device=device,
                out_model=Path("ppo_eval_model.pt"),
                log=Path("ppo_eval_log.csv"),
            ),
            device=device_t,
            ai_starts=bool(ep % 2 == 0),
            deterministic=True,
            seed_personality=seed_personality,
        )
        rows.append({
            "episode": ep,
            "target": result["target"],
            "ai_payout": result["ai_payout"],
            "score": result["score"],
            "ai_action": result.get("ai_action", ""),
            "target_action": result.get("target_action", ""),
            "both_stole": int(result.get("both_steal", 0)),
        })
    return rows


def summarize_payout(rows, jackpot=100.0):
    mean_payout = mean(float(r["ai_payout"]) for r in rows) if rows else 0.0
    win_rate = mean(1.0 if float(r.get("ai_payout", 0.0)) > 0.0 else 0.0 for r in rows) if rows else 0.0
    mean_normalized = mean_payout / jackpot if jackpot else 0.0
    return {"mean_payout": mean_payout, "mean_normalized": mean_normalized, "win_rate": win_rate}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained-dir", type=Path, default=Path("trained_personalities"))
    parser.add_argument("--train-episodes", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--out-csv", type=Path, default=Path("results/bayes_vs_ppo_all_10turns.csv"))
    parser.add_argument("--skip-train", action="store_true", help="If set, skip calling train_ppo.py and assume model already exists as <prefix>_model.pt")
    parser.add_argument("--extra-train-args", nargs="*", default=[])
    args = parser.parse_args()

    personalities = sorted(args.trained_dir.glob("*.csv"))
    summary_rows = []

    for p in personalities:
        name = p.stem
        prefix = f"ppo_{name}"
        model_path = Path(f"{prefix}_model.pt")
        if not args.skip_train:
            run_training(p, args.train_episodes, args.device, args.extra_train_args, prefix)
        if not model_path.exists():
            print(f"Model {model_path} not found, skipping evaluation for {name}")
            continue
        seed_personality = PersonalityVector.from_csv(p)
        ppo_rows = eval_model(model_path, seed_personality, args.eval_episodes, args.device, args.max_turns)
        s = summarize_payout(ppo_rows, jackpot=100.0)
        summary_rows.append({
            "personality_file": str(p),
            "name": name,
            "ppo_mean_payout": s["mean_payout"],
            "ppo_mean_normalized": s["mean_normalized"],
            "ppo_win_rate": s["win_rate"],
            "n_eval_episodes": len(ppo_rows),
        })
        write_rows(Path(f"results/{prefix}_eval.csv"), [{"episode": r["episode"], "target": r["target"], "ai_payout": r["ai_payout"], "score": r["score"], "both_stole": r["both_stole"]} for r in ppo_rows])

    write_rows(args.out_csv, summary_rows)
    print("Wrote consolidated results to:", args.out_csv)


if __name__ == "__main__":
    main()
