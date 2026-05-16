#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
from pathlib import Path
from statistics import mean
import sys

# ensure repo root on sys.path when run from scripts/
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from personalityClass import PersonalityVector
import train_bayesian as tb


def write_rows(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_payout(rows, jackpot=100.0):
    mp = mean(float(r["ai_payout"]) for r in rows) if rows else 0.0
    wr = mean(1.0 if float(r.get("ai_payout", 0.0)) > 0.0 else 0.0 for r in rows) if rows else 0.0
    return {"mean_payout": mp, "mean_normalized": mp / jackpot if jackpot else 0.0, "win_rate": wr}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained-dir", type=Path, default=Path("trained_personalities"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--out-csv", type=Path, default=Path("results/bayes_baseline_eval_10turns.csv"))
    parser.add_argument("--jackpot", type=float, default=100.0)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    personalities = sorted(args.trained_dir.glob("*.csv"))
    summary_rows = []

    targets = sorted(tb.TARGETS)
    for p in personalities:
        pv = PersonalityVector.from_csv(p)
        rows = []
        for ep in range(args.episodes):
            target_name = targets[ep % len(targets)]
            target = tb.TARGETS[target_name]
            row = tb.run_one_game(
                pv,
                target,
                ai_starts=bool(ep % 2 == 0),
                jackpot=args.jackpot,
                max_turns=args.max_turns,
                seed=args.seed + ep,
                both_steal_penalty=0.0,
                ai_steal_penalty=0.0,
                ai_exploit_penalty=0.0,
                target_exploit_penalty=0.0,
                both_split_bonus=0.0,
                ai_response_bonus=0.0,
            )
            rows.append(row)

        s = summarize_payout(rows, jackpot=args.jackpot)
        summary_rows.append({
            "personality_file": str(p),
            "name": p.stem,
            "bayes_mean_payout": s["mean_payout"],
            "bayes_mean_normalized": s["mean_normalized"],
            "bayes_win_rate": s["win_rate"],
            "n_episodes": len(rows),
        })

    write_rows(args.out_csv, summary_rows)
    print("Wrote Bayes baseline summaries to:", args.out_csv)


if __name__ == "__main__":
    main()
