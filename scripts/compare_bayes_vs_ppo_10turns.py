#!/usr/bin/env python3
from __future__ import annotations
import csv
from pathlib import Path
from statistics import mean
import math

def read_csv(path: Path):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return rows

def main():
    ppo = read_csv(Path('results/bayes_vs_ppo_all_10turns.csv'))
    bayes = read_csv(Path('results/bayes_baseline_eval_10turns.csv'))
    bayes_map = {Path(r['personality_file']).stem: r for r in bayes}

    out = []
    for r in ppo:
        name = Path(r['personality_file']).stem
        b = bayes_map.get(name)
        if not b:
            continue
        ppo_mean = float(r['ppo_mean_payout'])
        bay_mean = float(b['bayes_mean_payout'])
        diff = ppo_mean - bay_mean
        pct = (diff / bay_mean * 100.0) if bay_mean != 0 else math.nan
        out.append({
            'name': name,
            'personality_file': r['personality_file'],
            'bayes_mean_payout': bay_mean,
            'ppo_mean_payout': ppo_mean,
            'diff': diff,
            'diff_pct': pct,
            'bayes_win_rate': float(b['bayes_win_rate']),
            'ppo_win_rate': float(r['ppo_win_rate']),
            'n_episodes': int(r.get('n_eval_episodes', b.get('n_episodes', 0)))
        })

    out_path = Path('results/bayes_vs_ppo_10turns_summary.csv')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out:
        with out_path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(out[0].keys()))
            writer.writeheader()
            writer.writerows(out)

    for r in out:
        print(f"{r['name']}: Bayes={r['bayes_mean_payout']:.1f} PPO={r['ppo_mean_payout']:.1f} diff={r['diff']:.1f} ({r['diff_pct']:.1f}%)")

    print('Wrote', out_path)

if __name__ == '__main__':
    main()
