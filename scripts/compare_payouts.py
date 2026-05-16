#!/usr/bin/env python3
"""
Compare payouts between Bayesian baseline and PPO learned policy.
Computes means, stds, and statistical significance (Welch's t-test, Cohen's d).
"""

import csv
import os
import sys
from pathlib import Path
from scipy import stats
import numpy as np

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train_bayesian import run_one_game, TARGETS
from personalityClass import PersonalityVector

def load_episode_payouts(csv_path):
    """Load episode-level payouts from CSV."""
    payouts = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            payouts.append(float(row['ai_payout']))
    return np.array(payouts)

def compute_cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0

def main():
    results_dir = Path(__file__).parent.parent / 'results'
    
    # Personality names and corresponding files
    personalities = ['best_aggressive', 'best_deceiver', 'best_trusting']
    
    # Load Bayes baseline summary
    bayes_summary_path = results_dir / 'bayes_baseline_eval_10turns.csv'
    bayes_data = {}
    with open(bayes_summary_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['name']
            bayes_data[name] = {
                'mean_payout': float(row['bayes_mean_payout']),
                'win_rate': float(row['bayes_win_rate']),
                'n_episodes': int(row['n_episodes'])
            }
    
    # Load PPO summary
    ppo_summary_path = results_dir / 'bayes_vs_ppo_all_10turns.csv'
    ppo_summary = {}
    with open(ppo_summary_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['name']
            ppo_summary[name] = {
                'mean_payout': float(row['ppo_mean_payout']),
                'win_rate': float(row['ppo_win_rate']),
                'n_episodes': int(row['n_eval_episodes'])
            }
    
    # Import Bayes evaluator
    # Load episode-level PPO payouts and run Bayes evaluations
    bayes_payouts = {}
    ppo_payouts = {}
    
    for personality in personalities:
        # Load PPO episode-level payouts
        ppo_eval_path = results_dir / f'ppo_{personality}_eval.csv'
        if ppo_eval_path.exists():
            ppo_payouts[personality] = load_episode_payouts(ppo_eval_path)
        
        # Run Bayes evaluation to get episode-level payouts
        print(f"Evaluating Bayes for {personality}...", file=sys.stderr)
        
        # Load seed personality
        seed_path = Path(__file__).parent.parent / 'trained_personalities' / f'{personality}.csv'
        personality_vec = PersonalityVector.from_csv(seed_path)
        
        # Run 100 games with 10 turns max (same as PPO)
        payouts = []
        targets = sorted(TARGETS)
        for episode in range(100):
            target_name = targets[episode % len(targets)]
            target = TARGETS[target_name]
            row = run_one_game(
                personality_vec,
                target,
                ai_starts=bool(episode % 2 == 0),
                jackpot=100.0,
                max_turns=10,
                seed=7 + episode,
                both_steal_penalty=0.0,
                ai_steal_penalty=0.0,
                ai_exploit_penalty=0.0,
                target_exploit_penalty=0.0,
                both_split_bonus=0.0,
                ai_response_bonus=0.0,
            )
            payouts.append(row['ai_payout'])
        
        bayes_payouts[personality] = np.array(payouts)
    
    # Compute comparisons
    comparison_results = []
    
    print("\n" + "="*90)
    print("PAYOUT COMPARISON: BAYES vs PPO (10-turn games, 100 episodes per method)")
    print("="*90 + "\n")
    
    for personality in personalities:
        if personality not in bayes_payouts:
            continue
        if personality not in ppo_payouts:
            continue
        
        bayes_pay = bayes_payouts[personality]
        ppo_pay = ppo_payouts[personality]
        
        # Compute statistics
        bayes_mean = np.mean(bayes_pay)
        bayes_std = np.std(bayes_pay, ddof=1)
        ppo_mean = np.mean(ppo_pay)
        ppo_std = np.std(ppo_pay, ddof=1)
        
        # Welch's t-test
        t_stat, p_value = stats.ttest_ind(bayes_pay, ppo_pay, equal_var=False)
        
        # Cohen's d
        cohens_d = compute_cohens_d(bayes_pay, ppo_pay)
        
        # Percentage difference
        pct_diff = ((ppo_mean - bayes_mean) / bayes_mean * 100) if bayes_mean != 0 else 0
        
        print(f"{personality}:")
        print(f"  Bayes:  mean={bayes_mean:.2f}, std={bayes_std:.2f}, n={len(bayes_pay)}")
        print(f"  PPO:    mean={ppo_mean:.2f}, std={ppo_std:.2f}, n={len(ppo_pay)}")
        print(f"  Δ mean: {ppo_mean - bayes_mean:+.2f} ({pct_diff:+.1f}%)")
        print(f"  Welch's t-test: t={t_stat:.3f}, p={p_value:.4f} (significant: {'YES' if p_value < 0.05 else 'NO'})")
        print(f"  Cohen's d: {cohens_d:.3f} (effect size: {'negligible' if abs(cohens_d) < 0.2 else 'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'})")
        print()
        
        comparison_results.append({
            'personality': personality,
            'bayes_mean': bayes_mean,
            'bayes_std': bayes_std,
            'ppo_mean': ppo_mean,
            'ppo_std': ppo_std,
            'mean_diff': ppo_mean - bayes_mean,
            'pct_diff': pct_diff,
            't_stat': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'bayes_n': len(bayes_pay),
            'ppo_n': len(ppo_pay)
        })
    
    # Write summary CSV
    summary_path = results_dir / 'bayes_vs_ppo_comparison.csv'
    with open(summary_path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=comparison_results[0].keys())
        writer.writeheader()
        writer.writerows(comparison_results)
    
    print(f"Comparison summary written to: {summary_path}")
    print("\n" + "="*90)
    
    # Overall summary
    print("\nOVERALL SUMMARY:")
    total_bayes = sum(len(bayes_payouts[p]) for p in personalities if p in bayes_payouts)
    total_ppo = sum(len(ppo_payouts[p]) for p in personalities if p in ppo_payouts)
    all_bayes = np.concatenate([bayes_payouts[p] for p in personalities if p in bayes_payouts])
    all_ppo = np.concatenate([ppo_payouts[p] for p in personalities if p in ppo_payouts])
    
    print(f"Bayes (all personalities): mean={np.mean(all_bayes):.2f}, std={np.std(all_bayes, ddof=1):.2f}, n={len(all_bayes)}")
    print(f"PPO  (all personalities): mean={np.mean(all_ppo):.2f}, std={np.std(all_ppo, ddof=1):.2f}, n={len(all_ppo)}")
    print(f"Overall Δ: {np.mean(all_ppo) - np.mean(all_bayes):+.2f} ({(np.mean(all_ppo) - np.mean(all_bayes)) / np.mean(all_bayes) * 100:+.1f}%)")
    
    t_stat, p_value = stats.ttest_ind(all_bayes, all_ppo, equal_var=False)
    print(f"Welch's t-test (all): t={t_stat:.3f}, p={p_value:.4f}")

if __name__ == '__main__':
    main()
