"""
Cross-evaluation script for the Bidirectional Generalization Matrix.
Evaluates models trained on each scenario (Easy/Baseline, Medium, Hard)
on all other scenarios to determine transferability across seeds (42, 43, 44).
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.evaluate_rs_drl import evaluate_agent

def run_generalization_matrix():
    offline_data_dir = "./data/offline"
    experiments_base = "./experiments"
    
    source_scenarios = ["baseline", "medium", "hard"]
    target_scenarios = ["baseline", "medium", "hard"]
    seeds = [42, 43, 44]
    n_episodes = 50
    
    matrix_results = {}
    
    for source in source_scenarios:
        matrix_results[source] = {}
        
        for target in target_scenarios:
            print("\n" + "="*70)
            print(f"TRANSFER EVALUATION: Source={source.upper()} ---> Target={target.upper()}")
            print("="*70)
            
            rewards = []
            successes = []
            targets_detected = []
            
            for seed in seeds:
                # Resolve source model path
                model_dir = Path(experiments_base) / source / "rs_drl" / f"seed_{seed}" / "model"
                model_path = model_dir / "rs_drl_dqn_rho0.3"
                if not model_path.exists():
                    model_path = model_dir / "rs_drl_dqn_rho0.3.zip"
                    
                if not model_path.exists():
                    print(f"  Warning: Model not found at {model_path}")
                    continue
                
                print(f"  Evaluating seed {seed} model: {model_path} on {target}")
                
                env_kwargs = {
                    "obs_dim": 17,
                    "max_transitions": 10000,
                    "seed": seed
                }
                
                try:
                    metrics = evaluate_agent(
                        model_path=str(model_path),
                        env_kwargs=env_kwargs,
                        n_episodes=n_episodes,
                        deterministic=True,
                        render=False,
                        offline_data_dir=offline_data_dir,
                        offline_scenario=target
                    )
                    
                    rewards.append(metrics["mean_reward"])
                    successes.append(metrics["mission_success_rate"])
                    targets_detected.append(metrics["mean_targets_detected"])
                    
                    print(f"    Seed {seed}: Reward={metrics['mean_reward']:.4f}, Success={metrics['mission_success_rate']*100:.1f}%")
                except Exception as e:
                    print(f"    Error evaluating seed {seed}: {e}")
            
            if len(rewards) > 0:
                matrix_results[source][target] = {
                    "reward_mean": float(np.mean(rewards)),
                    "reward_std": float(np.std(rewards)),
                    "success_mean": float(np.mean(successes)),
                    "success_std": float(np.std(successes)),
                    "targets_mean": float(np.mean(targets_detected)),
                    "targets_std": float(np.std(targets_detected))
                }
                print(f"\n  Aggregated results for {source} -> {target}:")
                print(f"    Reward: {np.mean(rewards):.4f} \u00b1 {np.std(rewards):.4f}")
                print(f"    Success: {np.mean(successes)*100:.1f}% \u00b1 {np.std(successes)*100:.1f}%")
                
    # Save matrix results
    out_path = Path(experiments_base) / "generalization_matrix.json"
    with open(out_path, 'w') as f:
        json.dump(matrix_results, f, indent=2)
    print(f"\nSaved generalization matrix to: {out_path}")

if __name__ == "__main__":
    run_generalization_matrix()
