"""
Robustness evaluation script for Medium and Hard RS-DRL models.
Evaluates the models under sensor noise levels (10%, 30%, 50%)
across multiple seeds (42, 43, 44) and aggregates the statistics.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.evaluate_rs_drl import evaluate_agent

def run_robustness_eval():
    offline_data_dir = "./data/offline"
    experiments_base = "./experiments"
    
    scenarios = ["medium", "hard"]
    seeds = [42, 43, 44]
    noise_levels = [0.1, 0.3, 0.5]
    n_episodes = 50
    
    for scenario in scenarios:
        print("\n" + "="*70)
        print(f"EVALUATING SENSOR NOISE ROBUSTNESS FOR SCENARIO: {scenario.upper()}")
        print("="*70)
        
        scenario_results = {
            "scenario": scenario,
            "noise_levels": {}
        }
        
        for noise in noise_levels:
            print(f"\nEvaluating noise level: {noise} (probability of sensor failure)")
            
            rewards = []
            successes = []
            destructions = []
            targets_detected = []
            
            for seed in seeds:
                model_dir = Path(experiments_base) / scenario / "rs_drl" / f"seed_{seed}" / "model"
                model_path = model_dir / "rs_drl_dqn_rho0.3"
                if not model_path.exists():
                    model_path = model_dir / "rs_drl_dqn_rho0.3.zip"
                
                print(f"  Evaluating seed {seed} model: {model_path}")
                
                env_kwargs = {
                    "obs_dim": 17,
                    "max_transitions": 10000,
                    "seed": seed,
                    "sensor_noise": noise
                }
                
                try:
                    metrics = evaluate_agent(
                        model_path=str(model_path),
                        env_kwargs=env_kwargs,
                        n_episodes=n_episodes,
                        deterministic=True,
                        render=False,
                        offline_data_dir=offline_data_dir,
                        offline_scenario=scenario
                    )
                    
                    rewards.append(metrics["mean_reward"])
                    successes.append(metrics["mission_success_rate"])
                    destructions.append(metrics["team_destruction_rate"])
                    targets_detected.append(metrics["mean_targets_detected"])
                    
                    print(f"    Seed {seed}: Reward={metrics['mean_reward']:.4f}, Success={metrics['mission_success_rate']*100:.1f}%, Targets={metrics['mean_targets_detected']:.2f}")
                except Exception as e:
                    print(f"    Error evaluating seed {seed}: {e}")
            
            if len(rewards) > 0:
                noise_str = str(noise)
                scenario_results["noise_levels"][noise_str] = {
                    "reward_mean": float(np.mean(rewards)),
                    "reward_std": float(np.std(rewards)),
                    "success_mean": float(np.mean(successes)),
                    "success_std": float(np.std(successes)),
                    "destruction_mean": float(np.mean(destructions)),
                    "destruction_std": float(np.std(destructions)),
                    "targets_mean": float(np.mean(targets_detected)),
                    "targets_std": float(np.std(targets_detected))
                }
                
                print(f"\n  Aggregated results for noise {noise}:")
                print(f"    Reward: {np.mean(rewards):.4f} \u00b1 {np.std(rewards):.4f}")
                print(f"    Success: {np.mean(successes)*100:.1f}% \u00b1 {np.std(successes)*100:.1f}%")
                print(f"    Targets: {np.mean(targets_detected):.2f} \u00b1 {np.std(targets_detected):.2f}")
        
        # Save results to scenario folder
        out_dir = Path(experiments_base) / scenario
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "robustness_results.json"
        
        with open(out_path, 'w') as f:
            json.dump(scenario_results, f, indent=2)
        print(f"\nSaved aggregated robustness results for {scenario} to {out_path}")

if __name__ == "__main__":
    run_robustness_eval()
