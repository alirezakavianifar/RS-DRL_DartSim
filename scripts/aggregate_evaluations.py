import json
import os
from pathlib import Path

def aggregate():
    results_dir = Path("results")
    
    # Paths to the evaluation metrics files
    paths = {
        "Baseline (No Noise)": results_dir / "zero_shot" / "baseline" / "evaluation_metrics.json",
        "Zero-Shot Medium": results_dir / "zero_shot" / "medium" / "evaluation_metrics.json",
        "Zero-Shot Hard": results_dir / "zero_shot" / "hard" / "evaluation_metrics.json",
        "Noise 0.1": results_dir / "noise" / "0.1" / "evaluation_metrics.json",
        "Noise 0.3": results_dir / "noise" / "0.3" / "evaluation_metrics.json",
        "Noise 0.5": results_dir / "noise" / "0.5" / "evaluation_metrics.json"
    }
    
    print("==========================================================================================")
    print("AGGREGATED EVALUATION SUMMARY FOR PRE-TRAINED RS-DRL (RHO=0.3) MODEL")
    print("==========================================================================================")
    print(f"{'Evaluation Case':<25} | {'Mean Reward':<15} | {'Success Rate':<12} | {'Destruction':<12} | {'Targets Detected':<18}")
    print("-" * 90)
    
    for name, path in paths.items():
        if not path.exists():
            print(f"{name:<25} | {'N/A (File missing)':<15} | {'-':<12} | {'-':<12} | {'-':<18}")
            continue
            
        try:
            with open(path, "r") as f:
                data = json.load(f)
            
            mean_reward = f"{data.get('mean_reward', 0.0):.4f} ± {data.get('std_reward', 0.0):.4f}"
            success_rate = f"{data.get('mission_success_rate', 0.0)*100:.1f}%"
            destruction = f"{data.get('team_destruction_rate', 0.0)*100:.1f}%"
            targets = f"{data.get('mean_targets_detected', 0.0):.2f} ± {data.get('std_targets_detected', 0.0):.2f}"
            
            print(f"{name:<25} | {mean_reward:<15} | {success_rate:<12} | {destruction:<12} | {targets:<18}")
        except Exception as e:
            print(f"{name:<25} | {'Error loading':<15} | {'-':<12} | {'-':<12} | {'-':<18}")
            
    print("==========================================================================================")

if __name__ == "__main__":
    aggregate()
