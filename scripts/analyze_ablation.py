"""
Analysis script for the shaping parameter (rho) ablation grid.
Loads all ablation models (rho=0.0, 0.1, 0.3, 0.5) across seeds 42, 43, 44
for both Medium and Hard scenarios, computes their final Q-values on probe observations,
and generates an aggregated summary report.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.rs_drl_dqn import RSDRLDQN
from src.dartsim_env import OfflineDARTSimEnv
from scripts.statistical_analysis import load_probe_observations

def analyze_ablation_grid():
    experiments_base = "./experiments"
    offline_data_dir = "./data/offline"
    
    scenarios = ["medium", "hard"]
    seeds = [42, 43, 44]
    rho_values = [0.0, 0.1, 0.3, 0.5]
    
    summary_results = {}
    
    for scenario in scenarios:
        print("\n" + "="*70)
        print(f"ANALYZING ABLATION GRID FOR SCENARIO: {scenario.upper()}")
        print("="*70)
        
        probe_obs = load_probe_observations(offline_data_dir, scenario)
        probe_t = torch.FloatTensor(probe_obs)
        
        scenario_key = f"{scenario}_ablation"
        summary_results[scenario_key] = {}
        
        ablation_dir = Path(experiments_base) / f"{scenario}_ablation" / "rs_drl"
        
        dummy_env = OfflineDARTSimEnv(obs_dim=17)
        
        for rho in rho_values:
            rho_str = f"rho{rho}"
            print(f"\nProcessing rho = {rho}...")
            
            q_means = []
            
            for seed in seeds:
                model_dir = ablation_dir / f"seed_{seed}" / "model"
                model_path = model_dir / f"rs_drl_dqn_rho{rho}"
                if not model_path.exists():
                    model_path = model_dir / f"rs_drl_dqn_rho{rho}.zip"
                    
                if not model_path.exists():
                    print(f"  Warning: Model not found at {model_path}")
                    continue
                
                print(f"  Loading model: {model_path}")
                try:
                    model = RSDRLDQN.load(str(model_path), env=dummy_env)
                    
                    with torch.no_grad():
                        q_vals = model.q_net(probe_t.to(model.device)).cpu().numpy().max(axis=1)
                    
                    mean_q = float(q_vals.mean())
                    q_means.append(mean_q)
                    print(f"    Seed {seed} Mean max-Q: {mean_q:.4f}")
                except Exception as e:
                    print(f"    Error loading model {model_path}: {e}")
            
            if q_means:
                summary_results[scenario_key][str(rho)] = {
                    "q_means": [float(q) for q in q_means],
                    "mean": float(np.mean(q_means)),
                    "std": float(np.std(q_means))
                }
                print(f"  Aggregated for rho={rho}: {np.mean(q_means):.4f} \u00b1 {np.std(q_means):.4f}")
        
        dummy_env.close()
                
    # Save summary report
    out_path = Path(experiments_base) / "ablation_summary.json"
    with open(out_path, 'w') as f:
        json.dump(summary_results, f, indent=2)
    print(f"\nSaved comprehensive ablation summary to: {out_path}")

if __name__ == "__main__":
    analyze_ablation_grid()
