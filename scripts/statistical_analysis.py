"""
Statistical significance analysis for RS-DRL vs Baseline DQN models.
Loads models trained on Medium and Hard scenarios, computes their final Q-values
on a common set of probe observations from the offline dataset, and performs
normality, variance equality, t-test, and Cohen's d effect size analyses.
"""

import os
import sys
import json
import numpy as np
import scipy.stats as stats
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.rs_drl_dqn import RSDRLDQN
from stable_baselines3 import DQN
from src.dartsim_env import OfflineDARTSimEnv
from scripts.offline_rl_training import load_offline_dataset_generator

def load_probe_observations(offline_data_dir: str, scenario: str, num_samples: int = 1000) -> np.ndarray:
    """Load a fixed set of observation vectors from the offline dataset as probe states."""
    print(f"Loading probe observations for scenario '{scenario}'...")
    data_path = Path(offline_data_dir)
    pattern = f"episodes_{scenario}_*.json" if scenario else "episodes_*.json"
    episode_files = sorted(list(data_path.glob(pattern)))
    
    if not episode_files:
        raise FileNotFoundError(f"No episode files found matching {pattern} in {offline_data_dir}")
        
    observations = []
    # Read only the first file to get probe observations quickly
    with open(episode_files[0], 'r') as f:
        data = json.load(f)
        for transition in data:
            if 'state' in transition and len(transition['state']) == 17:
                observations.append(transition['state'])
            if len(observations) >= num_samples:
                break
                
    if not observations:
        print("Warning: No observations found in offline data. Creating random dummy observations.")
        return np.random.normal(0, 1, (num_samples, 17))
        
    obs_arr = np.array(observations[:num_samples], dtype=np.float32)
    print(f"Loaded {len(obs_arr)} probe observations with shape {obs_arr.shape}")
    return obs_arr


def analyze_scenario(scenario: str, experiments_base: str, offline_data_dir: str):
    print("\n" + "="*70)
    print(f"Running Statistical Analysis for Scenario: {scenario.upper()}")
    print("="*70)
    
    probe_obs = load_probe_observations(offline_data_dir, scenario)
    probe_t = torch.FloatTensor(probe_obs)
    
    base_path = Path(experiments_base) / scenario
    seeds = [42, 43, 44]
    
    rs_drl_q_means = []
    baseline_q_means = []
    
    # Load RS-DRL models
    for seed in seeds:
        # Check files/directories
        rs_dir = base_path / "rs_drl" / f"seed_{seed}" / "model"
        # We saw file rs_drl_dqn_rho0.3
        rs_path = rs_dir / "rs_drl_dqn_rho0.3"
        if not rs_path.exists():
            # Try with .zip
            rs_path = rs_dir / "rs_drl_dqn_rho0.3.zip"
            
        print(f"Loading RS-DRL model: {rs_path}")
        try:
            env = OfflineDARTSimEnv(obs_dim=17, data_dir=offline_data_dir, scenario=scenario, max_transitions=1000, seed=seed)
            model = RSDRLDQN.load(str(rs_path), env=env)
            # Evaluate Q-values
            with torch.no_grad():
                q_vals = model.q_net(probe_t.to(model.device)).cpu().numpy().max(axis=1)
            mean_q = float(q_vals.mean())
            rs_drl_q_means.append(mean_q)
            print(f"  Seed {seed} RS-DRL Mean max-Q: {mean_q:.4f}")
            env.close()
        except Exception as e:
            print(f"  Error loading RS-DRL seed {seed}: {e}")
            
    # Load Baseline models
    for seed in seeds:
        bl_dir = base_path / "baseline" / f"seed_{seed}" / "model"
        bl_path = bl_dir / "dqn_baseline"
        if not bl_path.exists():
            bl_path = bl_dir / "dqn_baseline.zip"
            
        print(f"Loading Baseline DQN model: {bl_path}")
        try:
            env = OfflineDARTSimEnv(obs_dim=17, data_dir=offline_data_dir, scenario=scenario, max_transitions=1000, seed=seed)
            model = DQN.load(str(bl_path), env=env)
            with torch.no_grad():
                q_vals = model.q_net(probe_t.to(model.device)).cpu().numpy().max(axis=1)
            mean_q = float(q_vals.mean())
            baseline_q_means.append(mean_q)
            print(f"  Seed {seed} Baseline Mean max-Q: {mean_q:.4f}")
            env.close()
        except Exception as e:
            print(f"  Error loading Baseline seed {seed}: {e}")
            
    if len(rs_drl_q_means) < 2 or len(baseline_q_means) < 2:
        print("Error: Not enough data points to run statistical significance tests.")
        return
        
    # Normality Test (Shapiro-Wilk)
    _, p_shapiro_rs = stats.shapiro(rs_drl_q_means)
    _, p_shapiro_bl = stats.shapiro(baseline_q_means)
    
    # Variance Equality Test (Levene)
    _, p_levene = stats.levene(rs_drl_q_means, baseline_q_means)
    
    # Independent t-test
    equal_var = p_levene > 0.05
    t_stat, p_ttest = stats.ttest_ind(rs_drl_q_means, baseline_q_means, equal_var=equal_var)
    
    # Cohen's d
    rs_mean, rs_std = np.mean(rs_drl_q_means), np.std(rs_drl_q_means, ddof=1)
    bl_mean, bl_std = np.mean(baseline_q_means), np.std(baseline_q_means, ddof=1)
    n1, n2 = len(rs_drl_q_means), len(baseline_q_means)
    
    pooled_std = np.sqrt(((n1 - 1) * (rs_std ** 2) + (n2 - 1) * (bl_std ** 2)) / (n1 + n2 - 2))
    cohen_d = (rs_mean - bl_mean) / pooled_std if pooled_std > 0 else 0
    
    pct_gain = ((rs_mean / bl_mean) - 1) * 100
    
    results = {
        "scenario": scenario,
        "rs_drl": {
            "q_means": [float(x) for x in rs_drl_q_means],
            "mean": float(rs_mean),
            "std": float(rs_std),
            "p_shapiro": float(p_shapiro_rs)
        },
        "baseline": {
            "q_means": [float(x) for x in baseline_q_means],
            "mean": float(bl_mean),
            "std": float(bl_std),
            "p_shapiro": float(p_shapiro_bl)
        },
        "statistics": {
            "levene_p_value": float(p_levene),
            "equal_variance_assumed": bool(equal_var),
            "t_statistic": float(t_stat),
            "p_value": float(p_ttest),
            "cohens_d": float(cohen_d),
            "percentage_gain": float(pct_gain)
        }
    }
    
    # Print results
    print("\nResults Summary:")
    print(f"  RS-DRL Final Q Mean: {rs_mean:.4f} ± {rs_std:.4f}")
    print(f"  Baseline Final Q Mean: {bl_mean:.4f} ± {bl_std:.4f}")
    print(f"  Improvement: {pct_gain:+.2f}%")
    print(f"  Shapiro-Wilk Normality p-values: RS-DRL={p_shapiro_rs:.4f}, Baseline={p_shapiro_bl:.4f}")
    print(f"  Levene's Variance Equality p-value: {p_levene:.4f}")
    print(f"  t-test statistic: {t_stat:.4f}, p-value: {p_ttest:.4f} (Significant if < 0.05)")
    print(f"  Cohen's d effect size: {cohen_d:.4f} (Large if > 0.8)")
    
    # Save to JSON
    out_dir = base_path
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "statistical_analysis.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved statistical analysis report to: {out_path}")

def main():
    experiments_base = "./experiments"
    offline_data_dir = "./data/offline"
    
    analyze_scenario("medium", experiments_base, offline_data_dir)
    analyze_scenario("hard", experiments_base, offline_data_dir)

if __name__ == "__main__":
    main()
