"""
Generate plots for DARTSim case study report.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path


def plot_ablation_comparison(data_file: str = "./results/tensorboard_data.json",
                             output_file: str = "./results/plots/ablation_comparison.png"):
    """Plot comparison of different rho values."""
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    plt.figure(figsize=(12, 8))
    
    # Group by rho
    by_rho = {}
    for key, exp_data in data.items():
        rho = exp_data["rho"]
        if rho not in by_rho:
            by_rho[rho] = []
        by_rho[rho].append(exp_data)
    
    for rho in sorted(by_rho.keys()):
        experiments = by_rho[rho]
        
        # Align learning curves
        all_curves = []
        for exp in experiments:
            if len(exp["eval_rewards"]) > 0:
                rewards = np.array(exp["eval_rewards"])
                # Interpolate to common length if needed
                all_curves.append(rewards)
        
        if len(all_curves) == 0:
            continue
        
        # Compute mean and std
        min_len = min(len(c) for c in all_curves)
        curves_aligned = [c[:min_len] for c in all_curves]
        mean_curve = np.mean(curves_aligned, axis=0)
        std_curve = np.std(curves_aligned, axis=0)
        
        # Plot
        steps = experiments[0]["steps"][:min_len] if len(experiments[0]["steps"]) >= min_len else np.arange(min_len)
        plt.plot(steps, mean_curve, label=f"ρ = {rho}", linewidth=2)
        plt.fill_between(steps, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)
    
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Mean Evaluation Reward", fontsize=12)
    plt.title("RS-DRL Performance vs. Reshaping Factor (rho)", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved ablation comparison plot to {output_path}")
    plt.close()


def plot_rho_sensitivity(metrics_file: str = "./results/metrics_summary.csv",
                        output_file: str = "./results/plots/rho_sensitivity.png"):
    """Plot rho sensitivity analysis."""
    df = pd.read_csv(metrics_file)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Total Performance
    axes[0].errorbar(df["rho"], df["tp_mean"], yerr=df["tp_std"],
                     marker='o', capsize=5, linewidth=2, markersize=8)
    axes[0].set_xlabel("Reshaping Factor (rho)", fontsize=12)
    axes[0].set_ylabel("Total Performance (TP)", fontsize=12)
    axes[0].set_title("Total Performance vs. rho", fontsize=13)
    axes[0].grid(True, alpha=0.3)
    
    # Final Reward
    axes[1].errorbar(df["rho"], df["final_reward_mean"], yerr=df["final_reward_std"],
                     marker='s', capsize=5, linewidth=2, markersize=8)
    axes[1].set_xlabel("Reshaping Factor (rho)", fontsize=12)
    axes[1].set_ylabel("Final Reward", fontsize=12)
    axes[1].set_title("Final Reward vs. rho", fontsize=13)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved rho sensitivity plot to {output_path}")
    plt.close()


def create_summary_table(metrics_file: str = "./results/metrics_summary.csv",
                        output_file: str = "./results/tables/ablation_summary.tex"):
    """Create LaTeX table for ablation study."""
    df = pd.read_csv(metrics_file)
    
    # Format table
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create markdown table
    with open(output_path.with_suffix('.md'), 'w') as f:
        f.write("| rho | n_seeds | TP Mean | TP Std | Final Reward Mean | Final Reward Std |\n")
        f.write("|-----|---------|---------|--------|-------------------|------------------|\n")
        
        for _, row in df.iterrows():
            f.write(f"| {row['rho']:.1f} | {int(row['n_seeds'])} | "
                   f"{row['tp_mean']:.3f} | {row['tp_std']:.3f} | "
                   f"{row['final_reward_mean']:.2f} | {row['final_reward_std']:.2f} |\n")
    
    print(f"Saved summary table to {output_path.with_suffix('.md')}")


def main():
    print("Generating case study plots...")
    
    # Check if data files exist
    data_file = Path("./results/tensorboard_data.json")
    metrics_file = Path("./results/metrics_summary.csv")
    
    if not data_file.exists():
        print(f"ERROR: Data file not found: {data_file}")
        print("Run: python extract_tensorboard_data.py first")
        return
    
    if not metrics_file.exists():
        print(f"ERROR: Metrics file not found: {metrics_file}")
        return
    
    # Generate plots
    plot_ablation_comparison(str(data_file))
    plot_rho_sensitivity(str(metrics_file))
    create_summary_table(str(metrics_file))
    
    print("\nAll plots and tables generated!")


if __name__ == "__main__":
    main()

