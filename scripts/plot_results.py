"""
Plotting utilities for RS-DRL evaluation results.
Creates learning curves, comparison plots, and statistical visualizations.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List
import seaborn as sns


def plot_learning_curves_comparison(
    experiment_dirs: Dict[str, str],
    output_path: str = "./results/learning_curves.png",
    window: int = 100
) -> None:
    """
    Plot learning curves for multiple methods/models.
    
    Args:
        experiment_dirs: Dictionary mapping names to experiment directories
        output_path: Output path for plot
        window: Moving average window
    """
    plt.figure(figsize=(14, 8))
    
    for name, exp_dir in experiment_dirs.items():
        # Load all seed runs
        exp_path = Path(exp_dir)
        seed_dirs = [d for d in exp_path.iterdir() if d.is_dir() and d.name.startswith("seed_")]
        
        all_curves = []
        
        for seed_dir in seed_dirs:
            monitor_file = seed_dir / "logs" / "monitor.csv"
            if monitor_file.exists():
                try:
                    df = pd.read_csv(monitor_file, skiprows=1)
                    if 'r' in df.columns:
                        rewards = df['r'].values
                        # Smooth with moving average
                        if len(rewards) > window:
                            smoothed = pd.Series(rewards).rolling(window=window, min_periods=1).mean().values
                        else:
                            smoothed = rewards
                        all_curves.append(smoothed)
                except:
                    continue
        
        if len(all_curves) > 0:
            # Compute mean and std across seeds
            min_len = min(len(c) for c in all_curves)
            curves_truncated = [c[:min_len] for c in all_curves]
            mean_curve = np.mean(curves_truncated, axis=0)
            std_curve = np.std(curves_truncated, axis=0)
            
            # Plot
            x = np.arange(len(mean_curve))
            plt.plot(x, mean_curve, label=name, linewidth=2)
            plt.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)
    
    plt.xlabel("Timesteps", fontsize=12)
    plt.ylabel("Mean Reward", fontsize=12)
    plt.title("Learning Curves Comparison (Mean ± Std across seeds)", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Learning curves plot saved to {output_path}")
    plt.close()


def plot_metric_comparison(
    results_df: pd.DataFrame,
    metrics: List[str],
    output_path: str = "./results/metric_comparison.png"
) -> None:
    """
    Plot bar chart comparing metrics across methods.
    
    Args:
        results_df: DataFrame with evaluation results
        metrics: List of metric names to plot
        output_path: Output path
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
    
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        if metric not in results_df.columns:
            continue
        
        # Group by model if available
        if "model" in results_df.columns:
            data = results_df.groupby("model")[metric].agg(['mean', 'std'])
            models = data.index
            means = data['mean'].values
            stds = data['std'].values
        else:
            means = [results_df[metric].mean()]
            stds = [results_df[metric].std()]
            models = ["All"]
        
        axes[idx].bar(models, means, yerr=stds, capsize=5, alpha=0.7)
        axes[idx].set_ylabel(metric.replace("_", " ").title(), fontsize=10)
        axes[idx].set_title(f"{metric.replace('_', ' ').title()}", fontsize=11)
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Metric comparison plot saved to {output_path}")
    plt.close()


def plot_time_to_threshold(
    experiment_dirs: Dict[str, str],
    thresholds: List[float] = [0.7, 0.8, 0.9],
    output_path: str = "./results/time_to_threshold.png"
) -> None:
    """
    Plot time-to-threshold comparison.
    
    Args:
        experiment_dirs: Dictionary mapping names to experiment directories
        thresholds: List of reward thresholds
        output_path: Output path
    """
    from evaluate_rs_drl import load_learning_curve, compute_time_to_threshold
    
    results = []
    
    for name, exp_dir in experiment_dirs.items():
        exp_path = Path(exp_dir)
        seed_dirs = [d for d in exp_path.iterdir() if d.is_dir() and d.name.startswith("seed_")]
        
        ttt_values = {t: [] for t in thresholds}
        
        for seed_dir in seed_dirs:
            log_dir = str(seed_dir / "logs")
            learning_curve = load_learning_curve(log_dir)
            
            if len(learning_curve) > 0:
                for threshold in thresholds:
                    ttt, _ = compute_time_to_threshold(learning_curve, threshold)
                    if ttt is not None:
                        ttt_values[threshold].append(ttt)
        
        for threshold in thresholds:
            if len(ttt_values[threshold]) > 0:
                results.append({
                    "method": name,
                    "threshold": threshold,
                    "mean_ttt": np.mean(ttt_values[threshold]),
                    "std_ttt": np.std(ttt_values[threshold])
                })
    
    if len(results) == 0:
        print("No data for time-to-threshold plot")
        return
    
    df = pd.DataFrame(results)
    
    plt.figure(figsize=(10, 6))
    
    for method in df["method"].unique():
        method_data = df[df["method"] == method]
        plt.errorbar(
            method_data["threshold"],
            method_data["mean_ttt"],
            yerr=method_data["std_ttt"],
            marker='o',
            label=method,
            capsize=5,
            linewidth=2
        )
    
    plt.xlabel("Reward Threshold", fontsize=12)
    plt.ylabel("Time to Threshold (Timesteps)", fontsize=12)
    plt.title("Time-to-Threshold Comparison", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Time-to-threshold plot saved to {output_path}")
    plt.close()


def main():
    """Main plotting function."""
    parser = argparse.ArgumentParser(description="Plot evaluation results")
    
    parser.add_argument("--experiments", type=str, nargs="+",
                       help="Experiment directories (format: name:path)")
    parser.add_argument("--results-csv", type=str,
                       help="Path to results CSV file")
    parser.add_argument("--output-dir", type=str, default="./results",
                       help="Output directory for plots")
    parser.add_argument("--plot-type", type=str, default="all",
                       choices=["learning_curves", "metrics", "time_to_threshold", "all"],
                       help="Type of plot to generate")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.experiments:
        exp_dict = {}
        for pair in args.experiments:
            name, path = pair.split(":")
            exp_dict[name] = path
        
        if args.plot_type in ["learning_curves", "all"]:
            plot_learning_curves_comparison(
                exp_dict,
                str(output_dir / "learning_curves.png")
            )
        
        if args.plot_type in ["time_to_threshold", "all"]:
            plot_time_to_threshold(
                exp_dict,
                output_path=str(output_dir / "time_to_threshold.png")
            )
    
    if args.results_csv and args.plot_type in ["metrics", "all"]:
        df = pd.read_csv(args.results_csv)
        plot_metric_comparison(
            df,
            metrics=["mean_reward", "mission_success_rate", "mean_targets_detected"],
            output_path=str(output_dir / "metric_comparison.png")
        )
    
    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()

