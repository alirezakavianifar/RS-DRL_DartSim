"""
Comprehensive analysis script for Phase 4 experiments.
Computes time-to-threshold, aggregates results, and prepares data for plotting.
"""

import argparse
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.evaluate_rs_drl import compute_time_to_threshold, compute_total_performance, load_learning_curve


def extract_eval_rewards(log_dir: str) -> np.ndarray:
    """
    Extract evaluation rewards from TensorBoard logs or monitor files.
    
    Args:
        log_dir: Directory containing logs
        
    Returns:
        Array of evaluation mean rewards
    """
    log_path = Path(log_dir)
    
    # Try to find evaluation logs in TensorBoard format
    # SB3 stores eval results in monitor files
    monitor_files = list(log_path.rglob("*.monitor.csv"))
    
    if not monitor_files:
        # Check for eval results in tensorboard events
        return np.array([])
    
    # Read monitor files
    eval_rewards = []
    for monitor_file in monitor_files:
        try:
            df = pd.read_csv(monitor_file, skiprows=1)
            if 'r' in df.columns and 'l' in df.columns:
                # Group by episode and take mean
                episode_rewards = df.groupby(df.index // 1)['r'].mean().values
                eval_rewards.extend(episode_rewards)
        except Exception as e:
            continue
    
    if len(eval_rewards) == 0:
        return np.array([])
    
    return np.array(eval_rewards)


def analyze_experiment_directory(exp_dir: str, method_name: str = None) -> Dict:
    """
    Analyze all seeds in an experiment directory.
    
    Args:
        exp_dir: Experiment directory path
        method_name: Method name (extracted from path if None)
        
    Returns:
        Dictionary with aggregated metrics
    """
    exp_path = Path(exp_dir)
    
    if method_name is None:
        method_name = exp_path.name
    
    seed_dirs = sorted([d for d in exp_path.iterdir() if d.is_dir() and d.name.startswith("seed_")])
    
    if len(seed_dirs) == 0:
        print(f"Warning: No seed directories found in {exp_dir}")
        return {}
    
    all_ttt_results = []
    all_tp_results = []
    all_final_rewards = []
    all_learning_curves = []
    
    for seed_dir in seed_dirs:
        seed_num = int(seed_dir.name.split("_")[1])
        
        # Load learning curve
        log_dir = str(seed_dir / "logs")
        
        # Try multiple log file locations
        eval_rewards = extract_eval_rewards(log_dir)
        
        if len(eval_rewards) > 0:
            all_learning_curves.append(eval_rewards)
            all_final_rewards.append(float(np.mean(eval_rewards[-10:])))  # Last 10 evaluations
            
            # Compute time-to-threshold for different thresholds
            for threshold in [0.5, 0.7, 0.8, 0.9]:
                ttt, achieved = compute_time_to_threshold(eval_rewards, threshold)
                if ttt is not None:
                    all_ttt_results.append({
                        "seed": seed_num,
                        "threshold": threshold,
                        "ttt": ttt,
                        "achieved": achieved
                    })
            
            # Compute total performance
            tp = compute_total_performance(eval_rewards)
            all_tp_results.append({"seed": seed_num, "tp": tp})
    
    # Aggregate results
    results = {
        "method": method_name,
        "n_seeds": len(seed_dirs),
        "n_curves": len(all_learning_curves)
    }
    
    # Time-to-threshold statistics
    if len(all_ttt_results) > 0:
        ttt_df = pd.DataFrame(all_ttt_results)
        for threshold in [0.5, 0.7, 0.8, 0.9]:
            threshold_data = ttt_df[ttt_df["threshold"] == threshold]
            if len(threshold_data) > 0:
                results[f"ttt_{threshold}_mean"] = float(threshold_data["ttt"].mean())
                results[f"ttt_{threshold}_std"] = float(threshold_data["ttt"].std())
                results[f"ttt_{threshold}_count"] = len(threshold_data)
    
    # Total performance statistics
    if len(all_tp_results) > 0:
        tp_df = pd.DataFrame(all_tp_results)
        results["tp_mean"] = float(tp_df["tp"].mean())
        results["tp_std"] = float(tp_df["tp"].std())
    
    # Final reward statistics
    if len(all_final_rewards) > 0:
        results["final_reward_mean"] = float(np.mean(all_final_rewards))
        results["final_reward_std"] = float(np.std(all_final_rewards))
    
    # Combined learning curve
    if len(all_learning_curves) > 0:
        min_len = min(len(c) for c in all_learning_curves)
        curves_aligned = [c[:min_len] for c in all_learning_curves]
        mean_curve = np.mean(curves_aligned, axis=0)
        std_curve = np.std(curves_aligned, axis=0)
        results["learning_curve_mean"] = mean_curve.tolist()
        results["learning_curve_std"] = std_curve.tolist()
        results["curve_length"] = len(mean_curve)
    
    return results


def compute_time_to_threshold_analysis(experiments_base: str, output_file: str = None):
    """
    Compute time-to-threshold metrics for all experiments.
    
    Args:
        experiments_base: Base directory containing experiment subdirectories
        output_file: Path to save results JSON
    """
    base_path = Path(experiments_base)
    
    all_results = []
    
    # Find all method directories
    method_dirs = [d for d in base_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
    
    for method_dir in method_dirs:
        print(f"\nAnalyzing {method_dir.name}...")
        result = analyze_experiment_directory(str(method_dir), method_dir.name)
        if result:
            all_results.append(result)
            print(f"  Seeds analyzed: {result.get('n_seeds', 0)}")
            print(f"  Curves found: {result.get('n_curves', 0)}")
            if "final_reward_mean" in result:
                print(f"  Final reward: {result['final_reward_mean']:.4f} +/- {result.get('final_reward_std', 0):.4f}")
    
    # Save results
    if output_file is None:
        output_file = base_path / "analysis_results.json"
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nAnalysis results saved to {output_file}")
    
    # Create summary DataFrame
    summary_data = []
    for result in all_results:
        row = {"method": result["method"]}
        
        # Add time-to-threshold for 0.9 threshold
        if "ttt_0.9_mean" in result:
            row["ttt_0.9_mean"] = result["ttt_0.9_mean"]
            row["ttt_0.9_std"] = result["ttt_0.9_std"]
        
        # Add total performance
        if "tp_mean" in result:
            row["tp_mean"] = result["tp_mean"]
            row["tp_std"] = result["tp_std"]
        
        # Add final reward
        if "final_reward_mean" in result:
            row["final_reward_mean"] = result["final_reward_mean"]
            row["final_reward_std"] = result["final_reward_std"]
        
        summary_data.append(row)
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        summary_file = base_path / "analysis_summary.csv"
        df.to_csv(summary_file, index=False)
        print(f"Summary CSV saved to {summary_file}")
        print("\nSummary:")
        print(df.to_string())
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Analyze Phase 4 experiment results")
    parser.add_argument("--experiments-dir", type=str, default="./experiments_phase4",
                       help="Base directory containing experiment results")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file for analysis results")
    
    args = parser.parse_args()
    
    results = compute_time_to_threshold_analysis(args.experiments_dir, args.output)
    
    print(f"\nAnalysis complete! Analyzed {len(results)} methods.")


if __name__ == "__main__":
    main()

