"""
Extract evaluation metrics from TensorBoard event files.
This is needed to compute time-to-threshold and create learning curves.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")


def extract_tensorboard_scalars(event_file: Path, scalar_tag: str = "eval/mean_reward") -> List[Tuple[int, float]]:
    """
    Extract scalar values from TensorBoard event file.
    
    Args:
        event_file: Path to TensorBoard event file
        scalar_tag: Tag name to extract (e.g., "eval/mean_reward")
        
    Returns:
        List of (step, value) tuples
    """
    if not TENSORBOARD_AVAILABLE:
        return []
    
    if not event_file.exists():
        return []
    
    try:
        ea = EventAccumulator(str(event_file.parent))
        ea.Reload()
        
        if scalar_tag not in ea.Tags()['scalars']:
            return []
        
        scalars = ea.Scalars(scalar_tag)
        return [(int(s.step), float(s.value)) for s in scalars]
    except Exception as e:
        return []


def extract_all_eval_data(experiments_dir: str) -> Dict:
    """
    Extract all evaluation data from TensorBoard logs.
    
    Args:
        experiments_dir: Base experiments directory
        
    Returns:
        Dictionary with experiment data
    """
    exp_path = Path(experiments_dir)
    all_data = {}
    
    # Find all seed directories
    seed_dirs = sorted([d for d in exp_path.rglob("seed_*") if d.is_dir()])
    
    for seed_dir in seed_dirs:
        seed_num = int(seed_dir.name.split("_")[1])
        
        # Find all TensorBoard event files
        event_files = list(seed_dir.rglob("events.out.tfevents.*"))
        
        for event_file in event_files:
            # Extract rho from path
            path_parts = event_file.parts
            rho = None
            for part in path_parts:
                if "rho" in part:
                    try:
                        rho_str = part.split("rho")[1]
                        rho = float(rho_str)
                        break
                    except:
                        continue
            
            if rho is None:
                continue
            
            # Extract evaluation metrics
            eval_rewards = extract_tensorboard_scalars(event_file, "eval/mean_reward")
            eval_lengths = extract_tensorboard_scalars(event_file, "eval/mean_ep_length")
            
            key = f"seed_{seed_num}_rho_{rho}"
            if key not in all_data:
                all_data[key] = {
                    "seed": seed_num,
                    "rho": rho,
                    "eval_rewards": [],
                    "eval_lengths": [],
                    "steps": []
                }
            
            if eval_rewards:
                steps, rewards = zip(*eval_rewards)
                all_data[key]["steps"].extend(steps)
                all_data[key]["eval_rewards"].extend(rewards)
                all_data[key]["eval_lengths"].extend([l[1] for l in eval_lengths] if eval_lengths else [])
    
    # Sort and organize data
    for key in all_data:
        if all_data[key]["steps"]:
            # Sort by step
            sorted_pairs = sorted(zip(all_data[key]["steps"], 
                                     all_data[key]["eval_rewards"],
                                     all_data[key]["eval_lengths"] if all_data[key]["eval_lengths"] else [0] * len(all_data[key]["steps"])))
            steps, rewards, lengths = zip(*sorted_pairs)
            all_data[key]["steps"] = list(steps)
            all_data[key]["eval_rewards"] = list(rewards)
            all_data[key]["eval_lengths"] = list(lengths) if all_data[key]["eval_lengths"] else []
    
    return all_data


def compute_aggregated_metrics(all_data: Dict) -> pd.DataFrame:
    """
    Compute aggregated metrics from extracted data.
    
    Args:
        all_data: Dictionary with experiment data
        
    Returns:
        DataFrame with aggregated metrics
    """
    from evaluate_rs_drl import compute_time_to_threshold, compute_total_performance
    
    results = []
    
    # Group by rho
    by_rho = {}
    for key, data in all_data.items():
        rho = data["rho"]
        if rho not in by_rho:
            by_rho[rho] = []
        by_rho[rho].append(data)
    
    for rho, experiments in by_rho.items():
        all_ttt = {0.5: [], 0.7: [], 0.8: [], 0.9: []}
        all_tp = []
        all_final_rewards = []
        
        for exp in experiments:
            rewards = np.array(exp["eval_rewards"])
            if len(rewards) == 0:
                continue
            
            # Time-to-threshold
            for threshold in [0.5, 0.7, 0.8, 0.9]:
                ttt, achieved = compute_time_to_threshold(rewards, threshold)
                if ttt is not None:
                    all_ttt[threshold].append(ttt)
            
            # Total performance
            tp = compute_total_performance(rewards)
            all_tp.append(tp)
            
            # Final reward (mean of last 10%)
            final_reward = np.mean(rewards[-max(1, len(rewards)//10):])
            all_final_rewards.append(final_reward)
        
        # Aggregate statistics
        result = {"rho": rho, "n_seeds": len(experiments)}
        
        for threshold in [0.5, 0.7, 0.8, 0.9]:
            if len(all_ttt[threshold]) > 0:
                result[f"ttt_{threshold}_mean"] = np.mean(all_ttt[threshold])
                result[f"ttt_{threshold}_std"] = np.std(all_ttt[threshold])
        
        if len(all_tp) > 0:
            result["tp_mean"] = np.mean(all_tp)
            result["tp_std"] = np.std(all_tp)
        
        if len(all_final_rewards) > 0:
            result["final_reward_mean"] = np.mean(all_final_rewards)
            result["final_reward_std"] = np.std(all_final_rewards)
        
        results.append(result)
    
    df = pd.DataFrame(results)
    return df


def main():
    parser = argparse.ArgumentParser(description="Extract TensorBoard data for analysis")
    parser.add_argument("--experiments-dir", type=str, default="./experiments_ablation",
                       help="Experiments directory")
    parser.add_argument("--output", type=str, default="./results/tensorboard_data.json",
                       help="Output JSON file")
    parser.add_argument("--summary-csv", type=str, default="./results/metrics_summary.csv",
                       help="Output CSV summary")
    
    args = parser.parse_args()
    
    if not TENSORBOARD_AVAILABLE:
        print("ERROR: TensorBoard not available. Install with: pip install tensorboard")
        return
    
    print("Extracting TensorBoard data...")
    all_data = extract_all_eval_data(args.experiments_dir)
    
    print(f"Extracted data from {len(all_data)} experiment configurations")
    
    # Save raw data
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    json_data = {}
    for key, data in all_data.items():
        json_data[key] = {
            "seed": int(data["seed"]),
            "rho": float(data["rho"]),
            "steps": [int(s) for s in data["steps"]],
            "eval_rewards": [float(r) for r in data["eval_rewards"]],
            "eval_lengths": [float(l) for l in data["eval_lengths"]] if data["eval_lengths"] else []
        }
    
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Raw data saved to {output_path}")
    
    # Compute aggregated metrics
    print("Computing aggregated metrics...")
    df = compute_aggregated_metrics(all_data)
    
    # Save summary
    csv_path = Path(args.summary_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    
    print(f"Summary saved to {csv_path}")
    print("\nSummary:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

