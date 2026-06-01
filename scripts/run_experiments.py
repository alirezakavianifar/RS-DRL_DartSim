"""
Multi-seed experiment runner for RS-DRL evaluation.
Runs multiple experiments with different seeds and collects results.

Usage:
    python run_experiments.py --method rs_drl --seeds 10 --timesteps 20000
"""

import argparse
import subprocess
import sys
from pathlib import Path
import json
import time
from typing import List, Dict

# Ensure output is flushed immediately
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None


def run_single_experiment(
    method: str,
    seed: int,
    timesteps: int,
    rho: float = 0.3,
    output_base: str = "./experiments",
    **kwargs
) -> Dict:
    """
    Run a single experiment.
    
    Args:
        method: Training method (rs_drl or baseline)
        seed: Random seed
        timesteps: Training timesteps
        rho: RS-DRL reshaping factor
        output_base: Base output directory
        **kwargs: Additional training arguments
    
    Returns:
        Dictionary with experiment info
    """
    # Create output directory for this seed
    exp_dir = Path(output_base) / method / f"seed_{seed}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable,
        "train_rs_drl.py",
        "--method", method,
        "--seed", str(seed),
        "--timesteps", str(timesteps),
        "--log-dir", str(exp_dir / "logs"),
        "--save-path", str(exp_dir / "model"),
    ]
    
    if method == "rs_drl":
        cmd.extend(["--rho", str(rho)])
    
    # Add additional arguments
    for key, value in kwargs.items():
        if value is not None:
            key = key.replace("_", "-")
            cmd.append(f"--{key}")
            if not isinstance(value, bool):
                cmd.append(str(value))
            elif value:
                cmd.append("")
    
    print(f"\n{'='*60}")
    print(f"Running experiment: {method}, seed={seed}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    
    # Run training
    start_time = time.time()
    try:
        # Use capture_output=False to show output in real-time
        result = subprocess.run(
            cmd,
            capture_output=False,  # Show output in real-time
            text=True,
            check=True
        )
        elapsed = time.time() - start_time
        
        # Save experiment info
        exp_info = {
            "method": method,
            "seed": seed,
            "timesteps": timesteps,
            "rho": rho if method == "rs_drl" else None,
            "status": "success",
            "elapsed_time": elapsed,
            "output_dir": str(exp_dir)
        }
        
        exp_info_file = exp_dir / "experiment_info.json"
        with open(exp_info_file, 'w') as f:
            json.dump(exp_info, f, indent=2)
        
        print(f"[OK] Experiment completed in {elapsed:.2f} seconds")
        return exp_info
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"[FAILED] Experiment failed after {elapsed:.2f} seconds")
        print(f"Error: {e.stderr}")
        
        exp_info = {
            "method": method,
            "seed": seed,
            "status": "failed",
            "elapsed_time": elapsed,
            "error": str(e.stderr)
        }
        
        exp_info_file = exp_dir / "experiment_info.json"
        with open(exp_info_file, 'w') as f:
            json.dump(exp_info, f, indent=2)
        
        return exp_info


def run_experiment_grid(
    methods: List[str],
    seeds: List[int],
    timesteps: int,
    rho_values: List[float] = [0.3],
    output_base: str = "./experiments",
    parallel: bool = False,
    **kwargs
) -> List[Dict]:
    """
    Run grid of experiments.
    
    Args:
        methods: List of methods to test
        seeds: List of random seeds
        timesteps: Training timesteps per experiment
        rho_values: List of rho values to test (for RS-DRL)
        output_base: Base output directory
        parallel: Run experiments in parallel (not implemented yet)
        **kwargs: Additional training arguments
    
    Returns:
        List of experiment info dictionaries
    """
    all_experiments = []
    total_experiments = 0
    completed_experiments = 0
    
    # Count total experiments
    for method in methods:
        for seed in seeds:
            if method == "rs_drl":
                total_experiments += len(rho_values)
            else:
                total_experiments += 1
    
    print(f"\nTotal experiments to run: {total_experiments}")
    print(f"Estimated time: ~{total_experiments * 3} minutes\n")
    
    for method in methods:
        for seed in seeds:
            if method == "rs_drl":
                for rho in rho_values:
                    completed_experiments += 1
                    print(f"\n[{completed_experiments}/{total_experiments}] Starting: {method} seed={seed} rho={rho}")
                    sys.stdout.flush()  # Ensure output is visible immediately
                    
                    exp_info = run_single_experiment(
                        method=method,
                        seed=seed,
                        timesteps=timesteps,
                        rho=rho,
                        output_base=output_base,
                        **kwargs
                    )
                    all_experiments.append(exp_info)
                    
                    print(f"[{completed_experiments}/{total_experiments}] Completed: {method} seed={seed} rho={rho}")
                    sys.stdout.flush()
            else:
                exp_info = run_single_experiment(
                    method=method,
                    seed=seed,
                    timesteps=timesteps,
                    rho=0.0,  # Not used for baseline
                    output_base=output_base,
                    **kwargs
                )
                all_experiments.append(exp_info)
    
    # Save summary
    summary_file = Path(output_base) / "experiment_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_experiments, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("Experiment Summary")
    print("="*60)
    successful = sum(1 for exp in all_experiments if exp.get("status") == "success")
    failed = len(all_experiments) - successful
    print(f"Total experiments: {len(all_experiments)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Summary saved to {summary_file}")
    
    return all_experiments


def main():
    """Main experiment runner."""
    parser = argparse.ArgumentParser(description="Run multi-seed RS-DRL experiments")
    
    parser.add_argument("--methods", type=str, nargs="+", default=["rs_drl", "baseline"],
                       choices=["rs_drl", "baseline"],
                       help="Methods to test")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)),
                       help="Random seeds (e.g., --seeds 42 43 44)")
    parser.add_argument("--seed-range", type=str,
                       help="Seed range (e.g., 42:52)")
    parser.add_argument("--timesteps", type=int, default=20000,
                       help="Training timesteps per experiment")
    parser.add_argument("--rho-values", type=float, nargs="+", default=[0.3],
                       help="Rho values to test for RS-DRL")
    parser.add_argument("--output-base", type=str, default="./experiments",
                       help="Base output directory")
    
    # Additional training arguments
    parser.add_argument("--learning-rate", type=float,
                       help="Learning rate")
    parser.add_argument("--gamma", type=float,
                       help="Discount factor")
    parser.add_argument("--batch-size", type=int,
                       help="Batch size")
    parser.add_argument("--eval-freq", type=int, default=1000,
                       help="Evaluation frequency")
    parser.add_argument("--eval-episodes", type=int, default=2,
                       help="Number of episodes for evaluation")
    
    args = parser.parse_args()
    
    # Parse seed range if provided
    if args.seed_range:
        start, end = map(int, args.seed_range.split(":"))
        seeds = list(range(start, end))
    else:
        seeds = args.seeds
    
    # Prepare kwargs
    kwargs = {}
    if args.learning_rate is not None:
        kwargs["learning_rate"] = args.learning_rate
    if args.gamma is not None:
        kwargs["gamma"] = args.gamma
    if args.batch_size is not None:
        kwargs["batch_size"] = args.batch_size
    kwargs["eval_freq"] = args.eval_freq
    kwargs["eval_episodes"] = args.eval_episodes
    
    print(f"Running experiments:")
    print(f"  Methods: {args.methods}")
    print(f"  Seeds: {seeds}")
    print(f"  Timesteps: {args.timesteps}")
    print(f"  Rho values: {args.rho_values}")
    print(f"  Output: {args.output_base}")
    
    # Run experiments
    experiments = run_experiment_grid(
        methods=args.methods,
        seeds=seeds,
        timesteps=args.timesteps,
        rho_values=args.rho_values,
        output_base=args.output_base,
        **kwargs
    )
    
    print(f"\nAll experiments completed!")
    print(f"Results saved to {args.output_base}")


if __name__ == "__main__":
    main()

