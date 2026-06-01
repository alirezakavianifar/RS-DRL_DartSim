"""
Hyperparameter grid search for RS-DRL.
Implements training/validation split protocol from RS-DRL paper.

Usage:
    python grid_search.py --config config.json
"""

import argparse
import json
import numpy as np
from pathlib import Path
from train_rs_drl import train_rs_drl, train_baseline_dqn
from evaluate_rs_drl import evaluate_agent, compute_total_performance, load_learning_curve
from typing import Dict, List, Tuple
import itertools


def grid_search_rs_drl(
    param_grid: Dict[str, List],
    env_kwargs: Dict,
    validation_timesteps: int = 2000,
    n_validation_runs: int = 3,
    output_dir: str = "./grid_search"
) -> Dict:
    """
    Perform grid search over hyperparameters.
    
    Args:
        param_grid: Dictionary mapping parameter names to lists of values
        env_kwargs: Environment configuration
        validation_timesteps: Timesteps for validation split
        n_validation_runs: Number of validation runs per config
        output_dir: Output directory
    
    Returns:
        Dictionary with best configuration and results
    """
    # Extract fixed parameters and grid parameters
    grid_params = {}
    fixed_params = {}
    
    for key, value in param_grid.items():
        if isinstance(value, list) and len(value) > 1:
            grid_params[key] = value
        else:
            fixed_params[key] = value[0] if isinstance(value, list) else value
    
    # Generate all combinations
    keys = list(grid_params.keys())
    values = list(grid_params.values())
    combinations = list(itertools.product(*values))
    
    print(f"Grid search: {len(combinations)} configurations to test")
    print(f"Grid parameters: {keys}")
    print(f"Fixed parameters: {fixed_params}")
    
    results = []
    
    for idx, combo in enumerate(combinations):
        # Create config from combination
        config = fixed_params.copy()
        config.update(dict(zip(keys, combo)))
        
        print(f"\n{'='*60}")
        print(f"Configuration {idx+1}/{len(combinations)}")
        print(f"{'='*60}")
        for key in keys:
            print(f"  {key}: {dict(zip(keys, combo))[key]}")
        
        # Train with validation split
        validation_scores = []
        
        for val_run in range(n_validation_runs):
            # Create temporary output
            temp_dir = Path(output_dir) / f"config_{idx}" / f"run_{val_run}"
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Train for validation_timesteps
            try:
                if config.get("method", "rs_drl") == "rs_drl":
                    train_rs_drl(
                        total_timesteps=validation_timesteps,
                        seed=42 + val_run,  # Different seed per run
                        log_dir=str(temp_dir / "logs"),
                        save_path=str(temp_dir / "model"),
                        eval_freq=validation_timesteps // 5,
                        verbose=0,
                        **{k: v for k, v in config.items() if k != "method"},
                        **env_kwargs
                    )
                else:
                    train_baseline_dqn(
                        total_timesteps=validation_timesteps,
                        seed=42 + val_run,
                        log_dir=str(temp_dir / "logs"),
                        save_path=str(temp_dir / "model"),
                        eval_freq=validation_timesteps // 5,
                        verbose=0,
                        **{k: v for k, v in config.items() if k != "method"},
                        **env_kwargs
                    )
                
                # Evaluate
                model_path = str(temp_dir / "model.zip")
                metrics = evaluate_agent(
                    model_path,
                    env_kwargs,
                    n_episodes=5,
                    deterministic=True
                )
                
                # Use mean reward as validation score
                validation_scores.append(metrics["mean_reward"])
                
            except Exception as e:
                print(f"  Error in validation run {val_run}: {e}")
                validation_scores.append(-np.inf)
        
        # Average validation score
        mean_score = np.mean(validation_scores)
        std_score = np.std(validation_scores)
        
        result = {
            "config": config,
            "validation_score": float(mean_score),
            "validation_std": float(std_score),
            "validation_scores": [float(s) for s in validation_scores]
        }
        results.append(result)
        
        print(f"  Validation score: {mean_score:.4f} ± {std_score:.4f}")
    
    # Find best configuration
    best_idx = np.argmax([r["validation_score"] for r in results])
    best_result = results[best_idx]
    
    print(f"\n{'='*60}")
    print("Grid Search Results")
    print(f"{'='*60}")
    print(f"Best configuration:")
    for key, value in best_result["config"].items():
        print(f"  {key}: {value}")
    print(f"Validation score: {best_result['validation_score']:.4f} ± {best_result['validation_std']:.4f}")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "grid_search_results.json", 'w') as f:
        json.dump({
            "results": results,
            "best_config": best_result["config"],
            "best_validation_score": best_result["validation_score"]
        }, f, indent=2)
    
    print(f"\nResults saved to {output_path / 'grid_search_results.json'}")
    
    return best_result


def main():
    """Main grid search function."""
    parser = argparse.ArgumentParser(description="Hyperparameter grid search")
    
    parser.add_argument("--config", type=str, required=True,
                       help="Path to grid search config JSON")
    parser.add_argument("--validation-timesteps", type=int, default=2000,
                       help="Timesteps for validation")
    parser.add_argument("--n-validation-runs", type=int, default=3,
                       help="Number of validation runs per config")
    parser.add_argument("--output-dir", type=str, default="./grid_search",
                       help="Output directory")
    
    # Environment parameters
    parser.add_argument("--host", type=str, default="localhost",
                       help="DARTSim TCP host")
    parser.add_argument("--port", type=int, default=5418,
                       help="DARTSim TCP port")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    env_kwargs = {
        "host": args.host,
        "port": args.port,
        "sensor_lookahead": config.get("sensor_lookahead", 5)
    }
    
    # Run grid search
    best_result = grid_search_rs_drl(
        param_grid=config["param_grid"],
        env_kwargs=env_kwargs,
        validation_timesteps=args.validation_timesteps,
        n_validation_runs=args.n_validation_runs,
        output_dir=args.output_dir
    )
    
    print(f"\nBest configuration: {best_result['config']}")


if __name__ == "__main__":
    main()

