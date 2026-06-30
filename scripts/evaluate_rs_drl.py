"""
Evaluation script for RS-DRL on DARTSim.
Implements evaluation metrics from Phase 0 and RS-DRL paper:
- Asymptotic performance
- Time-to-threshold
- Total Performance (area under curve)
- Domain-specific metrics

Usage:
    python evaluate_rs_drl.py --model-path <path> --episodes 10
"""

import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dartsim_env import OfflineDARTSimEnv
from stable_baselines3 import DQN
from src.rs_drl_dqn import RSDRLDQN
import src.rs_drl_dqn as _rs_drl_module
# Allow cloudpickle to resolve 'rs_drl_dqn' saved without 'src.' prefix
sys.modules.setdefault('rs_drl_dqn', _rs_drl_module)
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt


def evaluate_agent(
    model_path: str,
    env_kwargs: Dict,
    n_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    offline_data_dir: str = "./data/offline",
    offline_scenario: str = None
) -> Dict:
    """
    Evaluate a trained agent using offline DARTSim environment.
    
    Args:
        model_path: Path to saved model
        env_kwargs: Environment configuration (obs_dim, etc.)
        n_episodes: Number of evaluation episodes
        deterministic: Use deterministic policy
        render: Render episodes (not supported in offline mode)
        offline_data_dir: Directory containing offline collected data
        offline_scenario: Optional scenario filter (baseline, medium, hard)
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Create offline environment using collected data
    obs_dim = env_kwargs.get("obs_dim", 17)
    max_transitions = env_kwargs.get("max_transitions", 50000)
    seed = env_kwargs.get("seed", 42)
    
    env = OfflineDARTSimEnv(
        obs_dim=obs_dim,
        data_dir=offline_data_dir,
        scenario=offline_scenario,
        max_transitions=max_transitions,
        seed=seed
    )
    env.sensor_noise = env_kwargs.get("sensor_noise", 0.0)
    env = Monitor(env, filename=None, allow_early_resets=True)
    
    # Load model
    try:
        model = RSDRLDQN.load(model_path, env=env)
    except:
        # Try baseline DQN if RS-DRL fails
        model = DQN.load(model_path, env=env)
    
    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    mission_successes = []
    targets_detected = []
    team_destroyed = []
    decision_times = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0
        
        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=deterministic)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1
            
            if done:
                # Extract final results
                if "results" in info and info["results"]:
                    results = info["results"]
                    mission_successes.append(1.0 if results.get("missionSuccess", False) else 0.0)
                    targets_detected.append(results.get("targetsDetected", 0))
                    team_destroyed.append(1.0 if results.get("destroyed", False) else 0.0)
                    decision_times.append(results.get("decisionTimeAvg", 0.0))
                else:
                    mission_successes.append(0.0)
                    targets_detected.append(0)
                    team_destroyed.append(1.0)
                    decision_times.append(0.0)
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    env.close()
    
    # Compute statistics
    metrics = {
        "n_episodes": n_episodes,
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_episode_length": np.mean(episode_lengths),
        "mission_success_rate": np.mean(mission_successes),
        "mean_targets_detected": np.mean(targets_detected),
        "std_targets_detected": np.std(targets_detected),
        "team_destruction_rate": np.mean(team_destroyed),
        "mean_decision_time": np.mean(decision_times),
        "std_decision_time": np.std(decision_times),
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "mission_successes": mission_successes,
        "targets_detected": targets_detected
    }
    
    return metrics


def compute_time_to_threshold(
    learning_curve: np.ndarray,
    threshold: float = 0.9
) -> Tuple[int, float]:
    """
    Compute time-to-threshold (steps to reach reward threshold).
    
    Args:
        learning_curve: Array of mean rewards over time
        threshold: Reward threshold (default: 0.9)
    
    Returns:
        (steps, achieved_value) or (None, max_value) if threshold not reached
    """
    # Smooth curve (moving average)
    window = min(100, len(learning_curve) // 10)
    if window > 1:
        smoothed = np.convolve(learning_curve, np.ones(window)/window, mode='valid')
        # Pad to original length
        smoothed = np.pad(smoothed, (window//2, len(learning_curve) - len(smoothed) - window//2), mode='edge')
    else:
        smoothed = learning_curve
    
    # Find first point where smoothed value >= threshold
    above_threshold = np.where(smoothed >= threshold)[0]
    
    if len(above_threshold) > 0:
        return int(above_threshold[0]), float(smoothed[above_threshold[0]])
    else:
        max_value = float(np.max(smoothed))
        return None, max_value


def compute_total_performance(learning_curve: np.ndarray) -> float:
    """
    Compute Total Performance (TP) - area under learning curve.
    
    Args:
        learning_curve: Array of mean rewards over time
    
    Returns:
        Total performance (area under curve)
    """
    # Normalize by timesteps
    trapz_fn = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')
    return float(trapz_fn(learning_curve) / len(learning_curve))


def load_learning_curve(log_dir: str) -> np.ndarray:
    """
    Load learning curve from TensorBoard logs or CSV.
    
    Args:
        log_dir: Directory containing logs
    
    Returns:
        Array of mean rewards over time
    """
    # Try to load from monitor CSV first
    monitor_file = Path(log_dir) / "monitor.csv"
    if monitor_file.exists():
        df = pd.read_csv(monitor_file, skiprows=1)
        rewards = df['r'].values
        # Compute rolling mean
        window = min(100, len(rewards) // 10)
        if window > 1:
            return np.convolve(rewards, np.ones(window)/window, mode='valid')
        return rewards
    
    # Fallback: return zeros if no data
    return np.array([0.0] * 1000)


def compare_models(
    model_paths: Dict[str, str],
    env_kwargs: Dict,
    n_episodes: int = 10,
    output_dir: str = "./results",
    offline_data_dir: str = "./data/offline",
    offline_scenario: str = None
) -> pd.DataFrame:
    """
    Compare multiple models and generate comparison report.
    
    Args:
        model_paths: Dictionary mapping model names to paths
        env_kwargs: Environment configuration
        n_episodes: Number of evaluation episodes per model
        output_dir: Output directory for results
    
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for model_name, model_path in model_paths.items():
        print(f"\nEvaluating {model_name}...")
        metrics = evaluate_agent(
            model_path, env_kwargs, n_episodes,
            offline_data_dir=offline_data_dir,
            offline_scenario=offline_scenario
        )
        
        # Try to load learning curve
        log_dir = str(Path(model_path).parent.parent / "logs" / model_name)
        learning_curve = load_learning_curve(log_dir)
        
        # Compute time-to-threshold
        ttt_09, value_09 = compute_time_to_threshold(learning_curve, threshold=0.9)
        ttt_08, value_08 = compute_time_to_threshold(learning_curve, threshold=0.8)
        ttt_07, value_07 = compute_time_to_threshold(learning_curve, threshold=0.7)
        
        # Compute total performance
        tp = compute_total_performance(learning_curve)
        
        results.append({
            "model": model_name,
            "mean_reward": metrics["mean_reward"],
            "std_reward": metrics["std_reward"],
            "mission_success_rate": metrics["mission_success_rate"],
            "mean_targets_detected": metrics["mean_targets_detected"],
            "team_destruction_rate": metrics["team_destruction_rate"],
            "mean_decision_time": metrics["mean_decision_time"],
            "time_to_threshold_0.9": ttt_09 if ttt_09 is not None else -1,
            "time_to_threshold_0.8": ttt_08 if ttt_08 is not None else -1,
            "time_to_threshold_0.7": ttt_07 if ttt_07 is not None else -1,
            "total_performance": tp,
            "n_episodes": n_episodes
        })
        
        print(f"  Mean reward: {metrics['mean_reward']:.4f} ± {metrics['std_reward']:.4f}")
        print(f"  Mission success rate: {metrics['mission_success_rate']:.4f}")
        print(f"  Targets detected: {metrics['mean_targets_detected']:.2f} ± {metrics['std_targets_detected']:.2f}")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path / "comparison_results.csv", index=False)
    df.to_json(output_path / "comparison_results.json", indent=2)
    
    print(f"\nComparison results saved to {output_path}")
    print("\n" + "="*60)
    print("Comparison Summary")
    print("="*60)
    print(df.to_string(index=False))
    
    return df


def plot_learning_curves(
    log_dirs: Dict[str, str],
    output_path: str = "./results/learning_curves.png"
) -> None:
    """
    Plot learning curves for multiple models.
    
    Args:
        log_dirs: Dictionary mapping model names to log directories
        output_path: Output path for plot
    """
    plt.figure(figsize=(12, 6))
    
    for model_name, log_dir in log_dirs.items():
        learning_curve = load_learning_curve(log_dir)
        if len(learning_curve) > 0:
            plt.plot(learning_curve, label=model_name, alpha=0.7)
    
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Reward")
    plt.title("Learning Curves Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Learning curves plot saved to {output_path}")
    plt.close()


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate RS-DRL models")
    
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to model file")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of evaluation episodes")
    parser.add_argument("--output-dir", type=str, default="./results",
                       help="Output directory for results")
    parser.add_argument("--deterministic", action="store_true", default=True,
                       help="Use deterministic policy")
    
    # Offline environment parameters
    parser.add_argument("--offline-data-dir", type=str, default="./data/offline",
                       help="Directory containing offline collected data")
    parser.add_argument("--offline-scenario", type=str, default=None,
                       help="Filter offline data by scenario (baseline, medium, hard)")
    parser.add_argument("--obs-dim", type=int, default=17,
                       help="Observation dimension (state vector size)")
    parser.add_argument("--max-transitions", type=int, default=50000,
                       help="Maximum transitions to load for evaluation")
    
    # Comparison mode
    parser.add_argument("--compare", type=str, nargs="+",
                       help="Compare multiple models (format: name1:path1 name2:path2 ...)")
    parser.add_argument("--sensor-noise", type=float, default=0.0,
                       help="Threat sensor failure rate / masking probability (default: 0.0)")
    
    args = parser.parse_args()
    
    env_kwargs = {
        "obs_dim": args.obs_dim,
        "max_transitions": args.max_transitions,
        "seed": 42,
        "sensor_noise": args.sensor_noise
    }
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.compare:
        # Comparison mode
        model_paths = {}
        for pair in args.compare:
            name, path = pair.split(":")
            model_paths[name] = path
        
        compare_models(
            model_paths, env_kwargs, args.episodes, str(output_dir),
            offline_data_dir=args.offline_data_dir,
            offline_scenario=args.offline_scenario
        )
    else:
        # Single model evaluation
        print(f"Evaluating model: {args.model_path}")
        metrics = evaluate_agent(
            args.model_path,
            env_kwargs,
            args.episodes,
            args.deterministic,
            offline_data_dir=args.offline_data_dir,
            offline_scenario=args.offline_scenario
        )
        
        # Save metrics
        metrics_file = output_dir / "evaluation_metrics.json"
        # Convert numpy arrays to lists for JSON
        metrics_serializable = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in metrics.items()
        }
        with open(metrics_file, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        
        print("\n" + "="*60)
        print("Evaluation Results")
        print("="*60)
        print(f"Mean Reward: {metrics['mean_reward']:.4f} ± {metrics['std_reward']:.4f}")
        print(f"Mission Success Rate: {metrics['mission_success_rate']:.4f}")
        print(f"Mean Targets Detected: {metrics['mean_targets_detected']:.2f} ± {metrics['std_targets_detected']:.2f}")
        print(f"Team Destruction Rate: {metrics['team_destruction_rate']:.4f}")
        print(f"Mean Decision Time: {metrics['mean_decision_time']:.4f} ± {metrics['std_decision_time']:.4f}")
        print(f"\nResults saved to {metrics_file}")


if __name__ == "__main__":
    main()

