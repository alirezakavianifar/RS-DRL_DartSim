"""
Training script with MAPE-K integration.
Demonstrates RS-DRL with on-demand retraining via MAPE-K loop.

Usage:
    python train_with_mapek.py --timesteps 20000 --threshold-min-reward 0.7
"""

import argparse
import numpy as np
from pathlib import Path
import time
from typing import Dict, Optional
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dartsim_env import OfflineDARTSimEnv
from src.rs_drl_dqn import RSDRLDQN
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from src.mape_k_architecture import (
    MAPEKManager, Knowledge, Thresholds,
    AdaptationTrigger, SystemMetrics
)


def train_with_mapek(
    total_timesteps: int = 20000,
    initial_model_path: Optional[str] = None,
    rho: float = 0.3,
    learning_rate: float = 1e-4,
    gamma: float = 0.99,
    batch_size: int = 32,
    buffer_size: int = 100000,
    seed: int = 42,
    log_dir: str = "./logs/mapek",
    save_path: str = "./models/mapek_rs_drl",
    threshold_min_reward: float = 0.7,
    threshold_min_success_rate: float = 0.8,
    retrain_threshold_violations: int = 3,
    retrain_interval: int = 1000,
    env_kwargs: Dict = None,
    verbose: int = 1
):
    """
    Train RS-DRL with MAPE-K on-demand retraining.
    
    Args:
        total_timesteps: Total training timesteps
        initial_model_path: Path to initial model (if exists)
        rho: RS-DRL reshaping factor
        learning_rate: Learning rate
        gamma: Discount factor
        batch_size: Batch size
        buffer_size: Replay buffer size
        seed: Random seed
        log_dir: Logging directory
        save_path: Model save path
        threshold_min_reward: Minimum reward threshold
        threshold_min_success_rate: Minimum mission success rate
        retrain_threshold_violations: Violations before retraining
        retrain_interval: Minimum timesteps between retrains
        env_kwargs: Environment configuration
        verbose: Verbosity level
    """
    if env_kwargs is None:
        env_kwargs = {
            "offline_data_dir": "./data/offline",
            "offline_scenario": None,
            "obs_dim": 17,
            "max_transitions": 100000
        }
    
    # Create directories
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize knowledge repository
    knowledge = Knowledge(
        thresholds=Thresholds(
            min_reward=threshold_min_reward,
            min_mission_success_rate=threshold_min_success_rate
        ),
        retrain_threshold_violations=retrain_threshold_violations,
        retrain_interval=retrain_interval
    )
    
    # Create offline environment
    print("Creating offline DARTSim environment...")
    offline_data_dir = env_kwargs.get("offline_data_dir", "./data/offline")
    offline_scenario = env_kwargs.get("offline_scenario", None)
    obs_dim = env_kwargs.get("obs_dim", 17)
    max_transitions = env_kwargs.get("max_transitions", 100000)
    
    env = OfflineDARTSimEnv(
        obs_dim=obs_dim,
        data_dir=offline_data_dir,
        scenario=offline_scenario,
        max_transitions=max_transitions,
        seed=seed
    )
    env = Monitor(env, filename=None, allow_early_resets=True)
    
    # Initialize or load model
    if initial_model_path and Path(initial_model_path).exists():
        print(f"Loading initial model from {initial_model_path}...")
        try:
            model = RSDRLDQN.load(initial_model_path, env=env)
            knowledge.current_model_path = initial_model_path
        except:
            model = DQN.load(initial_model_path, env=env)
            knowledge.current_model_path = initial_model_path
    else:
        print("Creating new RS-DRL DQN model...")
        model = RSDRLDQN(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            gamma=gamma,
            batch_size=batch_size,
            buffer_size=buffer_size,
            learning_starts=max(100, batch_size),  # Start learning after buffer has some samples
            train_freq=4,  # Train every 4 steps
            gradient_steps=1,  # Gradient steps per training call
            tensorboard_log=log_dir,
            verbose=verbose,
            seed=seed,
            rho=rho,
            optimistic_reward=1.0
        )
        # Save initial model
        initial_path = save_path + "_initial.zip"
        model.save(initial_path)
        knowledge.current_model_path = initial_path
    
    # Initialize MAPE-K Manager
    mapek = MAPEKManager(knowledge, env=env, model_path=knowledge.current_model_path)
    
    # Training loop with MAPE-K
    print(f"\nStarting training with MAPE-K (on-demand retraining)...")
    print(f"Thresholds: min_reward={threshold_min_reward}, min_success_rate={threshold_min_success_rate}")
    print(f"Retrain after {retrain_threshold_violations} violations (min interval: {retrain_interval} timesteps)")
    print(f"Total timesteps: {total_timesteps}\n")
    
    obs, info = env.reset()
    episode_reward = 0.0
    timestep = 0
    episode_num = 0
    
    adaptation_log = []
    
    while timestep < total_timesteps:
        # Get action from model
        action, _ = model.predict(obs, deterministic=False)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        timestep += 1
        
        # MAPE-K cycle: Monitor, Analyze, Plan, Execute
        adaptation_result = mapek.step(timestep, reward, info)
        
        if adaptation_result and adaptation_result.get("executed"):
            if verbose > 0:
                print(f"\n[Timestep {timestep}] Adaptation triggered: {adaptation_result.get('action')}")
                if "new_model_path" in adaptation_result:
                    print(f"  New model saved: {adaptation_result['new_model_path']}")
            
            # Reload model if retrained
            if "new_model_path" in adaptation_result:
                try:
                    model = RSDRLDQN.load(adaptation_result["new_model_path"], env=env)
                    knowledge.current_model_path = adaptation_result["new_model_path"]
                except:
                    pass
            
            adaptation_log.append(adaptation_result)
        
        # Reset if episode done
        if done:
            episode_num += 1
            if verbose > 0 and episode_num % 10 == 0:
                stats = mapek.get_status()
                print(f"Episode {episode_num}: Reward={episode_reward:.2f}, "
                      f"Metrics={stats['metrics_count']}, "
                      f"Adaptations={stats['adaptations_count']}")
            
            episode_reward = 0.0
            obs, info = env.reset()
    
    # Save final model
    final_path = save_path + "_final.zip"
    model.save(final_path)
    knowledge.current_model_path = final_path
    
    # Save knowledge repository
    knowledge_path = Path(save_path).parent / "mapek_knowledge.json"
    mapek.save_knowledge(str(knowledge_path))
    
    # Print summary
    print("\n" + "="*60)
    print("Training with MAPE-K Completed")
    print("="*60)
    stats = mapek.get_status()
    print(f"Total timesteps: {timestep}")
    print(f"Episodes: {episode_num}")
    print(f"Adaptations performed: {stats['adaptations_count']}")
    print(f"Final model: {final_path}")
    print(f"Knowledge saved: {knowledge_path}")
    
    if len(adaptation_log) > 0:
        print(f"\nAdaptation log ({len(adaptation_log)} adaptations):")
        for i, adapt in enumerate(adaptation_log[:5]):  # Show first 5
            print(f"  {i+1}. Timestep {adapt.get('timestep')}: {adapt.get('action')}")
        if len(adaptation_log) > 5:
            print(f"  ... and {len(adaptation_log) - 5} more")
    
    env.close()
    
    return model, mapek


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train RS-DRL with MAPE-K integration")
    
    # Training parameters
    parser.add_argument("--timesteps", type=int, default=20000,
                       help="Total training timesteps")
    parser.add_argument("--initial-model", type=str,
                       help="Path to initial model (if exists)")
    parser.add_argument("--rho", type=float, default=0.3,
                       help="RS-DRL reshaping factor")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # MAPE-K parameters
    parser.add_argument("--threshold-min-reward", type=float, default=0.7,
                       help="Minimum reward threshold")
    parser.add_argument("--threshold-min-success-rate", type=float, default=0.8,
                       help="Minimum mission success rate threshold")
    parser.add_argument("--retrain-violations", type=int, default=3,
                       help="Number of violations before retraining")
    parser.add_argument("--retrain-interval", type=int, default=1000,
                       help="Minimum timesteps between retrains")
    
    # DQN hyperparameters
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    
    # Output
    parser.add_argument("--log-dir", type=str, default="./logs/mapek",
                       help="Log directory")
    parser.add_argument("--save-path", type=str, default="./models/mapek_rs_drl",
                       help="Model save path")
    
    # Offline environment parameters
    parser.add_argument("--offline-data-dir", type=str, default="./data/offline",
                       help="Directory containing offline collected data")
    parser.add_argument("--offline-scenario", type=str, default=None,
                       help="Filter offline data by scenario (baseline, medium, hard)")
    parser.add_argument("--obs-dim", type=int, default=17,
                       help="Observation dimension (state vector size)")
    parser.add_argument("--max-transitions", type=int, default=100000,
                       help="Maximum transitions to load")
    
    parser.add_argument("--verbose", type=int, default=1,
                       help="Verbosity level")
    
    args = parser.parse_args()
    
    env_kwargs = {
        "offline_data_dir": args.offline_data_dir,
        "offline_scenario": args.offline_scenario,
        "obs_dim": args.obs_dim,
        "max_transitions": args.max_transitions
    }
    
    train_with_mapek(
        total_timesteps=args.timesteps,
        initial_model_path=args.initial_model,
        rho=args.rho,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        batch_size=args.batch_size,
        seed=args.seed,
        log_dir=args.log_dir,
        save_path=args.save_path,
        threshold_min_reward=args.threshold_min_reward,
        threshold_min_success_rate=args.threshold_min_success_rate,
        retrain_threshold_violations=args.retrain_violations,
        retrain_interval=args.retrain_interval,
        env_kwargs=env_kwargs,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()

