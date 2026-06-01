"""
Training script for RS-DRL on DARTSim using offline RL approach.

This script implements RS-DRL (Randomized Reward Shaping for Deep Reinforcement Learning)
following the plan.md approach: offline training from pre-collected data.

Key features:
- Offline RL training from pre-collected simulation data (primary mode)
- RS-DRL Algorithm 2 implementation (reward reshaping during minibatch replay)
- Memory-efficient data loading using generators
- Supports multiple scenarios (baseline, medium, hard)

Based on plan.md Phase 3: Offline RL training from replay buffer.

Usage:
    # Offline RL training (default and recommended)
    python train_rs_drl.py --offline --offline-data-dir ./data/offline --rho 0.3 --timesteps 10000 --seed 42
    
    # With scenario filter
    python train_rs_drl.py --offline --offline-data-dir ./data/offline --offline-scenario baseline --rho 0.3 --timesteps 10000

Example:
    python train_rs_drl.py --offline --offline-data-dir ./data/offline --rho 0.3 --timesteps 10000
"""

import argparse
import os
import numpy as np
import sys
import gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.rs_drl_dqn import RSDRLDQN, RSDRLTrainingCallback
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.buffers import ReplayBuffer
from scripts.offline_rl_training import (
    load_offline_dataset_generator,
    convert_episode_generator_to_buffer_format
)
import gymnasium as gym

# Import custom callbacks
try:
    from callbacks_with_early_stopping import ConvergenceCallback, TrainingProgressCallback
    CUSTOM_CALLBACKS_AVAILABLE = True
except ImportError:
    CUSTOM_CALLBACKS_AVAILABLE = False
    ConvergenceCallback = None
    TrainingProgressCallback = None


def create_env(env_id: str = "DARTSim", **env_kwargs) -> gym.Env:
    """
    Create and wrap DARTSim environment for offline training.
    
    Note: Online mode is deprecated. This function is kept for compatibility
    but will raise an error if called (offline mode should be used instead).
    
    Args:
        env_id: Environment identifier
        **env_kwargs: Additional arguments (ignored for offline mode)
    
    Returns:
        Wrapped Gymnasium environment
    """
    raise NotImplementedError(
        "Online mode with DARTSim library is no longer supported. "
        "Please use offline mode (--offline) with pre-collected data instead."
    )


def train_rs_drl(
    total_timesteps: int = 20000,
    rho: float = 0.3,
    learning_rate: float = 1e-4,
    gamma: float = 0.99,
    batch_size: int = 32,
    buffer_size: int = 100000,
    exploration_fraction: float = 0.1,
    exploration_initial_eps: float = 1.0,
    exploration_final_eps: float = 0.05,
    target_update_interval: int = 1000,
    train_freq: int = 4,
    gradient_steps: int = 1,
    seed: int = 42,
    log_dir: str = "./logs/rs_drl",
    save_path: str = "./models/rs_drl_dqn",
    eval_freq: int = 1000,
    eval_episodes: int = 5,
    verbose: int = 1,
    early_stopping: bool = True,
    convergence_patience: int = 10,
    convergence_min_delta: float = 0.01,
    offline_mode: bool = True,  # Default to offline mode per plan.md
    offline_data_dir: str = "./data/offline",
    offline_scenario: str = None,
    **env_kwargs
):
    """
    Train RS-DRL DQN agent using offline RL approach (plan.md Phase 3).
    
    Implements RS-DRL (Randomized Reward Shaping) for offline RL training:
    - Loads pre-collected offline data from JSON files
    - Populates replay buffer incrementally using generators (memory efficient)
    - Trains RS-DRL DQN by calling train() directly on replay buffer
    - Applies Algorithm 2 reward reshaping during minibatch replay
    
    This follows plan.md approach: offline training from replay buffer avoids
    TCP connection issues, enables faster/reproducible experiments, and supports
    large-scale training without simulator overhead.
    
    Args:
        total_timesteps: Number of gradient steps to train (converted to epochs)
        rho: Reshaping factor (fraction of failed transitions to reshape, Algorithm 2)
        learning_rate: Learning rate for DQN optimizer
        gamma: Discount factor
        batch_size: Minibatch size for training
        buffer_size: Maximum replay buffer size
        offline_mode: If True, use offline training (default, recommended per plan.md)
        offline_data_dir: Directory containing collected offline data
        offline_scenario: Optional scenario filter (baseline, medium, hard)
        seed: Random seed for reproducibility
        exploration_fraction: Fraction of timesteps for exploration decay (online mode only)
        exploration_initial_eps: Initial epsilon for ε-greedy (online mode only)
        exploration_final_eps: Final epsilon for ε-greedy (online mode only)
        target_update_interval: Target network update frequency
        train_freq: Training frequency (online mode only)
        gradient_steps: Gradient steps per training call
        seed: Random seed for reproducibility
        log_dir: Directory for TensorBoard logs
        save_path: Path to save trained model
        eval_freq: Evaluation frequency (online mode only)
        eval_episodes: Number of episodes for evaluation (online mode only)
        verbose: Verbosity level
        early_stopping: Enable early stopping (online mode only)
        convergence_patience: Patience for convergence detection (online mode only)
        convergence_min_delta: Minimum delta for convergence (online mode only)
        **env_kwargs: Additional environment arguments (online mode only)
    """
    # Set random seeds
    np.random.seed(seed)
    
    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    
    # Offline RL training mode (primary approach per plan.md)
    if offline_mode:
        # OFFLINE RL MODE: Load pre-collected data
        if not offline_data_dir:
            raise ValueError("offline_data_dir must be provided when offline_mode=True")
        
        print("=" * 60)
        print("OFFLINE RL TRAINING MODE")
        print("=" * 60)
        print(f"Loading offline data from: {offline_data_dir}")
        
        # Load offline dataset using generator (memory efficient)
        print("Loading offline dataset incrementally (using generator)...")
        episode_generator = load_offline_dataset_generator(offline_data_dir, scenario=offline_scenario)
        
        # Convert generator to buffer format batches
        print("Converting to replay buffer format (streaming batches)...")
        buffer_batch_generator = convert_episode_generator_to_buffer_format(
            episode_generator, 
            batch_size=10000  # Process 10k transitions at a time
        )
        
        # First, determine state dimension from first batch (before creating environment)
        print("\nDetermining state dimensions from first batch...")
        first_batch = next(buffer_batch_generator)
        obs_dim = len(first_batch["observations"][0])
        print(f"State dimension: {obs_dim}")
        print(f"First batch size: {len(first_batch['observations'])} transitions")
        
        # Create offline environment that uses real collected data
        print("\nCreating offline environment using collected data...")
        from src.dartsim_env import OfflineDARTSimEnv
        
        # Create environment with offline data - limit transitions for memory efficiency
        max_env_transitions = min(100000, total_timesteps * 20)  # Reasonable limit
        dummy_env = OfflineDARTSimEnv(
            obs_dim=obs_dim,
            data_dir=offline_data_dir,
            scenario=offline_scenario,
            max_transitions=max_env_transitions,
            seed=seed
        )
        obs_space = dummy_env.observation_space
        action_space = dummy_env.action_space
        
        # Extract reward stats from first batch before processing (to free memory)
        sample_rewards = first_batch['rewards'].tolist() if len(first_batch['rewards']) > 0 else []
        
        # Create RS-DRL DQN agent with dummy environment
        print(f"\nCreating RS-DRL DQN agent (rho={rho}) for offline training...")
        # Use large buffer size for offline data (will be populated incrementally)
        # But limit it based on actual data size to avoid excessive memory usage
        model = RSDRLDQN(
            "MlpPolicy",
            dummy_env,
            learning_rate=learning_rate,
            gamma=gamma,
            batch_size=batch_size,
            buffer_size=max(buffer_size, 1000000),  # Large buffer for offline data
            learning_starts=0,  # No exploration needed in offline mode
            exploration_fraction=0.0,  # No exploration
            exploration_initial_eps=0.0,
            exploration_final_eps=0.0,
            target_update_interval=target_update_interval,
            train_freq=1,  # Train every step
            gradient_steps=gradient_steps,
            tensorboard_log=log_dir,
            verbose=verbose,
            seed=seed,
            # RS-DRL specific parameters
            rho=rho,
            optimistic_reward=1.0
        )
        
        # Populate replay buffer incrementally from generator
        print(f"\nPopulating replay buffer incrementally from generator...")
        replay_buffer = model.replay_buffer
        
        # Add first batch
        total_transitions = 0
        batch_num = 1
        
        # Limit buffer size based on training needs (for 1000 timesteps, we don't need millions)
        # Use a reasonable multiplier: 10x training timesteps, but cap at buffer_size
        max_transitions_needed = min(buffer_size, total_timesteps * 10)
        print(f"Limiting replay buffer to {max_transitions_needed} transitions (for {total_timesteps} training timesteps)")
        
        # Process first batch
        batch_data = first_batch
        batch_size_add = len(batch_data["observations"])
        
        # Vectorized addition: prepare all data as arrays first
        obs_batch = batch_data["observations"]
        next_obs_batch = batch_data["next_observations"]
        actions_batch = batch_data["actions"].astype(np.int64)
        rewards_batch = batch_data["rewards"].astype(np.float32)
        dones_batch = batch_data["dones"].astype(np.bool_)
        
        # Add transitions in batches (more efficient than one-by-one)
        batch_add_size = min(batch_size_add, max_transitions_needed - total_transitions)
        for i in range(batch_add_size):
            replay_buffer.add(
                obs_batch[i:i+1], 
                next_obs_batch[i:i+1], 
                np.array([actions_batch[i]], dtype=np.int64),
                np.array([rewards_batch[i]], dtype=np.float32), 
                np.array([dones_batch[i]], dtype=np.bool_),
                [{}]
            )
            total_transitions += 1
        
        print(f"  Batch {batch_num}: Added {batch_add_size} transitions (total: {total_transitions}/{max_transitions_needed})")
        
        # Free first_batch memory now that we've processed it
        del first_batch, batch_data, obs_batch, next_obs_batch, actions_batch, rewards_batch, dones_batch
        gc.collect()
        
        # Process remaining batches from generator (until we have enough transitions)
        if total_transitions < max_transitions_needed:
            for batch_data in buffer_batch_generator:
                if total_transitions >= max_transitions_needed:
                    print(f"\nReached transition limit ({max_transitions_needed}). Stopping data loading.")
                    break
                    
                batch_num += 1
                batch_size_add = len(batch_data["observations"])
                
                # Vectorized preparation
                obs_batch = batch_data["observations"]
                next_obs_batch = batch_data["next_observations"]
                actions_batch = batch_data["actions"].astype(np.int64)
                rewards_batch = batch_data["rewards"].astype(np.float32)
                dones_batch = batch_data["dones"].astype(np.bool_)
                
                # Add only what we need
                batch_add_size = min(batch_size_add, max_transitions_needed - total_transitions)
                for i in range(batch_add_size):
                    replay_buffer.add(
                        obs_batch[i:i+1],
                        next_obs_batch[i:i+1],
                        np.array([actions_batch[i]], dtype=np.int64),
                        np.array([rewards_batch[i]], dtype=np.float32),
                        np.array([dones_batch[i]], dtype=np.bool_),
                        [{}]
                    )
                    total_transitions += 1
                
                # Progress indicator
                if batch_num % 10 == 0 or batch_size_add < 10000 or total_transitions >= max_transitions_needed:
                    print(f"  Batch {batch_num}: Added {batch_add_size} transitions (total: {total_transitions}/{max_transitions_needed})")
                
                # Free batch data after processing to reduce memory usage
                del batch_data, obs_batch, next_obs_batch, actions_batch, rewards_batch, dones_batch
                # Periodically trigger garbage collection every 50 batches
                if batch_num % 50 == 0:
                    gc.collect()
        else:
            print(f"Buffer already populated with {total_transitions} transitions (limit: {max_transitions_needed})")
        
        # Final garbage collection after populating buffer
        gc.collect()
        
        # ReplayBuffer doesn't support len(), so we use our tracked counter
        print(f"\nReplay buffer populated: {total_transitions} transitions")
        
        # Display reward stats from first batch (already extracted)
        if sample_rewards:
            print(f"Sample reward stats (first batch): avg={np.mean(sample_rewards):.4f}, min={np.min(sample_rewards):.4f}, max={np.max(sample_rewards):.4f}")
        del sample_rewards
        
        # Store replay_buffer and transition count for later use in training loop
        offline_replay_buffer = replay_buffer
        offline_num_transitions = total_transitions  # Store count for later use
        
        # For offline mode, we don't need a real evaluation environment
        # Use the dummy environment (evaluation can be skipped or done differently)
        eval_env = dummy_env  # Use dummy environment for offline mode
        train_env = dummy_env  # Keep dummy for compatibility
        
        # In offline mode, we'll train by calling train() directly
        print(f"\nStarting offline training for {total_timesteps} gradient steps...")
        print(f"RS-DRL parameters: rho={rho}, optimistic_reward=1.0")
        print(f"Hyperparameters: lr={learning_rate}, gamma={gamma}, batch_size={batch_size}")
        
        # Initialize offline_replay_buffer for offline mode (already set above)
        pass
    
    # Create callbacks
    callbacks = []
    
    # Evaluation callback (skip for offline mode - no real environment)
    # Offline mode always uses dummy environment, so skip evaluation callbacks
    if False:  # Online mode no longer supported
        # Limit evaluation episodes to avoid long hangs (max 2-3 episodes)
        safe_eval_episodes = min(eval_episodes, 3)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=save_path + "_best",
            log_path=log_dir,
            eval_freq=eval_freq,
            n_eval_episodes=safe_eval_episodes,
            deterministic=True,
            render=False,
            verbose=verbose,
            warn=True
        )
        callbacks.append(eval_callback)
        
        # Add convergence detection if available and enabled
        if CUSTOM_CALLBACKS_AVAILABLE and early_stopping and ConvergenceCallback is not None:
            convergence_callback = ConvergenceCallback(
                eval_callback,
                patience=convergence_patience,  # Stop if no improvement for N evaluations
                min_delta=convergence_min_delta,  # Minimum improvement threshold
                verbose=verbose
            )
            callbacks.append(convergence_callback)
            if verbose > 0:
                print(f"Early stopping enabled: patience={convergence_patience}, min_delta={convergence_min_delta}")
    else:
        # For offline mode, create a minimal eval_callback for compatibility
        eval_callback = None
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=eval_freq * 5,
        save_path=save_path + "_checkpoints",
        name_prefix="rs_drl_dqn"
    )
    callbacks.append(checkpoint_callback)
    
    # RS-DRL statistics callback
    rs_drl_callback = RSDRLTrainingCallback(verbose=verbose)
    callbacks.append(rs_drl_callback)
    
    # Add training progress logging if available
    if CUSTOM_CALLBACKS_AVAILABLE and TrainingProgressCallback is not None:
        progress_callback = TrainingProgressCallback(
            log_interval=min(eval_freq, 1000),  # Log every eval_freq or 1000 steps
            verbose=verbose
        )
        callbacks.append(progress_callback)
    
    # OFFLINE RL: Train directly from replay buffer
    print(f"Logging to: {log_dir}")
    print(f"Model will be saved to: {save_path}\n")
    
    # Train by calling train() method directly
    # Convert total_timesteps to gradient steps
    # Use stored transition count (ReplayBuffer doesn't support len())
    num_transitions = offline_num_transitions
    gradient_steps_per_epoch = max(1, num_transitions // batch_size)
    num_epochs = max(1, total_timesteps // gradient_steps_per_epoch)
    
    print(f"Training for {num_epochs} epochs ({gradient_steps_per_epoch} gradient steps per epoch)...")
    
    # Set model on all callbacks for offline training
    for callback in callbacks:
        if hasattr(callback, 'init_callback'):
            callback.init_callback(model)  # Only takes model argument
        elif hasattr(callback, 'model'):
            callback.model = model
    
    for epoch in range(num_epochs):
        # Train on all available data
        model.train(gradient_steps=gradient_steps_per_epoch, batch_size=batch_size)
        
        # Periodic evaluation and logging
        if (epoch + 1) % max(1, num_epochs // 10) == 0 or epoch == num_epochs - 1:
            # Run callbacks (checkpointing, statistics, etc.)
            # Note: Evaluation callbacks are skipped for offline mode
            for callback in callbacks:
                if hasattr(callback, '_on_step'):
                    callback._on_step()
            
            # Log progress
            if verbose > 0:
                stats = model.get_reshaping_stats()
                print(f"Epoch {epoch + 1}/{num_epochs} - Reshaping rate: {stats['reshaping_rate']:.4f}")
    
    print("\nOffline training completed!")
    
    # Save final model
    model.save(save_path)
    print(f"\nTraining completed! Model saved to {save_path}")
    
    # Print final statistics
    stats = model.get_reshaping_stats()
    print(f"\nReward Reshaping Statistics:")
    print(f"  Reshaping rate: {stats['reshaping_rate']:.4f}")
    print(f"  Total transitions: {stats.get('total_transitions', 0)}")
    print(f"  Reshaped transitions: {stats.get('reshaped_transitions', 0)}")
    
    # Cleanup
    train_env.close()
    eval_env.close()
    
    return model


def train_baseline_dqn(
    total_timesteps: int = 20000,
    learning_rate: float = 1e-4,
    gamma: float = 0.99,
    batch_size: int = 32,
    buffer_size: int = 100000,
    exploration_fraction: float = 0.1,
    exploration_initial_eps: float = 1.0,
    exploration_final_eps: float = 0.05,
    target_update_interval: int = 1000,
    train_freq: int = 4,
    gradient_steps: int = 1,
    seed: int = 42,
    log_dir: str = "./logs/dqn_baseline",
    save_path: str = "./models/dqn_baseline",
    eval_freq: int = 1000,
    eval_episodes: int = 5,
    verbose: int = 1,
    offline_mode: bool = True,  # Default to offline mode per plan.md
    offline_data_dir: str = "./data/offline",
    offline_scenario: str = None,
    **env_kwargs
):
    """
    Train baseline DQN agent (without RS-DRL) for comparison.
    
    Uses offline RL training by default (per plan.md). Can be used for
    comparison with RS-DRL in offline training mode.
    
    Note: This is equivalent to train_rs_drl with rho=0 (no reward reshaping).
    """
    # Use train_rs_drl with rho=0 for baseline (no reward reshaping)
    return train_rs_drl(
        total_timesteps=total_timesteps,
        rho=0.0,  # No reward reshaping = baseline DQN
        learning_rate=learning_rate,
        gamma=gamma,
        batch_size=batch_size,
        buffer_size=buffer_size,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=exploration_initial_eps,
        exploration_final_eps=exploration_final_eps,
        target_update_interval=target_update_interval,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        seed=seed,
        log_dir=log_dir,
        save_path=save_path,
        eval_freq=eval_freq,
        eval_episodes=eval_episodes,
        verbose=verbose,
        early_stopping=False,  # Baseline doesn't use early stopping
        offline_mode=offline_mode,
        offline_data_dir=offline_data_dir,
        offline_scenario=offline_scenario,
        **env_kwargs
    )


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train RS-DRL DQN on DARTSim")
    
    # Training parameters
    parser.add_argument("--method", type=str, default="rs_drl", choices=["rs_drl", "baseline"],
                       help="Training method: rs_drl or baseline")
    parser.add_argument("--timesteps", type=int, default=20000,
                       help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # RS-DRL parameters
    parser.add_argument("--rho", type=float, default=0.3,
                       help="Reshaping factor for RS-DRL")
    
    # DQN hyperparameters
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Minibatch size")
    parser.add_argument("--buffer-size", type=int, default=100000,
                       help="Replay buffer size")
    
    # Exploration parameters
    parser.add_argument("--exploration-fraction", type=float, default=0.1,
                       help="Fraction of timesteps for exploration decay")
    parser.add_argument("--exploration-initial-eps", type=float, default=1.0,
                       help="Initial epsilon")
    parser.add_argument("--exploration-final-eps", type=float, default=0.05,
                       help="Final epsilon")
    
    # Evaluation parameters
    parser.add_argument("--eval-freq", type=int, default=1000,
                       help="Evaluation frequency")
    parser.add_argument("--eval-episodes", type=int, default=5,
                       help="Number of episodes for evaluation")
    
    # Logging and saving
    parser.add_argument("--log-dir", type=str, default="./logs",
                       help="Directory for logs")
    parser.add_argument("--save-path", type=str, default="./models",
                       help="Path to save model")
    
    # Environment parameters
    parser.add_argument("--host", type=str, default="localhost",
                       help="DARTSim TCP host")
    parser.add_argument("--port", type=int, default=5418,
                       help="DARTSim TCP port")
    parser.add_argument("--sensor-lookahead", type=int, default=5,
                       help="Sensor lookahead cells")
    
    # Offline RL parameters (default mode per plan.md)
    parser.add_argument("--offline", action="store_true", default=True,
                       help="Enable offline RL training mode (default, uses pre-collected data)")
    parser.add_argument("--online", action="store_true",
                       help="Use online training mode (DEPRECATED - not supported, requires DARTSim library)")
    parser.add_argument("--offline-data-dir", type=str, default="./data/offline",
                       help="Directory containing offline collected data")
    parser.add_argument("--offline-scenario", type=str, default=None,
                       help="Filter offline data by scenario (baseline, medium, hard)")
    
    parser.add_argument("--verbose", type=int, default=1,
                       help="Verbosity level")
    
    args = parser.parse_args()
    
    # Set offline mode as default (unless --online is explicitly specified)
    if args.online:
        raise NotImplementedError(
            "Online training mode is no longer supported. "
            "Please use offline mode with pre-collected data instead. "
            "See plan.md for offline RL training approach."
        )
    args.offline = True  # Always use offline mode per plan.md
    
    # Setup paths
    if args.method == "rs_drl":
        log_dir = f"{args.log_dir}/rs_drl_rho{args.rho}"
        save_path = f"{args.save_path}/rs_drl_dqn_rho{args.rho}"
    else:
        log_dir = f"{args.log_dir}/dqn_baseline"
        save_path = f"{args.save_path}/dqn_baseline"
    
    # Environment kwargs (library interface doesn't need host/port)
    env_kwargs = {
        "sensor_lookahead": args.sensor_lookahead
    }
    
    # Add simulator arguments if provided
    if hasattr(args, 'sim_args') and args.sim_args:
        env_kwargs["sim_args"] = args.sim_args
    
    # Train
    if args.method == "rs_drl":
        train_rs_drl(
            total_timesteps=args.timesteps,
            rho=args.rho,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            exploration_fraction=args.exploration_fraction,
            exploration_initial_eps=args.exploration_initial_eps,
            exploration_final_eps=args.exploration_final_eps,
            seed=args.seed,
            log_dir=log_dir,
            save_path=save_path,
            eval_freq=args.eval_freq,
            eval_episodes=args.eval_episodes,
            verbose=args.verbose,
            offline_mode=args.offline,
            offline_data_dir=args.offline_data_dir,
            offline_scenario=args.offline_scenario,
            **env_kwargs
        )
    else:
        train_baseline_dqn(
            total_timesteps=args.timesteps,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            exploration_fraction=args.exploration_fraction,
            exploration_initial_eps=args.exploration_initial_eps,
            exploration_final_eps=args.exploration_final_eps,
            seed=args.seed,
            log_dir=log_dir,
            save_path=save_path,
            eval_freq=args.eval_freq,
            eval_episodes=args.eval_episodes,
            verbose=args.verbose,
            **env_kwargs
        )


if __name__ == "__main__":
    main()

