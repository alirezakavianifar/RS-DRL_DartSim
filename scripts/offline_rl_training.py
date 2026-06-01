"""
Offline Data Loading Utilities for RS-DRL Training

Provides functions for loading and processing pre-collected offline data
for RS-DRL offline training (plan.md Phase 3).

Key functions:
- load_offline_dataset_generator: Memory-efficient generator for loading episodes
- convert_episode_generator_to_buffer_format: Converts episodes to replay buffer format
- normalize_state: Normalizes state vectors to fixed dimension

Used by train_rs_drl.py for offline RL training from replay buffer.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Generator, Iterator
import sys

# Try to use ijson for streaming JSON parsing (optional, more memory efficient)
try:
    import ijson
    IJSON_AVAILABLE = True
except ImportError:
    IJSON_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
    from stable_baselines3.common.buffers import ReplayBuffer
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Warning: Stable-Baselines3 not available")


def load_offline_dataset(data_dir: str, scenario: str = None) -> List[Dict[str, Any]]:
    """
    Load collected offline dataset.
    
    Args:
        data_dir: Directory containing collected data
        scenario: Optional scenario filter
        
    Returns:
        List of transitions (state, action, reward, next_state, done)
    """
    data_path = Path(data_dir)
    
    # Find episode files
    if scenario:
        pattern = f"episodes_{scenario}_*.json"
    else:
        pattern = "episodes_*.json"
    
    episode_files = list(data_path.glob(pattern))
    
    if not episode_files:
        raise FileNotFoundError(f"No episode files found in {data_dir}")
    
    # Load most recent or all files
    print(f"Found {len(episode_files)} episode files")
    
    all_episodes = []
    total_size = 0
    for idx, ep_file in enumerate(episode_files):
        file_size = ep_file.stat().st_size / (1024 * 1024)  # Size in MB
        total_size += file_size
        
        # Progress indicator for large datasets
        if len(episode_files) > 10 and (idx + 1) % max(1, len(episode_files) // 10) == 0:
            progress = ((idx + 1) / len(episode_files)) * 100
            print(f"  Loading: {idx + 1}/{len(episode_files)} files ({progress:.1f}%) - {total_size:.1f} MB loaded...")
        
        try:
            with open(ep_file, 'r') as f:
                episodes = json.load(f)
                all_episodes.extend(episodes)
        except Exception as e:
            print(f"  Warning: Failed to load {ep_file.name}: {e}")
            continue
    
    print(f"Loaded {len(all_episodes)} transitions from {len(episode_files)} files ({total_size:.1f} MB total)")
    return all_episodes


def load_offline_dataset_generator(data_dir: str, scenario: str = None):
    """
    Generator that loads episodes one file at a time.
    
    Yields transitions from each file as they're loaded, reducing memory usage.
    
    Args:
        data_dir: Directory containing collected data
        scenario: Optional scenario filter
        
    Yields:
        Individual transitions (state, action, reward, next_state, done)
    """
    data_path = Path(data_dir)
    
    # Find episode files
    if scenario:
        pattern = f"episodes_{scenario}_*.json"
    else:
        pattern = "episodes_*.json"
    
    episode_files = sorted(list(data_path.glob(pattern)))
    
    if not episode_files:
        raise FileNotFoundError(f"No episode files found in {data_dir}")
    
    print(f"Found {len(episode_files)} episode files - loading incrementally...")
    
    total_loaded = 0
    total_size = 0
    
    for idx, ep_file in enumerate(episode_files):
        file_size = ep_file.stat().st_size / (1024 * 1024)  # Size in MB
        total_size += file_size
        
        try:
            # Load file and immediately yield episodes one by one
            # This way, we only keep one file's data in memory at a time
            file_episode_count = 0
            
            # Option 1: Use streaming JSON parser if available (for very large files)
            if IJSON_AVAILABLE and file_size > 50:  # Use streaming for files > 50MB
                with open(ep_file, 'rb') as f:
                    parser = ijson.items(f, 'item')
                    for episode in parser:
                        total_loaded += 1
                        file_episode_count += 1
                        yield episode
            else:
                # Option 2: Load entire file (faster for smaller files)
                with open(ep_file, 'r') as f:
                    episodes = json.load(f)
                    
                    # Process and yield episodes from this file
                    # After yielding, Python garbage collector can free the memory
                    for episode in episodes:
                        total_loaded += 1
                        file_episode_count += 1
                        yield episode
                        # After yielding, this episode is no longer needed in memory
                    
                    # Clear the episodes list to free memory before next file
                    del episodes
            
            # Progress indicator (outside the if/else, for both methods)
            if len(episode_files) > 10 and (idx + 1) % max(1, len(episode_files) // 10) == 0:
                progress = ((idx + 1) / len(episode_files)) * 100
                print(f"  Processed: {idx + 1}/{len(episode_files)} files ({progress:.1f}%) - {total_loaded} transitions ({file_episode_count} from this file) - {total_size:.1f} MB")
            elif file_size > 10:  # Show progress for large files
                print(f"  Processed file {idx + 1}/{len(episode_files)}: {ep_file.name} ({file_episode_count} episodes, {file_size:.1f} MB)")
        
        except Exception as e:
            print(f"  Warning: Failed to load {ep_file.name}: {e}")
            continue
    
    print(f"Generator complete: {total_loaded} transitions from {len(episode_files)} files ({total_size:.1f} MB total)")


def normalize_state(state, max_state_dim: int):
    """Normalize state to fixed dimension."""
    state_array = np.array(state, dtype=np.float32)
    if len(state_array) < max_state_dim:
        # Pad with zeros
        padded = np.zeros(max_state_dim, dtype=np.float32)
        padded[:len(state_array)] = state_array
        return padded
    elif len(state_array) > max_state_dim:
        # Truncate
        return state_array[:max_state_dim]
    return state_array


def convert_to_replay_buffer_format(episodes: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """
    Convert episodes to format suitable for Stable-Baselines3 replay buffer.
    
    Returns:
        Dictionary with observations, actions, rewards, next_observations, dones
    """
    if not episodes:
        return {}
    
    # Find maximum state dimension (first pass)
    print("Determining state dimensions...")
    max_state_dim = max(len(ep["state"]) for ep in episodes)
    print(f"Max state dimension: {max_state_dim}")
    
    # Extract and normalize arrays (with progress for large datasets)
    print(f"Normalizing {len(episodes)} transitions...")
    if len(episodes) > 100000:
        # For very large datasets, show progress
        states_list = []
        next_states_list = []
        batch_size = 50000
        for i in range(0, len(episodes), batch_size):
            batch = episodes[i:i+batch_size]
            states_list.extend([normalize_state(ep["state"], max_state_dim) for ep in batch])
            next_states_list.extend([normalize_state(ep["next_state"], max_state_dim) for ep in batch])
            if (i + batch_size) % 100000 == 0 or i + batch_size >= len(episodes):
                progress = min((i + batch_size) / len(episodes) * 100, 100)
                print(f"  Normalizing: {min(i + batch_size, len(episodes))}/{len(episodes)} ({progress:.1f}%)")
        states = np.array(states_list, dtype=np.float32)
        next_states = np.array(next_states_list, dtype=np.float32)
    else:
        states = np.array([normalize_state(ep["state"], max_state_dim) for ep in episodes], dtype=np.float32)
        next_states = np.array([normalize_state(ep["next_state"], max_state_dim) for ep in episodes], dtype=np.float32)
    
    # Action mapping
    action_map = {
        "IncAlt": 0, "DecAlt": 1, "IncAlt2": 2, "DecAlt2": 3,
        "GoTight": 4, "GoLoose": 5, "EcmOn": 6, "EcmOff": 7,
        "Unknown": 0, "Finished": 0
    }
    
    actions = np.array([action_map.get(ep["action"], 0) for ep in episodes], dtype=np.int64)
    
    rewards = np.array([ep["reward"] for ep in episodes], dtype=np.float32)
    dones = np.array([ep["done"] for ep in episodes], dtype=np.bool_)
    
    return {
        "observations": states,
        "next_observations": next_states,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "max_state_dim": max_state_dim  # Include for generator version
    }


def convert_episode_generator_to_buffer_format(
    episode_generator: Iterator[Dict[str, Any]],
    max_state_dim: int = None,
    batch_size: int = 10000
) -> Generator[Dict[str, np.ndarray], None, None]:
    """
    Convert episode generator to replay buffer format in batches.
    
    Yields batches of normalized transitions ready for replay buffer.
    
    Args:
        episode_generator: Generator yielding episode dictionaries
        max_state_dim: Maximum state dimension (if None, will determine from first batch)
        batch_size: Number of transitions per batch
        
    Yields:
        Dictionary with observations, actions, rewards, next_observations, dones
    """
    # Action mapping
    action_map = {
        "IncAlt": 0, "DecAlt": 1, "IncAlt2": 2, "DecAlt2": 3,
        "GoTight": 4, "GoLoose": 5, "EcmOn": 6, "EcmOff": 7,
        "Unknown": 0, "Finished": 0
    }
    
    batch = []
    total_processed = 0
    
    for episode in episode_generator:
        batch.append(episode)
        
        # Yield batch when full
        if len(batch) >= batch_size:
            # Determine max_state_dim from batch if not provided
            if max_state_dim is None:
                max_state_dim = max(len(ep["state"]) for ep in batch)
            
            # Convert batch
            states = np.array([normalize_state(ep["state"], max_state_dim) for ep in batch], dtype=np.float32)
            next_states = np.array([normalize_state(ep["next_state"], max_state_dim) for ep in batch], dtype=np.float32)
            actions = np.array([action_map.get(ep["action"], 0) for ep in batch], dtype=np.int64)
            rewards = np.array([ep["reward"] for ep in batch], dtype=np.float32)
            dones = np.array([ep["done"] for ep in batch], dtype=np.bool_)
            
            total_processed += len(batch)
            if total_processed % 50000 == 0:
                print(f"  Processed: {total_processed} transitions...")
            
            yield {
                "observations": states,
                "next_observations": next_states,
                "actions": actions,
                "rewards": rewards,
                "dones": dones
            }
            
            # Clear batch to free memory
            del batch
            batch = []
    
    # Yield remaining batch
    if batch:
        if max_state_dim is None:
            max_state_dim = max(len(ep["state"]) for ep in batch)
        
        states = np.array([normalize_state(ep["state"], max_state_dim) for ep in batch], dtype=np.float32)
        next_states = np.array([normalize_state(ep["next_state"], max_state_dim) for ep in batch], dtype=np.float32)
        actions = np.array([action_map.get(ep["action"], 0) for ep in batch], dtype=np.int64)
        rewards = np.array([ep["reward"] for ep in batch], dtype=np.float32)
        dones = np.array([ep["done"] for ep in batch], dtype=np.bool_)
        
        yield {
            "observations": states,
            "next_observations": next_states,
            "actions": actions,
            "rewards": rewards,
            "dones": dones
        }



