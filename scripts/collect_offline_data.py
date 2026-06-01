"""
Offline Data Collection for DARTSim

Collects simulation data offline by running multiple simulations
and saving trajectory data for batch/offline RL training.

This avoids TCP connection issues by using the library interface
through Docker container execution.
"""

import argparse
import subprocess
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import sys
import time
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.visualize_from_simulation import parse_simulation_output


def run_simulation(
    container_name: str = "dartsim",
    seed: int = None,
    map_size: int = None,
    num_targets: int = None,
    num_threats: int = None,
    timeout: int = 60
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Run a single DARTSim simulation and return trajectory + results.
    
    Returns:
        (trajectory, results)
    """
    # Build command
    cmd_parts = ["docker", "exec", container_name, "bash", "-c"]
    
    sim_opts = []
    if seed is not None:
        sim_opts.append(f"--seed={seed}")
    if map_size is not None:
        sim_opts.append(f"--map-size={map_size}")
    if num_targets is not None:
        sim_opts.append(f"--num-targets={num_targets}")
    if num_threats is not None:
        sim_opts.append(f"--num-threats={num_threats}")
    
    sim_opts_str = " ".join(sim_opts) if sim_opts else ""
    
    bash_cmd = f"cd /headless/dartsim/examples/simple-cpp && timeout {timeout} ./run.sh {sim_opts_str} 2>&1"
    cmd_parts.append(bash_cmd)
    
    try:
        result = subprocess.run(
            cmd_parts,
            capture_output=True,
            text=True,
            timeout=timeout + 10
        )
        
        output = result.stdout + result.stderr
        
        if len(output) < 100:
            print(f"Warning: Simulation output too short, may have failed")
            return [], {}
        
        trajectory, results = parse_simulation_output(output)
        
        # Add metadata
        results["seed"] = seed
        results["map_size"] = map_size
        results["num_targets"] = num_targets
        results["num_threats"] = num_threats
        
        return trajectory, results
        
    except subprocess.TimeoutExpired:
        print(f"Simulation timed out")
        return [], {}
    except Exception as e:
        print(f"Error running simulation: {e}")
        return [], {}


def collect_episode_data(
    trajectory: List[Dict[str, Any]],
    results: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Convert trajectory to RL episode format (state, action, reward, next_state, done).
    
    Returns:
        List of transitions (state, action, reward, next_state, done, info)
    """
    episodes = []
    
    if not trajectory:
        return episodes
    
    # Simple reward function based on results
    # Reward = 0 during episode, terminal reward at end
    for i, step in enumerate(trajectory):
        # State vector (simplified - can be expanded)
        state = [
            step["position_x"],
            step["position_y"],
            step["altitude"],
            1.0 if step["formation"] == "TIGHT" else 0.0,
            1.0 if step["ecm"] else 0.0,
            step["direction_x"],
            step["direction_y"],
            *[float(x) for x in step.get("threats_ahead", [])],
            *[float(x) for x in step.get("targets_ahead", [])]
        ]
        
        # Action (convert action name to index if needed)
        action = step.get("action", "Unknown")
        
        # Reward (sparse - only at end)
        if i == len(trajectory) - 1:
            # Terminal reward
            reward = 0.0
            if results.get("missionSuccess", False):
                reward += 0.4
            if results.get("targetsDetected", 0) > 0:
                reward += 0.3 * min(results.get("targetsDetected", 0) / 10.0, 1.0)
            if not results.get("destroyed", False):
                reward += 0.2
            else:
                reward -= 0.5
        else:
            reward = -0.01  # Step penalty
        
        # Next state
        if i < len(trajectory) - 1:
            next_step = trajectory[i + 1]
            next_state = [
                next_step["position_x"],
                next_step["position_y"],
                next_step["altitude"],
                1.0 if next_step["formation"] == "TIGHT" else 0.0,
                1.0 if next_step["ecm"] else 0.0,
                next_step["direction_x"],
                next_step["direction_y"],
                *[float(x) for x in next_step.get("threats_ahead", [])],
                *[float(x) for x in next_step.get("targets_ahead", [])]
            ]
        else:
            next_state = state  # Terminal state
        
        # Done flag
        done = (i == len(trajectory) - 1)
        
        episodes.append({
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "info": {
                "step": step["step"],
                "position": (step["position_x"], step["position_y"]),
                "altitude": step["altitude"],
                "formation": step["formation"],
                "ecm": step["ecm"]
            }
        })
    
    return episodes


def collect_offline_dataset(
    num_episodes: int = 100,
    output_dir: str = "./data/offline",
    seed_start: int = 42,
    container_name: str = "dartsim",
    scenario: str = "baseline"
) -> None:
    """
    Collect offline dataset by running multiple simulations.
    
    Args:
        num_episodes: Number of episodes to collect
        output_dir: Directory to save collected data
        seed_start: Starting seed (incremented for each episode)
        container_name: Docker container name
        scenario: Scenario configuration ("baseline", "medium", "hard")
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Scenario configurations
    scenarios = {
        "baseline": {"map_size": 40, "num_targets": 3, "num_threats": 5},
        "medium": {"map_size": 50, "num_targets": 5, "num_threats": 10},
        "hard": {"map_size": 60, "num_targets": 8, "num_threats": 15}
    }
    
    scenario_config = scenarios.get(scenario, scenarios["baseline"])
    
    print(f"Collecting offline dataset: {num_episodes} episodes")
    print(f"Scenario: {scenario} ({scenario_config})")
    print(f"Output directory: {output_dir}")
    print()
    
    all_episodes = []
    all_trajectories = []
    all_results = []
    
    successful = 0
    failed = 0
    
    for episode in range(num_episodes):
        seed = seed_start + episode
        
        print(f"Episode {episode + 1}/{num_episodes} (seed={seed})... ", end="", flush=True)
        
        trajectory, results = run_simulation(
            container_name=container_name,
            seed=seed,
            **scenario_config
        )
        
        if trajectory and results:
            # Convert to episode format
            episodes = collect_episode_data(trajectory, results)
            
            all_episodes.extend(episodes)
            all_trajectories.append(trajectory)
            all_results.append(results)
            
            successful += 1
            print(f"OK (steps={len(trajectory)}, targets={results.get('targetsDetected', 0)})")
        else:
            failed += 1
            print("FAILED")
        
        # Save intermediate results every 10 episodes
        if (episode + 1) % 10 == 0:
            _save_dataset(
                all_episodes, all_trajectories, all_results,
                output_path, episode + 1, scenario
            )
            print(f"  Saved intermediate checkpoint ({episode + 1} episodes)")
    
    # Final save
    print(f"\nCollection complete: {successful} successful, {failed} failed")
    _save_dataset(
        all_episodes, all_trajectories, all_results,
        output_path, num_episodes, scenario
    )
    
    # Print statistics
    if all_results:
        targets_detected = [r.get("targetsDetected", 0) for r in all_results]
        mission_success = sum(1 for r in all_results if r.get("missionSuccess", False))
        destroyed = sum(1 for r in all_results if r.get("destroyed", False))
        
        print(f"\nDataset Statistics:")
        print(f"  Total episodes: {len(all_results)}")
        print(f"  Total transitions: {len(all_episodes)}")
        print(f"  Avg targets detected: {np.mean(targets_detected):.2f}")
        print(f"  Mission success rate: {mission_success / len(all_results) * 100:.1f}%")
        print(f"  Destruction rate: {destroyed / len(all_results) * 100:.1f}%")
        print(f"  Avg episode length: {np.mean([len(t) for t in all_trajectories]):.1f} steps")


def _save_dataset(
    episodes: List[Dict[str, Any]],
    trajectories: List[List[Dict[str, Any]]],
    results: List[Dict[str, Any]],
    output_path: Path,
    episode_count: int,
    scenario: str
) -> None:
    """Save collected dataset."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save episodes (RL format)
    episodes_file = output_path / f"episodes_{scenario}_{episode_count}_{timestamp}.json"
    with open(episodes_file, 'w') as f:
        json.dump(episodes, f, indent=2)
    
    # Save trajectories (raw simulation data)
    trajectories_file = output_path / f"trajectories_{scenario}_{episode_count}_{timestamp}.json"
    with open(trajectories_file, 'w') as f:
        json.dump(trajectories, f, indent=2)
    
    # Save results summary
    results_file = output_path / f"results_{scenario}_{episode_count}_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save metadata
    metadata = {
        "scenario": scenario,
        "num_episodes": episode_count,
        "num_transitions": len(episodes),
        "timestamp": timestamp,
        "episodes_file": str(episodes_file.name),
        "trajectories_file": str(trajectories_file.name),
        "results_file": str(results_file.name)
    }
    
    metadata_file = output_path / f"metadata_{scenario}_{episode_count}_{timestamp}.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)


def collect_offline_dataset_tcp(
    num_episodes: int = 100,
    output_dir: str = "./data/offline",
    seed_start: int = 0,
    container_name: str = "dartsim",
    scenario: str = "baseline",
    epsilon: float = 1.0,
    sim_args: str = "",
) -> None:
    """
    Collect offline data via the TCP adaptation-manager interface.

    Unlike the library-based collector (which uses simple-cpp and only exercises
    4 of 8 actions), this function connects to DARTSim through the TCP interface
    and sends actions chosen by an ε-greedy random policy.  With epsilon=1.0
    (default) every action is chosen uniformly at random, giving full 8-action
    coverage in the resulting dataset.

    Prerequisites
    -------------
    * Docker container running with port 5418 exposed.
    * Call ``utils/start_dartsim_live.ps1`` first, or pass ``container_name``
      so that this function manages DARTSim restarts automatically.

    Parameters
    ----------
    num_episodes : int
        Number of complete episodes to collect.
    output_dir : str
        Directory to write the dataset files.
    seed_start : int
        First seed value (episodes use seed_start, seed_start+1, …).
    container_name : str
        Docker container name used to restart DARTSim between episodes.
    scenario : str
        One of ``'baseline'``, ``'medium'``, ``'hard'`` — determines sim_args
        if ``sim_args`` is empty.
    epsilon : float
        Probability of choosing a random action (1.0 = fully random).
    sim_args : str
        Extra command-line flags forwarded to DARTSim's run.sh.
    """
    from src.live_dartsim_env import LiveDARTSimEnv, ACTIONS

    scenario_sim_args = {
        "baseline": "--map-size=40 --num-targets=3 --num-threats=5",
        "medium":   "--map-size=50 --num-targets=5 --num-threats=10",
        "hard":     "--map-size=60 --num-targets=8 --num-threats=15",
    }
    if not sim_args:
        sim_args = scenario_sim_args.get(scenario, "")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nTCP offline data collection — {num_episodes} episodes")
    print(f"  Scenario : {scenario} ({sim_args})")
    print(f"  Epsilon  : {epsilon}  (1.0 = fully random → all 8 actions)")
    print(f"  Output   : {output_dir}\n")

    env = LiveDARTSimEnv(
        container_name=container_name,
        sim_args=sim_args,
        connect_timeout=40.0,
    )

    all_episodes_rl: List[Dict[str, Any]] = []
    all_results:     List[Dict[str, Any]] = []
    action_counts = {a: 0 for a in ACTIONS}
    successful = 0
    failed = 0
    rng = np.random.default_rng(seed_start)

    for ep_idx in range(num_episodes):
        seed = seed_start + ep_idx
        print(f"Episode {ep_idx + 1}/{num_episodes} (seed={seed}) ... ", end="", flush=True)

        try:
            obs, _ = env.reset(seed=seed)
        except Exception as e:
            print(f"RESET FAILED: {e}")
            failed += 1
            continue

        transitions: List[Dict[str, Any]] = []
        done = False
        results: Dict[str, Any] = {}
        step_count = 0

        while not done:
            # ε-greedy action selection
            if rng.random() < epsilon:
                action = int(rng.integers(0, env.action_space.n))
            else:
                action = 0  # default fallback

            try:
                next_obs, reward, terminated, truncated, info = env.step(action)
            except Exception as e:
                print(f"STEP FAILED at step {step_count}: {e}")
                done = True
                info = {}
                reward = 0.0
                terminated = True
                next_obs = obs

            done = terminated or truncated
            if done:
                results = info.get("results", {})

            # Build state vector same format as library collector
            state_vec = obs.tolist()
            next_state_vec = next_obs.tolist()

            transitions.append({
                "state": state_vec,
                "action": ACTIONS[action],
                "reward": float(reward),
                "next_state": next_state_vec,
                "done": bool(done),
                "info": {"step": step_count},
            })

            action_counts[ACTIONS[action]] += 1
            obs = next_obs
            step_count += 1

        if transitions and results:
            all_episodes_rl.extend(transitions)
            all_results.append(results)
            successful += 1
            print(
                f"OK  steps={step_count}  "
                f"success={results.get('missionSuccess', '?')}  "
                f"targets={results.get('targetsDetected', '?')}"
            )
        else:
            failed += 1
            print("EMPTY")

        # Checkpoint every 50 episodes
        if (ep_idx + 1) % 50 == 0 and all_episodes_rl:
            _save_dataset(
                all_episodes_rl, [], all_results,
                output_path, ep_idx + 1, f"{scenario}_tcp"
            )
            print(f"  [checkpoint saved at episode {ep_idx + 1}]")

    env.close()

    print(f"\nCollection complete: {successful} OK, {failed} failed")

    # Action distribution summary
    total_actions = sum(action_counts.values())
    print("\nAction distribution (ground-truth behaviour policy):")
    for a, c in action_counts.items():
        pct = 100 * c / total_actions if total_actions else 0
        bar = "█" * int(pct / 2)
        print(f"  {a:12s}: {c:6d}  ({pct:5.1f}%)  {bar}")

    if all_episodes_rl:
        _save_dataset(
            all_episodes_rl, [], all_results,
            output_path, num_episodes, f"{scenario}_tcp"
        )
        print(f"\nDataset saved ({len(all_episodes_rl)} transitions, {len(all_results)} episodes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect offline DARTSim dataset")
    parser.add_argument("--episodes", type=int, default=100,
                       help="Number of episodes to collect")
    parser.add_argument("--output-dir", type=str, default="./data/offline",
                       help="Output directory for collected data")
    parser.add_argument("--seed-start", type=int, default=42,
                       help="Starting seed (incremented for each episode)")
    parser.add_argument("--container", type=str, default="dartsim",
                       help="Docker container name")
    parser.add_argument("--scenario", type=str, default="baseline",
                       choices=["baseline", "medium", "hard"],
                       help="Scenario configuration")
    parser.add_argument("--use-tcp", action="store_true",
                       help="Collect via TCP interface (all 8 actions, random policy). "
                            "Requires DARTSim container with port 5418 exposed.")
    parser.add_argument("--epsilon", type=float, default=1.0,
                       help="ε-greedy exploration rate for TCP collection (default: 1.0)")
    parser.add_argument("--sim-args", default="",
                       help="Extra arguments for DARTSim run.sh (TCP mode only)")

    args = parser.parse_args()

    if args.use_tcp:
        collect_offline_dataset_tcp(
            num_episodes=args.episodes,
            output_dir=args.output_dir,
            seed_start=args.seed_start,
            container_name=args.container,
            scenario=args.scenario,
            epsilon=args.epsilon,
            sim_args=args.sim_args,
        )
    else:
        collect_offline_dataset(
            num_episodes=args.episodes,
            output_dir=args.output_dir,
            seed_start=args.seed_start,
            container_name=args.container,
            scenario=args.scenario
        )

