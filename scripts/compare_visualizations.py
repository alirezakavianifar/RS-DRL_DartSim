"""
Compare multiple DARTSim simulation runs visually.

Usage:
    python scripts/compare_visualizations.py --runs 3 --steps 50
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.visualize_dartsim import DARTSimVisualizer, run_simulation_with_visualization
from src.dartsim_env import DARTSimEnv


def main():
    parser = argparse.ArgumentParser(description="Compare multiple DARTSim runs")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs to compare")
    parser.add_argument("--steps", type=int, default=100, help="Steps per run")
    parser.add_argument("--save-dir", type=str, default="./results/comparisons",
                       help="Directory to save comparison")
    
    args = parser.parse_args()
    
    print(f"Running {args.runs} simulations for comparison...")
    
    visualizers = []
    results_list = []
    labels = []
    
    for i in range(args.runs):
        print(f"\n{'='*60}")
        print(f"Run {i+1}/{args.runs}")
        print(f"{'='*60}")
        
        # Create environment
        env = DARTSimEnv(verbose=False, ensure_dartsim_running=True)
        visualizer = DARTSimVisualizer(env)
        
        # Reset and record initial state
        obs, info = env.reset()
        state_data = info.get("state_data", {})
        threats = env._read_sensor("Threat", env.sensor_lookahead)
        targets = env._read_sensor("Target", env.sensor_lookahead)
        visualizer.record_step(state_data, 0, threats, targets, info)
        
        # Run simulation
        for step in range(args.steps):
            action = env.action_space.sample()  # Random policy for demo
            obs, reward, terminated, truncated, info = env.step(action)
            
            state_data = info.get("state_data", {})
            threats = env._read_sensor("Threat", env.sensor_lookahead)
            targets = env._read_sensor("Target", env.sensor_lookahead)
            visualizer.record_step(state_data, action, threats, targets, info)
            
            if terminated or truncated:
                break
        
        # Get results
        results = env._get_results()
        results_list.append(results)
        visualizers.append(visualizer)
        labels.append(f"Run {i+1}")
        
        print(f"Run {i+1} Results:")
        print(f"  Targets: {results.get('targetsDetected', 0)}")
        print(f"  Success: {results.get('missionSuccess', False)}")
        print(f"  Destroyed: {results.get('destroyed', False)}")
        
        env.close()
    
    # Create comparison visualization
    print(f"\n{'='*60}")
    print("Creating comparison visualization...")
    print(f"{'='*60}")
    
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    DARTSimVisualizer.plot_comparison(
        visualizers,
        labels,
        results_list,
        str(save_path / "comparison.png")
    )
    
    print(f"\nComparison saved to {save_path / 'comparison.png'}")
    print("\nDone!")


if __name__ == "__main__":
    main()

