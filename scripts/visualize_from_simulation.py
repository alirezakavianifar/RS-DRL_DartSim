"""
Extract trajectory data from DARTSim simulation output and create visualizations.

This runs the simple-cpp example inside Docker and parses the output to create
visualizations with real simulation data.
"""

import argparse
import subprocess
import re
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.visualize_dartsim import DARTSimVisualizer


def parse_simulation_output(output: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Parse DARTSim enhanced simple-cpp output to extract trajectory and results.
    
    Returns:
        (trajectory, results)
    """
    trajectory = []
    results = {}
    
    # Patterns for enhanced output
    state_block_pattern = re.compile(r'STATE:')
    position_pattern = re.compile(r'  position: (\d+);(\d+)')
    direction_pattern = re.compile(r'  direction: (-?\d+);(-?\d+)')
    altitude_pattern = re.compile(r'  altitude: (\d+)')
    formation_pattern = re.compile(r'  formation: (LOOSE|TIGHT)')
    ecm_pattern = re.compile(r'  ecm: (true|false)')
    threats_pattern = re.compile(r'  threats: \[([0-9,]+)\]')
    targets_pattern = re.compile(r'  targets: \[([0-9,]+)\]')
    
    # Parse tactics
    tactic_pattern = re.compile(r'executing tactic (\w+)')
    
    # Parse results
    targets_detected_pattern = re.compile(r'Total targets detected: (\d+)')
    destroyed_pattern = re.compile(r'out:destroyed=(\d+)')
    targets_out_pattern = re.compile(r'out:targetsDetected=(\d+)')
    success_pattern = re.compile(r'out:missionSuccess=(\d+)')
    csv_pattern = re.compile(r'csv,(\d+),(\d+),(\d+),(\d+),([\d.]+),([\d.e-]+)')
    
    lines = output.split('\n')
    current_state = None
    in_state_block = False
    step_num = 0
    
    for i, line in enumerate(lines):
        # Check for STATE block start
        if state_block_pattern.search(line):
            in_state_block = True
            current_state = {}
            continue
        
        if in_state_block:
            # Parse state information
            pos_match = position_pattern.search(line)
            if pos_match:
                current_state["position_x"] = int(pos_match.group(1))
                current_state["position_y"] = int(pos_match.group(2))
                continue
            
            dir_match = direction_pattern.search(line)
            if dir_match:
                current_state["direction_x"] = int(dir_match.group(1))
                current_state["direction_y"] = int(dir_match.group(2))
                continue
            
            alt_match = altitude_pattern.search(line)
            if alt_match:
                current_state["altitude"] = int(alt_match.group(1))
                continue
            
            form_match = formation_pattern.search(line)
            if form_match:
                current_state["formation"] = form_match.group(1)
                continue
            
            ecm_match = ecm_pattern.search(line)
            if ecm_match:
                current_state["ecm"] = (ecm_match.group(1) == "true")
                continue
            
            threats_match = threats_pattern.search(line)
            if threats_match:
                threats_str = threats_match.group(1)
                current_state["threats_ahead"] = [bool(int(x)) for x in threats_str.split(',') if x]
                continue
            
            targets_match = targets_pattern.search(line)
            if targets_match:
                targets_str = targets_match.group(1)
                current_state["targets_ahead"] = [bool(int(x)) for x in targets_str.split(',') if x]
                continue
            
            # Check if we've left the STATE block (next non-indented line)
            if line.strip() and not line.startswith('  '):
                # Save the state we just collected
                if current_state and "position_x" in current_state:
                    current_state["step"] = step_num
                    current_state["action"] = "Unknown"
                    current_state["info"] = {}
                    trajectory.append(current_state.copy())
                    step_num += 1
                in_state_block = False
                current_state = None
        
        # Check for tactics (these modify the state)
        tactic_match = tactic_pattern.search(line)
        if tactic_match and trajectory:
            tactic = tactic_match.group(1)
            # Update last trajectory entry with action
            trajectory[-1]["action"] = tactic
        
        # Check for target detection
        if "Target detected" in line:
            if trajectory:
                trajectory[-1]["targets_ahead"] = [True]  # Simplified
        
        # Parse results
        targets_match = targets_detected_pattern.search(line)
        if targets_match:
            results["targetsDetected"] = int(targets_match.group(1))
        
        destroyed_match = destroyed_pattern.search(line)
        if destroyed_match:
            results["destroyed"] = bool(int(destroyed_match.group(1)))
        
        targets_out_match = targets_out_pattern.search(line)
        if targets_out_match:
            results["targetsDetected"] = int(targets_out_match.group(1))
        
        success_match = success_pattern.search(line)
        if success_match:
            results["missionSuccess"] = bool(int(success_match.group(1)))
        
        csv_match = csv_pattern.search(line)
        if csv_match:
            results["targetsDetected"] = int(csv_match.group(1))
            results["destroyed"] = bool(int(csv_match.group(2)))
            results["destruction positionX"] = int(csv_match.group(3))
            results["missionSuccess"] = bool(int(csv_match.group(4)))
            results["decisionTimeAvg"] = float(csv_match.group(5))
            results["decisionTimeVar"] = float(csv_match.group(6))
    
    # Save final state if we're still in a state block
    if in_state_block and current_state and "position_x" in current_state:
        current_state["step"] = step_num
        current_state["action"] = "Finished"
        current_state["info"] = {}
        trajectory.append(current_state)
    
    return trajectory, results


def run_simulation_and_visualize(
    n_steps: int = None,
    save_dir: str = "./results/visualizations",
    container_name: str = "dartsim"
) -> None:
    """Run DARTSim simulation and create visualizations."""
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print("Running DARTSim simulation in container...")
    
    # Run simple-cpp example
    # Use proper command format for Windows PowerShell
    docker_cmd = [
        "docker", "exec", container_name,
        "bash", "-c",
        "cd /headless/dartsim/examples/simple-cpp && timeout 30 ./run.sh 2>&1"
    ]
    
    try:
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        output = result.stdout + result.stderr
        
        if "Simulator instantiated" not in output and len(output) < 100:
            print(f"Error: Simulation may not have run properly")
            print(f"Output: {output[:500]}")
            return
        
        print("Parsing simulation output...")
        trajectory, results = parse_simulation_output(output)
        
        if not trajectory:
            print("No trajectory data extracted from simulation")
            print(f"Output preview: {output[:500]}")
            return
        
        print(f"Extracted {len(trajectory)} trajectory points")
        print(f"Results: {results}")
        
        # Create a mock environment for the visualizer
        class MockEnv:
            ACTIONS = ["IncAlt", "DecAlt", "GoTight", "GoLoose", "EcmOn", "EcmOff"]
        
        # Create visualizer with trajectory data
        visualizer = DARTSimVisualizer(MockEnv())
        visualizer.trajectory = trajectory
        visualizer.step_count = len(trajectory)
        
        print("\nCreating visualizations...")
        
        # Save trajectory
        visualizer.save_trajectory(str(save_path / "trajectory.json"))
        
        # Create plots
        visualizer.plot_2d_trajectory(
            results,
            str(save_path / "trajectory_2d.png"),
            show_threats=True,
            show_targets=True
        )
        
        visualizer.plot_3d_trajectory(
            results,
            str(save_path / "trajectory_3d.png")
        )
        
        visualizer.plot_altitude_timeline(
            str(save_path / "altitude_timeline.png")
        )
        
        # Try interactive plot if available
        try:
            visualizer.plot_interactive(
                results,
                str(save_path / "trajectory_interactive.html")
            )
        except Exception as e:
            print(f"Skipping interactive plot: {e}")
        
        print(f"\nAll visualizations saved to {save_path}")
        print(f"\nMission Results:")
        print(f"  Targets Detected: {results.get('targetsDetected', 0)}")
        print(f"  Mission Success: {results.get('missionSuccess', False)}")
        print(f"  Destroyed: {results.get('destroyed', False)}")
        
    except subprocess.TimeoutExpired:
        print("Error: Simulation timed out")
    except Exception as e:
        print(f"Error running simulation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize DARTSim from real simulation")
    parser.add_argument("--save-dir", type=str, default="./results/visualizations",
                       help="Directory to save visualizations")
    parser.add_argument("--container", type=str, default="dartsim",
                       help="Docker container name")
    
    args = parser.parse_args()
    
    run_simulation_and_visualize(
        save_dir=args.save_dir,
        container_name=args.container
    )

