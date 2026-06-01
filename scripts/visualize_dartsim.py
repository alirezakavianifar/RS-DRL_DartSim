"""
DARTSim Visualization Tool
Creates visually appealing 2D/3D visualizations of DARTSim simulations.

Features:
- Real-time trajectory tracking
- 2D top-down view with altitude coloring
- 3D path visualization
- Interactive Plotly visualizations
- Static matplotlib plots
- Animation support
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Plotly is optional for interactive visualizations
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Interactive visualizations will be skipped.")
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import sys
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.dartsim_env import DARTSimEnv
    TCP_AVAILABLE = True
except ImportError:
    TCP_AVAILABLE = False

try:
    from src.dartsim_env_library import DARTSimEnvLibrary
    LIBRARY_AVAILABLE = True
except ImportError:
    LIBRARY_AVAILABLE = False


class DARTSimVisualizer:
    """Visualizes DARTSim simulation data."""
    
    def __init__(self, env: DARTSimEnv):
        """Initialize visualizer with environment."""
        self.env = env
        self.trajectory: List[Dict[str, Any]] = []
        self.step_count = 0
        
    def record_step(self, state_data: Dict[str, Any], action: int, 
                   threats: List[bool], targets: List[bool], 
                   info: Dict[str, Any]):
        """Record a simulation step for visualization."""
        step_data = {
            "step": self.step_count,
            "position_x": state_data.get("positionX", 0),
            "position_y": state_data.get("positionY", 0),
            "altitude": state_data.get("altitudeLevel", 1),
            "formation": "TIGHT" if state_data.get("formation", 0) == 1 else "LOOSE",
            "ecm": state_data.get("ecm", False),
            "direction_x": state_data.get("directionX", 0),
            "direction_y": state_data.get("directionY", 0),
            "threats_ahead": threats,
            "targets_ahead": targets,
            "action": self.env.ACTIONS[action] if action < len(self.env.ACTIONS) else "Unknown",
            "info": info
        }
        self.trajectory.append(step_data)
        self.step_count += 1
    
    def get_plot_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[bool]]:
        """Extract data for plotting."""
        if not self.trajectory:
            return np.array([]), np.array([]), np.array([]), [], []
        
        x = np.array([step["position_x"] for step in self.trajectory])
        y = np.array([step["position_y"] for step in self.trajectory])
        z = np.array([step["altitude"] for step in self.trajectory])
        formations = [step["formation"] for step in self.trajectory]
        ecm_status = [step["ecm"] for step in self.trajectory]
        
        return x, y, z, formations, ecm_status
    
    def plot_2d_trajectory(self, results: Optional[Dict[str, Any]] = None, 
                          save_path: Optional[str] = None,
                          show_threats: bool = True,
                          show_targets: bool = True) -> None:
        """Create 2D top-down view with altitude coloring and threat/target overlays."""
        x, y, z, formations, ecm_status = self.get_plot_data()
        
        if len(x) == 0:
            print("No trajectory data to plot")
            return
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Extract threat and target positions from trajectory
        threat_positions = []
        target_positions = []
        detected_targets = []
        
        for step in self.trajectory:
            # Check threats ahead in sensor readings
            threats_ahead = step.get("threats_ahead", [])
            targets_ahead = step.get("targets_ahead", [])
            pos_x = step["position_x"]
            pos_y = step["position_y"]
            direction_x = step.get("direction_x", 0)
            direction_y = step.get("direction_y", 0)
            
            # Project threats and targets ahead based on direction
            for i, threat in enumerate(threats_ahead):
                if threat:
                    # Project threat position ahead
                    threat_x = pos_x + (i + 1) * direction_x
                    threat_y = pos_y + (i + 1) * direction_y
                    threat_positions.append((threat_x, threat_y))
            
            for i, target in enumerate(targets_ahead):
                if target:
                    # Project target position ahead
                    target_x = pos_x + (i + 1) * direction_x
                    target_y = pos_y + (i + 1) * direction_y
                    target_positions.append((target_x, target_y))
                    # Check if target was detected (in results)
                    detected_targets.append(False)  # Will update from results if available
        
        # Plot threat positions
        if show_threats and threat_positions:
            threat_xs, threat_ys = zip(*threat_positions) if threat_positions else ([], [])
            ax.scatter(threat_xs, threat_ys, marker='^', s=300, color='red', 
                      alpha=0.6, label='Threats', zorder=8, edgecolors='darkred', linewidths=2)
        
        # Plot target positions
        if show_targets and target_positions:
            target_xs, target_ys = zip(*target_positions) if target_positions else ([], [])
            # Use different colors for detected vs undetected
            ax.scatter(target_xs, target_ys, marker='D', s=300, color='gold', 
                      alpha=0.7, label='Targets', zorder=8, edgecolors='orange', linewidths=2)
        
        # Plot path with altitude coloring
        scatter = ax.scatter(x, y, c=z, cmap='viridis', s=100, 
                            edgecolors='black', linewidths=0.5, alpha=0.7, zorder=5)
        
        # Draw path lines
        for i in range(len(x) - 1):
            color = plt.cm.viridis(z[i] / max(z) if max(z) > 0 else 0)
            ax.plot([x[i], x[i+1]], [y[i], y[i+1]], 
                   color=color, linewidth=2, alpha=0.5, zorder=4)
        
        # Mark formation changes
        for i in range(len(formations)):
            marker = 'o' if formations[i] == "TIGHT" else 's'
            size = 150 if ecm_status[i] else 100
            ax.scatter(x[i], y[i], marker=marker, s=size, 
                      edgecolors='red' if ecm_status[i] else 'black',
                      linewidths=2, facecolors='none', zorder=6)
        
        # Mark start and end
        ax.scatter(x[0], y[0], marker='*', s=500, color='green', 
                  label='Start', zorder=10, edgecolors='black', linewidths=2)
        ax.scatter(x[-1], y[-1], marker='*', s=500, color='red', 
                  label='End', zorder=10, edgecolors='black', linewidths=2)
        
        # Add destruction point if destroyed
        if results and results.get("destroyed", False):
            destroy_x = results.get("destruction positionX", x[-1])
            destroy_y = results.get("destruction positionY", y[-1])
            ax.scatter(destroy_x, destroy_y, marker='X', s=600, color='darkred',
                      label='Destroyed', zorder=11, edgecolors='black', linewidths=3)
        
        # Add results info
        if results:
            title = f"DARTSim Mission Trajectory\n"
            title += f"Targets Detected: {results.get('targetsDetected', 0)} | "
            title += f"Mission Success: {results.get('missionSuccess', False)} | "
            title += f"Destroyed: {results.get('destroyed', False)}"
            ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.set_xlabel("Position X", fontsize=12)
        ax.set_ylabel("Position Y", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        
        # Colorbar for altitude
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Altitude Level', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved 2D plot to {save_path}")
        else:
            plt.show()
    
    def plot_3d_trajectory(self, results: Optional[Dict[str, Any]] = None,
                          save_path: Optional[str] = None) -> None:
        """Create 3D trajectory plot."""
        x, y, z, formations, ecm_status = self.get_plot_data()
        
        if len(x) == 0:
            print("No trajectory data to plot")
            return
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot 3D path
        colors = plt.cm.viridis(z / max(z))
        ax.plot(x, y, z, 'b-', linewidth=2, alpha=0.6, label='Path')
        
        # Scatter points with altitude coloring
        scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=100, 
                            edgecolors='black', linewidths=0.5)
        
        # Mark formation changes
        for i in range(len(formations)):
            marker = 'o' if formations[i] == "TIGHT" else 's'
            ax.scatter([x[i]], [y[i]], [z[i]], marker=marker, 
                      s=150, edgecolors='red' if ecm_status[i] else 'black',
                      linewidths=2, facecolors='none', zorder=5)
        
        # Mark start and end
        ax.scatter([x[0]], [y[0]], [z[0]], marker='*', s=500, 
                  color='green', label='Start', zorder=10)
        ax.scatter([x[-1]], [y[-1]], [z[-1]], marker='*', s=500, 
                  color='red', label='End', zorder=10)
        
        ax.set_xlabel("Position X", fontsize=12)
        ax.set_ylabel("Position Y", fontsize=12)
        ax.set_zlabel("Altitude Level", fontsize=12)
        
        if results:
            title = f"DARTSim 3D Mission Trajectory\n"
            title += f"Targets: {results.get('targetsDetected', 0)} | "
            title += f"Success: {results.get('missionSuccess', False)}"
            ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved 3D plot to {save_path}")
        else:
            plt.show()
    
    def plot_interactive(self, results: Optional[Dict[str, Any]] = None,
                        save_path: Optional[str] = None) -> None:
        """Create interactive Plotly visualization."""
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Skipping interactive visualization.")
            print("Install with: pip install plotly")
            return
        
        x, y, z, formations, ecm_status = self.get_plot_data()
        
        if len(x) == 0:
            print("No trajectory data to plot")
            return
        
        # Create hover text
        hover_texts = []
        for i, step in enumerate(self.trajectory):
            text = f"Step: {step['step']}<br>"
            text += f"Position: ({step['position_x']}, {step['position_y']})<br>"
            text += f"Altitude: {step['altitude']}<br>"
            text += f"Formation: {step['formation']}<br>"
            text += f"ECM: {'ON' if step['ecm'] else 'OFF'}<br>"
            text += f"Action: {step['action']}"
            hover_texts.append(text)
        
        # Create 3D scatter plot
        fig = go.Figure()
        
        # Main trajectory
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines+markers',
            marker=dict(
                size=8,
                color=z,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Altitude"),
                line=dict(width=1, color='black')
            ),
            line=dict(width=4, color='blue'),
            text=hover_texts,
            hoverinfo='text',
            name='Trajectory'
        ))
        
        # Mark formation changes
        tight_x = [x[i] for i in range(len(x)) if formations[i] == "TIGHT"]
        tight_y = [y[i] for i in range(len(y)) if formations[i] == "TIGHT"]
        tight_z = [z[i] for i in range(len(z)) if formations[i] == "TIGHT"]
        
        loose_x = [x[i] for i in range(len(x)) if formations[i] == "LOOSE"]
        loose_y = [y[i] for i in range(len(y)) if formations[i] == "LOOSE"]
        loose_z = [z[i] for i in range(len(z)) if formations[i] == "LOOSE"]
        
        if tight_x:
            fig.add_trace(go.Scatter3d(
                x=tight_x, y=tight_y, z=tight_z,
                mode='markers',
                marker=dict(size=12, symbol='circle', color='red', 
                          line=dict(width=2, color='black')),
                name='Tight Formation',
                hoverinfo='skip'
            ))
        
        if loose_x:
            fig.add_trace(go.Scatter3d(
                x=loose_x, y=loose_y, z=loose_z,
                mode='markers',
                marker=dict(size=12, symbol='square', color='orange',
                          line=dict(width=2, color='black')),
                name='Loose Formation',
                hoverinfo='skip'
            ))
        
        # Start and end markers
        fig.add_trace(go.Scatter3d(
            x=[x[0]], y=[y[0]], z=[z[0]],
            mode='markers',
            marker=dict(size=20, symbol='star', color='green',
                      line=dict(width=3, color='black')),
            name='Start',
            hoverinfo='text',
            text=[f"Start: ({x[0]}, {y[0]}, {z[0]})"]
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[x[-1]], y=[y[-1]], z=[z[-1]],
            mode='markers',
            marker=dict(size=20, symbol='star', color='red',
                      line=dict(width=3, color='black')),
            name='End',
            hoverinfo='text',
            text=[f"End: ({x[-1]}, {y[-1]}, {z[-1]})"]
        ))
        
        # Layout
        title = "DARTSim Interactive 3D Trajectory"
        if results:
            title += f"<br>Targets: {results.get('targetsDetected', 0)} | "
            title += f"Success: {results.get('missionSuccess', False)} | "
            title += f"Destroyed: {results.get('destroyed', False)}"
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="Position X",
                yaxis_title="Position Y",
                zaxis_title="Altitude Level",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            ),
            width=1200,
            height=800,
            hovermode='closest'
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"Saved interactive plot to {save_path}")
        else:
            fig.show()
    
    def plot_altitude_timeline(self, save_path: Optional[str] = None) -> None:
        """Plot altitude changes over time."""
        if not self.trajectory:
            print("No trajectory data to plot")
            return
        
        steps = [step["step"] for step in self.trajectory]
        altitudes = [step["altitude"] for step in self.trajectory]
        formations = [step["formation"] for step in self.trajectory]
        ecm_status = [step["ecm"] for step in self.trajectory]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Altitude plot
        ax1.plot(steps, altitudes, 'b-', linewidth=2, marker='o', markersize=6)
        ax1.fill_between(steps, altitudes, alpha=0.3)
        ax1.set_ylabel("Altitude Level", fontsize=12)
        ax1.set_title("Altitude Over Time", fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)
        
        # Formation and ECM status
        formation_colors = ['red' if f == "TIGHT" else 'blue' for f in formations]
        ecm_colors = ['orange' if e else 'gray' for e in ecm_status]
        
        ax2.scatter(steps, [1]*len(steps), c=formation_colors, 
                   s=100, marker='s', label='Formation', alpha=0.7)
        ax2.scatter(steps, [0]*len(steps), c=ecm_colors,
                   s=100, marker='o', label='ECM', alpha=0.7)
        ax2.set_ylabel("Status", fontsize=12)
        ax2.set_xlabel("Step", fontsize=12)
        ax2.set_title("Formation and ECM Status Over Time", fontsize=14, fontweight='bold')
        ax2.set_ylim(-0.5, 1.5)
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['ECM', 'Formation'])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved timeline plot to {save_path}")
        else:
            plt.show()
    
    def save_trajectory(self, filepath: str) -> None:
        """Save trajectory data to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.trajectory, f, indent=2)
        print(f"Saved trajectory data to {filepath}")
    
    @staticmethod
    def plot_comparison(visualizers: List['DARTSimVisualizer'], 
                       labels: List[str],
                       results_list: Optional[List[Dict[str, Any]]] = None,
                       save_path: Optional[str] = None) -> None:
        """
        Create comparison visualization of multiple simulation runs.
        
        Args:
            visualizers: List of DARTSimVisualizer instances
            labels: Labels for each run
            results_list: Optional list of results dictionaries
            save_path: Path to save comparison plot
        """
        if len(visualizers) == 0:
            print("No visualizers provided for comparison")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # Color map for different runs
        colors = plt.cm.tab10(np.linspace(0, 1, len(visualizers)))
        
        # Plot 1: 2D trajectory comparison
        ax1 = axes[0]
        for i, (viz, label, color) in enumerate(zip(visualizers, labels, colors)):
            x, y, z, _, _ = viz.get_plot_data()
            if len(x) > 0:
                ax1.plot(x, y, color=color, linewidth=2, alpha=0.7, label=label)
                ax1.scatter(x[0], y[0], marker='*', s=300, color=color, 
                           edgecolors='black', linewidths=1, zorder=10)
                ax1.scatter(x[-1], y[-1], marker='s', s=200, color=color,
                           edgecolors='black', linewidths=1, zorder=10)
        
        ax1.set_xlabel("Position X", fontsize=12)
        ax1.set_ylabel("Position Y", fontsize=12)
        ax1.set_title("Trajectory Comparison (Top-Down View)", fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=10)
        
        # Plot 2: Metrics comparison
        ax2 = axes[1]
        if results_list:
            metrics = {
                'Targets Detected': [r.get('targetsDetected', 0) for r in results_list],
                'Mission Success': [1 if r.get('missionSuccess', False) else 0 for r in results_list],
                'Destroyed': [1 if r.get('destroyed', False) else 0 for r in results_list]
            }
            
            x_pos = np.arange(len(labels))
            width = 0.25
            multiplier = 0
            
            for metric, values in metrics.items():
                offset = width * multiplier
                rects = ax2.bar(x_pos + offset, values, width, label=metric, alpha=0.8)
                multiplier += 1
            
            ax2.set_ylabel('Value', fontsize=12)
            ax2.set_title('Metrics Comparison', fontsize=14, fontweight='bold')
            ax2.set_xticks(x_pos + width)
            ax2.set_xticklabels(labels, rotation=45, ha='right')
            ax2.legend(loc='best', fontsize=10)
            ax2.grid(True, alpha=0.3, axis='y')
        else:
            # Altitude comparison over time
            for i, (viz, label, color) in enumerate(zip(visualizers, labels, colors)):
                if viz.trajectory:
                    steps = [s["step"] for s in viz.trajectory]
                    altitudes = [s["altitude"] for s in viz.trajectory]
                    ax2.plot(steps, altitudes, color=color, linewidth=2, 
                           alpha=0.7, label=label, marker='o', markersize=4)
            
            ax2.set_xlabel("Step", fontsize=12)
            ax2.set_ylabel("Altitude Level", fontsize=12)
            ax2.set_title("Altitude Comparison Over Time", fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='best', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved comparison plot to {save_path}")
        else:
            plt.show()


def run_simulation_with_visualization(
    n_steps: int = 100,
    visualize: bool = True,
    save_dir: str = "./results/visualizations",
    interactive: bool = True,
    auto_start_dartsim: bool = True
) -> Dict[str, Any]:
    """
    Run DARTSim simulation and create visualizations.
    
    Args:
        n_steps: Maximum number of steps
        visualize: Create visualizations
        save_dir: Directory to save outputs
        interactive: Create interactive Plotly plots
        auto_start_dartsim: Automatically start DARTSim if not running
    """
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Try library interface first (doesn't need TCP port), fallback to TCP
    if LIBRARY_AVAILABLE:
        try:
            print("Using DARTSim library interface (no TCP required)...")
            env = DARTSimEnvLibrary(verbose=False)
            print("Library interface initialized successfully")
        except Exception as e:
            print(f"Library interface failed: {e}, trying TCP interface...")
            if TCP_AVAILABLE:
                env = DARTSimEnv(verbose=True, ensure_dartsim_running=auto_start_dartsim)
            else:
                raise RuntimeError("Neither library nor TCP interface available")
    elif TCP_AVAILABLE:
        env = DARTSimEnv(verbose=True, ensure_dartsim_running=auto_start_dartsim)
    else:
        raise RuntimeError("No DARTSim interface available")
    
    visualizer = DARTSimVisualizer(env)
    
    # Reset environment
    obs, info = env.reset()
    state_data = info.get("state_data", {})
    threats = env._read_sensor("Threat", env.sensor_lookahead)
    targets = env._read_sensor("Target", env.sensor_lookahead)
    
    # Record initial state
    visualizer.record_step(state_data, 0, threats, targets, info)
    
    print(f"Running simulation for up to {n_steps} steps...")
    
    # Run simulation
    for step in range(n_steps):
        # Simple policy: random actions for demo
        action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        state_data = info.get("state_data", {})
        threats = env._read_sensor("Threat", env.sensor_lookahead)
        targets = env._read_sensor("Target", env.sensor_lookahead)
        
        visualizer.record_step(state_data, action, threats, targets, info)
        
        if step % 10 == 0:
            print(f"Step {step}: Position ({state_data.get('positionX', 0)}, "
                  f"{state_data.get('positionY', 0)}), "
                  f"Altitude {state_data.get('altitudeLevel', 1)}")
        
        if terminated or truncated:
            print(f"Simulation finished at step {step}")
            break
    
    # Get final results
    results = env._get_results()
    
    print(f"\nMission Results:")
    print(f"  Targets Detected: {results.get('targetsDetected', 0)}")
    print(f"  Mission Success: {results.get('missionSuccess', False)}")
    print(f"  Destroyed: {results.get('destroyed', False)}")
    
    # Create visualizations
    if visualize:
        print("\nCreating visualizations...")
        
        # Save trajectory data
        visualizer.save_trajectory(str(save_path / "trajectory.json"))
        
        # 2D plot
        visualizer.plot_2d_trajectory(
            results, 
            str(save_path / "trajectory_2d.png")
        )
        
        # 3D plot
        visualizer.plot_3d_trajectory(
            results,
            str(save_path / "trajectory_3d.png")
        )
        
        # Timeline
        visualizer.plot_altitude_timeline(
            str(save_path / "altitude_timeline.png")
        )
        
        # Interactive plot
        if interactive:
            visualizer.plot_interactive(
                results,
                str(save_path / "trajectory_interactive.html")
            )
        
        print(f"\nAll visualizations saved to {save_path}")
    
    env.close()
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize DARTSim simulations")
    parser.add_argument("--steps", type=int, default=100,
                       help="Maximum simulation steps")
    parser.add_argument("--save-dir", type=str, default="./results/visualizations",
                       help="Directory to save visualizations")
    parser.add_argument("--no-interactive", action="store_true",
                       help="Skip interactive Plotly plots")
    parser.add_argument("--no-visualize", action="store_true",
                       help="Skip visualization generation")
    
    args = parser.parse_args()
    
    results = run_simulation_with_visualization(
        n_steps=args.steps,
        visualize=not args.no_visualize,
        save_dir=args.save_dir,
        interactive=not args.no_interactive
    )
    
    print("\nVisualization complete!")

