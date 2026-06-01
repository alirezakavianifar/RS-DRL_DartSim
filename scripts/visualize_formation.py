"""
Visualize DARTSim team formation with individual drone representations.

Since DARTSim models a team as a single entity, this visualization
represents the formation by showing multiple drone icons arranged
according to the formation type (LOOSE = spread out, TIGHT = close together).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.visualize_dartsim import DARTSimVisualizer


def visualize_team_formation(
    trajectory_file: str,
    output_file: str,
    num_drones: int = 3,
    loose_spacing: float = 0.4,
    tight_spacing: float = 0.15
) -> None:
    """
    Visualize team formation showing individual drone positions.
    
    Args:
        trajectory_file: Path to trajectory JSON file
        output_file: Path to save visualization
        num_drones: Number of drones to visualize (default: 3)
        loose_spacing: Spacing between drones in loose formation
        tight_spacing: Spacing between drones in tight formation
    """
    # Load trajectory
    with open(trajectory_file, 'r') as f:
        trajectory = json.load(f)
    
    if not trajectory:
        print("No trajectory data to visualize")
        return
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Generate drone positions for each step
    for step in trajectory:
        x = step["position_x"]
        y = step["position_y"]
        altitude = step["altitude"]
        formation = step["formation"]
        ecm = step["ecm"]
        direction_x = step.get("direction_x", 1)
        direction_y = step.get("direction_y", 0)
        
        # Determine spacing based on formation
        spacing = loose_spacing if formation == "LOOSE" else tight_spacing
        
        # Calculate perpendicular direction for drone arrangement
        # Perpendicular to movement direction
        perp_x = -direction_y
        perp_y = direction_x
        
        # Generate drone positions in formation
        drone_positions = []
        if num_drones == 1:
            drone_positions = [(x, y)]
        elif num_drones == 2:
            offset = spacing / 2
            drone_positions = [
                (x + perp_x * offset, y + perp_y * offset),
                (x - perp_x * offset, y - perp_y * offset)
            ]
        else:
            # For 3+ drones, arrange in line perpendicular to movement
            start_offset = -spacing * (num_drones - 1) / 2
            for i in range(num_drones):
                offset = start_offset + i * spacing
                drone_positions.append(
                    (x + perp_x * offset, y + perp_y * offset)
                )
        
        # Plot each drone
        for i, (drone_x, drone_y) in enumerate(drone_positions):
            # Color by altitude
            color = plt.cm.viridis(altitude / 4.0)
            
            # Size and marker based on formation and ECM
            if formation == "TIGHT":
                size = 80 if ecm else 60
                marker = 'o'  # Circle
            else:
                size = 100 if ecm else 80
                marker = 's'  # Square
            
            # Edge color for ECM
            edge_color = 'red' if ecm else 'black'
            
            ax.scatter(drone_x, drone_y, s=size, c=[color], 
                      marker=marker, edgecolors=edge_color,
                      linewidths=1.5, alpha=0.7, zorder=5)
    
    # Draw path lines connecting team centers
    x_coords = [s["position_x"] for s in trajectory]
    y_coords = [s["position_y"] for s in trajectory]
    z_coords = [s["altitude"] for s in trajectory]
    
    for i in range(len(x_coords) - 1):
        color = plt.cm.viridis(z_coords[i] / max(z_coords) if max(z_coords) > 0 else 0)
        ax.plot([x_coords[i], x_coords[i+1]], [y_coords[i], y_coords[i+1]],
               color=color, linewidth=2, alpha=0.3, zorder=1)
    
    # Mark start and end
    ax.scatter(x_coords[0], y_coords[0], marker='*', s=600, 
              color='green', label='Start', zorder=10,
              edgecolors='black', linewidths=2)
    ax.scatter(x_coords[-1], y_coords[-1], marker='*', s=600,
              color='red', label='End', zorder=10,
              edgecolors='black', linewidths=2)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', edgecolor='black', label='Loose Formation'),
        Patch(facecolor='blue', edgecolor='red', label='Loose Formation (ECM)'),
        Patch(facecolor='blue', edgecolor='black', label='Tight Formation'),
        Patch(facecolor='blue', edgecolor='red', label='Tight Formation (ECM)'),
        plt.scatter([], [], marker='*', s=600, color='green', 
                   edgecolors='black', label='Start'),
        plt.scatter([], [], marker='*', s=600, color='red',
                   edgecolors='black', label='End')
    ]
    ax.legend(handles=legend_elements[:4], loc='upper right', fontsize=10)
    
    ax.set_xlabel("Position X", fontsize=12)
    ax.set_ylabel("Position Y", fontsize=12)
    ax.set_title(f"DARTSim Team Formation Visualization ({num_drones} Drones)\n"
                f"Loose Formation: Spread Out | Tight Formation: Close Together",
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved formation visualization to {output_file}")


def visualize_formation_evolution(
    trajectory_file: str,
    output_file: str,
    num_drones: int = 3
) -> None:
    """
    Create animation-style visualization showing formation changes over time.
    Shows multiple snapshots of the team formation.
    """
    with open(trajectory_file, 'r') as f:
        trajectory = json.load(f)
    
    if not trajectory:
        print("No trajectory data to visualize")
        return
    
    # Select key moments (formation changes, ECM changes, etc.)
    key_steps = []
    last_formation = None
    last_ecm = None
    
    for i, step in enumerate(trajectory):
        if (step["formation"] != last_formation or 
            step["ecm"] != last_ecm or 
            i == 0 or 
            i == len(trajectory) - 1 or
            i % 10 == 0):  # Every 10th step
            key_steps.append((i, step))
            last_formation = step["formation"]
            last_ecm = step["ecm"]
    
    # Create subplot grid
    n_plots = min(len(key_steps), 12)  # Max 12 snapshots
    cols = 4
    rows = (n_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    axes = axes.flatten()
    
    for idx, (step_num, step) in enumerate(key_steps[:n_plots]):
        ax = axes[idx]
        
        x = step["position_x"]
        y = step["position_y"]
        altitude = step["altitude"]
        formation = step["formation"]
        ecm = step["ecm"]
        direction_x = step.get("direction_x", 1)
        direction_y = step.get("direction_y", 0)
        
        # Calculate drone positions
        spacing = 0.4 if formation == "LOOSE" else 0.15
        perp_x = -direction_y
        perp_y = direction_x
        
        start_offset = -spacing * (num_drones - 1) / 2
        for i in range(num_drones):
            offset = start_offset + i * spacing
            drone_x = x + perp_x * offset
            drone_y = y + perp_y * offset
            
            color = plt.cm.viridis(altitude / 4.0)
            size = 100 if formation == "LOOSE" else 70
            marker = 's' if formation == "LOOSE" else 'o'
            edge_color = 'red' if ecm else 'black'
            
            ax.scatter(drone_x, drone_y, s=size, c=[color],
                      marker=marker, edgecolors=edge_color,
                      linewidths=2, alpha=0.8)
        
        # Show context (path leading to this point)
        for prev_step in trajectory[:step_num]:
            ax.scatter(prev_step["position_x"], prev_step["position_y"],
                      s=20, c='gray', alpha=0.2, marker='.')
        
        ax.set_title(f"Step {step_num}\n{formation} | Alt:{altitude} | ECM:{'ON' if ecm else 'OFF'}",
                    fontsize=9)
        ax.set_xlim(min(s["position_x"] for s in trajectory) - 2,
                   max(s["position_x"] for s in trajectory) + 2)
        ax.set_ylim(min(s["position_y"] for s in trajectory) - 2,
                   max(s["position_y"] for s in trajectory) + 2)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved formation evolution to {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize team formation")
    parser.add_argument("--trajectory", type=str, 
                       default="./results/test_visualizations/trajectory.json",
                       help="Path to trajectory JSON file")
    parser.add_argument("--output", type=str,
                       default="./results/test_visualizations/formation_visualization.png",
                       help="Output file path")
    parser.add_argument("--num-drones", type=int, default=3,
                       help="Number of drones to visualize")
    parser.add_argument("--evolution", action="store_true",
                       help="Create evolution snapshot visualization")
    
    args = parser.parse_args()
    
    if args.evolution:
        visualize_formation_evolution(
            args.trajectory,
            args.output.replace('.png', '_evolution.png'),
            args.num_drones
        )
    else:
        visualize_team_formation(
            args.trajectory,
            args.output,
            args.num_drones
        )

