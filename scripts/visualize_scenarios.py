"""
Visualize DARTSim scenarios showing map layout, route, and threat/target positions.

This creates visual representations of the different scenarios (baseline, medium, hard)
to help understand the challenges faced in each scenario.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from pathlib import Path
from typing import List, Tuple, Dict, Any
import json


def generate_route(map_size: int, square_map: bool = False) -> List[Tuple[int, int]]:
    """
    Generate route based on map type.
    
    Args:
        map_size: Size of the map
        square_map: If True, creates snake pattern. If False, creates linear route.
    
    Returns:
        List of (x, y) coordinates representing the route
    """
    route = []
    
    if square_map:
        # Snake pattern: zigzag through the square map
        for y in range(map_size):
            if y % 2 == 0:
                # Left to right
                for x in range(map_size):
                    route.append((x, y))
            else:
                # Right to left
                for x in range(map_size - 1, -1, -1):
                    route.append((x, y))
    else:
        # Linear route: straight line from (0,0) to (map_size, 0)
        for x in range(map_size):
            route.append((x, 0))
    
    return route


def generate_threat_target_positions(
    map_size: int,
    num_threats: int,
    num_targets: int,
    square_map: bool = False,
    seed: int = 42
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Generate representative threat and target positions.
    
    Note: Actual positions in DARTSim are randomly generated, but this
    creates a representative visualization based on the scenario parameters.
    
    Args:
        map_size: Size of the map
        num_threats: Number of threats
        num_targets: Number of targets
        square_map: Whether map is square or linear
        seed: Random seed for reproducibility
    
    Returns:
        (threat_positions, target_positions) as lists of (x, y) tuples
    """
    np.random.seed(seed)
    
    if square_map:
        max_x, max_y = map_size, map_size
    else:
        max_x, max_y = map_size, 1
    
    # Generate threat positions
    threat_positions = []
    attempts = 0
    while len(threat_positions) < num_threats and attempts < 1000:
        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)
        if (x, y) not in threat_positions:
            threat_positions.append((x, y))
        attempts += 1
    
    # Generate target positions (avoid overlap with threats)
    target_positions = []
    attempts = 0
    while len(target_positions) < num_targets and attempts < 1000:
        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)
        if (x, y) not in threat_positions and (x, y) not in target_positions:
            target_positions.append((x, y))
        attempts += 1
    
    return threat_positions, target_positions


def visualize_scenario(
    scenario_name: str,
    map_size: int,
    num_targets: int,
    num_threats: int,
    square_map: bool = False,
    altitude_levels: int = 4,
    seed: int = 42,
    output_path: str = None
) -> None:
    """
    Create a visualization of a DARTSim scenario.
    
    Args:
        scenario_name: Name of the scenario (e.g., "baseline", "medium", "hard")
        map_size: Size of the map
        num_targets: Number of targets
        num_threats: Number of threats
        square_map: Whether map is square (True) or linear (False)
        altitude_levels: Number of altitude levels
        seed: Random seed
        output_path: Path to save the visualization
    """
    # Generate route
    route = generate_route(map_size, square_map)
    
    # Generate threat and target positions
    threat_positions, target_positions = generate_threat_target_positions(
        map_size, num_threats, num_targets, square_map, seed
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Set up plot limits
    if square_map:
        ax.set_xlim(-1, map_size)
        ax.set_ylim(-1, map_size)
        ax.set_aspect('equal')
    else:
        ax.set_xlim(-1, map_size)
        ax.set_ylim(-2, 3)
        ax.set_aspect('equal')
    
    # Draw grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xticks(range(0, map_size, max(1, map_size // 10)))
    ax.set_yticks(range(0, map_size if square_map else 2, max(1, (map_size if square_map else 2) // 5)))
    
    # Draw route
    route_x = [r[0] for r in route]
    route_y = [r[1] for r in route]
    ax.plot(route_x, route_y, 'b-', linewidth=3, alpha=0.5, label='Route', zorder=1)
    
    # Mark route start and end
    if route:
        start = route[0]
        end = route[-1]
        ax.scatter([start[0]], [start[1]], s=300, c='green', marker='*', 
                  edgecolors='black', linewidths=2, label='Start', zorder=10)
        ax.scatter([end[0]], [end[1]], s=300, c='red', marker='*',
                  edgecolors='black', linewidths=2, label='End', zorder=10)
    
    # Draw threats
    for threat_x, threat_y in threat_positions:
        threat_circle = Circle((threat_x, threat_y), 0.3, 
                              color='red', alpha=0.7, zorder=5)
        ax.add_patch(threat_circle)
        # Add warning symbol
        ax.text(threat_x, threat_y, '⚠', fontsize=12, ha='center', va='center', 
               weight='bold', zorder=6)
    
    # Draw targets
    for target_x, target_y in target_positions:
        target_circle = Circle((target_x, target_y), 0.3,
                              color='gold', alpha=0.7, zorder=5)
        ax.add_patch(target_circle)
        # Add target symbol
        ax.text(target_x, target_y, 'T', fontsize=10, ha='center', va='center',
               weight='bold', zorder=6)
    
    # Add scenario information
    map_type = "Square (Snake Pattern)" if square_map else "Linear"
    info_text = f"""
Scenario: {scenario_name.upper()}
Map Size: {map_size}×{map_size if square_map else 1}
Map Type: {map_type}
Route Length: {len(route)} steps
Threats: {num_threats}
Targets: {num_targets}
Altitude Levels: {altitude_levels}
Seed: {seed}
    """
    
    # Add text box with scenario info
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black')
    ax.text(0.02, 0.98, info_text.strip(), transform=ax.transAxes,
           fontsize=10, verticalalignment='top', family='monospace',
           bbox=props)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        plt.Line2D([0], [0], color='blue', linewidth=3, alpha=0.5, label='Route'),
        plt.scatter([], [], s=300, c='green', marker='*', edgecolors='black', label='Start'),
        plt.scatter([], [], s=300, c='red', marker='*', edgecolors='black', label='End'),
        Circle((0, 0), 0.3, color='red', alpha=0.7, label='Threat'),
        Circle((0, 0), 0.3, color='gold', alpha=0.7, label='Target'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Labels
    ax.set_xlabel('Position X', fontsize=12, fontweight='bold')
    ax.set_ylabel('Position Y', fontsize=12, fontweight='bold')
    ax.set_title(f'DARTSim Scenario: {scenario_name.upper()}\n'
                f'Map Layout with Threats and Targets',
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved scenario visualization to {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_all_scenarios(output_dir: str = "./results/test_visualizations") -> None:
    """
    Create visualizations for all three scenarios.
    
    Args:
        output_dir: Directory to save visualizations
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Scenario configurations
    scenarios = {
        "baseline": {
            "map_size": 40,
            "num_targets": 3,
            "num_threats": 5,
            "square_map": False,
            "altitude_levels": 4,
            "seed": 42
        },
        "medium": {
            "map_size": 50,
            "num_targets": 5,
            "num_threats": 10,
            "square_map": True,
            "altitude_levels": 5,
            "seed": 123
        },
        "hard": {
            "map_size": 60,
            "num_targets": 8,
            "num_threats": 15,
            "square_map": True,
            "altitude_levels": 6,
            "seed": 456
        }
    }
    
    print("Generating scenario visualizations...")
    
    for scenario_name, config in scenarios.items():
        output_file = output_path / f"scenario_{scenario_name}.png"
        print(f"  Creating {scenario_name} scenario visualization...")
        
        visualize_scenario(
            scenario_name=scenario_name,
            output_path=str(output_file),
            **config
        )
    
    print(f"\nAll scenario visualizations saved to {output_path}")
    
    # Create a comparison figure
    create_scenario_comparison(scenarios, output_path)


def create_scenario_comparison(
    scenarios: Dict[str, Dict[str, Any]],
    output_dir: Path
) -> None:
    """
    Create a side-by-side comparison of all scenarios.
    
    Args:
        scenarios: Dictionary of scenario configurations
        output_dir: Directory to save the comparison
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    for idx, (scenario_name, config) in enumerate(scenarios.items()):
        ax = axes[idx]
        
        # Generate route and positions
        route = generate_route(config["map_size"], config["square_map"])
        threat_positions, target_positions = generate_threat_target_positions(
            config["map_size"],
            config["num_threats"],
            config["num_targets"],
            config["square_map"],
            config["seed"]
        )
        
        # Set up plot
        if config["square_map"]:
            ax.set_xlim(-1, config["map_size"])
            ax.set_ylim(-1, config["map_size"])
        else:
            ax.set_xlim(-1, config["map_size"])
            ax.set_ylim(-2, 3)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Draw route
        route_x = [r[0] for r in route]
        route_y = [r[1] for r in route]
        ax.plot(route_x, route_y, 'b-', linewidth=2, alpha=0.5, zorder=1)
        
        # Mark start and end
        if route:
            start = route[0]
            end = route[-1]
            ax.scatter([start[0]], [start[1]], s=200, c='green', marker='*',
                      edgecolors='black', linewidths=1.5, zorder=10)
            ax.scatter([end[0]], [end[1]], s=200, c='red', marker='*',
                      edgecolors='black', linewidths=1.5, zorder=10)
        
        # Draw threats
        for threat_x, threat_y in threat_positions:
            threat_circle = Circle((threat_x, threat_y), 0.25,
                                 color='red', alpha=0.7, zorder=5)
            ax.add_patch(threat_circle)
            ax.text(threat_x, threat_y, '⚠', fontsize=10, ha='center', va='center',
                   weight='bold', zorder=6)
        
        # Draw targets
        for target_x, target_y in target_positions:
            target_circle = Circle((target_x, target_y), 0.25,
                                  color='gold', alpha=0.7, zorder=5)
            ax.add_patch(target_circle)
            ax.text(target_x, target_y, 'T', fontsize=9, ha='center', va='center',
                   weight='bold', zorder=6)
        
        # Title
        map_type = "Square" if config["square_map"] else "Linear"
        ax.set_title(f'{scenario_name.upper()}\n'
                    f'{config["map_size"]}×{config["map_size"] if config["square_map"] else 1} {map_type} | '
                    f'{config["num_threats"]} Threats | {config["num_targets"]} Targets',
                   fontsize=11, fontweight='bold')
        ax.set_xlabel('Position X', fontsize=10)
        if idx == 0:
            ax.set_ylabel('Position Y', fontsize=10)
    
    plt.suptitle('DARTSim Scenario Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_file = output_dir / "scenario_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved scenario comparison to {output_file}")
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize DARTSim scenarios")
    parser.add_argument("--scenario", type=str, choices=["baseline", "medium", "hard", "all"],
                       default="all", help="Scenario to visualize")
    parser.add_argument("--output-dir", type=str,
                       default="./results/test_visualizations",
                       help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    if args.scenario == "all":
        visualize_all_scenarios(args.output_dir)
    else:
        scenarios = {
            "baseline": {
                "map_size": 40, "num_targets": 3, "num_threats": 5,
                "square_map": False, "altitude_levels": 4, "seed": 42
            },
            "medium": {
                "map_size": 50, "num_targets": 5, "num_threats": 10,
                "square_map": True, "altitude_levels": 5, "seed": 123
            },
            "hard": {
                "map_size": 60, "num_targets": 8, "num_threats": 15,
                "square_map": True, "altitude_levels": 6, "seed": 456
            }
        }
        
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        config = scenarios[args.scenario]
        output_file = output_path / f"scenario_{args.scenario}.png"
        
        visualize_scenario(
            scenario_name=args.scenario,
            output_path=str(output_file),
            **config
        )

