# DARTSim Visualization Guide

This guide explains how to create visually appealing visualizations of DARTSim simulations.

## Quick Start

### Basic Usage

Run a simulation and generate visualizations:

```bash
python scripts/visualize_dartsim.py --steps 100
```

This will:
- Run a DARTSim simulation for up to 100 steps
- Capture trajectory data (position, altitude, formation, ECM status)
- Generate multiple visualization formats:
  - 2D top-down view with altitude coloring
  - 3D trajectory plot
  - Altitude timeline
  - Interactive Plotly visualization (HTML file)

### Options

```bash
# Specify custom save directory
python scripts/visualize_dartsim.py --steps 200 --save-dir ./my_visualizations

# Skip interactive plots (faster)
python scripts/visualize_dartsim.py --steps 100 --no-interactive

# Only run simulation, don't generate visualizations
python scripts/visualize_dartsim.py --steps 100 --no-visualize
```

## Visualization Types

### 1. 2D Top-Down View (`trajectory_2d.png`)
- Shows drone path in X-Y plane
- Color-coded by altitude level (viridis colormap)
- **NEW**: Threat and target overlays
- Markers indicate:
  - Green star: Start position
  - Red star: End position
  - Circles: Tight formation
  - Squares: Loose formation
  - Red outline: ECM active
  - Red triangles (^): Threats detected ahead
  - Gold diamonds (D): Targets detected ahead
  - Dark red X: Destruction point (if destroyed)

### 2. 3D Trajectory (`trajectory_3d.png`)
- Full 3D view showing X, Y, and altitude
- Same marker conventions as 2D plot
- Better visualization of altitude changes

### 3. Altitude Timeline (`altitude_timeline.png`)
- Two subplots:
  - Altitude changes over time
  - Formation and ECM status timeline

### 4. Interactive Plotly Visualization (`trajectory_interactive.html`)
- **Best for presentations and analysis!**
- Fully interactive 3D plot
- Hover to see step details:
  - Position coordinates
  - Altitude level
  - Formation type
  - ECM status
  - Action taken
- Rotatable, zoomable view
- Open in any web browser

## Using with Trained Models

To visualize a trained agent's behavior:

```python
from scripts.visualize_dartsim import DARTSimVisualizer
from src.dartsim_env import DARTSimEnv
from stable_baselines3 import DQN

# Create environment
env = DARTSimEnv()
visualizer = DARTSimVisualizer(env)

# Load trained model
model = DQN.load("path/to/model.zip")

# Run simulation
obs, info = env.reset()
for step in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Record step
    state_data = info.get("state_data", {})
    threats = env._read_sensor("Threat", env.sensor_lookahead)
    targets = env._read_sensor("Target", env.sensor_lookahead)
    visualizer.record_step(state_data, action, threats, targets, info)
    
    if terminated or truncated:
        break

# Get results and create visualizations
results = env._get_results()
visualizer.plot_2d_trajectory(results, "agent_trajectory_2d.png")
visualizer.plot_3d_trajectory(results, "agent_trajectory_3d.png")
visualizer.plot_interactive(results, "agent_trajectory_interactive.html")

env.close()
```

## Understanding the Visualizations

### Color Coding
- **Path color**: Altitude level (dark = low, bright = high)
- **Formation markers**:
  - Circle (○) = Tight formation
  - Square (□) = Loose formation
- **ECM status**: Red outline = ECM active

### What to Look For
1. **Efficient path**: Minimal zigzagging, smooth altitude changes
2. **Altitude adaptation**: Lower altitude near targets, higher near threats
3. **Formation changes**: Tight formation when threats detected
4. **ECM usage**: Strategic use of ECM when needed

## Requirements

Install visualization dependencies:

```bash
pip install plotly matplotlib
```

Plotly is optional but recommended for interactive visualizations.

## Comparison Visualizations

Compare multiple simulation runs side-by-side:

```bash
python scripts/compare_visualizations.py --runs 3 --steps 50
```

This creates:
- Side-by-side trajectory comparison
- Metrics comparison (targets detected, mission success, destroyed)
- Altitude timeline comparison

### Using Comparison in Code

```python
from scripts.visualize_dartsim import DARTSimVisualizer

# Run multiple simulations and collect visualizers
visualizers = [viz1, viz2, viz3]
labels = ["RS-DRL", "Baseline DQN", "Random Policy"]
results = [results1, results2, results3]

# Create comparison
DARTSimVisualizer.plot_comparison(
    visualizers, labels, results, "comparison.png"
)
```

## Tips

1. **For presentations**: Use the interactive HTML file - it's the most impressive
2. **For papers**: Use the 2D or 3D PNG plots with high DPI (300)
3. **For analysis**: Save trajectory JSON and create custom visualizations
4. **Performance**: Use `--no-interactive` for faster generation during development
5. **Comparisons**: Use comparison script to evaluate different policies/agents

## Troubleshooting

### Plotly not working
```bash
pip install plotly kaleido
```

### DARTSim connection errors
Make sure DARTSim container is running:
```bash
docker ps --filter "name=dartsim"
docker start dartsim  # if not running
```

### Missing dependencies
```bash
pip install -r requirements.txt
```

