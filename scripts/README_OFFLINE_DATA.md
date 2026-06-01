# Offline Data Collection for DARTSim

This guide explains how to collect simulation data offline and use it for batch/offline RL training, avoiding TCP connection issues.

## Why Offline Data Collection?

**Problem**: TCP connection to DARTSim can be unreliable for online RL training.

**Solution**: Collect simulation data offline by running simulations in batch, then use the collected data for offline/batch RL training.

## Quick Start

### 1. Collect Offline Dataset

```bash
# Collect 100 episodes (baseline scenario)
python scripts/collect_offline_data.py --episodes 100 --output-dir ./data/offline

# Collect more episodes with different scenario
python scripts/collect_offline_data.py --episodes 500 --scenario medium --output-dir ./data/offline

# Custom seed range
python scripts/collect_offline_data.py --episodes 50 --seed-start 1000
```

### 2. View Collected Data

The script creates:
- `episodes_*.json` - RL transitions (state, action, reward, next_state, done)
- `trajectories_*.json` - Raw simulation trajectories
- `results_*.json` - Mission results for each episode
- `metadata_*.json` - Dataset metadata

### 3. Use for Offline RL

```bash
# Analyze collected dataset
python scripts/offline_rl_training.py --data-dir ./data/offline
```

## Data Format

### Episode Format (RL Transitions)
```json
{
  "state": [x, y, altitude, formation, ecm, dir_x, dir_y, threats..., targets...],
  "action": "IncAlt",
  "reward": -0.01,
  "next_state": [...],
  "done": false,
  "info": {...}
}
```

### Trajectory Format (Raw Simulation)
```json
{
  "step": 0,
  "position_x": 0,
  "position_y": 0,
  "altitude": 4,
  "formation": "LOOSE",
  "ecm": false,
  "threats_ahead": [true, false, ...],
  "targets_ahead": [false, false, ...]
}
```

## Scenarios

### Baseline (Easy)
- Map size: 40
- Targets: 3
- Threats: 5

### Medium
- Map size: 50
- Targets: 5
- Threats: 10

### Hard
- Map size: 60
- Targets: 8
- Threats: 15

## Usage Examples

### Collect Diverse Dataset

```bash
# Collect from multiple scenarios
python scripts/collect_offline_data.py --episodes 100 --scenario baseline --seed-start 42
python scripts/collect_offline_data.py --episodes 100 --scenario medium --seed-start 200
python scripts/collect_offline_data.py --episodes 100 --scenario hard --seed-start 400
```

### Large Dataset Collection

```bash
# Collect 1000 episodes (takes longer but creates comprehensive dataset)
python scripts/collect_offline_data.py --episodes 1000 --output-dir ./data/offline_large
```

## Advantages of Offline Collection

1. **No TCP Issues**: Uses library interface via Docker (more reliable)
2. **Batch Processing**: Run many simulations efficiently
3. **Reproducibility**: Fixed seeds ensure consistent data
4. **Offline RL**: Can use batch/offline RL algorithms
5. **Data Analysis**: Analyze collected data before training

## Integration with RL Training

The collected data can be used with:
- **Behavioral Cloning**: Learn from demonstration data
- **Offline RL Algorithms**: BCQ, CQL, etc. (requires additional libraries)
- **Data Augmentation**: Combine with online learning
- **Pre-training**: Initialize online RL with offline data

## File Structure

```
data/offline/
├── episodes_baseline_100_20241104_123456.json
├── trajectories_baseline_100_20241104_123456.json
├── results_baseline_100_20241104_123456.json
├── metadata_baseline_100_20241104_123456.json
└── ...
```

## Troubleshooting

### Simulations Failing
- Check Docker container is running: `docker ps --filter "name=dartsim"`
- Check container logs: `docker logs dartsim`
- Try shorter timeout: `--timeout 30` (in code)

### Low Success Rate
- Adjust scenario difficulty
- Check seed values
- Verify simulation parameters

