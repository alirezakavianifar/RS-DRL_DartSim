# DARTSim RS-DRL Case Study

This project implements RS-DRL (Randomized Reward Shaping for Deep Reinforcement Learning) on DARTSim as a case study, following the plan outlined in `plan.md`.

## Quick Start

The main entry point is `main.py`, which provides a unified interface for all project phases:

```bash
# Show help and available commands
python main.py

# Setup DARTSim environment
python main.py setup

# Train an RS-DRL agent
python main.py train --method rs_drl --rho 0.3 --timesteps 20000 --auto

# Run experiments
python main.py evaluate --methods baseline,rs_drl --seeds 42,43,44 --auto

# Analyze results
python main.py analyze --auto
```

## Project Structure

```
.
├── main.py                 # Main entry point - start here!
├── plan.md                 # Detailed project plan
├── requirements.txt        # Python dependencies
│
├── src/                    # Core implementation
│   ├── dartsim_env.py      # DARTSim environment (offline mode using pre-collected data)
│   ├── rs_drl_dqn.py       # RS-DRL DQN implementation for offline RL training
│   └── mape_k_architecture.py  # MAPE-K architecture
│
├── scripts/                # Execution scripts
│   ├── train_rs_drl.py     # Train RS-DRL agents
│   ├── run_experiments.py  # Run experiment suites
│   ├── grid_search.py      # Hyperparameter grid search
│   ├── evaluate_rs_drl.py  # Evaluate trained agents
│   ├── analyze_experiments.py  # Analyze experiment results
│   ├── plot_results.py     # Generate plots
│   ├── generate_case_study_plots.py  # Case study visualizations
│   ├── train_with_mapek.py # MAPE-K integration demo
│   └── extract_tensorboard_data.py  # Extract TensorBoard data
│
├── utils/                  # Utility scripts
│   ├── start_dartsim_for_training.ps1  # Start DARTSim
│   └── start_tensorboard.ps1  # Start TensorBoard
│
├── experiments_phase4/     # Phase 4 experiment results
├── experiments_ablation/    # Ablation study results
├── models/                 # Trained model checkpoints
├── logs/                   # Training logs
└── results/                # Analysis results and plots
```

## Workflow

### Phase 1: Setup
```bash
python main.py setup
# Follow the instructions to start DARTSim Docker container
```

### Phase 3: Training
```bash
# Train a single agent
python main.py train --method rs_drl --rho 0.3 --timesteps 20000 --auto

# Run grid search
python main.py grid-search --auto
```

### Phase 4: Evaluation
```bash
# Run experiments
python main.py evaluate --methods baseline,rs_drl --seeds 42,43,44 --auto

# Analyze results
python main.py analyze --experiments-dir ./experiments_phase4 --auto
```

### Phase 5: MAPE-K (Optional)
```bash
python main.py mapek --timesteps 20000 --auto
```

## Dependencies

Install dependencies:
```bash
pip install -r requirements.txt
```

## Documentation

- `plan.md` - Complete project plan with phases and implementation details
- `src/` - Core implementation code with docstrings
- `scripts/` - Execution scripts with command-line help

## Notes

- All scripts can be run directly or through `main.py`
- Use `--auto` flag to execute commands, otherwise commands are displayed only
- Results are saved in `experiments_*/`, `models/`, and `results/` directories





