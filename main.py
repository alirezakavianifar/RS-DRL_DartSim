"""
DARTSim RS-DRL Project - Main Entry Point

This script provides a unified interface for running all phases of the DARTSim case study
as outlined in plan.md. The project uses offline RL training from pre-collected data.

Usage:
    python main.py [phase] [options]

Phases:
    collect-data  - Phase 1: Collect offline dataset from DARTSim simulations
    train         - Phase 3: Train RS-DRL agents using offline RL (default mode)
    evaluate      - Phase 4: Run experiments and evaluate trained models
    analyze       - Phase 4: Analyze results and generate plots
    mapek         - Phase 5: Run MAPE-K integration demo (optional)

Workflow:
    1. Collect offline data: python main.py collect-data --episodes 1000 --auto
    2. Train model: python main.py train --method rs_drl --rho 0.3 --auto
    3. Evaluate: python main.py evaluate --auto
    4. Analyze: python main.py analyze --auto

All training uses offline RL mode by default (no DARTSim library or TCP required).
"""

import argparse
import sys
import subprocess
from pathlib import Path


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def phase_collect_data(args):
    """Phase 1: Collect offline dataset from DARTSim simulations."""
    print_header("PHASE 1: Collect Offline Dataset")
    
    print("This phase collects simulation data offline for batch/offline RL training.")
    print("The collected data will be used for offline training (no live DARTSim needed during training).")
    print("\nNote: Requires DARTSim Docker container for data collection.")
    print("      Training will use this collected data and does NOT require DARTSim.")
    
    cmd = [
        sys.executable,
        "scripts/collect_offline_data.py"
    ]
    
    # Add arguments
    if args.episodes:
        cmd.extend(["--episodes", str(args.episodes)])
    if args.scenario:
        cmd.extend(["--scenario", args.scenario])
    if args.output_dir:
        cmd.extend(["--output-dir", args.output_dir])
    if args.container:
        cmd.extend(["--container", args.container])
    if args.seed:
        cmd.extend(["--seed-start", str(args.seed)])
    
    print(f"\nRunning: {' '.join(cmd)}\n")
    
    if args.auto:
        subprocess.run(cmd)
    else:
        print("Command to run:")
        print("  " + " ".join(cmd))
        print("\nExample:")
        print("  python main.py collect-data --episodes 1000 --scenario baseline --auto")


def phase_train(args):
    """Phase 3: Train RS-DRL agents using offline RL (default mode)."""
    print_header("PHASE 3: Train RS-DRL Agents (Offline RL Mode)")
    
    print("Training using offline RL from pre-collected data.")
    print("No DARTSim library or TCP connection required during training.")
    
    cmd = [
        sys.executable,
        "scripts/train_rs_drl.py",
        "--offline"  # Explicitly use offline mode
    ]
    
    # Add arguments
    if args.method:
        cmd.extend(["--method", args.method])
    if args.rho:
        cmd.extend(["--rho", str(args.rho)])
    if args.timesteps:
        cmd.extend(["--timesteps", str(args.timesteps)])
    if args.seed:
        cmd.extend(["--seed", str(args.seed)])
    if args.save_path:
        cmd.extend(["--save-path", args.save_path])
    if args.offline_data_dir:
        cmd.extend(["--offline-data-dir", args.offline_data_dir])
    if args.offline_scenario:
        cmd.extend(["--offline-scenario", args.offline_scenario])
    if args.learning_rate:
        cmd.extend(["--learning-rate", str(args.learning_rate)])
    if args.batch_size:
        cmd.extend(["--batch-size", str(args.batch_size)])
    
    print(f"\nRunning: {' '.join(cmd)}\n")
    
    if args.auto:
        subprocess.run(cmd)
    else:
        print("Command to run:")
        print("  " + " ".join(cmd))
        print("\nExample:")
        print("  python main.py train --method rs_drl --rho 0.3 --timesteps 10000 --auto")


def phase_grid_search(args):
    """Phase 3/4: Run hyperparameter grid search."""
    print_header("PHASE 3/4: Hyperparameter Grid Search")
    
    cmd = [sys.executable, "scripts/grid_search.py"]
    
    if args.config:
        cmd.extend(["--config", args.config])
    if args.output_dir:
        cmd.extend(["--output-dir", args.output_dir])
    
    print(f"Running: {' '.join(cmd)}\n")
    
    if args.auto:
        subprocess.run(cmd)
    else:
        print("Command to run:")
        print("  " + " ".join(cmd))


def phase_evaluate(args):
    """Phase 4: Run experiments and evaluation."""
    print_header("PHASE 4: Run Experiments")
    
    cmd = [sys.executable, "scripts/run_experiments.py"]
    
    if args.methods:
        cmd.extend(["--methods"] + args.methods.split(","))
    if args.seeds:
        cmd.extend(["--seeds"] + [str(s) for s in args.seeds])
    if args.timesteps:
        cmd.extend(["--timesteps", str(args.timesteps)])
    if args.output_base:
        cmd.extend(["--output-base", args.output_base])
    if args.rho_values:
        cmd.extend(["--rho-values"] + [str(r) for r in args.rho_values])
    
    print(f"Running: {' '.join(cmd)}\n")
    
    if args.auto:
        subprocess.run(cmd)
    else:
        print("Command to run:")
        print("  " + " ".join(cmd))


def phase_analyze(args):
    """Phase 4: Analyze results and generate plots."""
    print_header("PHASE 4: Analyze Results")
    
    # Step 1: Analyze experiments
    print("Step 1: Analyzing experiments...")
    cmd_analyze = [
        sys.executable,
        "scripts/analyze_experiments.py",
        "--experiments-dir", args.experiments_dir or "./experiments_phase4"
    ]
    
    if args.auto:
        subprocess.run(cmd_analyze)
    else:
        print(f"  Command: {' '.join(cmd_analyze)}")
    
    # Step 2: Generate plots
    print("\nStep 2: Generating plots...")
    cmd_plot = [
        sys.executable,
        "scripts/plot_results.py",
        "--experiments-dir", args.experiments_dir or "./experiments_phase4",
        "--output-dir", args.output_dir or "./results/plots"
    ]
    
    if args.auto:
        subprocess.run(cmd_plot)
    else:
        print(f"  Command: {' '.join(cmd_plot)}")
    
    # Step 3: Generate case study plots (optional)
    if args.case_study:
        print("\nStep 3: Generating case study plots...")
        cmd_case = [
            sys.executable,
            "scripts/generate_case_study_plots.py",
            "--output-dir", args.output_dir or "./results"
        ]
        
        if args.auto:
            subprocess.run(cmd_case)
        else:
            print(f"  Command: {' '.join(cmd_case)}")


def phase_mapek(args):
    """Phase 5: Run MAPE-K integration demo (optional, uses offline RL)."""
    print_header("PHASE 5: MAPE-K Integration Demo (Offline RL Mode)")
    
    print("MAPE-K adaptive training using offline RL from pre-collected data.")
    
    cmd = [
        sys.executable,
        "scripts/train_with_mapek.py"
    ]
    
    if args.timesteps:
        cmd.extend(["--timesteps", str(args.timesteps)])
    if args.threshold:
        cmd.extend(["--threshold-min-reward", str(args.threshold)])
    if args.offline_data_dir:
        cmd.extend(["--offline-data-dir", args.offline_data_dir])
    if args.offline_scenario:
        cmd.extend(["--offline-scenario", args.offline_scenario])
    
    print(f"Running: {' '.join(cmd)}\n")
    print("Note: This is an optional advanced feature using offline RL.")
    
    if args.auto:
        subprocess.run(cmd)
    else:
        print("Command to run:")
        print("  " + " ".join(cmd))


def phase_benchmark_competing(args):
    """Phase 6: Benchmark against competing literature approaches."""
    print_header("PHASE 6: Benchmark Against Competing Literature Approaches")
    
    cmd = [
        sys.executable,
        "scripts/evaluate_competing_baselines.py"
    ]
    
    if args.episodes:
        cmd.extend(["--episodes", str(args.episodes)])
    if args.scenario:
        cmd.extend(["--scenario", args.scenario])
    if args.output_dir:
        cmd.extend(["--output-dir", args.output_dir])
        
    print(f"Running: {' '.join(cmd)}\n")
    
    if args.auto:
        subprocess.run(cmd)
    else:
        print("Command to run:")
        print("  " + " ".join(cmd))



def main():
    parser = argparse.ArgumentParser(
        description="DARTSim RS-DRL Project - Main Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Phase 1: Collect offline dataset (requires DARTSim Docker container)
  python main.py collect-data --episodes 1000 --scenario baseline --auto

  # Phase 3: Train RS-DRL agent (offline mode, no DARTSim needed)
  python main.py train --method rs_drl --rho 0.3 --timesteps 10000 --auto

  # Train baseline DQN for comparison
  python main.py train --method baseline --timesteps 10000 --auto

  # Train with specific scenario data
  python main.py train --method rs_drl --offline-scenario baseline --auto

  # Run grid search (offline mode)
  python main.py grid-search --auto

  # Run full experiment suite
  python main.py evaluate --methods baseline,rs_drl --seeds 42,43,44 --auto

  # Analyze results
  python main.py analyze --experiments-dir ./experiments_phase4 --auto

  # Run MAPE-K demo (offline mode)
  python main.py mapek --timesteps 10000 --auto
        """
    )
    
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Automatically execute commands (default: show commands only)"
    )
    
    subparsers = parser.add_subparsers(dest="phase", help="Phase to run")
    
    # Collect data phase (Phase 1)
    parser_collect = subparsers.add_parser("collect-data", help="Phase 1: Collect offline dataset")
    parser_collect.add_argument("--episodes", type=int, default=1000, help="Number of episodes to collect")
    parser_collect.add_argument("--scenario", type=str, choices=["baseline", "medium", "hard"], default="baseline", help="Scenario type")
    parser_collect.add_argument("--output-dir", type=str, default="./data/offline", help="Output directory for collected data")
    parser_collect.add_argument("--container", type=str, default="dartsim", help="DARTSim Docker container name")
    parser_collect.add_argument("--seed", type=int, help="Random seed start value for data collection (maps to --seed-start)")
    parser_collect.set_defaults(func=phase_collect_data)
    
    # Train phase (Phase 3)
    parser_train = subparsers.add_parser("train", help="Phase 3: Train agents (offline RL mode)")
    parser_train.add_argument("--method", choices=["baseline", "rs_drl"], default="rs_drl", help="Training method")
    parser_train.add_argument("--rho", type=float, default=0.3, help="RS-DRL reshaping factor")
    parser_train.add_argument("--timesteps", type=int, default=10000, help="Number of gradient steps")
    parser_train.add_argument("--seed", type=int, default=42, help="Random seed")
    parser_train.add_argument("--save-path", type=str, default="./models", help="Path to save model")
    parser_train.add_argument("--offline-data-dir", type=str, default="./data/offline", help="Directory with offline data")
    parser_train.add_argument("--offline-scenario", type=str, help="Filter data by scenario (baseline, medium, hard)")
    parser_train.add_argument("--learning-rate", type=float, help="Learning rate")
    parser_train.add_argument("--batch-size", type=int, help="Batch size")
    parser_train.set_defaults(func=phase_train)
    
    # Grid search phase
    parser_grid = subparsers.add_parser("grid-search", help="Phase 3/4: Grid search")
    parser_grid.add_argument("--config", type=str, default="grid_search_config.json", help="Path to grid search config (can be in root or scripts/)")
    parser_grid.add_argument("--output-dir", type=str, default="./experiments_grid")
    parser_grid.set_defaults(func=phase_grid_search)
    
    # Evaluate phase
    parser_eval = subparsers.add_parser("evaluate", help="Phase 4: Run experiments")
    parser_eval.add_argument("--methods", type=str, default="baseline,rs_drl")
    parser_eval.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser_eval.add_argument("--timesteps", type=int, default=20000)
    parser_eval.add_argument("--output-base", type=str, default="./experiments_phase4")
    parser_eval.add_argument("--rho-values", type=float, nargs="+", default=[0.3])
    parser_eval.set_defaults(func=phase_evaluate)
    
    # Analyze phase
    parser_analyze = subparsers.add_parser("analyze", help="Phase 4: Analyze results")
    parser_analyze.add_argument("--experiments-dir", type=str, default="./experiments_phase4")
    parser_analyze.add_argument("--output-dir", type=str, default="./results/plots")
    parser_analyze.add_argument("--case-study", action="store_true", help="Generate case study plots")
    parser_analyze.set_defaults(func=phase_analyze)
    
    # MAPE-K phase
    parser_mapek = subparsers.add_parser("mapek", help="Phase 5: MAPE-K demo (optional, offline RL)")
    parser_mapek.add_argument("--timesteps", type=int, default=10000, help="Number of training timesteps")
    parser_mapek.add_argument("--threshold", type=float, default=0.7, help="Minimum reward threshold")
    parser_mapek.add_argument("--offline-data-dir", type=str, default="./data/offline", help="Directory with offline data")
    parser_mapek.add_argument("--offline-scenario", type=str, help="Filter data by scenario")
    parser_mapek.set_defaults(func=phase_mapek)

    # Benchmark competing phase
    parser_competing = subparsers.add_parser("benchmark-competing", help="Phase 6: Benchmark competing literature approaches")
    parser_competing.add_argument("--episodes", type=int, default=50, help="Number of evaluation episodes")
    parser_competing.add_argument("--scenario", type=str, choices=["baseline", "medium", "hard"], default="baseline", help="Scenario type")
    parser_competing.add_argument("--output-dir", type=str, default="./results/competing_benchmarks", help="Output directory")
    parser_competing.set_defaults(func=phase_benchmark_competing)

    
    args = parser.parse_args()
    
    if not args.phase:
        parser.print_help()
        print("\n" + "="*70)
        print("Quick Start Guide (Offline RL Workflow):")
        print("="*70)
        print("1. Collect Data:  python main.py collect-data --episodes 1000 --auto")
        print("   (Requires DARTSim Docker container for data collection)")
        print("")
        print("2. Train Model:   python main.py train --method rs_drl --rho 0.3 --auto")
        print("   (Offline RL - no DARTSim library needed during training)")
        print("")
        print("3. Evaluate:      python main.py evaluate --auto")
        print("4. Analyze:       python main.py analyze --auto")
        print("\nNote: Training uses offline RL mode by default (no DARTSim library required).")
        print("      Only data collection requires DARTSim Docker container.")
        print("\nFor detailed workflow, see plan.md")
        sys.exit(0)
    
    args.func(args)


if __name__ == "__main__":
    main()

