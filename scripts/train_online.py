"""
Online RS-DRL Training against Live DARTSim
============================================
Trains the RS-DRL (DQN + reward reshaping) agent directly against a running
DARTSim Docker container via the TCP adaptation-manager interface.

This resolves the training-collapse issue observed with offline RL:
the offline dataset only contained 4 of the 8 available actions
(IncAlt2, DecAlt2, EcmOn, EcmOff were never exercised), so the
policy could not correct random Q-initialization for those actions.

With online training all 8 actions are tried via ε-greedy exploration
and the agent receives real simulator feedback for every tactic.

Prerequisites
-------------
  1. Docker Desktop running
  2. DARTSim container started with port 5418 exposed:
         .\\utils\\start_dartsim_live.ps1
     (or manually:)
         docker run -d -p 5418:5418 -p 5901:5901 -p 6901:6901 \\
             --name dartsim gabrielmoreno/dartsim:1.0

Usage
-----
  python scripts/train_online.py
  python scripts/train_online.py --timesteps 200000 --rho 0.3
  python scripts/train_online.py --sim-args "--map-size=50 --num-threats=10"
  python scripts/train_online.py --no-reward-shaping   # plain DQN baseline
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor

from src.live_dartsim_env import LiveDARTSimEnv
from src.rs_drl_dqn import RSDRLDQN
from stable_baselines3 import DQN


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Online RS-DRL training against live DARTSim")

    p.add_argument("--host", default="localhost",
                   help="DARTSim host (default: localhost)")
    p.add_argument("--port", type=int, default=5418,
                   help="DARTSim TCP port (default: 5418)")
    p.add_argument("--container", default="dartsim",
                   help="Docker container name (default: dartsim). "
                        "Pass empty string '' to manage DARTSim externally.")

    p.add_argument("--sim-args", default="",
                   help="Extra arguments forwarded to DARTSim run.sh "
                        "(e.g. '--map-size=40 --num-targets=3 --num-threats=5')")

    p.add_argument("--timesteps", type=int, default=100_000,
                   help="Total environment steps to train for (default: 100 000)")
    p.add_argument("--rho", type=float, default=0.3,
                   help="RS-DRL reward-reshaping factor ρ (default: 0.3)")
    p.add_argument("--no-reward-shaping", action="store_true",
                   help="Use a plain SB3 DQN without reward reshaping")

    p.add_argument("--lr", type=float, default=1e-4,
                   help="Learning rate (default: 1e-4)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--buffer-size", type=int, default=50_000)
    p.add_argument("--exploration-fraction", type=float, default=0.2,
                   help="Fraction of training spent decreasing ε (default: 0.2)")
    p.add_argument("--exploration-final-eps", type=float, default=0.05)
    p.add_argument("--gamma", type=float, default=0.99)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-path", default="models/rs_drl_online",
                   help="Directory to save the trained model")
    p.add_argument("--checkpoint-freq", type=int, default=10_000,
                   help="Save a checkpoint every N steps")
    p.add_argument("--eval-freq", type=int, default=5_000,
                   help="Evaluate the policy every N steps (0 to disable)")
    p.add_argument("--eval-episodes", type=int, default=10)

    p.add_argument("--connect-timeout", type=float, default=30.0,
                   help="Seconds to wait for DARTSim to start per episode")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helper: build environment
# ---------------------------------------------------------------------------

def make_env(args: argparse.Namespace, container: str) -> LiveDARTSimEnv:
    env = LiveDARTSimEnv(
        host=args.host,
        port=args.port,
        container_name=container or None,
        sim_args=args.sim_args,
        connect_timeout=args.connect_timeout,
        step_timeout=30.0,
    )
    return env


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    log_dir = save_path / "tensorboard"
    ckpt_dir = save_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    container = args.container.strip()

    print()
    print("=" * 65)
    print("  Online RS-DRL Training — Live DARTSim")
    print("=" * 65)
    print(f"  Host          : {args.host}:{args.port}")
    print(f"  Container     : {container or '(external)'}")
    print(f"  Sim args      : {args.sim_args or '(defaults)'}")
    print(f"  Timesteps     : {args.timesteps:,}")
    print(f"  Reward shaping: {'disabled (plain DQN)' if args.no_reward_shaping else f'RS-DRL ρ={args.rho}'}")
    print(f"  Save path     : {save_path}")
    print()

    # ── Build training env ────────────────────────────────────────────────
    print("Building training environment...")
    train_env = Monitor(make_env(args, container), filename=str(save_path / "monitor_train.csv"))

    # ── Build eval env (separate container instance not possible on same
    #    port, so we skip if port is shared; a second container would need
    #    a different port) ─────────────────────────────────────────────────
    eval_callback = None
    if args.eval_freq > 0:
        print("Building eval environment...")
        # Eval env manages its own DARTSim restarts via the same container
        eval_env = Monitor(make_env(args, container), filename=str(save_path / "monitor_eval.csv"))
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(save_path / "best"),
            log_path=str(save_path / "eval_log"),
            eval_freq=args.eval_freq,
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            verbose=1,
        )

    # ── Build callbacks ───────────────────────────────────────────────────
    ckpt_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(ckpt_dir),
        name_prefix="rs_drl_online",
        verbose=1,
    )
    callbacks = [ckpt_callback]
    if eval_callback:
        callbacks.append(eval_callback)

    # ── Build model ───────────────────────────────────────────────────────
    dqn_kwargs = dict(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        tensorboard_log=str(log_dir),
        verbose=1,
        seed=args.seed,
    )

    if args.no_reward_shaping:
        print("Using plain SB3 DQN (no reward reshaping)\n")
        model = DQN(**dqn_kwargs)
    else:
        print(f"Using RS-DRL DQN (ρ={args.rho})\n")
        model = RSDRLDQN(rho=args.rho, **dqn_kwargs)

    # ── Train ─────────────────────────────────────────────────────────────
    t0 = time.time()
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=CallbackList(callbacks),
            reset_num_timesteps=True,
            tb_log_name="online_training",
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        elapsed = time.time() - t0

    # ── Save final model ──────────────────────────────────────────────────
    final_path = save_path / "rs_drl_online_final"
    model.save(str(final_path))
    print(f"\nModel saved → {final_path}.zip")
    print(f"Training elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # ── Quick evaluation summary ──────────────────────────────────────────
    print("\nRunning quick evaluation (20 episodes)...")
    eval_env_quick = make_env(args, container)
    successes, destructions, rewards = [], [], []

    for ep in range(20):
        obs, _ = eval_env_quick.reset(seed=args.seed + ep)
        ep_reward = 0.0
        done = False
        results = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env_quick.step(int(action))
            ep_reward += reward
            if done:
                results = info.get("results", {})
        rewards.append(ep_reward)
        successes.append(bool(results.get("missionSuccess", False)))
        destructions.append(bool(results.get("destroyed", False)))

    eval_env_quick.close()

    print(f"  Mean reward    : {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
    print(f"  Mission success: {100*np.mean(successes):.1f}%")
    print(f"  Team destroyed : {100*np.mean(destructions):.1f}%")
    print()
    print("Done. Use TensorBoard to view training curves:")
    print(f"  tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    main()
