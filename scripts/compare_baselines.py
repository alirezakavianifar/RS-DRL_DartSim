"""
Baseline comparison for RS-DRL DQN using Off-Policy Evaluation (OPE).

Because the offline environment replays recorded trajectories regardless of
the agent's chosen action, raw episode rewards are identical for all policies.
We therefore use three complementary metrics to distinguish policies:

  1. Action Agreement Rate (AAR)
     Fraction of steps where policy π chooses the same action as the
     behavior policy that collected the data. Higher = more aligned with
     the best-performing data-collector.

  2. Per-Decision Importance-Sampled return (PDIS)
     Weighted episode return using cumulative IS ratio at each step:
         w_t = Π_{i=0}^{t} π(a_i|s_i) / μ(a_i|s_i)
     Since π is deterministic we treat π(a|s)=1 if policy selects a,
     else 0 (hard IS). The weight is clipped at 1 and decays to 0 fast
     for long mismatches. We report per-step weighted reward sum.
     A policy that consistently agrees with the behavior policy earns
     higher IS weights and thus higher PDIS.

  3. Soft IS with temperature τ
     Same as PDIS but π is treated as a soft policy via Q-value softmax,
     giving non-zero probability to non-greedy actions and reducing
     variance of the IS estimator for DQN-based policies.

Policies evaluated:
  1. RS-DRL DQN (ρ=0.3) — trained model
  2. Untrained DQN (0 steps) — same architecture, random weights
  3. Majority-action (always IncAlt=0) — most frequent action in data
  4. Random — uniform over 8 actions

Usage:
    python scripts/compare_baselines.py [--episodes N] [--obs-dim D]
"""
import argparse
import sys
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dartsim_env import OfflineDARTSimEnv
from src.rs_drl_dqn import RSDRLDQN
import src.rs_drl_dqn as _rs_mod
sys.modules.setdefault("rs_drl_dqn", _rs_mod)
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import torch

N_ACTIONS = 8
GAMMA = 0.99


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def make_env(obs_dim: int, max_transitions, seed: int) -> OfflineDARTSimEnv:
    return OfflineDARTSimEnv(
        obs_dim=obs_dim,
        data_dir="./data/offline",
        scenario=None,
        max_transitions=max_transitions,
        seed=seed,
    )


def load_sb3_model(path: str, env: OfflineDARTSimEnv):
    """Try RSDRLDQN first, fall back to DQN."""
    wrapped = Monitor(env, filename=None, allow_early_resets=True)
    try:
        return RSDRLDQN.load(path, env=wrapped)
    except Exception:
        return DQN.load(path, env=wrapped)


# ---------------------------------------------------------------------------
# Behavior policy estimation
# ---------------------------------------------------------------------------

def estimate_behavior_policy(episodes: list) -> dict:
    """
    Estimate μ(a|s) empirically as the fraction of times each action was
    taken from each (discretised) state bucket. Falls back to the global
    marginal if a state is unseen.
    """
    action_counts = np.zeros(N_ACTIONS)
    for ep in episodes:
        for t in ep["transitions"]:
            action_counts[t["action"]] += 1
    total = action_counts.sum()
    global_mu = action_counts / total  # shape (8,)
    return global_mu  # we use the global marginal as μ(a|·)


# ---------------------------------------------------------------------------
# Policy wrappers
# ---------------------------------------------------------------------------

def make_greedy_policy(model):
    """Returns policy_fn and q_value_fn from an SB3 DQN model."""
    def policy_fn(obs: np.ndarray) -> int:
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    def q_values_fn(obs: np.ndarray) -> np.ndarray:
        """Return Q-values for all actions."""
        obs_tensor = torch.as_tensor(obs[None], dtype=torch.float32)
        with torch.no_grad():
            q = model.q_net(obs_tensor)
        return q.cpu().numpy().flatten()

    return policy_fn, q_values_fn


def softmax_probs(q_values: np.ndarray, tau: float = 1.0) -> np.ndarray:
    q = q_values / tau
    q -= q.max()
    e = np.exp(q)
    return e / e.sum()


# ---------------------------------------------------------------------------
# OPE metrics for one episode
# ---------------------------------------------------------------------------

def evaluate_episode_ope(
    transitions: list,
    behavior_mu: np.ndarray,
    policy_fn,
    q_values_fn=None,
    tau: float = 1.0,
    clip_rho: float = 10.0,
) -> dict:
    """
    Compute OPE metrics for a single episode.

    Returns dict with:
      agree_rate   : fraction of steps where policy a == behavior a
      pdis_return  : per-decision IS weighted discounted return
      soft_is_return : soft IS return using Q-value softmax
      steps        : episode length
    """
    agree = 0
    pdis_return = 0.0
    soft_is_return = 0.0
    cumulative_rho = 1.0
    cumulative_soft_rho = 1.0

    for t_idx, t in enumerate(transitions):
        obs = t["state"]
        behavior_action = t["action"]
        policy_action = policy_fn(obs)
        reward = t["reward"]
        disc = GAMMA ** t_idx

        # --- Hard IS (deterministic policy) ---
        # π(a|s) = 1 if policy chooses a, else 0
        pi_a = 1.0 if policy_action == behavior_action else 0.0
        mu_a = float(behavior_mu[behavior_action])
        hard_rho = (pi_a / mu_a) if mu_a > 1e-9 else 0.0
        hard_rho = min(hard_rho, clip_rho)
        cumulative_rho *= hard_rho
        pdis_return += disc * cumulative_rho * reward

        # --- Soft IS (Q-value softmax) ---
        if q_values_fn is not None:
            q_vals = q_values_fn(obs)
            soft_pi = softmax_probs(q_vals, tau=tau)
            soft_pi_a = float(soft_pi[behavior_action])
            soft_rho = (soft_pi_a / mu_a) if mu_a > 1e-9 else 0.0
            soft_rho = min(soft_rho, clip_rho)
            cumulative_soft_rho *= soft_rho
            soft_is_return += disc * cumulative_soft_rho * reward

        if policy_action == behavior_action:
            agree += 1

    steps = len(transitions)
    return {
        "agree_rate": agree / steps if steps > 0 else 0.0,
        "pdis_return": pdis_return,
        "soft_is_return": soft_is_return if q_values_fn else float("nan"),
        "steps": steps,
    }


def evaluate_policy_ope(
    policy_fn,
    env: OfflineDARTSimEnv,
    episode_indices: list,
    behavior_mu: np.ndarray,
    q_values_fn=None,
    tau: float = 1.0,
) -> dict:
    agree_rates, pdis_returns, soft_returns = [], [], []
    lengths = []

    for idx in episode_indices:
        ep = env.episodes[idx]
        res = evaluate_episode_ope(
            ep["transitions"], behavior_mu, policy_fn, q_values_fn, tau
        )
        agree_rates.append(res["agree_rate"])
        pdis_returns.append(res["pdis_return"])
        if not np.isnan(res["soft_is_return"]):
            soft_returns.append(res["soft_is_return"])
        lengths.append(res["steps"])

    # Dataset-level ground-truth metrics (NOT affected by policy choice)
    outcomes = [env.episodes[i]["results"] for i in episode_indices]
    success_rate = np.mean([1.0 if r.get("missionSuccess", False) else 0.0 for r in outcomes])
    targets_det  = np.mean([r.get("targetsDetected", 0) for r in outcomes])
    destroyed    = np.mean([1.0 if r.get("destroyed", False) else 0.0 for r in outcomes])
    raw_return   = np.mean([
        sum(t["reward"] for t in env.episodes[i]["transitions"])
        for i in episode_indices
    ])

    return {
        # Policy-discriminating metrics
        "action_agreement":  round(float(np.mean(agree_rates)), 4),
        "pdis_return":       round(float(np.mean(pdis_returns)), 6),
        "soft_is_return":    round(float(np.mean(soft_returns)) if soft_returns else float("nan"), 6),
        # Ground-truth (same for all policies — shows dataset quality)
        "dataset_success_rate":  round(float(success_rate), 4),
        "dataset_raw_return":    round(float(raw_return), 4),
        "dataset_targets_det":   round(float(targets_det), 4),
        "dataset_destroyed":     round(float(destroyed), 4),
        "mean_ep_length":        round(float(np.mean(lengths)), 1),
        "n_episodes":            len(episode_indices),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--obs-dim", type=int, default=17)
    parser.add_argument("--max-transitions", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--softmax-tau", type=float, default=1.0,
                        help="Temperature for soft IS (lower = more greedy)")
    parser.add_argument("--output-dir", type=str, default="./results/baseline_comparison")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Loading offline environment …")
    env = make_env(args.obs_dim, args.max_transitions, args.seed)
    n_available = len(env.episodes)
    n_eval = min(args.episodes, n_available)
    print(f"Loaded {n_available} episodes. Evaluating on {n_eval}.\n")

    episode_indices = random.sample(range(n_available), n_eval)

    # Estimate behavior policy from ALL episodes
    print("Estimating behavior policy …")
    behavior_mu = estimate_behavior_policy(env.episodes)
    print("  Action distribution (behavior μ):")
    for a_id, name in enumerate(OfflineDARTSimEnv.ACTIONS):
        print(f"    {name:10s}: {behavior_mu[a_id]:.3f}")
    print()

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    print("Loading models …")
    model_rsdrl    = load_sb3_model("models/rs_drl_dqn_rho0.3", env)
    model_untrained = load_sb3_model(
        "models/rs_drl_dqn_rho0.3_checkpoints/rs_drl_dqn_0_steps", env
    )
    policy_rsdrl,     qfn_rsdrl     = make_greedy_policy(model_rsdrl)
    policy_untrained, qfn_untrained = make_greedy_policy(model_untrained)
    policy_majority  = lambda _: 0                       # always IncAlt
    policy_random    = lambda _: env.action_space.sample()

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    configs = [
        ("RS-DRL DQN (ρ=0.3)",         policy_rsdrl,     qfn_rsdrl),
        ("Untrained DQN (0 steps)",     policy_untrained, qfn_untrained),
        ("Majority action (IncAlt)",    policy_majority,  None),
        ("Random",                      policy_random,    None),
    ]

    results = {}
    for name, pfn, qfn in configs:
        print(f"Evaluating: {name} …")
        m = evaluate_policy_ope(pfn, env, episode_indices, behavior_mu, qfn, args.softmax_tau)
        results[name] = m
        print(f"  AAR={m['action_agreement']:.3f}  PDIS={m['pdis_return']:.5f}"
              + (f"  SoftIS={m['soft_is_return']:.5f}" if not np.isnan(m['soft_is_return']) else ""))

    # ------------------------------------------------------------------
    # Oracle
    # ------------------------------------------------------------------
    all_outcomes = [ep["results"] for ep in env.episodes]
    oracle_success  = float(np.mean([1.0 if r.get("missionSuccess", False) else 0.0 for r in all_outcomes]))
    oracle_targets  = float(np.mean([r.get("targetsDetected", 0) for r in all_outcomes]))
    oracle_destroyed = float(np.mean([1.0 if r.get("destroyed", False) else 0.0 for r in all_outcomes]))

    # ------------------------------------------------------------------
    # Print table
    # ------------------------------------------------------------------
    rows = []
    for name, m in results.items():
        rows.append({
            "Policy":        name,
            "AAR":           f"{m['action_agreement']:.3f}",
            "PDIS":          f"{m['pdis_return']:.5f}",
            "SoftIS":        f"{m['soft_is_return']:.5f}" if not np.isnan(m.get('soft_is_return', float('nan'))) else "—",
        })
    df = pd.DataFrame(rows)

    print("\n" + "=" * 72)
    print(f"OFF-POLICY EVALUATION  ({n_eval} episodes, seed={args.seed})")
    print("=" * 72)
    print(df.to_string(index=False))
    print()
    print("Columns:")
    print("  AAR    = Action Agreement Rate: fraction of steps where policy matches behavior")
    print("  PDIS   = Per-Decision IS return (hard, deterministic π)")
    print("  SoftIS = Per-Decision IS return (soft, Q-value softmax π)")
    print()
    print("Dataset ground truth (same for all policies):")
    m0 = next(iter(results.values()))
    print(f"  Mission success:   {m0['dataset_success_rate']:.1%}")
    print(f"  Raw return:        {m0['dataset_raw_return']:.3f}")
    print(f"  Targets detected:  {m0['dataset_targets_det']:.2f}")
    print(f"  Team destroyed:    {m0['dataset_destroyed']:.1%}")
    print()
    print(f"Dataset oracle (all {n_available} episodes):")
    print(f"  Mission success: {oracle_success:.1%}  |  Targets: {oracle_targets:.2f}  |  Destroyed: {oracle_destroyed:.1%}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "ope_comparison_table.csv", index=False)
    with open(out / "ope_comparison_full.json", "w") as f:
        json.dump({
            "results": {k: {kk: (vv if not isinstance(vv, float) or not np.isnan(vv) else None)
                             for kk, vv in v.items()} for k, v in results.items()},
            "oracle": {"success": oracle_success, "targets": oracle_targets, "destroyed": oracle_destroyed},
            "behavior_mu": {OfflineDARTSimEnv.ACTIONS[i]: float(behavior_mu[i]) for i in range(N_ACTIONS)},
            "n_eval": n_eval, "n_total": n_available, "seed": args.seed,
        }, f, indent=2)
    print(f"\nResults saved to {out}/")


if __name__ == "__main__":
    main()
