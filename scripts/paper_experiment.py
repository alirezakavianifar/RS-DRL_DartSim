"""
Paper Experiment: RS-DRL vs Baseline DQN — Multi-Seed Convergence Study
========================================================================
Runs both methods across multiple seeds, captures per-epoch training
loss and Q-value statistics, then generates publication-quality figures
with confidence intervals (mean ± std shaded bands).

Designed to be memory-efficient (fits in 8 GB RAM) and crash-resilient
(saves intermediate results after every seed/method combo).

Usage:
    python scripts/paper_experiment.py --timesteps 50000 --seeds 42 43 44
"""

import sys, os, argparse, json, gc, time, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import uniform_filter1d

from src.rs_drl_dqn import RSDRLDQN
from stable_baselines3 import DQN
from src.dartsim_env import OfflineDARTSimEnv
from stable_baselines3.common.monitor import Monitor
from scripts.offline_rl_training import (
    load_offline_dataset_generator,
    convert_episode_generator_to_buffer_format,
)

# ── Colour palette ────────────────────────────────────────────────────
BG     = '#0D0D1A'
PANEL  = '#13132B'
PURPLE = '#7C6EFA'
PINK   = '#FF5E8A'
TEAL   = '#00D4AA'
GRID   = '#1E1E3A'
WHITE  = '#E8E8FF'
GREY   = '#9090B0'


def build_replay_buffer(model, offline_data_dir, offline_scenario,
                        total_timesteps, buffer_size, seed):
    """Load offline data into the model's replay buffer."""
    episode_generator = load_offline_dataset_generator(
        offline_data_dir, scenario=offline_scenario
    )
    buffer_batch_generator = convert_episode_generator_to_buffer_format(
        episode_generator, batch_size=10000
    )
    first_batch = next(buffer_batch_generator)
    max_transitions = min(buffer_size, total_timesteps * 10)
    total = 0

    def _add_batch(batch, limit):
        nonlocal total
        obs   = batch["observations"]
        nobs  = batch["next_observations"]
        acts  = batch["actions"].astype(np.int64)
        rews  = batch["rewards"].astype(np.float32)
        dones = batch["dones"].astype(np.bool_)
        n = min(len(obs), limit - total)
        for i in range(n):
            model.replay_buffer.add(
                obs[i:i+1], nobs[i:i+1],
                np.array([acts[i]]), np.array([rews[i]]),
                np.array([dones[i]]), [{}]
            )
        total += n

    _add_batch(first_batch, max_transitions)
    del first_batch; gc.collect()

    for batch in buffer_batch_generator:
        if total >= max_transitions:
            break
        _add_batch(batch, max_transitions)
        del batch
        gc.collect()

    return total


def train_one_seed(method, rho, seed, total_timesteps,
                   offline_data_dir, batch_size=32,
                   learning_rate=1e-4, buffer_size=50_000):
    """Train one model; return per-epoch loss, max-Q, and reshaping rate."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    obs_dim = 17
    env = OfflineDARTSimEnv(
        obs_dim=obs_dim, data_dir=offline_data_dir,
        max_transitions=min(buffer_size, total_timesteps * 10), seed=seed
    )
    env = Monitor(env)

    # Build model — use buffer_size directly (no inflating to 1M)
    model_cls = RSDRLDQN if method == 'rs_drl' else DQN
    model_kwargs = dict(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        gamma=0.99,
        batch_size=batch_size,
        buffer_size=buffer_size,
        learning_starts=0,
        exploration_fraction=0.0,
        exploration_initial_eps=0.0,
        exploration_final_eps=0.0,
        target_update_interval=1000,
        train_freq=1,
        gradient_steps=1,
        verbose=0,
        seed=seed,
    )
    if method == 'rs_drl':
        model_kwargs['rho'] = rho
        model_kwargs['optimistic_reward'] = 1.0

    model = model_cls(**model_kwargs)

    # Populate buffer
    n_transitions = build_replay_buffer(
        model, offline_data_dir, None, total_timesteps, buffer_size, seed
    )
    print(f"    Buffer populated with {n_transitions} transitions")

    # Training loop — record per-epoch metrics
    gradient_steps_per_epoch = max(1, n_transitions // batch_size)
    n_epochs = max(1, total_timesteps // gradient_steps_per_epoch)

    epoch_losses      = []
    epoch_max_q_mean  = []
    epoch_max_q_std   = []
    epoch_reshape_rate = []

    # Probe observations (sampled once)
    probe_obs = []
    obs_env, _ = env.reset()
    for _ in range(300):
        probe_obs.append(obs_env.copy())
        a, _ = model.predict(obs_env, deterministic=True)
        obs_env, _, t, tr, _ = env.step(a)
        if t or tr:
            obs_env, _ = env.reset()
    probe_t = torch.FloatTensor(np.array(probe_obs))

    t0 = time.time()
    for epoch in range(n_epochs):
        # --- one epoch of gradient steps ---
        losses_this_epoch = []
        model.policy.set_training_mode(True)
        for _ in range(gradient_steps_per_epoch):
            replay_data = model.replay_buffer.sample(batch_size, env=model._vec_normalize_env)

            # Reward reshaping (RS-DRL only)
            if method == 'rs_drl':
                rews_np = replay_data.rewards.cpu().numpy()
                orig = rews_np.copy()
                rews_np = model.reward_shaping.reshape_rewards(rews_np)
                model.total_transitions += len(orig)
                model.num_reshaped_transitions += np.sum(rews_np != orig)
                rews_t = torch.tensor(rews_np, dtype=torch.float32,
                                      device=replay_data.rewards.device)
                fields = type(replay_data)._fields
                kw = {f: getattr(replay_data, f) for f in fields}
                kw['rewards'] = rews_t
                replay_data = type(replay_data)(**kw)

            with torch.no_grad():
                nq = model.q_net_target(replay_data.next_observations)
                nq, _ = nq.max(dim=1)
                nq = nq.reshape(-1, 1)
                target_q = replay_data.rewards + (1 - replay_data.dones) * model.gamma * nq

            cq = model.q_net(replay_data.observations)
            cq = torch.gather(cq, dim=1, index=replay_data.actions.long())
            loss = torch.nn.functional.mse_loss(cq, target_q)
            losses_this_epoch.append(loss.item())

            model.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), model.max_grad_norm)
            model.policy.optimizer.step()

            if not hasattr(model, '_n_updates'):
                model._n_updates = 0
            if model._n_updates % model.target_update_interval == 0:
                from stable_baselines3.common.utils import polyak_update
                polyak_update(model.q_net.parameters(),
                              model.q_net_target.parameters(), model.tau)
            model._n_updates += 1

        # --- record epoch metrics ---
        epoch_losses.append(np.mean(losses_this_epoch))

        with torch.no_grad():
            q_vals = model.q_net(probe_t).numpy().max(axis=1)
        epoch_max_q_mean.append(float(q_vals.mean()))
        epoch_max_q_std.append(float(q_vals.std()))

        if method == 'rs_drl':
            r = model.get_reshaping_stats()
            epoch_reshape_rate.append(r['reshaping_rate'])
        else:
            epoch_reshape_rate.append(0.0)

        pct = int((epoch + 1) / n_epochs * 100)
        if (epoch + 1) % max(1, n_epochs // 10) == 0 or epoch == n_epochs - 1:
            elapsed = time.time() - t0
            print(f"    [{method} seed={seed}] Epoch {epoch+1}/{n_epochs} "
                  f"({pct}%) | loss={epoch_losses[-1]:.4f} | "
                  f"max-Q={epoch_max_q_mean[-1]:.3f} | "
                  f"reshape={epoch_reshape_rate[-1]:.3f} | "
                  f"time={elapsed:.0f}s")

    env.close()
    # Free model memory
    del model, probe_t
    gc.collect()

    return {
        'losses':        np.array(epoch_losses),
        'max_q_mean':    np.array(epoch_max_q_mean),
        'max_q_std':     np.array(epoch_max_q_std),
        'reshape_rate':  np.array(epoch_reshape_rate),
        'n_epochs':      n_epochs,
    }


def smooth(arr, w=3):
    return uniform_filter1d(arr, size=max(1, w))


def plot_paper_figure(results, output_dir, timesteps, seeds, rho):
    """Generate publication-quality 2×3 figure."""
    os.makedirs(output_dir, exist_ok=True)

    rs_results = results['rs_drl']   # list of dicts, one per seed
    bl_results = results['baseline']

    def stack(key):
        return np.array([r[key] for r in rs_results]), \
               np.array([r[key] for r in bl_results])

    rs_loss,  bl_loss  = stack('losses')
    rs_qmean, bl_qmean = stack('max_q_mean')
    rs_qstd,  bl_qstd  = stack('max_q_std')
    rs_rate,  _        = stack('reshape_rate')

    n_epochs = rs_results[0]['n_epochs']
    x = np.arange(1, n_epochs + 1)

    def band(ax, x, arr, color, label):
        """Plot mean ± std band."""
        mu  = smooth(arr.mean(axis=0))
        std = arr.std(axis=0)
        ax.plot(x, mu, color=color, lw=2, label=label, zorder=3)
        ax.fill_between(x, mu - std, mu + std, color=color,
                        alpha=0.18, zorder=2)
        return mu[-1], std[-1]

    def style(ax, title, xlabel='Epoch', ylabel=''):
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
        ax.tick_params(colors=GREY, labelsize=8.5)
        ax.set_title(title, color=WHITE, fontsize=10.5, fontweight='bold', pad=8)
        ax.set_xlabel(xlabel, color=GREY, fontsize=9)
        ax.set_ylabel(ylabel, color=GREY, fontsize=9)
        ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)

    fig = plt.figure(figsize=(18, 10), facecolor=BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.35,
                             left=0.06, right=0.97, top=0.88, bottom=0.09)

    # ── (0,0) TD Loss convergence ────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    style(ax, 'TD Loss (MSE) Convergence', ylabel='Loss')
    rs_mu, rs_s = band(ax, x, rs_loss, PURPLE, f'RS-DRL ρ={rho}')
    bl_mu, bl_s = band(ax, x, bl_loss, PINK,   'Baseline DQN')
    ax.legend(fontsize=8.5, facecolor='#1A1A3A', edgecolor=GRID, labelcolor=WHITE)

    # ── (0,1) Q-value mean evolution ────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    style(ax2, 'Max Q-Value Mean (↑ better)', ylabel='Mean max-Q')
    band(ax2, x, rs_qmean, PURPLE, f'RS-DRL ρ={rho}')
    band(ax2, x, bl_qmean, PINK,   'Baseline DQN')
    ax2.legend(fontsize=8.5, facecolor='#1A1A3A', edgecolor=GRID, labelcolor=WHITE)

    # ── (0,2) Q-value std evolution ─────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    style(ax3, 'Q-Value Std Dev (policy spread)', ylabel='Std max-Q')
    band(ax3, x, rs_qstd, PURPLE, f'RS-DRL ρ={rho}')
    band(ax3, x, bl_qstd, PINK,   'Baseline DQN')
    ax3.legend(fontsize=8.5, facecolor='#1A1A3A', edgecolor=GRID, labelcolor=WHITE)

    # ── (1,0) Reshaping rate over training ──────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    style(ax4, 'Reward Reshaping Rate (RS-DRL)', ylabel='Reshaping rate')
    mu_rate = smooth(rs_rate.mean(axis=0))
    ax4.plot(x, mu_rate, color=TEAL, lw=2, label=f'RS-DRL ρ={rho}', zorder=3)
    ax4.fill_between(x, mu_rate - rs_rate.std(axis=0),
                        mu_rate + rs_rate.std(axis=0),
                     color=TEAL, alpha=0.18, zorder=2)
    ax4.axhline(rho, color=GREY, lw=1, ls='--', alpha=0.6, label=f'Target ρ={rho}')
    ax4.legend(fontsize=8.5, facecolor='#1A1A3A', edgecolor=GRID, labelcolor=WHITE)

    # ── (1,1) Loss ratio RS-DRL / Baseline ──────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    style(ax5, 'Loss Ratio  RS-DRL / Baseline  (<1 = RS-DRL better)', ylabel='Ratio')
    ratio = rs_loss.mean(axis=0) / np.maximum(bl_loss.mean(axis=0), 1e-8)
    ratio_s = smooth(ratio)
    ax5.plot(x, ratio_s, color=TEAL, lw=2, zorder=3)
    ax5.axhline(1.0, color=WHITE, lw=1, ls='--', alpha=0.4, label='Parity')
    ax5.fill_between(x, ratio_s, 1.0,
                     where=(ratio_s < 1.0), alpha=0.2, color=PURPLE,
                     label='RS-DRL advantage')
    ax5.fill_between(x, ratio_s, 1.0,
                     where=(ratio_s >= 1.0), alpha=0.2, color=PINK,
                     label='Baseline advantage')
    ax5.legend(fontsize=8, facecolor='#1A1A3A', edgecolor=GRID, labelcolor=WHITE)

    # ── (1,2) Summary table ──────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    style(ax6, 'Summary Table (mean ± std, all seeds)')
    ax6.axis('off')

    final_rs_loss = rs_loss[:, -1]
    final_bl_loss = bl_loss[:, -1]
    final_rs_q    = rs_qmean[:, -1]
    final_bl_q    = bl_qmean[:, -1]
    pct_loss_imp  = (1 - final_rs_loss.mean() / max(final_bl_loss.mean(), 1e-8)) * 100
    pct_q_imp     = (final_rs_q.mean() / max(final_bl_q.mean(), 1e-8) - 1) * 100

    rows = [
        ('Metric',                'RS-DRL',                               'Baseline'),
        ('Final TD Loss',
         f'{final_rs_loss.mean():.4f}±{final_rs_loss.std():.4f}',
         f'{final_bl_loss.mean():.4f}±{final_bl_loss.std():.4f}'),
        ('Final Mean max-Q',
         f'{final_rs_q.mean():.3f}±{final_rs_q.std():.3f}',
         f'{final_bl_q.mean():.3f}±{final_bl_q.std():.3f}'),
        ('Loss improvement', f'{pct_loss_imp:+.1f}%', '—'),
        ('Q-value gain',    f'{pct_q_imp:+.1f}%',    '—'),
        ('Reshape rate',    f'{rho:.0%} target',      'N/A'),
        ('Seeds',           str(len(seeds)),           str(len(seeds))),
        ('Timesteps',       str(timesteps),            str(timesteps)),
    ]
    col_x = [0.01, 0.44, 0.74]
    for i, row in enumerate(rows):
        y = 0.97 - i * 0.115
        for j, (val, cx) in enumerate(zip(row, col_x)):
            c = WHITE if (i == 0 or j == 0) else (PURPLE if j == 1 else PINK)
            fw = 'bold' if i == 0 else 'normal'
            ax6.text(cx, y, val, transform=ax6.transAxes, color=c,
                     fontsize=8.5, fontweight=fw, va='top')
        if i == 0:
            ax6.plot([0.01, 0.99], [y - 0.025, y - 0.025],
                     transform=ax6.transAxes, color=GRID, lw=1)

    # ── Super title ──────────────────────────────────────────────────
    seed_str = ', '.join(map(str, seeds))
    fig.text(0.5, 0.945,
             'RS-DRL vs Baseline DQN — Convergence & Policy Quality Analysis',
             ha='center', color=WHITE, fontsize=15, fontweight='bold')
    fig.text(0.5, 0.915,
             f'DARTSim Offline RL  |  {timesteps:,} timesteps  |  '
             f'seeds={seed_str}  |  ρ={rho}  |  shaded = ±1 std across seeds',
             ha='center', color=GREY, fontsize=9.5)

    out_path = Path(output_dir) / 'paper_convergence_figure.png'
    plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=BG)
    print(f'\n✅ Publication figure saved → {out_path}')
    plt.close()

    # ── Also save raw numbers as JSON ────────────────────────────────
    summary = {
        'timesteps': timesteps, 'seeds': seeds, 'rho': rho,
        'rs_drl': {
            'final_loss_mean': float(final_rs_loss.mean()),
            'final_loss_std':  float(final_rs_loss.std()),
            'final_q_mean':    float(final_rs_q.mean()),
            'final_q_std':     float(final_rs_q.std()),
            'pct_loss_improvement_vs_baseline': float(pct_loss_imp),
            'pct_q_gain_vs_baseline':           float(pct_q_imp),
        },
        'baseline': {
            'final_loss_mean': float(final_bl_loss.mean()),
            'final_loss_std':  float(final_bl_loss.std()),
            'final_q_mean':    float(final_bl_q.mean()),
            'final_q_std':     float(final_bl_q.std()),
        }
    }
    json_path = Path(output_dir) / 'paper_results_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'✅ Numerical summary saved → {json_path}')

    # ── Save per-epoch data as CSV for supplementary material ────────
    csv_path = Path(output_dir) / 'per_epoch_data.csv'
    with open(csv_path, 'w') as f:
        f.write('epoch,rs_drl_loss_mean,rs_drl_loss_std,baseline_loss_mean,baseline_loss_std,'
                'rs_drl_q_mean,rs_drl_q_std,baseline_q_mean,baseline_q_std,reshape_rate_mean\n')
        for i in range(n_epochs):
            f.write(f'{i+1},'
                    f'{rs_loss[:, i].mean():.6f},{rs_loss[:, i].std():.6f},'
                    f'{bl_loss[:, i].mean():.6f},{bl_loss[:, i].std():.6f},'
                    f'{rs_qmean[:, i].mean():.6f},{rs_qmean[:, i].std():.6f},'
                    f'{bl_qmean[:, i].mean():.6f},{bl_qmean[:, i].std():.6f},'
                    f'{rs_rate[:, i].mean():.6f}\n')
    print(f'✅ Per-epoch CSV saved → {csv_path}')

    # ── LaTeX table for direct article insertion ─────────────────────
    latex_path = Path(output_dir) / 'results_table.tex'
    with open(latex_path, 'w') as f:
        f.write('% Auto-generated results table\n')
        f.write('\\begin{table}[ht]\n\\centering\n')
        f.write(f'\\caption{{RS-DRL vs Baseline DQN ({timesteps:,} timesteps, '
                f'{len(seeds)} seeds, $\\rho={rho}$)}}\n')
        f.write('\\label{tab:results}\n')
        f.write('\\begin{tabular}{lcc}\n\\toprule\n')
        f.write('Metric & RS-DRL & Baseline DQN \\\\\n\\midrule\n')
        f.write(f'Final TD Loss & ${final_rs_loss.mean():.4f} \\pm {final_rs_loss.std():.4f}$'
                f' & ${final_bl_loss.mean():.4f} \\pm {final_bl_loss.std():.4f}$ \\\\\n')
        f.write(f'Final max-Q & ${final_rs_q.mean():.3f} \\pm {final_rs_q.std():.3f}$'
                f' & ${final_bl_q.mean():.3f} \\pm {final_bl_q.std():.3f}$ \\\\\n')
        f.write(f'Loss improvement & ${pct_loss_imp:+.1f}\\%$ & --- \\\\\n')
        f.write(f'Q-value gain & ${pct_q_imp:+.1f}\\%$ & --- \\\\\n')
        f.write(f'Reshape rate & ${rho:.0%}$ target & N/A \\\\\n')
        f.write('\\bottomrule\n\\end{tabular}\n\\end{table}\n')
    print(f'✅ LaTeX table saved → {latex_path}')

    return summary


def save_intermediate(results, output_dir, method, seed):
    """Save one seed/method result as JSON for crash-resilience."""
    os.makedirs(output_dir, exist_ok=True)
    path = Path(output_dir) / f'intermediate_{method}_seed{seed}.json'
    data = {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in results.items()}
    with open(path, 'w') as f:
        json.dump(data, f)
    print(f"    💾 Saved intermediate: {path.name}")


def load_intermediate(output_dir, method, seed):
    """Load a previously saved intermediate result (for resume)."""
    path = Path(output_dir) / f'intermediate_{method}_seed{seed}.json'
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    # Convert lists back to numpy arrays
    for k in ('losses', 'max_q_mean', 'max_q_std', 'reshape_rate'):
        if k in data and isinstance(data[k], list):
            data[k] = np.array(data[k])
    print(f"    📂 Resumed from saved: {path.name}")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps',    type=int,   default=50000)
    parser.add_argument('--seeds',        type=int,   nargs='+', default=[42, 43, 44])
    parser.add_argument('--rho',          type=float, default=0.3)
    parser.add_argument('--offline-data-dir', type=str, default='./data/offline')
    parser.add_argument('--output-dir',   type=str,   default='./results/paper')
    parser.add_argument('--batch-size',   type=int,   default=32)
    parser.add_argument('--learning-rate',type=float, default=1e-4)
    parser.add_argument('--buffer-size',  type=int,   default=50_000)
    parser.add_argument('--resume',       action='store_true', default=True,
                        help='Resume from saved intermediate results')
    args = parser.parse_args()

    print('=' * 65)
    print('  RS-DRL Publication Experiment')
    print(f'  Timesteps: {args.timesteps:,}  |  Seeds: {args.seeds}  |  ρ={args.rho}')
    print(f'  Buffer size: {args.buffer_size:,}  |  Batch: {args.batch_size}  |  LR: {args.learning_rate}')
    print('=' * 65)

    results = {'rs_drl': [], 'baseline': []}
    total_start = time.time()

    for seed in args.seeds:
        for method in ('rs_drl', 'baseline'):
            # Try to resume from saved intermediate
            if args.resume:
                saved = load_intermediate(args.output_dir, method, seed)
                if saved is not None:
                    results[method].append(saved)
                    continue

            print(f'\n▶ Training {method.upper()} | seed={seed}')
            try:
                r = train_one_seed(
                    method=method, rho=args.rho, seed=seed,
                    total_timesteps=args.timesteps,
                    offline_data_dir=args.offline_data_dir,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    buffer_size=args.buffer_size,
                )
                results[method].append(r)
                # Save intermediate for crash-resilience
                save_intermediate(r, args.output_dir, method, seed)
                gc.collect()
            except Exception as e:
                print(f"    ❌ FAILED: {method} seed={seed}: {e}")
                traceback.print_exc()
                # Continue with other seeds
                continue

    # Verify we have enough results
    if not results['rs_drl'] or not results['baseline']:
        print("\n❌ Not enough results to generate figure. Aborting.")
        return

    # Ensure equal number of results (trim to min)
    n = min(len(results['rs_drl']), len(results['baseline']))
    results['rs_drl'] = results['rs_drl'][:n]
    results['baseline'] = results['baseline'][:n]

    total_elapsed = time.time() - total_start
    print(f'\n{"=" * 65}')
    print(f'  All training runs complete ({total_elapsed:.0f}s) — generating paper figure...')
    print('=' * 65)

    summary = plot_paper_figure(
        results, args.output_dir,
        args.timesteps, args.seeds[:n], args.rho
    )

    print('\n📊 KEY RESULTS:')
    print(f"   RS-DRL  final loss : {summary['rs_drl']['final_loss_mean']:.4f} ± {summary['rs_drl']['final_loss_std']:.4f}")
    print(f"   Baseline final loss: {summary['baseline']['final_loss_mean']:.4f} ± {summary['baseline']['final_loss_std']:.4f}")
    print(f"   Loss improvement   : {summary['rs_drl']['pct_loss_improvement_vs_baseline']:+.1f}%")
    print(f"   RS-DRL  final Q    : {summary['rs_drl']['final_q_mean']:.3f} ± {summary['rs_drl']['final_q_std']:.3f}")
    print(f"   Baseline final Q   : {summary['baseline']['final_q_mean']:.3f} ± {summary['baseline']['final_q_std']:.3f}")
    print(f"   Q-value gain       : {summary['rs_drl']['pct_q_gain_vs_baseline']:+.1f}%")
    print(f"\n   Total time: {total_elapsed:.0f}s")


if __name__ == '__main__':
    main()
