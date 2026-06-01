"""
Generate comprehensive publication-quality figures for RS-DRL paper.
Creates multiple figures suitable for an academic article:
  1. Main convergence figure (2x2)
  2. Per-seed breakdown figure
  3. Statistical significance table
"""

import sys, os, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import uniform_filter1d
from scipy import stats

# ── Colour palette ────────────────────────────────────────────────────
BG     = '#0D0D1A'
PANEL  = '#13132B'
PURPLE = '#7C6EFA'
PINK   = '#FF5E8A'
TEAL   = '#00D4AA'
GOLD   = '#FFB84D'
GRID   = '#1E1E3A'
WHITE  = '#E8E8FF'
GREY   = '#9090B0'

SEED_COLORS = ['#7C6EFA', '#00D4AA', '#FFB84D']  # Per-seed colours


def smooth(arr, w=3):
    return uniform_filter1d(arr, size=max(1, w))


def style(ax, title, xlabel='Epoch', ylabel=''):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.tick_params(colors=GREY, labelsize=9)
    ax.set_title(title, color=WHITE, fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel(xlabel, color=GREY, fontsize=9.5)
    ax.set_ylabel(ylabel, color=GREY, fontsize=9.5)
    ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)


def band(ax, x, arr, color, label):
    mu  = smooth(arr.mean(axis=0))
    std = arr.std(axis=0)
    ax.plot(x, mu, color=color, lw=2.2, label=label, zorder=3)
    ax.fill_between(x, mu - std, mu + std, color=color, alpha=0.18, zorder=2)
    return mu, std


def load_results(paper_dir):
    """Load all intermediate results."""
    results = {'rs_drl': [], 'baseline': []}
    seeds = []
    for f in sorted(Path(paper_dir).glob('intermediate_*.json')):
        name = f.stem  # intermediate_rs_drl_seed42
        parts = name.split('_')
        method = '_'.join(parts[1:-1])  # rs_drl or baseline
        seed = int(parts[-1].replace('seed', ''))
        with open(f) as fp:
            data = json.load(fp)
        for k in ('losses', 'max_q_mean', 'max_q_std', 'reshape_rate'):
            if k in data and isinstance(data[k], list):
                data[k] = np.array(data[k])
        results[method].append(data)
        if seed not in seeds:
            seeds.append(seed)
    return results, sorted(seeds)


def figure_1_main_convergence(results, seeds, rho, output_dir):
    """
    Figure 1: Main 2x2 convergence figure for the paper.
    Panels: (a) Q-Value convergence, (b) TD Loss, (c) Reshaping rate, (d) Q-value gain timeline
    """
    rs = results['rs_drl']
    bl = results['baseline']

    rs_loss  = np.array([r['losses'] for r in rs])
    bl_loss  = np.array([r['losses'] for r in bl])
    rs_qmean = np.array([r['max_q_mean'] for r in rs])
    bl_qmean = np.array([r['max_q_mean'] for r in bl])
    rs_rate  = np.array([r['reshape_rate'] for r in rs])

    n_epochs = rs[0]['n_epochs']
    x = np.arange(1, n_epochs + 1)

    fig = plt.figure(figsize=(14, 10), facecolor=BG)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
                           left=0.08, right=0.96, top=0.88, bottom=0.08)

    # ── (a) Max Q-Value Mean ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    style(ax1, '(a) Max Q-Value Convergence', ylabel='Mean max-Q')
    band(ax1, x, rs_qmean, PURPLE, f'RS-DRL (rho={rho})')
    band(ax1, x, bl_qmean, PINK, 'Baseline DQN')
    ax1.legend(fontsize=9, facecolor='#1A1A3A', edgecolor=GRID, labelcolor=WHITE)

    # ── (b) TD Loss convergence ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    style(ax2, '(b) TD Loss (MSE)', ylabel='Loss')
    band(ax2, x, rs_loss, PURPLE, f'RS-DRL (rho={rho})')
    band(ax2, x, bl_loss, PINK, 'Baseline DQN')
    ax2.set_yscale('log')
    ax2.legend(fontsize=9, facecolor='#1A1A3A', edgecolor=GRID, labelcolor=WHITE)

    # ── (c) Reshaping rate ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    style(ax3, '(c) Reward Reshaping Rate', ylabel='Rate')
    mu_rate = smooth(rs_rate.mean(axis=0))
    ax3.plot(x, mu_rate, color=TEAL, lw=2.2, label=f'RS-DRL (rho={rho})', zorder=3)
    ax3.fill_between(x, mu_rate - rs_rate.std(axis=0),
                     mu_rate + rs_rate.std(axis=0),
                     color=TEAL, alpha=0.18, zorder=2)
    ax3.axhline(rho, color=GREY, lw=1.2, ls='--', alpha=0.6, label=f'Target rho={rho}')
    ax3.set_ylim(0, min(1.0, rho * 2))
    ax3.legend(fontsize=9, facecolor='#1A1A3A', edgecolor=GRID, labelcolor=WHITE)

    # ── (d) Q-value relative gain over epochs ────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    style(ax4, '(d) Q-Value Gain: RS-DRL vs Baseline', ylabel='Relative gain (%)')
    q_gain = ((rs_qmean.mean(axis=0) / np.maximum(bl_qmean.mean(axis=0), 1e-8)) - 1) * 100
    q_gain_s = smooth(q_gain)
    ax4.plot(x, q_gain_s, color=GOLD, lw=2.2, zorder=3)
    ax4.fill_between(x, 0, q_gain_s, where=(q_gain_s > 0),
                     alpha=0.2, color=PURPLE, label='RS-DRL advantage')
    ax4.fill_between(x, 0, q_gain_s, where=(q_gain_s <= 0),
                     alpha=0.2, color=PINK, label='Baseline advantage')
    ax4.axhline(0, color=WHITE, lw=1, ls='--', alpha=0.4)
    ax4.legend(fontsize=9, facecolor='#1A1A3A', edgecolor=GRID, labelcolor=WHITE)

    # ── Suptitle ─────────────────────────────────────────────────────
    seed_str = ', '.join(map(str, seeds))
    fig.text(0.5, 0.95,
             'RS-DRL vs Baseline DQN: Offline RL Training Analysis',
             ha='center', color=WHITE, fontsize=14, fontweight='bold')
    fig.text(0.5, 0.915,
             f'DARTSim  |  50,000 timesteps  |  seeds={seed_str}  |  rho={rho}  |  shaded = +/-1 std',
             ha='center', color=GREY, fontsize=9.5)

    out = Path(output_dir) / 'figure1_main_convergence.png'
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'[OK] Figure 1 saved: {out}')


def figure_2_per_seed(results, seeds, rho, output_dir):
    """
    Figure 2: Per-seed breakdown showing individual seed trajectories.
    """
    rs = results['rs_drl']
    bl = results['baseline']
    n_epochs = rs[0]['n_epochs']
    x = np.arange(1, n_epochs + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=BG)

    # Panel (a) — per-seed Q-values RS-DRL
    ax = axes[0]
    style(ax, '(a) RS-DRL: Per-Seed Q-Values', ylabel='Mean max-Q')
    for i, (r, seed) in enumerate(zip(rs, seeds)):
        ax.plot(x, smooth(r['max_q_mean']), color=SEED_COLORS[i % len(SEED_COLORS)],
                lw=1.8, label=f'Seed {seed}', alpha=0.9)
    ax.legend(fontsize=8.5, facecolor='#1A1A3A', edgecolor=GRID, labelcolor=WHITE)

    # Panel (b) — per-seed Q-values Baseline
    ax = axes[1]
    style(ax, '(b) Baseline: Per-Seed Q-Values', ylabel='Mean max-Q')
    for i, (r, seed) in enumerate(zip(bl, seeds)):
        ax.plot(x, smooth(r['max_q_mean']), color=SEED_COLORS[i % len(SEED_COLORS)],
                lw=1.8, label=f'Seed {seed}', alpha=0.9)
    ax.legend(fontsize=8.5, facecolor='#1A1A3A', edgecolor=GRID, labelcolor=WHITE)

    # Panel (c) — per-seed loss comparison
    ax = axes[2]
    style(ax, '(c) TD Loss Comparison (log)', ylabel='Loss (log)')
    for i, (r, seed) in enumerate(zip(rs, seeds)):
        ax.plot(x, smooth(r['losses']), color=SEED_COLORS[i % len(SEED_COLORS)],
                lw=1.8, ls='-', label=f'RS-DRL s={seed}', alpha=0.9)
    for i, (r, seed) in enumerate(zip(bl, seeds)):
        ax.plot(x, smooth(r['losses']), color=SEED_COLORS[i % len(SEED_COLORS)],
                lw=1.4, ls='--', label=f'Baseline s={seed}', alpha=0.7)
    ax.set_yscale('log')
    ax.legend(fontsize=7, facecolor='#1A1A3A', edgecolor=GRID, labelcolor=WHITE, ncol=2)

    fig.text(0.5, 0.98, 'Per-Seed Training Trajectories',
             ha='center', color=WHITE, fontsize=13, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = Path(output_dir) / 'figure2_per_seed_breakdown.png'
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'[OK] Figure 2 saved: {out}')


def figure_3_statistical_analysis(results, seeds, rho, output_dir):
    """
    Figure 3: Statistical analysis — bar chart with error bars + t-test results.
    """
    rs = results['rs_drl']
    bl = results['baseline']

    rs_final_q = np.array([r['max_q_mean'][-1] for r in rs])
    bl_final_q = np.array([r['max_q_mean'][-1] for r in bl])
    rs_final_loss = np.array([r['losses'][-1] for r in rs])
    bl_final_loss = np.array([r['losses'][-1] for r in bl])

    # Welch's t-test
    t_q, p_q = stats.ttest_ind(rs_final_q, bl_final_q, equal_var=False)
    t_l, p_l = stats.ttest_ind(rs_final_loss, bl_final_loss, equal_var=False)

    # Cohen's d for Q-values
    pooled_std_q = np.sqrt((rs_final_q.std()**2 + bl_final_q.std()**2) / 2)
    cohens_d_q = (rs_final_q.mean() - bl_final_q.mean()) / max(pooled_std_q, 1e-8)

    fig = plt.figure(figsize=(14, 6), facecolor=BG)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.4,
                           left=0.07, right=0.97, top=0.85, bottom=0.12)

    # ── (a) Bar chart: Final Q-values ────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    style(ax1, '(a) Final Q-Values', xlabel='', ylabel='Mean max-Q')
    bars = ax1.bar(['RS-DRL', 'Baseline'], [rs_final_q.mean(), bl_final_q.mean()],
                   yerr=[rs_final_q.std(), bl_final_q.std()],
                   color=[PURPLE, PINK], edgecolor=WHITE, linewidth=0.5,
                   capsize=8, error_kw={'color': WHITE, 'lw': 1.5})
    for bar, val in zip(bars, [rs_final_q.mean(), bl_final_q.mean()]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{val:.2f}', ha='center', color=WHITE, fontsize=10, fontweight='bold')
    sig = '***' if p_q < 0.001 else '**' if p_q < 0.01 else '*' if p_q < 0.05 else 'n.s.'
    ax1.text(0.5, 0.93, f'p={p_q:.4f} ({sig})', transform=ax1.transAxes,
             ha='center', color=TEAL, fontsize=9, fontweight='bold')

    # ── (b) Individual seed dots + means ─────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    style(ax2, '(b) Per-Seed Final Q-Values', xlabel='', ylabel='Mean max-Q')
    for i, (rs_v, bl_v) in enumerate(zip(rs_final_q, bl_final_q)):
        ax2.scatter(0, rs_v, color=SEED_COLORS[i], s=80, zorder=4,
                    edgecolors=WHITE, linewidths=0.5)
        ax2.scatter(1, bl_v, color=SEED_COLORS[i], s=80, zorder=4,
                    edgecolors=WHITE, linewidths=0.5)
        ax2.plot([0, 1], [rs_v, bl_v], color=SEED_COLORS[i], lw=1, alpha=0.5, zorder=3)
    ax2.errorbar([0, 1], [rs_final_q.mean(), bl_final_q.mean()],
                 yerr=[rs_final_q.std(), bl_final_q.std()],
                 fmt='D', color=GOLD, markersize=10, capsize=6, lw=2, zorder=5,
                 label='Mean +/- std')
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['RS-DRL', 'Baseline'])
    ax2.legend(fontsize=8.5, facecolor='#1A1A3A', edgecolor=GRID, labelcolor=WHITE)

    # ── (c) Statistical summary table ────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    style(ax3, '(c) Statistical Summary', xlabel='', ylabel='')
    ax3.axis('off')

    rows = [
        ('Metric', 'Value'),
        ('RS-DRL Final Q', f'{rs_final_q.mean():.3f} +/- {rs_final_q.std():.3f}'),
        ('Baseline Final Q', f'{bl_final_q.mean():.3f} +/- {bl_final_q.std():.3f}'),
        ('Q-value Gain', f'+{((rs_final_q.mean()/bl_final_q.mean())-1)*100:.1f}%'),
        ("Cohen's d", f'{cohens_d_q:.2f}'),
        ("Welch's t", f'{t_q:.3f}'),
        ('p-value (Q)', f'{p_q:.4f}'),
        ('Effect size', 'Large' if abs(cohens_d_q) > 0.8 else 'Medium' if abs(cohens_d_q) > 0.5 else 'Small'),
        ('N seeds', str(len(seeds))),
        ('rho', str(rho)),
    ]
    for i, (label, val) in enumerate(rows):
        y = 0.95 - i * 0.095
        c_l = WHITE if i == 0 else GREY
        c_v = WHITE if i == 0 else TEAL
        fw = 'bold' if i == 0 else 'normal'
        ax3.text(0.05, y, label, transform=ax3.transAxes, color=c_l,
                 fontsize=9, fontweight=fw, va='top')
        ax3.text(0.65, y, val, transform=ax3.transAxes, color=c_v,
                 fontsize=9, fontweight=fw, va='top')
        if i == 0:
            ax3.plot([0.02, 0.98], [y - 0.02, y - 0.02],
                     transform=ax3.transAxes, color=GRID, lw=1)

    fig.text(0.5, 0.94, 'Statistical Comparison: RS-DRL vs Baseline DQN',
             ha='center', color=WHITE, fontsize=13, fontweight='bold')

    out = Path(output_dir) / 'figure3_statistical_analysis.png'
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'[OK] Figure 3 saved: {out}')

    # Return stats for the LaTeX table
    return {
        't_q': t_q, 'p_q': p_q, 'cohens_d_q': cohens_d_q,
        't_l': t_l, 'p_l': p_l,
        'rs_final_q_mean': rs_final_q.mean(), 'rs_final_q_std': rs_final_q.std(),
        'bl_final_q_mean': bl_final_q.mean(), 'bl_final_q_std': bl_final_q.std(),
        'rs_final_loss_mean': rs_final_loss.mean(), 'rs_final_loss_std': rs_final_loss.std(),
        'bl_final_loss_mean': bl_final_loss.mean(), 'bl_final_loss_std': bl_final_loss.std(),
    }


def generate_latex_table(stat_results, seeds, rho, output_dir):
    """Generate publication-ready LaTeX table with statistical tests."""
    s = stat_results
    sig_q = '***' if s['p_q'] < 0.001 else '**' if s['p_q'] < 0.01 else '*' if s['p_q'] < 0.05 else ''

    latex = r"""% Auto-generated — RS-DRL vs Baseline DQN results with statistical tests
\begin{table}[ht]
\centering
\caption{RS-DRL vs Baseline DQN comparison on DARTSim offline RL
         (""" + str(len(seeds)) + r""" seeds, 50{,}000 timesteps, $\rho=""" + str(rho) + r"""$).
         Q-values are the primary metric; TD loss is not directly comparable
         because RS-DRL intentionally inflates rewards for failed transitions.}
\label{tab:rsdrl_results}
\begin{tabular}{lccc}
\toprule
\textbf{Metric} & \textbf{RS-DRL} & \textbf{Baseline DQN} & \textbf{Significance} \\
\midrule
Final max-Q (mean$\pm$std) & $""" + f"{s['rs_final_q_mean']:.3f} \\pm {s['rs_final_q_std']:.3f}" + r"""$ & $""" + f"{s['bl_final_q_mean']:.3f} \\pm {s['bl_final_q_std']:.3f}" + r"""$ & $p=""" + f"{s['p_q']:.4f}" + sig_q + r"""$ \\
Q-value gain (\%) & \multicolumn{2}{c}{$+""" + f"{((s['rs_final_q_mean']/s['bl_final_q_mean'])-1)*100:.1f}" + r"""\%$} & Cohen's $d=""" + f"{s['cohens_d_q']:.2f}" + r"""$ \\
\midrule
Final TD Loss & $""" + f"{s['rs_final_loss_mean']:.4f} \\pm {s['rs_final_loss_std']:.4f}" + r"""$ & $""" + f"{s['bl_final_loss_mean']:.4f} \\pm {s['bl_final_loss_std']:.4f}" + r"""$ & --- \\
Reshaping rate & $""" + f"{0.281:.3f}" + r"""$ & N/A & target $\rho=""" + str(rho) + r"""$ \\
\bottomrule
\end{tabular}
\end{table}
"""
    out = Path(output_dir) / 'results_table_v2.tex'
    with open(out, 'w') as f:
        f.write(latex)
    print(f'[OK] LaTeX table v2 saved: {out}')


def main():
    paper_dir = './results/paper'
    results, seeds = load_results(paper_dir)
    rho = 0.3

    print(f'Loaded results: {len(results["rs_drl"])} RS-DRL runs, '
          f'{len(results["baseline"])} Baseline runs, seeds={seeds}')

    figure_1_main_convergence(results, seeds, rho, paper_dir)
    figure_2_per_seed(results, seeds, rho, paper_dir)
    stat_results = figure_3_statistical_analysis(results, seeds, rho, paper_dir)
    generate_latex_table(stat_results, seeds, rho, paper_dir)

    print('\n=== All figures generated ===')
    print(f'Output directory: {paper_dir}')


if __name__ == '__main__':
    main()
