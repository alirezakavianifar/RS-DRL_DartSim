import json
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -- Colour palette (consistent with paper figures) ---------------------
BG     = '#FFFFFF'  # Figure background
PANEL  = '#FFFFFF'  # Axes background
PURPLE = '#1F77B4'  # RS-DRL (Standard Matplotlib Blue)
PINK   = '#D62728'  # Baseline DQN (Standard Matplotlib Red)
TEAL   = '#2CA02C'  # Reshaping rate (Standard Matplotlib Green)
GOLD   = '#FF7F0E'  # Q-Value gain (Standard Matplotlib Orange)
GRID   = '#E5E5E5'  # Grid lines (Light grey)
WHITE  = '#1A1A1A'  # Text/Labels (Dark grey for high contrast)
GREY   = '#4A4A4A'  # Subtitles/secondary text (Medium-dark grey)

def style_axes(ax, title, xlabel='', ylabel=''):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor('#CCCCCC')
    ax.tick_params(colors=WHITE, labelsize=9)
    ax.set_title(title, color=WHITE, fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel(xlabel, color=WHITE, fontsize=9.5)
    ax.set_ylabel(ylabel, color=WHITE, fontsize=9.5)
    ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)

def style_twin_axes(ax, ylabel=''):
    for sp in ax.spines.values():
        sp.set_edgecolor('#CCCCCC')
    ax.tick_params(colors=WHITE, labelsize=9)
    ax.set_ylabel(ylabel, color=WHITE, fontsize=9.5)

def generate_plots():
    results_dir = Path("results")
    
    # 1. Load Scenario Generalization data
    scenarios = ["Baseline", "Medium", "Hard"]
    scenario_paths = [
        results_dir / "zero_shot" / "baseline" / "evaluation_metrics.json",
        results_dir / "zero_shot" / "medium" / "evaluation_metrics.json",
        results_dir / "zero_shot" / "hard" / "evaluation_metrics.json"
    ]
    
    sc_rewards = []
    sc_success = []
    
    for path in scenario_paths:
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            sc_rewards.append(data.get("mean_reward", 0.0))
            sc_success.append(data.get("mission_success_rate", 0.0) * 100)
        else:
            sc_rewards.append(0.0)
            sc_success.append(0.0)
            
    # 2. Load Sensor Noise data
    noises = ["0.0", "0.1", "0.3", "0.5"]
    noise_paths = [
        results_dir / "zero_shot" / "baseline" / "evaluation_metrics.json",
        results_dir / "noise" / "0.1" / "evaluation_metrics.json",
        results_dir / "noise" / "0.3" / "evaluation_metrics.json",
        results_dir / "noise" / "0.5" / "evaluation_metrics.json"
    ]
    
    ns_rewards = []
    ns_success = []
    
    for path in noise_paths:
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            ns_rewards.append(data.get("mean_reward", 0.0))
            ns_success.append(data.get("mission_success_rate", 0.0) * 100)
        else:
            ns_rewards.append(0.0)
            ns_success.append(0.0)

    # 3. Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
    
    # ---- Left Panel: Scenario Generalization (Map & Threat Scaling) ----
    x = np.arange(len(scenarios))
    width = 0.3
    
    style_axes(ax1, "Scenario Generalization\n(Map & Threat Scaling)", "Scenario", "Mean Reward")
    rects1 = ax1.bar(x - width/2, sc_rewards, width, label='Mean Reward', color=PURPLE, zorder=3)
    ax1.set_ylim(0.0, 0.6)
    
    ax1_twin = ax1.twinx()
    style_twin_axes(ax1_twin, "Success Rate (%)")
    rects2 = ax1_twin.bar(x + width/2, sc_success, width, label='Success Rate (%)', color=TEAL, zorder=3)
    ax1_twin.set_ylim(0.0, 100.0)
    
    # X-Axis Ticks
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, color=WHITE)
    
    # Combined Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, facecolor=PANEL, edgecolor='#CCCCCC', labelcolor=WHITE, loc='upper right')
    
    # ---- Right Panel: Threat Sensor Noise Degradation ----
    style_axes(ax2, "Sensor Failure Robustness\n(Noise Injection)", "Sensor Noise Level (Probability)", "Mean Reward")
    line1 = ax2.plot(noises, ns_rewards, marker='o', lw=2.2, label='Mean Reward', color=PINK, zorder=3)
    ax2.set_ylim(0.0, 0.6)
    
    ax2_twin = ax2.twinx()
    style_twin_axes(ax2_twin, "Success Rate (%)")
    line2 = ax2_twin.plot(noises, ns_success, marker='s', lw=2.2, linestyle='--', label='Success Rate (%)', color=GOLD, zorder=3)
    ax2_twin.set_ylim(0.0, 100.0)
    
    # Combined Legend
    lns = line1 + line2
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, facecolor=PANEL, edgecolor='#CCCCCC', labelcolor=WHITE, loc='upper right')
    
    plt.tight_layout()
    out_dir = results_dir / "paper"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "figure4_robustness_analysis.png"
    plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close()
    
    print(f"Robustness analysis figure successfully generated and saved to: {out_path}")

if __name__ == "__main__":
    generate_plots()
