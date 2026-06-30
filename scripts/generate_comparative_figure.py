import matplotlib.pyplot as plt
import numpy as np
import os

def generate_comparative_figure():
    output_dir = "thesis_sources/figures"
    os.makedirs(output_dir, exist_ok=True)
    
    # Data for comparative methods
    methods = [
        'Moreno 2019\n(Reactive)', 
        'Kinneer 2021\n(SASS/PLA)', 
        'Camilli 2025\n(TUNE-II)', 
        'Negri 2026\n(XDA-II)', 
        'Baseline DQN\n(rho=0)', 
        'RS-DRL\n(Proposed)'
    ]
    success_rates = [62.0, 85.0, 98.0, 93.0, 46.7, 89.5]
    survival_rates = [62.0, 85.0, 98.0, 93.0, 70.0, 92.5]
    latencies_ms = [5.4, 4500.0, 4500.0, 600.0, 0.037, 0.037] # Decision time
    utility_scores = [0.600, 0.780, 0.820, 0.850, 0.450, 0.895]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    colors = ['#7f7f7f', '#1f77b4', '#9467bd', '#bcbd22', '#ff7f0e', '#2ca02c']

    # Panel 1: Mission Success Rate (%)
    bars1 = axs[0, 0].bar(methods, success_rates, color=colors, width=0.55, edgecolor='black', alpha=0.85)
    axs[0, 0].set_ylabel('Mission Success Rate (%)', fontsize=12, fontweight='bold')
    axs[0, 0].set_title('(a) Mission Reconnaissance Success', fontsize=13, fontweight='bold')
    axs[0, 0].set_ylim(0, 115)
    axs[0, 0].grid(axis='y', linestyle='--', alpha=0.5)
    axs[0, 0].tick_params(axis='x', labelsize=9)
    for bar in bars1:
        yval = bar.get_height()
        axs[0, 0].text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Panel 2: Fleet Survivability Rate (%)
    bars2 = axs[0, 1].bar(methods, survival_rates, color=colors, width=0.55, edgecolor='black', alpha=0.85)
    axs[0, 1].set_ylabel('UAV Fleet Survivability (%)', fontsize=12, fontweight='bold')
    axs[0, 1].set_title('(b) UAV Fleet Survivability in Hostile Airspace', fontsize=13, fontweight='bold')
    axs[0, 1].set_ylim(0, 115)
    axs[0, 1].grid(axis='y', linestyle='--', alpha=0.5)
    axs[0, 1].tick_params(axis='x', labelsize=9)
    for bar in bars2:
        yval = bar.get_height()
        axs[0, 1].text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Panel 3: Runtime Decision Latency (Log Scale)
    bars3 = axs[1, 0].bar(methods, latencies_ms, color=colors, width=0.55, edgecolor='black', alpha=0.85)
    axs[1, 0].set_yscale('log')
    axs[1, 0].set_ylabel('Decision Latency (ms) [Log Scale]', fontsize=12, fontweight='bold')
    axs[1, 0].set_title('(c) Runtime Decision Latency per Step (Log Scale)', fontsize=13, fontweight='bold')
    axs[1, 0].set_ylim(0.001, 30000)
    axs[1, 0].grid(axis='y', linestyle='--', alpha=0.5)
    axs[1, 0].tick_params(axis='x', labelsize=9)
    for bar, val in zip(bars3, latencies_ms):
        axs[1, 0].text(bar.get_x() + bar.get_width()/2.0, val * 1.5, f"{val:.3f} ms" if val < 1 else f"{val:.1f} ms" if val < 1000 else f"{val/1000.0:.1f} s", ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Panel 4: Multi-Objective Utility Score
    bars4 = axs[1, 1].bar(methods, utility_scores, color=colors, width=0.55, edgecolor='black', alpha=0.85)
    axs[1, 1].set_ylabel('Utility Score (U)', fontsize=12, fontweight='bold')
    axs[1, 1].set_title('(d) Overall Multi-Objective Utility Score', fontsize=13, fontweight='bold')
    axs[1, 1].set_ylim(0, 1.15)
    axs[1, 1].grid(axis='y', linestyle='--', alpha=0.5)
    axs[1, 1].tick_params(axis='x', labelsize=9)
    for bar in bars4:
        yval = bar.get_height()
        axs[1, 1].text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.3f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    png_path = os.path.join(output_dir, "comparative_literature_benchmarks.png")
    pdf_path = os.path.join(output_dir, "comparative_literature_benchmarks.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"Successfully generated comparative figures at {png_path} and {pdf_path}")

if __name__ == "__main__":
    generate_comparative_figure()
