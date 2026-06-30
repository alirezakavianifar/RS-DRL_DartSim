"""
Visualize environmental uncertainty in DARTSim offline data.

Creates multiple visualizations showing:
1. Sensor uncertainty heatmaps
2. Distribution of stochastic transitions
3. Examples of state-action pairs with multiple outcomes
4. Sensor variability patterns
5. Uncertainty by action type
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
import re

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: Matplotlib not available. Please install: pip install matplotlib")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    # Seaborn is optional, we can work without it

# Try to import plotly for interactive visualizations
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Interactive visualizations will be skipped.")

def extract_number(filename):
    """Extract numeric value from filename for proper sorting"""
    match = re.search(r'baseline_(\d+)', filename.name)
    if match:
        return int(match.group(1))
    return 0

def state_to_tuple(state):
    """Convert state list to tuple for hashing"""
    return tuple(state)

def analyze_state_differences(state1, state2):
    """Analyze which components differ between two states"""
    differences = {
        'position': False,
        'altitude': False,
        'formation': False,
        'ecm': False,
        'direction': False,
        'threat_sensor': False,
        'target_sensor': False,
    }
    
    min_len = min(len(state1), len(state2))
    
    if min_len > 1 and (state1[0] != state2[0] or state1[1] != state2[1]):
        differences['position'] = True
    if min_len > 2 and state1[2] != state2[2]:
        differences['altitude'] = True
    if min_len > 3 and state1[3] != state2[3]:
        differences['formation'] = True
    if min_len > 4 and state1[4] != state2[4]:
        differences['ecm'] = True
    if min_len > 6 and (state1[5] != state2[5] or state1[6] != state2[6]):
        differences['direction'] = True
    if min_len > 11:
        if any(state1[i] != state2[i] for i in range(7, 12)):
            differences['threat_sensor'] = True
    if min_len > 16:
        if any(state1[i] != state2[i] for i in range(12, 17)):
            differences['target_sensor'] = True
    
    return differences

def load_episode_data(offline_dir, num_files=20):
    """Load episode data from files"""
    episode_files = sorted(
        [f for f in offline_dir.glob("episodes_*.json")],
        key=extract_number
    )[:num_files]
    
    state_action_to_next_states = defaultdict(set)
    state_action_transitions = defaultdict(list)
    all_transitions = []
    
    for episode_file in episode_files:
        with open(episode_file, 'r') as f:
            transitions = json.load(f)
        
        for trans in transitions:
            state_tup = state_to_tuple(trans["state"])
            action = trans["action"]
            next_state_tup = state_to_tuple(trans["next_state"])
            
            key = (state_tup, action)
            state_action_to_next_states[key].add(next_state_tup)
            state_action_transitions[key].append({
                "next_state": next_state_tup,
                "reward": trans["reward"],
                "done": trans["done"]
            })
            all_transitions.append(trans)
    
    return state_action_to_next_states, state_action_transitions, all_transitions

def plot_uncertainty_overview(state_action_to_next_states, state_action_transitions, output_dir):
    """Create overview plot of uncertainty types"""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available, skipping uncertainty overview plot")
        return
    
    fig = plt.figure(figsize=(18, 13))
    gs = GridSpec(3, 3, figure=fig, hspace=0.62, wspace=0.38)
    
    # 1. Distribution of stochasticity
    ax1 = fig.add_subplot(gs[0, 0])
    outcome_counts = [len(next_states) for next_states in state_action_to_next_states.values()]
    unique_counts = np.unique(outcome_counts, return_counts=True)
    ax1.bar(unique_counts[0][:10], unique_counts[1][:10], color='steelblue', alpha=0.7)
    ax1.set_xlabel('Number of Different Outcomes', fontsize=10)
    ax1.set_ylabel('Number of State-Action Pairs', fontsize=10)
    ax1.set_title('Distribution of Stochasticity\n(Outcomes per State-Action Pair)', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Uncertainty by action type
    ax2 = fig.add_subplot(gs[0, 1])
    action_stats = defaultdict(lambda: {'total': 0, 'stochastic': 0})
    for key, next_states in state_action_to_next_states.items():
        state, action = key
        action_stats[action]['total'] += 1
        if len(next_states) > 1:
            action_stats[action]['stochastic'] += 1
    
    actions = sorted(action_stats.keys())
    stochastic_ratios = [action_stats[a]['stochastic'] / action_stats[a]['total'] * 100 
                         for a in actions]
    
    bars = ax2.bar(actions, stochastic_ratios, color='coral', alpha=0.7)
    ax2.set_ylabel('Stochasticity (%)', fontsize=10)
    ax2.set_title('Uncertainty by Action Type', fontsize=11, fontweight='bold')
    ax2.set_xticks(range(len(actions)))
    ax2.set_xticklabels(actions, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, ratio in zip(bars, stochastic_ratios):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{ratio:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 3. Component uncertainty pie chart
    ax3 = fig.add_subplot(gs[0, 2])
    component_affected = defaultdict(set)
    
    for key, next_states in state_action_to_next_states.items():
        if len(next_states) > 1:
            next_states_list = list(next_states)
            for i in range(len(next_states_list)):
                for j in range(i+1, len(next_states_list)):
                    diff = analyze_state_differences(next_states_list[i], next_states_list[j])
                    for component, changed in diff.items():
                        if changed:
                            component_affected[component].add(key)
    
    component_counts = {k: len(v) for k, v in component_affected.items()}
    
    if component_counts:
        # Sort by size (largest first) so the legend reads cleanly
        sorted_items = sorted(component_counts.items(), key=lambda x: x[1], reverse=True)
        labels = [k for k, _ in sorted_items]
        sizes = [v for _, v in sorted_items]
        total = sum(sizes)
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

        # Only print a percentage on the wedge for slices large enough to be legible;
        # tiny slices would otherwise overlap into an unreadable blob.
        def _autopct(pct):
            return f'{pct:.1f}%' if pct >= 4 else ''

        wedges, _texts, _autotexts = ax3.pie(
            sizes, labels=None, autopct=_autopct, colors=colors,
            startangle=90, pctdistance=0.75,
            wedgeprops=dict(edgecolor='white', linewidth=0.8),
        )
        # Move labels to a legend on the side to avoid label collisions.
        legend_labels = [f'{lab} ({sz/total*100:.1f}%)' for lab, sz in zip(labels, sizes)]
        ax3.legend(wedges, legend_labels, title='State Component',
                   loc='center left', bbox_to_anchor=(0.98, 0.5),
                   fontsize=8, title_fontsize=9, frameon=False)
        ax3.set_title('Uncertainty by State Component', fontsize=11, fontweight='bold')
    
    # 4. Sensor uncertainty heatmap (example)
    ax4 = fig.add_subplot(gs[1, :])
    
    # Find most stochastic state-action pairs
    stochastic_pairs = [(k, v) for k, v in state_action_to_next_states.items() if len(v) > 1]
    stochastic_pairs.sort(key=lambda x: len(x[1]), reverse=True)
    
    # Show top 20 most stochastic transitions
    top_n = min(20, len(stochastic_pairs))
    if top_n > 0:
        # Extract sensor readings for visualization
        sensor_data = []
        labels = []
        
        for i, (key, next_states) in enumerate(stochastic_pairs[:top_n]):
            state, action = key
            labels.append(f"({state[0]},{state[1]}) {action}")
            
            # Get sensor readings from next states
            next_states_list = list(next_states)
            sensor_row = []
            
            # Show variability in threat sensor (indices 7-11)
            for next_state in next_states_list[:5]:  # Limit to 5 outcomes
                if len(next_state) > 11:
                    threat_sensor = next_state[7:12]
                    sensor_row.extend(threat_sensor)
                else:
                    sensor_row.extend([0] * 5)
            
            # Pad if needed
            while len(sensor_row) < 25:  # 5 outcomes * 5 sensors
                sensor_row.append(0)
            sensor_row = sensor_row[:25]
            
            sensor_data.append(sensor_row)
        
        if sensor_data:
            sensor_array = np.array(sensor_data)
            im = ax4.imshow(sensor_array, aspect='auto', cmap='YlOrRd', interpolation='nearest')
            ax4.set_yticks(range(len(labels)))
            ax4.set_yticklabels(labels, fontsize=7)
            ax4.tick_params(axis='y', pad=2)
            ax4.set_xlabel('Sensor Reading (5 outcomes × 5 threat sensors)', fontsize=10)
            ax4.set_title(f'Threat Sensor Variability\n(Top {top_n} Most Stochastic Transitions)', 
                         fontsize=11, fontweight='bold')
            plt.colorbar(im, ax=ax4, label='Sensor Value (0=No Threat, 1=Threat)')
    
    # 5. Reward variability
    ax5 = fig.add_subplot(gs[2, 0])
    reward_variability = defaultdict(list)
    for key, transitions_list in state_action_transitions.items():
        if len(transitions_list) > 1:
            rewards = [t["reward"] for t in transitions_list]
            unique_rewards = len(set(rewards))
            if unique_rewards > 1:
                reward_variability[unique_rewards].append(np.std(rewards))
    
    if reward_variability:
        reward_keys = sorted(reward_variability.keys())
        reward_data = [reward_variability[k] for k in reward_keys]
        bp = ax5.boxplot(reward_data, tick_labels=reward_keys, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightgreen')
            patch.set_alpha(0.7)
        ax5.set_xlabel('Number of Different Reward Values', fontsize=10)
        ax5.set_ylabel('Reward Standard Deviation', fontsize=10)
        ax5.set_title('Reward Variability Distribution', fontsize=11, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Outcome count distribution (detailed)
    ax6 = fig.add_subplot(gs[2, 1])
    outcome_counts_detailed = [len(next_states) for next_states in state_action_to_next_states.values() 
                               if len(next_states) > 1]
    if outcome_counts_detailed:
        ax6.hist(outcome_counts_detailed, bins=min(20, max(outcome_counts_detailed)), 
                color='purple', alpha=0.7, edgecolor='black')
        ax6.set_xlabel('Number of Outcomes', fontsize=10)
        ax6.set_ylabel('Frequency', fontsize=10)
        ax6.set_title('Stochastic Transitions\n(Outcomes Distribution)', fontsize=11, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. Summary statistics
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    
    total_pairs = len(state_action_to_next_states)
    stochastic_pairs = sum(1 for v in state_action_to_next_states.values() if len(v) > 1)
    deterministic_pairs = total_pairs - stochastic_pairs
    
    stats_text = f"""
    ENVIRONMENTAL UNCERTAINTY SUMMARY
    
    Total State-Action Pairs: {total_pairs:,}
    Deterministic: {deterministic_pairs:,} ({deterministic_pairs/total_pairs*100:.1f}%)
    Stochastic: {stochastic_pairs:,} ({stochastic_pairs/total_pairs*100:.1f}%)
    
    Max Outcomes: {max(len(v) for v in state_action_to_next_states.values()) if state_action_to_next_states else 0}
    Avg Outcomes (stochastic): {np.mean([len(v) for v in state_action_to_next_states.values() if len(v) > 1]):.2f}
    
    Main Uncertainty Sources:
    - Threat Sensor: {len(component_affected.get('threat_sensor', set()))} transitions
    - Target Sensor: {len(component_affected.get('target_sensor', set()))} transitions
    """
    
    ax7.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Environmental Uncertainty Analysis', fontsize=16, fontweight='bold', y=0.98)
    
    output_path = output_dir / 'uncertainty_overview.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved uncertainty overview to {output_path}")
    plt.close()

def plot_sensor_uncertainty_examples(state_action_to_next_states, output_dir):
    """Create detailed visualization of sensor uncertainty"""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available, skipping sensor uncertainty examples")
        return
    
    # Find examples of stochastic transitions with sensor variability
    examples = []
    
    for key, next_states in state_action_to_next_states.items():
        if len(next_states) > 1:
            state, action = key
            next_states_list = list(next_states)
            
            # Check if sensors vary
            sensor_variability = False
            for i in range(len(next_states_list)):
                for j in range(i+1, len(next_states_list)):
                    diff = analyze_state_differences(next_states_list[i], next_states_list[j])
                    if diff['threat_sensor'] or diff['target_sensor']:
                        sensor_variability = True
                        break
                if sensor_variability:
                    break
            
            if sensor_variability:
                examples.append((key, next_states_list))
                if len(examples) >= 5:
                    break
    
    if not examples:
        print("No sensor uncertainty examples found")
        return
    
    fig, axes = plt.subplots(len(examples), 2, figsize=(16, 4*len(examples)))
    if len(examples) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (key, next_states_list) in enumerate(examples):
        state, action = key
        
        # Plot threat sensors
        ax1 = axes[idx, 0]
        threat_data = []
        for next_state in next_states_list[:10]:  # Limit to 10 outcomes
            if len(next_state) > 11:
                threat_sensor = next_state[7:12]
            else:
                threat_sensor = [0] * 5
            threat_data.append(threat_sensor)
        
        threat_array = np.array(threat_data)
        im1 = ax1.imshow(threat_array, aspect='auto', cmap='Reds', vmin=0, vmax=1)
        ax1.set_ylabel(f'Outcome #{idx+1}', fontsize=10)
        ax1.set_xlabel('Threat Sensor Cell (1-5 ahead)', fontsize=10)
        ax1.set_title(f'Threat Sensor Variability\nState: pos=({state[0]},{state[1]}), alt={state[2]}, Action: {action}', 
                     fontsize=11, fontweight='bold')
        ax1.set_yticks(range(len(threat_data)))
        ax1.set_yticklabels([f'Outcome {i+1}' for i in range(len(threat_data))])
        plt.colorbar(im1, ax=ax1, label='Threat Detected')
        
        # Plot target sensors
        ax2 = axes[idx, 1]
        target_data = []
        for next_state in next_states_list[:10]:
            if len(next_state) > 16:
                target_sensor = next_state[12:17]
            else:
                target_sensor = [0] * 5
            target_data.append(target_sensor)
        
        target_array = np.array(target_data)
        im2 = ax2.imshow(target_array, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
        ax2.set_ylabel(f'Outcome #{idx+1}', fontsize=10)
        ax2.set_xlabel('Target Sensor Cell (1-5 ahead)', fontsize=10)
        ax2.set_title(f'Target Sensor Variability\nState: pos=({state[0]},{state[1]}), alt={state[2]}, Action: {action}', 
                     fontsize=11, fontweight='bold')
        ax2.set_yticks(range(len(target_data)))
        ax2.set_yticklabels([f'Outcome {i+1}' for i in range(len(target_data))])
        plt.colorbar(im2, ax=ax2, label='Target Detected')
    
    plt.suptitle('Sensor Uncertainty Examples', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = output_dir / 'sensor_uncertainty_examples.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved sensor uncertainty examples to {output_path}")
    plt.close()

def create_interactive_uncertainty_plot(state_action_to_next_states, output_dir):
    """Create interactive Plotly visualization of uncertainty"""
    if not PLOTLY_AVAILABLE:
        print("Plotly not available, skipping interactive visualization")
        return
    
    # Find stochastic transitions
    stochastic_data = []
    for key, next_states in state_action_to_next_states.items():
        if len(next_states) > 1:
            state, action = key
            stochastic_data.append({
                'position_x': state[0],
                'position_y': state[1],
                'altitude': state[2],
                'action': action,
                'num_outcomes': len(next_states),
                'state_key': str(state[:5])
            })
    
    if not stochastic_data:
        print("No stochastic data for interactive plot")
        return
    
    df_dict = {
        'x': [d['position_x'] for d in stochastic_data],
        'y': [d['position_y'] for d in stochastic_data],
        'altitude': [d['altitude'] for d in stochastic_data],
        'action': [d['action'] for d in stochastic_data],
        'outcomes': [d['num_outcomes'] for d in stochastic_data],
        'state': [d['state_key'] for d in stochastic_data]
    }
    
    fig = go.Figure()
    
    # Color by number of outcomes
    fig.add_trace(go.Scatter3d(
        x=df_dict['x'],
        y=df_dict['y'],
        z=df_dict['altitude'],
        mode='markers',
        marker=dict(
            size=5,
            color=df_dict['outcomes'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Number of<br>Outcomes"),
            line=dict(width=0.5, color='black')
        ),
        text=[f"Action: {a}<br>Outcomes: {o}<br>State: {s}" 
              for a, o, s in zip(df_dict['action'], df_dict['outcomes'], df_dict['state'])],
        hovertemplate='<b>Position:</b> (%{x}, %{y})<br>' +
                     '<b>Altitude:</b> %{z}<br>' +
                     '<extra>%{text}</extra>',
        name='Stochastic Transitions'
    ))
    
    fig.update_layout(
        title='Environmental Uncertainty: Stochastic Transitions<br><sub>Size and color indicate number of possible outcomes</sub>',
        scene=dict(
            xaxis_title='Position X',
            yaxis_title='Position Y',
            zaxis_title='Altitude',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        width=1200,
        height=800
    )
    
    output_path = output_dir / 'uncertainty_interactive.html'
    fig.write_html(str(output_path))
    print(f"Saved interactive uncertainty plot to {output_path}")

def main():
    offline_dir = Path("data/offline")
    output_dir = Path("results/uncertainty_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading episode data...")
    state_action_to_next_states, state_action_transitions, all_transitions = load_episode_data(offline_dir, num_files=20)
    
    print("Creating uncertainty overview...")
    plot_uncertainty_overview(state_action_to_next_states, state_action_transitions, output_dir)
    
    print("Creating sensor uncertainty examples...")
    plot_sensor_uncertainty_examples(state_action_to_next_states, output_dir)
    
    print("Creating interactive visualization...")
    create_interactive_uncertainty_plot(state_action_to_next_states, output_dir)
    
    print(f"\nAll visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()

