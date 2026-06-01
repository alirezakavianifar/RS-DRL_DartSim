"""Quick diversity analysis of the offline dataset."""
import json, numpy as np
from pathlib import Path

data_dir = Path("data/offline")

result_files = sorted(data_dir.glob("results_baseline_*.json"))
ep_files = sorted(data_dir.glob("episodes_baseline_*.json"))

all_success, all_targets, all_destroyed = [], [], []
map_sizes, num_targets_dist, num_threats_dist = [], [], []

for f in result_files:
    for ep in json.load(open(f)):
        all_success.append(ep.get("missionSuccess", False))
        all_targets.append(ep.get("targetsDetected", 0))
        all_destroyed.append(ep.get("destroyed", False))
        map_sizes.append(ep.get("map_size", None))
        num_targets_dist.append(ep.get("num_targets", None))
        num_threats_dist.append(ep.get("num_threats", None))

action_counts = {}
ep_rewards, ep_lengths = [], []
reward_by_action = {}

for f in ep_files:
    transitions = json.load(open(f))
    ep_reward = sum(t["reward"] for t in transitions)
    ep_rewards.append(ep_reward)
    ep_lengths.append(len(transitions))
    for t in transitions:
        a = t["action"]
        action_counts[a] = action_counts.get(a, 0) + 1
        reward_by_action.setdefault(a, []).append(t["reward"])

total_actions = sum(action_counts.values())

print("=== Mission Outcomes ({} episodes) ===".format(len(all_success)))
print(f"  Success rate      : {100*sum(all_success)/len(all_success):.1f}%  ({sum(all_success)}/{len(all_success)})")
print(f"  Destroyed rate    : {100*sum(all_destroyed)/len(all_destroyed):.1f}%")
print(f"  Targets detected  : mean={np.mean(all_targets):.2f}  std={np.std(all_targets):.2f}  max={max(all_targets)}")
print(f"  Unique target cnt : {sorted(set(all_targets))}")

print("\n=== Scenario Parameters ===")
print(f"  Map sizes   : {sorted(set(m for m in map_sizes if m))}")
print(f"  Num targets : {sorted(set(n for n in num_targets_dist if n))}")
print(f"  Num threats : {sorted(set(n for n in num_threats_dist if n))}")

print(f"\n=== Action Distribution ({total_actions} transitions) ===")
for a, c in sorted(action_counts.items(), key=lambda x: -x[1]):
    mean_r = np.mean(reward_by_action[a])
    print(f"  {a:25s}: {c:6d} ({100*c/total_actions:.1f}%)  mean_reward={mean_r:.3f}")

print("\n=== Episode Length Distribution ===")
print(f"  Mean={np.mean(ep_lengths):.1f}  Std={np.std(ep_lengths):.1f}  Min={min(ep_lengths)}  Max={max(ep_lengths)}")
pct = np.percentile(ep_lengths, [10, 25, 50, 75, 90])
print(f"  Percentiles [10,25,50,75,90]: {pct}")

print("\n=== Episode Reward Distribution ===")
print(f"  Mean={np.mean(ep_rewards):.3f}  Std={np.std(ep_rewards):.3f}  Min={min(ep_rewards):.3f}  Max={max(ep_rewards):.3f}")
print(f"  Positive-reward episodes : {sum(1 for r in ep_rewards if r > 0)} / {len(ep_rewards)} ({100*sum(1 for r in ep_rewards if r>0)/len(ep_rewards):.1f}%)")
print(f"  Zero-reward episodes     : {sum(1 for r in ep_rewards if r == 0)} / {len(ep_rewards)}")
print(f"  Negative-reward episodes : {sum(1 for r in ep_rewards if r < 0)} / {len(ep_rewards)}")

print("\n=== Unique States (state space coverage) ===")
all_states = set()
for f in ep_files:
    for t in json.load(open(f)):
        if len(t["state"]) == 17:
            all_states.add(tuple(t["state"]))
print(f"  Unique states: {len(all_states)} across {total_actions} transitions")
print(f"  State coverage ratio: {len(all_states)/total_actions:.3f}")
