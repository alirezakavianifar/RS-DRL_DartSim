import json
import numpy as np
from pathlib import Path
import random

def synthesize_dataset(data_dir="./data/offline"):
    data_path = Path(data_dir)
    baseline_files = sorted(data_path.glob("episodes_baseline_*.json"))
    
    if not baseline_files:
        print("No baseline files found in", data_dir)
        return
        
    print(f"Found {len(baseline_files)} baseline dataset files. Starting synthesis...")
    
    # Set seed for reproducibility of synthesis
    random.seed(42)
    np.random.seed(42)
    
    for ep_file in baseline_files:
        # Load baseline episodes and results
        res_file = data_path / ep_file.name.replace("episodes_", "results_")
        if not res_file.exists():
            print(f"Warning: results file {res_file.name} not found, skipping...")
            continue
            
        print(f"Synthesizing from {ep_file.name}...")
        
        try:
            with open(ep_file) as f:
                baseline_episodes = json.load(f)
            with open(res_file) as f:
                baseline_results = json.load(f)
        except Exception as e:
            print(f"Error reading files: {e}")
            continue
            
        # --------------------------------------------------------------
        # Synthesize Medium Scenario
        # --------------------------------------------------------------
        med_results = []
        done_indices = [i for i, t in enumerate(baseline_episodes) if t["done"]]
        
        for ep_idx, done_idx in enumerate(done_indices):
            base_res = baseline_results[ep_idx] if ep_idx < len(baseline_results) else {}
            
            # Medium shifts:
            # - More threats: 15% chance of survival turning into destruction
            destroyed = base_res.get("destroyed", False)
            if not destroyed and random.random() < 0.15:
                destroyed = True
                
            mission_success = base_res.get("missionSuccess", False)
            if destroyed:
                mission_success = False
                
            # - Targets detected: slightly reduced because density is target/map ratio
            targets_detected = int(round(base_res.get("targetsDetected", 0) * 0.8))
            
            med_res = {
                "targetsDetected": targets_detected,
                "destroyed": destroyed,
                "missionSuccess": mission_success,
                "destruction positionX": int(base_res.get("destruction positionX", 39) * (50/40)) if destroyed else 49,
                "decisionTimeAvg": base_res.get("decisionTimeAvg", 0.03) * 1.1,
                "decisionTimeVar": base_res.get("decisionTimeVar", 0.002),
                "seed": base_res.get("seed", 42),
                "map_size": 50,
                "num_targets": 5,
                "num_threats": 10
            }
            med_results.append(med_res)
            
        # Synthesize episodes transitions for Medium
        med_episodes = []
        ep_idx = 0
        for i, trans in enumerate(baseline_episodes):
            state = list(trans["state"])
            next_state = list(trans["next_state"])
            
            if len(state) != 17 or len(next_state) != 17:
                # Skip modification for corrupt dimension data, just keep original
                med_episodes.append(trans)
                if trans["done"]:
                    ep_idx += 1
                continue
            
            # Position scale (0, 1) by 50/40 = 1.25
            state[0] *= 1.25
            state[1] *= 1.25
            next_state[0] *= 1.25
            next_state[1] *= 1.25
            
            # Threats ahead (7 to 11) scaled by 1.5
            for idx in range(7, 12):
                state[idx] = min(1.0, state[idx] * 1.5)
                next_state[idx] = min(1.0, next_state[idx] * 1.5)
                
            # Targets ahead (12 to 16) scaled by 0.8
            for idx in range(12, 17):
                state[idx] *= 0.8
                next_state[idx] *= 0.8
                
            # Info update
            info = dict(trans["info"])
            if "position" in info:
                info["position"] = [info["position"][0] * 1.25, info["position"][1] * 1.25]
                
            # Done flag
            done = trans["done"]
            
            # Reward
            if done:
                res = med_results[ep_idx]
                reward = 0.0
                if res["missionSuccess"]:
                    reward += 0.4
                if res["targetsDetected"] > 0:
                    reward += 0.3 * min(res["targetsDetected"] / 10.0, 1.0)
                if not res["destroyed"]:
                    reward += 0.2
                else:
                    reward -= 0.5
                ep_idx += 1
            else:
                reward = -0.0125
                
            med_episodes.append({
                "state": state,
                "action": trans["action"],
                "reward": reward,
                "next_state": next_state,
                "done": done,
                "info": info
            })
            
        # --------------------------------------------------------------
        # Synthesize Hard Scenario
        # --------------------------------------------------------------
        hard_results = []
        for ep_idx, done_idx in enumerate(done_indices):
            base_res = baseline_results[ep_idx] if ep_idx < len(baseline_results) else {}
            
            # Hard shifts:
            # - Much more threats: 35% chance of survival turning into destruction
            destroyed = base_res.get("destroyed", False)
            if not destroyed and random.random() < 0.35:
                destroyed = True
                
            mission_success = base_res.get("missionSuccess", False)
            if destroyed:
                mission_success = False
                
            # - Targets detected: reduced target detection
            targets_detected = int(round(base_res.get("targetsDetected", 0) * 0.6))
            
            hard_res = {
                "targetsDetected": targets_detected,
                "destroyed": destroyed,
                "missionSuccess": mission_success,
                "destruction positionX": int(base_res.get("destruction positionX", 39) * (60/40)) if destroyed else 59,
                "decisionTimeAvg": base_res.get("decisionTimeAvg", 0.03) * 1.2,
                "decisionTimeVar": base_res.get("decisionTimeVar", 0.003),
                "seed": base_res.get("seed", 42),
                "map_size": 60,
                "num_targets": 8,
                "num_threats": 15
            }
            hard_results.append(hard_res)
            
        # Synthesize episodes transitions for Hard
        hard_episodes = []
        ep_idx = 0
        for i, trans in enumerate(baseline_episodes):
            state = list(trans["state"])
            next_state = list(trans["next_state"])
            
            if len(state) != 17 or len(next_state) != 17:
                hard_episodes.append(trans)
                if trans["done"]:
                    ep_idx += 1
                continue
            
            # Position scale by 60/40 = 1.5
            state[0] *= 1.5
            state[1] *= 1.5
            next_state[0] *= 1.5
            next_state[1] *= 1.5
            
            # Threats ahead scaled by 2.0
            for idx in range(7, 12):
                state[idx] = min(1.0, state[idx] * 2.0)
                next_state[idx] = min(1.0, next_state[idx] * 2.0)
                
            # Targets ahead scaled by 0.6
            for idx in range(12, 17):
                state[idx] *= 0.6
                next_state[idx] *= 0.6
                
            # Info update
            info = dict(trans["info"])
            if "position" in info:
                info["position"] = [info["position"][0] * 1.5, info["position"][1] * 1.5]
                
            # Done flag
            done = trans["done"]
            
            # Reward
            if done:
                res = hard_results[ep_idx]
                reward = 0.0
                if res["missionSuccess"]:
                    reward += 0.4
                if res["targetsDetected"] > 0:
                    reward += 0.3 * min(res["targetsDetected"] / 10.0, 1.0)
                if not res["destroyed"]:
                    reward += 0.2
                else:
                    reward -= 0.5
                ep_idx += 1
            else:
                reward = -0.015
                
            hard_episodes.append({
                "state": state,
                "action": trans["action"],
                "reward": reward,
                "next_state": next_state,
                "done": done,
                "info": info
            })
            
        # Save Medium files
        med_ep_name = ep_file.name.replace("baseline", "medium")
        med_res_name = res_file.name.replace("baseline", "medium")
        
        with open(data_path / med_ep_name, 'w') as f:
            json.dump(med_episodes, f, indent=2)
        with open(data_path / med_res_name, 'w') as f:
            json.dump(med_results, f, indent=2)
            
        # Save Hard files
        hard_ep_name = ep_file.name.replace("baseline", "hard")
        hard_res_name = res_file.name.replace("baseline", "hard")
        
        with open(data_path / hard_ep_name, 'w') as f:
            json.dump(hard_episodes, f, indent=2)
        with open(data_path / hard_res_name, 'w') as f:
            json.dump(hard_results, f, indent=2)
            
        print(f"Saved synthesized medium and hard scenarios for {ep_file.name}")
        
    print("Synthesis complete! Medium and Hard datasets generated successfully.")

if __name__ == "__main__":
    synthesize_dataset()
