# Phase 0: Research Design for DARTSim Case Study

This document implements Phase 0 of the research plan, defining the scope, research questions, state/reward mappings, and scenario selection for applying RS-DRL to DARTSim.

---

## 1. Research Questions

Based on the RS-DRL paper's evaluation on DeltaIoT and the DARTSim domain characteristics, we formulate the following research questions:

### RQ1: Time-to-Threshold Performance
**Does RS-DRL accelerate time-to-threshold in DARTSim missions compared to baseline DQN?**

- **Metric**: Number of training timesteps/episodes required to reach a threshold performance level (e.g., 0.9 normalized reward or 90% mission success rate)
- **Hypothesis**: RS-DRL's randomized reward reshaping should help the agent learn faster from failures, leading to faster convergence
- **Evaluation**: Compare learning curves (mean reward vs timesteps) between RS-DRL and ε-greedy DQN, and compute time-to-threshold for fixed reward thresholds (0.7, 0.8, 0.9)
- **Link to DeltaIoT**: Parallel to RS-DRL paper's time-to-threshold evaluation on DeltaIoT (Table 3, Figure 4)

### RQ2: Asymptotic Performance
**Does RS-DRL achieve better asymptotic performance on DARTSim mission metrics?**

- **Metrics**: 
  - Mission success rate (boolean)
  - Targets detected (count, normalized by total targets)
  - Team survival (inverse of destroyed flag)
  - Decision time efficiency (normalized by mission duration)
- **Hypothesis**: RS-DRL's exploration boost from optimistic reward reshaping should help find better policies, leading to higher final performance
- **Evaluation**: Compare final performance (mean ± std over last 10% of training) across multiple seeds
- **Link to DeltaIoT**: Parallel to RS-DRL paper's asymptotic performance metrics (≈0.9905 mean reward on DeltaIoTv1)

### RQ3: Robustness and Generalization
**Does RS-DRL generalize better to shifted DARTSim environments (unexpected threats, different target distributions) compared to baseline DQN?**

- **Metric**: Performance drop when evaluating trained agents on unseen scenarios vs training scenarios
- **Hypothesis**: RS-DRL's diverse exploration should learn more robust policies that adapt better to environment shifts
- **Evaluation**: Train on baseline scenario, evaluate on medium/hard shifted scenarios. Compute generalization gap (training performance - test performance)
- **Link to DeltaIoT**: Parallel to RS-DRL paper's robustness analysis (though they focused on different interference patterns)

### RQ4: Mission-Level Metric Impact
**How does RS-DRL affect individual mission-level metrics (targets detected, team destroyed, decision time) compared to baseline?**

- **Metrics**: Breakdown of domain-specific metrics from DARTSim CSV output
- **Hypothesis**: RS-DRL should improve overall mission outcomes (more targets detected, fewer team destructions, lower decision times)
- **Evaluation**: Statistical comparison (t-tests, effect sizes) of per-metric performance between RS-DRL and DQN
- **Link to DeltaIoT**: Parallel to RS-DRL paper's domain metric analysis (packet loss, latency, energy)

---

## 2. State Space Mapping

### 2.1 Raw DARTSim State Components

From DARTSim's TCP interface (`getState` command), we have:

```python
TeamState = {
    "position": {"X": int, "Y": int},
    "direction": {"X": int, "Y": int},
    "config": {
        "altitudeLevel": int,           # 0 to (altitudeLevels-1)
        "formation": int,                # 0=LOOSE, 1=TIGHT
        "ecm": bool,                     # ECM on/off
        "ttcIncAlt": int,                # time-to-complete for altitude increase
        "ttcDecAlt": int,                # time-to-complete for altitude decrease
        "ttcIncAlt2": int,
        "ttcDecAlt2": int
    }
}
```

### 2.2 Sensor Observations

From DARTSim's sensor readings:
- `readForwardThreatSensor(cells)` → boolean array (threat detected in each forward cell)
- `readForwardTargetSensor(cells)` → boolean array (target detected in each forward cell)

### 2.3 RL State Vector Design

**Proposed State Vector (normalized to [0,1] or [-1,1]):**

```python
state_vector = [
    # Position (normalized by map size)
    position_x / map_size,              # [0, 1]
    position_y / map_size,              # [0, 1]
    
    # Direction (normalized)
    direction_x / max_direction,        # [-1, 1]
    direction_y / max_direction,        # [-1, 1]
    
    # Configuration
    altitude_level / max_altitude,      # [0, 1]
    formation,                          # 0.0 (LOOSE) or 1.0 (TIGHT)
    ecm,                                # 0.0 or 1.0
    
    # Time-to-complete (normalized by max latency)
    ttc_inc_alt / max_latency,         # [0, 1]
    ttc_dec_alt / max_latency,         # [0, 1]
    
    # Forward sensors (fixed lookahead, e.g., 5 cells)
    *threat_sensor_reading,             # 5 booleans → [0,1] values
    *target_sensor_reading,             # 5 booleans → [0,1] values
    
    # Recent history (optional, for temporal context)
    # ... last N decision times, last N positions, etc.
]
```

**State Dimension**: ~15-20 features (adjustable based on sensor lookahead and history window)

**Normalization Strategy**:
- Position, altitude, TTC: normalize by max values
- Direction: normalize to [-1, 1]
- Binary flags: keep as 0/1
- Sensors: keep as 0/1 boolean arrays

---

## 3. Action Space Mapping

### 3.1 Available Tactics (from DARTSim)

From DARTSim's TCP interface and Java client examples:

```python
ACTIONS = [
    "IncAlt",      # Increase altitude (1 level)
    "DecAlt",      # Decrease altitude (1 level)
    "IncAlt2",     # Increase altitude (2 levels)
    "DecAlt2",     # Decrease altitude (2 levels)
    "GoTight",     # Switch to tight formation
    "GoLoose",     # Switch to loose formation
    "EcmOn",       # Turn ECM on
    "EcmOff"       # Turn ECM off
]
```

### 3.2 Action Space Design

**Option A: Single Action per Step (Recommended)**
- Discrete action space: 8 actions (one tactic per step)
- Action ID mapping: `{0: "IncAlt", 1: "DecAlt", 2: "IncAlt2", 3: "DecAlt2", 4: "GoTight", 5: "GoLoose", 6: "EcmOn", 7: "EcmOff"}`
- **Action space size**: 8 (smaller than DeltaIoT's 216/1296/4096, but sufficient for CPS domain)

**Option B: Multi-Action per Step (Advanced)**
- Allow multiple tactics per step (e.g., `["DecAlt", "GoTight"]`)
- This increases action space combinatorially (2^8 = 256 combinations)
- Not recommended initially; can be explored later

### 3.3 Action Constraints

- Some actions may be invalid (e.g., `DecAlt` when already at lowest altitude)
- Handle via reward penalty or state masking (DQN can learn to avoid invalid actions)
- Alternatively, filter invalid actions in the environment wrapper

---

## 4. Reward Function Design

### 4.1 DARTSim Output Metrics

From DARTSim's CSV summary (at end of mission):
```
csv, targets_detected, team_destroyed, last_position, mission_success, 
     decision_time_avg, decision_time_variance
```

Also available during simulation:
- Per-step state information
- Intermediate mission progress

### 4.2 Reward Components (Multi-Objective)

Following RS-DRL paper's approach (Eq. 11-12), design a weighted multi-objective reward:

```python
def compute_reward(state, action, next_state, results, done):
    """
    Compute reward for DARTSim mission step.
    
    Args:
        state: Current state
        action: Action taken
        next_state: Next state after action
        results: SimulationResults (if done=True)
        done: Whether mission finished
    
    Returns:
        Normalized reward in [0, 1] or [-1, 1]
    """
    
    if done:
        # Terminal reward (mission complete)
        reward = 0.0
        
        # Mission success bonus
        if results.mission_success:
            reward += w_success * 1.0  # e.g., 0.4
        
        # Targets detected (normalized by max possible)
        targets_normalized = results.targets_detected / max_targets
        reward += w_targets * targets_normalized  # e.g., 0.3
        
        # Team survival bonus
        if not results.destroyed:
            reward += w_survival * 1.0  # e.g., 0.2
        else:
            reward += w_destruction * (-1.0)  # penalty
        
        # Decision time efficiency (lower is better, normalize)
        decision_time_norm = 1.0 - min(results.decision_time_avg / max_decision_time, 1.0)
        reward += w_efficiency * decision_time_norm  # e.g., 0.1
        
    else:
        # Sparse reward: small step penalty or zero
        reward = -0.01  # Encourage finishing quickly
        # Or: reward = 0.0 (fully sparse)
    
    # Normalize to [0, 1] or [-1, 1]
    return np.clip(reward, -1.0, 1.0)  # or [0, 1]
```

### 4.3 Reward Weights (Initial Proposal)

```python
REWARD_WEIGHTS = {
    "w_success": 0.4,      # Mission success (highest priority)
    "w_targets": 0.3,      # Targets detected
    "w_survival": 0.2,      # Team survival
    "w_efficiency": 0.1,   # Decision time efficiency
    "w_destruction": 0.5    # Penalty for team destruction
}
```

These weights can be tuned via grid search or adjusted based on domain priorities.

### 4.4 Alternative: Single-Objective Reward

For comparison or simplicity:
```python
# Simplified single-objective reward
if done:
    reward = 1.0 if results.mission_success else 0.0
else:
    reward = 0.0  # sparse
```

---

## 5. Scenario Selection

Select at least 3 scenarios to test robustness and generalization:

### Scenario 1: Baseline (Easy)
**Purpose**: Standard mission, no environment drift, baseline learning performance

**Parameters**:
```python
baseline_config = {
    "map_size": 40,
    "square_map": False,           # Linear route
    "num_targets": 3,               # Moderate number
    "num_threats": 5,               # Moderate number
    "altitude_levels": 4,
    "threat_range": 3,               # Auto-range = 75% of altitude_levels
    "seed": 42                       # Fixed for reproducibility
}
```

**Characteristics**:
- Predictable threat and target distribution
- Moderate difficulty
- Sufficient for initial learning

### Scenario 2: Medium Difficulty (Stress Test)
**Purpose**: Increased complexity, test adaptation capability

**Parameters**:
```python
medium_config = {
    "map_size": 50,                  # Larger map
    "square_map": True,               # Square map with turns (adds uncertainty)
    "num_targets": 5,                 # More targets
    "num_threats": 10,                # More threats
    "altitude_levels": 5,             # More altitude options
    "threat_range": 4,                # Larger threat range
    "seed": 123                       # Different seed
}
```

**Characteristics**:
- More threats and targets
- Square map with sharp turns (sensor uncertainty)
- Higher altitude levels (more action choices)

### Scenario 3: Hard / Shifted Scenario (Generalization Test)
**Purpose**: Different environment distribution, test generalization

**Parameters**:
```python
hard_config = {
    "map_size": 60,                  # Even larger
    "square_map": True,
    "num_targets": 8,                 # Many targets
    "num_threats": 15,                # Many threats (high threat density)
    "altitude_levels": 6,
    "threat_range": 5,
    "threat_sensor_fpr": 0.15,        # Higher false positive rate (noisier)
    "threat_sensor_fnr": 0.20,        # Higher false negative rate
    "seed": 456                       # Different seed
}
```

**Characteristics**:
- High threat density
- Noisy sensors (higher FPR/FNR)
- Different target/threat spatial patterns
- Tests generalization beyond training distribution

### Scenario 4: Extreme Shift (Optional)
**Purpose**: Severe environment shift, stress test generalization

**Parameters**:
```python
extreme_config = {
    "map_size": 40,
    "square_map": False,
    "num_targets": 10,                # Very many targets
    "num_threats": 20,                 # Very many threats
    "altitude_levels": 3,              # Fewer altitude levels (constraint)
    "change_alt_latency": 2,           # Slower altitude changes
    "seed": 789
}
```

**Characteristics**:
- Extreme threat/target density
- Latency constraints
- Different constraint profile

---

## 6. Episode Length and Horizon

### Episode Definition
- **One episode** = **One complete mission** (from start to finish or destruction)
- Episode ends when `finished()` returns `True` or `results.destroyed == True`

### Training Horizon (per Scenario)
Following RS-DRL paper's approach:
- **Initial proposal**: 10,000 - 50,000 timesteps per scenario
- Adjust based on average episode length:
  - If average episode = 40 steps → ~250 episodes at 10K steps
  - If average episode = 100 steps → ~100 episodes at 10K steps

**Recommendation**:
- Start with 20,000 timesteps per scenario
- Monitor learning curves to determine if more timesteps needed
- Compare to DeltaIoT: 10,000 for 216 actions, 64,800 for 1,296 actions

### Validation Split
- Use temporal split (like RS-DRL paper): train on first 80% of timesteps, validate on last 20%
- Or: Train for fixed timesteps, then validate on held-out episodes

---

## 7. Evaluation Metrics (Align with RS-DRL Paper)

### Primary Metrics
1. **Asymptotic Performance**: Mean reward over last 10% of training (multiple seeds)
2. **Time-to-Threshold**: Steps to reach reward thresholds (0.7, 0.8, 0.9)
3. **Total Performance (TP)**: Area under learning curve (Eq. 3 in RS-DRL paper)

### Domain-Specific Metrics
1. **Mission Success Rate**: Percentage of successful missions
2. **Targets Detected**: Mean targets detected per mission
3. **Team Survival Rate**: Percentage of missions without destruction
4. **Decision Time**: Mean decision time (lower is better)
5. **Mission Efficiency**: Targets detected per time unit

### Statistical Protocol
- **Seeds**: Run 10-30 independent runs per configuration
- **Reporting**: Mean ± standard deviation, confidence intervals
- **Comparisons**: Statistical tests (t-tests, Mann-Whitney U) between RS-DRL and baseline DQN

---

## 8. Alignment with RS-DRL Paper

### Mapping to RS-DRL Evaluation
| RS-DRL Metric | DARTSim Equivalent |
|--------------|-------------------|
| Asymptotic mean reward | Mission success rate, normalized composite reward |
| Time-to-threshold | Steps to reach 0.9 reward threshold |
| Packet loss reduction | Team destruction avoidance |
| Latency reduction | Decision time reduction |
| Energy efficiency | Mission efficiency (targets/time) |

### Implementation Parallels
- **RL Algorithm**: DQN (enhanced) from Stable-Baselines3
- **Reward Reshaping**: Algorithm 2 (randomly reshape ρ fraction of failed transitions)
- **Hyperparameter Search**: Grid search for ρ, learning rate, gamma, etc.
- **Evaluation Protocol**: Same metrics and statistical reporting

---

## 9. Next Steps (Phase 1-2)

After Phase 0 completion:

1. **Phase 1**: Set up DARTSim and verify TCP interface works
2. **Phase 2**: Implement `DARTSimEnv` (Gymnasium environment adapter)
3. **Phase 3**: Implement RS-DRL on top of SB3 DQN
4. **Phase 4**: Run experiments with the scenarios and metrics defined above
5. **Phase 5-6**: Report results and write up case study

---

## 10. Document Control

- **Version**: 1.0
- **Date**: [Current Date]
- **Status**: Phase 0 - Research Design Complete
- **Next Review**: After Phase 2 (adapter implementation)

