# Phase 0 Quick Reference Card

## Research Questions
1. **RQ1**: Time-to-threshold performance (faster convergence?)
2. **RQ2**: Asymptotic performance (better final results?)
3. **RQ3**: Robustness/generalization (better on shifted scenarios?)
4. **RQ4**: Mission metrics impact (targets, survival, decision time?)

## State Vector (15-20 dim)
```python
[
    position_x/map_size,           # [0,1]
    position_y/map_size,           # [0,1]
    direction_x/max,               # [-1,1]
    direction_y/max,               # [-1,1]
    altitude/max_altitude,         # [0,1]
    formation,                     # 0.0 or 1.0
    ecm,                           # 0.0 or 1.0
    ttc_inc_alt/max_latency,       # [0,1]
    ttc_dec_alt/max_latency,       # [0,1]
    *threat_sensor[5],             # 5 booleans
    *target_sensor[5],             # 5 booleans
    # + optional history
]
```

## Action Space (8 actions)
```python
0: "IncAlt"     # Increase altitude 1 level
1: "DecAlt"     # Decrease altitude 1 level
2: "IncAlt2"    # Increase altitude 2 levels
3: "DecAlt2"    # Decrease altitude 2 levels
4: "GoTight"    # Switch to tight formation
5: "GoLoose"    # Switch to loose formation
6: "EcmOn"      # Turn ECM on
7: "EcmOff"     # Turn ECM off
```

## Reward Function
```python
if done:
    reward = (
        0.4 * mission_success +
        0.3 * (targets_detected / max_targets) +
        0.2 * (1.0 if not destroyed else -1.0) +
        0.1 * (1.0 - decision_time_norm)
    )
else:
    reward = -0.01  # Sparse step penalty
```

## Scenarios

### Baseline (Easy)
- map_size=40, targets=3, threats=5, linear map
- seed=42

### Medium
- map_size=50, targets=5, threats=10, square map
- seed=123

### Hard (Shifted)
- map_size=60, targets=8, threats=15, noisy sensors
- seed=456

## DARTSim TCP Commands
- `finished` → check if done
- `getState` → get team state (JSON)
- `readForwardThreatSensor(N)` → threat array
- `readForwardTargetSensor(N)` → target array
- `step(tactics, decision_time)` → execute action
- `getResults` → final metrics (JSON)
- Port: 5418

## Evaluation Metrics
- Asymptotic performance (mean ± std)
- Time-to-threshold (steps to 0.9 reward)
- Total Performance (area under curve)
- Mission success rate
- Targets detected
- Team survival rate
- Decision time efficiency

## Training Setup
- **Algorithm**: DQN (SB3) + RS-DRL
- **Horizon**: 20,000 timesteps per scenario
- **Seeds**: 10-30 independent runs
- **Validation**: Temporal split (80/20)

---
See `phase0-research-design.md` for full details.

