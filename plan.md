Great — nice choice. Below I give a focused, **actionable, reproducible plan** to use the *DARTSim exemplar* (the GitHub repo you pointed to) as a **second case study** in your article (so it complements the DeltaIoT RS-DRL case). The plan is split into phases (study → reproduce → extend → evaluate → write-up) and each step gives concrete commands, code/architecture points to implement, and the exact places in the RS-DRL article and DARTSim repo that justify/enable those steps. I cite the article snippets you uploaded and the DARTSim repo page so you can jump straight to the exact references.

---

# Executive summary (one-liner)

Use DARTSim as a second case study by (A) mapping your research questions and metrics to DARTSim's mission/metrics, (B) collecting a large offline dataset of simulation episodes via DARTSim's TCP API, (C) building an offline RL environment (`OfflineDARTSimEnv`) that replays collected transitions, (D) implementing RS-DRL on top of Stable-Baselines3 DQN for offline training from the replay buffer, (E) running systematic hyperparameter grid searches and the same evaluation metrics/time horizons used for DeltaIoT, and (F) reporting comparisons and robustness experiments (environment shifts, generalization). The offline approach avoids TCP connection issues during training, enables faster/reproducible experiments, and supports large-scale training. ([GitHub](https://github.com/cps-sei/dartsim "GitHub - cps-sei/dartsim: DARTSim Mission Simulator"))

---

# Phase 0 — Decide scope & mapping (1–2 days)

1. **Decide which research questions DARTSim will answer** (examples):
   * Does RS-DRL speed up time-to-threshold in DARTSim missions (decision time / mission success)?
   * Does RS-DRL generalize to environment shifts in the mission (unexpected threats / extra targets)?
   * How does RS-DRL affect mission-level metrics (targets detected, team destroyed, mission success) vs baseline DQN?

     (These mirror DeltaIoT’s goals—packet loss/latency/energy—and allow apples-to-apples comparative discussion in your paper.)
2. **Map DARTSim outputs → RL state / reward** :

* DARTSim already reports a CSV summary line (example in README: `csv, targets detected, team destroyed, last team position, mission success, decision time avg, decision time variance`). Use these to build states and rewards (e.g., normalized mission success, negative penalty for team destroyed, decision-time penalty). ([GitHub](https://github.com/cps-sei/dartsim "GitHub - cps-sei/dartsim: DARTSim Mission Simulator"))

1. **Select scenario set** (at least 3):
   * Baseline scenario (no environment drift),
   * Medium difficulty (extra threats/spread targets),
   * Hard / shifted scenario (different target patterns).

     These let you test robustness / generalization like with DeltaIoT’s interference/load shifts.

---

# Phase 1 — Collect offline dataset (1–2 days)

Goal: collect a large dataset of simulation episodes offline for batch/offline RL training.

**Approach**: Instead of training with live DARTSim interaction, we collect simulation data in batch mode, then train offline from the collected transitions. This approach:
- Avoids TCP connection reliability issues during long training runs
- Enables faster training (no simulator wait time)
- Allows reproducible experiments with the same dataset
- Supports large-scale data collection across multiple scenarios

Commands:

```bash
# Collect offline data (1000 episodes, baseline scenario)
python scripts/collect_offline_data.py --episodes 1000 --scenario baseline --output-dir ./data/offline

# Collect data for multiple scenarios
python scripts/collect_offline_data.py --episodes 500 --scenario medium --output-dir ./data/offline
python scripts/collect_offline_data.py --episodes 500 --scenario hard --output-dir ./data/offline
```

**Data Format**: Each episode file contains transitions with:
- `state`: State vector (position, altitude, formation, sensors, etc.)
- `action`: Action taken (IncAlt, DecAlt, GoTight, etc.)
- `reward`: Computed reward (sparse: -0.01 per step, terminal reward based on mission results)
- `next_state`: Resulting state after action
- `done`: Episode termination flag
- `info`: Additional metadata (step, position, altitude, etc.)

**Data Collection Details**:
- DARTSim runs in Docker container via TCP interface for data collection
- Each episode is a complete mission simulation
- Rewards computed using multi-objective function (mission success, targets detected, survival, efficiency)
- Data saved as JSON files (`episodes_{scenario}_{episode_id}_{timestamp}.json`)

**Dataset Size**: Target 5000-10000+ episodes for robust offline training. Current dataset: ~179 episodes per file, multiple files totaling ~6.85 GB.

---

# Phase 2 — Design the offline RL environment adapter (1 day)

Goal: create an offline RL environment that replays collected simulation data (Gymnasium API) so you can plug in Stable-Baselines3 DQN and your RS-DRL modifications.

**Approach**: Build an offline environment that:
- Loads pre-collected episode transitions from JSON files
- Provides Gymnasium API (`reset()`, `step()`) that replays transitions
- Looks up state-action pairs to return actual next_state, reward, done from data
- Simulates the real environment without requiring DARTSim library or TCP connection

**Implementation**: `OfflineDARTSimEnv` in `src/dartsim_env.py`

Key features:
* **State vector**: Loaded from collected data (17-dimensional: position, altitude, formation, ECM, sensors)
* **Action space**: Discrete(8) - maps to DARTSim tactics (IncAlt, DecAlt, GoTight, GoLoose, EcmOn, EcmOff, etc.)
* **Reward**: Retrieved from collected data (pre-computed during data collection)
* **Transition lookup**: Builds hash table of (state, action) → transitions for fast lookup
* **Stochastic replay**: When multiple transitions match, randomly selects one (accounts for environment stochasticity)

**Data Collection** (Phase 1):
- Rewards computed during collection using multi-objective function:
  - Step penalty: -0.01 per non-terminal step
  - Terminal reward: weighted combination of:
    - Mission success: +0.4
    - Targets detected: +0.3 * min(targets/10, 1.0)
    - Team survival: +0.2 (if not destroyed) or -0.5 (if destroyed)

**Environment Usage**:
```python
from src.dartsim_env import OfflineDARTSimEnv

env = OfflineDARTSimEnv(
    obs_dim=17,
    data_dir="./data/offline",
    scenario="baseline",
    max_transitions=100000,  # Limit for memory efficiency
    seed=42
)
```

**Advantages of offline approach**:
- No TCP connection issues during training
- Faster training (no simulator wait time)
- Reproducible experiments (same dataset across runs)
- Memory efficient (can limit transitions loaded)
- Supports large-scale training without simulator overhead

---

# Phase 3 — Implement RS-DRL on top of SB3 DQN with offline training (2–3 days)

Goal: reproduce the paper's implementation approach: **Stable-Baselines3 DQN extended with randomized reward reshaping in replay**, using offline collected data.

Key references from the article:

* The authors implemented environment with Gymnasium and used Stable Baselines3 DQN extended to apply the reward reshaping during minibatch replay.
* Algorithm 1/2 describe the RS-DRL and RewardShaping functions (randomly replace up to ρ fraction of failed transitions in minibatch with optimistic reward 1).

Implementation steps:

1. **Offline environment** : Use `OfflineDARTSimEnv` which loads collected data and replays transitions. No live DARTSim interaction needed.

2. **Data loading** : Load offline data incrementally using generators to manage memory:
   ```python
   from scripts.offline_rl_training import load_offline_dataset_generator
   episode_generator = load_offline_dataset_generator(
       data_dir="./data/offline",
       scenario="baseline"
   )
   ```

3. **Replay buffer population** : Populate Stable-Baselines3 replay buffer from offline data:
   - Convert episodes to (state, action, reward, next_state, done) format
   - Add transitions to replay buffer (limited to reasonable size based on training needs)
   - Memory-efficient: only load needed transitions (e.g., 10k for 1k timesteps training)

4. **RS-DRL DQN extension** : Implement `RSDRLDQN` class extending SB3 DQN:
   ```python
   class RSDRLDQN(DQN):
       def train(self, gradient_steps, batch_size):
           # Sample from replay buffer
           replay_data = self.replay_buffer.sample(batch_size)
           # Apply reward reshaping (Algorithm 2)
           reshaped_rewards = self.reward_shaping.reshape_rewards(rewards)
           # Continue with standard DQN training
   ```
   Reward reshaping function:
   ```python
   def reshape_rewards(rewards, rho=0.3):
       # Find failed transitions (reward <= 0)
       # Randomly select up to K = floor(rho * N) failed transitions
       # Replace their reward with optimistic value (1.0)
   ```

5. **Offline training loop** : Train directly from replay buffer without environment interaction:
   ```python
   # Populate replay buffer from offline data
   # Train by calling model.train() directly with gradient steps
   for epoch in range(num_epochs):
       model.train(gradient_steps=gradient_steps_per_epoch, batch_size=batch_size)
   ```

6. **Hyperparameters** : Follow the article's approach — grid search for `rho` (reshaping fraction), learning rate, discount `gamma`, minibatch size, target update frequency. Use same systematic grid search protocol.

7. **Training horizon** : For offline training:
   - Limit replay buffer size based on training timesteps (e.g., 10x timesteps)
   - Training directly from replay buffer (no environment interaction)
   - Faster than online training (no simulator wait time)

8. **Baselines** : Implement these for comparison:
   * Vanilla ε-greedy DQN (SB3 out-of-the-box) - offline training mode
   * RS-DRL (your modified DQN) - offline training mode

---

# Phase 4 — Evaluation plan (2–4 days to run experiments)

Follow the **same evaluation metrics and statistical protocol** used in the RS-DRL article so your comparison is valid.

**Offline Training Approach**:
- Training uses pre-collected offline data (no live DARTSim during training)
- Evaluation can use either:
  1. **Offline evaluation**: Evaluate on held-out test set from collected data
  2. **Online evaluation**: Load trained model and evaluate on live DARTSim (for final performance assessment)

From the article:

* **Metrics** : Asymptotic performance, Total Performance (TP area metric defined in Eq.3), Time-to-threshold (steps to reach reward threshold). Also domain metrics (mission success, targets detected, average decision time). Report mean ± std over multiple seeds.
* **Runs & seeds** : The paper used multiple independent runs and reported averages + confidence intervals; replicate that (e.g., 10–30 seeds depending on compute).
* **Hyperparameter grid search protocol** : Use a training/validation split from offline data (split collected episodes into train/validation sets) and choose the best hyperparams by validation metrics. Then report final test runs.

Experiments to run:

1. **Baseline learning curves** : RS-DRL vs DQN over the full training horizon (plot mean reward ± CI). Training from offline replay buffer.
2. **Time-to-threshold** : compute gradient steps to achieve some threshold of mission performance (analogous to 0.9 reward threshold used in paper).
3. **Robustness** : 
   - Train on one scenario (e.g., baseline), evaluate on different scenarios (medium, hard)
   - Use offline environment to evaluate on collected test data from different scenarios
   - Or evaluate trained models on live DARTSim with shifted scenarios
4. **Ablations** : vary `rho` (reshaping factor) and the reshaped reward value (paper mostly used 1; test 0.5 and 1.5) to show sensitivity. The article mentioned ablation study.

Experimental logistics:

* **Offline training**: No DARTSim container needed during training (uses pre-collected data)
* **Data collection**: Run DARTSim in Docker once to collect dataset (batch collection)
* **Evaluation**: Can evaluate trained models offline (from test data) or online (live DARTSim)
* **Parallel runs**: Run multiple training experiments in parallel (different seeds) - no simulator bottleneck
* **Reproducibility**: Same dataset used across all runs ensures fair comparison

---

# Phase 5 — Integrate MAPE-K & runtime verification (optional / advanced)

If you want to mirror the article's architecture tightly (MAPE-K + ActivFORMS + UPPAAL-SMC), do this after you have a working RL loop:

1. **MAPE-K skeleton** : implement a simple Monitor/Analyzer/Planner/Executor in Python:

* **Monitor**: Collects DARTSim telemetry (mission metrics) - can use offline environment or live DARTSim
* **Analyzer**: Detects deviations (time-to-threshold violation) and triggers retraining or model swap
* **Planner/Executor**: Fetches chosen tactic/action and applies to simulation via adapter
* **Knowledge**: Stores current model, performance thresholds, retraining triggers

The RS-DRL paper integrates retraining only upon threshold violation (on-demand retraining). For offline approach:
- Initial training: Offline from collected data
- Retraining: Can trigger offline retraining with new data, or collect new data and retrain
- Runtime adaptation: Use trained model for online decision-making (via offline env or live DARTSim)

1. **Runtime verification** : the article used UPPAAL-SMC and ActivFORMS to analyze adaptation options. If you want to match that rigor, model key subsystems as timed automata and use ActivFORMS/UPPAAL-SMC to verify candidate adaptation options before enactment. (This is optional and significantly increases effort; include as a future work if you lack time.)

---

# Phase 6 — Reporting & writing (2–4 days)

Structure the case study section to match how you used DeltaIoT:

1. **Motivation & mapping** : describe why DARTSim is a complementary exemplar (CPS domain, mission-level metrics vs network metrics). Cite DARTSim repo and the RS-DRL article's rationale for DeltaIoT. ([GitHub](https://github.com/cps-sei/dartsim "GitHub - cps-sei/dartsim: DARTSim Mission Simulator"))
2. **Method** : 
   - **Offline data collection**: Describe batch collection process, dataset size, scenarios collected
   - **Offline environment**: Describe `OfflineDARTSimEnv` that replays collected transitions
   - **Action/state mappings**: Same as original plan
   - **Reward design**: Describe reward function used during data collection (link to article's Eq.11/12)
   - **RS-DRL modifications**: Algorithm 1/2 implementation for offline training
   - **Training approach**: Offline training from replay buffer (no live simulator during training)
3. **Experimental protocol** : 
   - Dataset: Size, scenarios, train/validation/test splits
   - Training: Gradient steps, epochs, replay buffer size, seeds
   - Grid search: Same hyperparameters (rho, lr, gamma, etc.)
   - Metrics: Same as RS-DRL article (TP, time-to-threshold, etc.)
   - Use the same phrasing and metrics definitions as the RS-DRL article so reviewers can compare. (Use Table and Figures analogous to theirs.)
4. **Results** : same plots (learning curves, time-to-threshold, per-metric tables) and statistical tests. Highlight where RS-DRL helps (e.g., faster time-to-threshold, better generalization under shifted mission scenarios).
5. **Discussion** : 
   - Discuss differences (DARTSim mission semantics vs DeltaIoT network semantics)
   - Advantages of offline approach (reproducibility, scalability, avoids TCP issues)
   - Limitations (dataset coverage, generalization to unseen scenarios)
   - Report limits and threats-to-validity

---

# Concrete code & commands I can provide immediately

**Status**: The following components are already implemented:

* ✅ **`OfflineDARTSimEnv`** (`src/dartsim_env.py`) - Gymnasium environment that loads and replays collected offline data
* ✅ **Offline data collection script** (`scripts/collect_offline_data.py`) - Collects simulation episodes via TCP and saves as JSON
* ✅ **RS-DRL DQN extension** (`src/rs_drl_dqn.py`) - Stable-Baselines3 DQN subclass with reward reshaping (Algorithm 2)
* ✅ **Offline training script** (`scripts/train_rs_drl.py`) - Training script that populates replay buffer from offline data and trains offline

**Ready to use**:
- Collect offline data: `python scripts/collect_offline_data.py --episodes 1000 --output-dir ./data/offline`
- Train offline: `python scripts/train_rs_drl.py --offline --offline-data-dir ./data/offline --rho 0.3 --timesteps 1000`

**Additional tools that could be added**:
* A **bash script** that collects offline datasets in batch (multiple scenarios, parallel collection), then launches N parallel training runs (different seeds), and aggregates results into a summary table matching the paper's metrics.
* A **data analysis script** to analyze collected dataset statistics (reward distributions, scenario coverage, etc.)

---

# Quick checklist & timeline estimate (suggested minimal schedule)

(You didn't ask for a timeline but here's a compact plan you can follow.)

* Day 0: clone repo, start Docker, run `simple-java`/`simple-cpp` examples. ([GitHub](https://github.com/cps-sei/dartsim "GitHub - cps-sei/dartsim: DARTSim Mission Simulator"))
* Day 1: **Collect offline dataset** - run batch data collection script to gather 5000+ episodes across scenarios.
* Day 2: implement `OfflineDARTSimEnv` (Gym) that loads and replays collected data; verify it works with random actions.
* Day 3–4: implement SB3 DQN + RS-DRL extension for offline training; verify learning from offline replay buffer.
* Day 5–8: run grid search and main experiments using offline data (parallel seeds, faster than online training).
* Day 9–11: robustness / ablation tests (evaluate on different scenarios from collected data or live DARTSim).
* Day 12–14: write case study section, figures and tables, link reproducibility artifacts (dataset + code + seed list).

---

# Useful references (from your uploaded article + DARTSim)

* RS-DRL algorithm & reward shaping — Algorithms 1 & 2 in the article.
* Implementation note: RS-DRL used Gymnasium & Stable-Baselines3 (DQN) and extended replay to apply randomized reward reshaping.
* Training horizons and timestep-splitting used for DeltaIoT (216 / 1,296 / 4,096 action sizes; training horizons 10,000 / 64,800 etc.) — use as inspiration for choosing DARTSim horizons and splitting.
* DARTSim repo (readme + examples): Docker container, TCP interface, example adaptation managers and `simple-cpp`/`simple-java` starters. Useful entry point for adapter code. ([GitHub](https://github.com/cps-sei/dartsim "GitHub - cps-sei/dartsim: DARTSim Mission Simulator"))

---
