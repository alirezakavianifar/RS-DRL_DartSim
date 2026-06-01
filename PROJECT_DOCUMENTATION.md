# DARTSim RS-DRL Project Documentation

## 1. Project Overview

This project implements **RS-DRL (Randomized Reward Shaping for Deep Reinforcement Learning)** on the **DARTSim** mission simulator. It serves as a complementary case study to evaluate RS-DRL on Cyber-Physical Systems (CPS) with mission-level metrics, demonstrating the method's capability to improve reinforcement learning efficiency and generalization.

### Key Innovations:
1. **Offline Reinforcement Learning:** Unlike traditional DARTSim projects that rely on live TCP connections during training (which cause latency and disconnection issues), this project operates in an **offline mode**. Large-scale datasets of simulation episodes are collected first, and the agent then learns directly from a replay buffer.
2. **Reward Reshaping (RS-DRL):** An extension of the standard Stable-Baselines3 Deep Q-Network (DQN). Using a randomized reshape mechanism, unsuccessful transitions in minibatches randomly receive optimistic rewards to accelerate policy learning and reach target threshold performances faster.
3. **MAPE-K Architecture Integration:** Features a Monitor-Analyze-Plan-Execute over a shared Knowledge base (MAPE-K) feedback loop to adapt runtime behaviors efficiently.

---

## 2. Directory Structure

```text
e:\projects\dart
│
├── main.py                     # Centralized CLI entry point for all project phases
├── plan.md                     # The theoretical research methodology & execution master plan
├── requirements.txt            # Python dependencies (Gymnasium, Stable-Baselines3, etc.)
│
├── src/                        # Core algorithmic implementations
│   ├── dartsim_env.py          # Gymnasium API compatible OfflineDARTSimEnv adapter for replay
│   ├── rs_drl_dqn.py           # RS-DRL algorithm extended from Stable-Baselines3 DQN
│   └── mape_k_architecture.py  # MAPE-K loop implementation
│
├── scripts/                    # Helper scripts managed via main.py
│   ├── collect_offline_data.py   # Connects to DARTSim Docker for offline batch generation
│   ├── train_rs_drl.py           # Populates replay buffers and initiates offline training
│   ├── grid_search.py            # Hyperparameter tuning runner
│   ├── run_experiments.py        # Complete execution suite spanning multiple seeds/modes
│   ├── evaluate_rs_drl.py        # Post-training evaluation framework
│   ├── analyze_experiments.py    # Metric aggregation
│   └── plot_results.py           # Matplotlib/Plotly rendering of evaluation datasets
│
├── utils/                      # Helper scripts (PowerShell tools, setup, Docker interactions)
├── data/offline/               # Collected JSON archives representing simulation episodes
├── models/                     # Compiled/trained Stable-Baselines3 binaries
├── logs/                       # TensorBoard compatible event tracking 
└── results/                    # Aggregated plots & numeric metric spreadsheets
```

---

## 3. How to Setup

### Prerequisites
- Python 3.9+
- Docker (Required only for collecting offline simulation data)
- Windows OS (Tested with PowerShell tools available in the root)

### Step-by-step Installation

1. **Install dependencies:**
   Launch a terminal or PowerShell session to pull all Python requirements.
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify DARTSim Simulation Environment:**
   Run the PowerShell verifier script to ensure the DARTSim Docker engine operates effectively.
   ```powershell
   ./verify-dartsim.ps1
   ```

---

## 4. How to Run the Project (Workflow)

The application flow strictly follows the `main.py` entrypoint. Everything is executed through specific "phases."

### Phase 1: Collect Offline Dataset
We generate transitions via live DARTSim interaction and save them locally. *Note: Ensure your DARTSim Docker container is running during this phase.*

```bash
# Collect default 1000 episodes into offline JSON cache
python main.py collect-data --episodes 1000 --scenario baseline --auto
```

*(You can modify scenarios using `--scenario baseline|medium|hard` to generate training data spanning variable difficulty profiles).*

### Phase 2 & 3: Train RS-DRL Agents 
We execute Offline RL decoupled completely from the Docker container. 

```bash
# Train RS-DRL model
python main.py train --method rs_drl --rho 0.3 --timesteps 10000 --auto

# Or train baseline standard DQN for accurate paper comparisons
python main.py train --method baseline --timesteps 10000 --auto
```
*Note: The `--rho` parameter specifies the RS-DRL reshaping factor percentage limits.*

**Looking to tune hyperparameters?** 
```bash
# Orchestrate algorithmic grid searching 
python main.py grid-search --auto
```

### Phase 4: Evaluation and Analysis
Test the trained models to compile performance thresholds and plot learning curve differentials via multiple synchronized seeds.

```bash
# Test trained agents across scenarios implicitly 
python main.py evaluate --methods baseline,rs_drl --seeds 42,43,44 --auto

# Analyze raw evaluation caches
python main.py analyze --experiments-dir ./experiments_phase4 --auto
```
The resulting visualizations and case study diagrams are automatically deposited to the `results/` folder.

### Phase 5: MAPE-K Validation Demo (Optional)
Run the advanced MAPE-K demonstration simulating intelligent adaptation over metrics thresholds utilizing offline mechanisms.

```bash
python main.py mapek --timesteps 10000 --threshold 0.7 --auto
```

---

## 5. Architectural Breakdown

### 5.1 Environment Adaptor (`src/dartsim_env.py`)
Rather than risking latency bottlenecks on a live TCP line through thousands of RL timesteps, the `OfflineDARTSimEnv`:
- Mocks the Gymnasium paradigm (`reset()`, `step()`).
- Matches requested action topologies to the closest pre-collected outcome in `data/offline`.
- Includes stochastic state sampling capabilities on uniform overlaps natively.

### 5.2 RS-DRL Implementation (`src/rs_drl_dqn.py`)
Extended directly from `stable_baselines3.DQN`:
- Automatically triggers `reshape_rewards()` dynamically over the mini-batches within replay buffers.
- Only transforms historically negative trajectory segments optimizing survival without distorting core mission-completion objectives.

### 5.3 Rewards Mechanics
Calculated explicitly during Phase 1 (Data Collection) allowing offline networks to target explicitly:
- Successful Missions: `+0.4` factor. 
- Survivability: `+0.2` if alive, `-0.5` terminal negative if annihilated. 
- Step penalty: `-0.01` discouraging extreme latency architectures.
