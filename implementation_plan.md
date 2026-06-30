
Listed directory dart
Listed directory scripts
Viewed train_rs_drl.py:1-649
Viewed evaluate_rs_drl.py:1-414
Viewed run_experiments.py:1-289

Here is a detailed, step-by-step implementation plan to carry out the missing evaluations for the **Medium** and **Hard** scenarios:

---

# Implementation Plan: Medium & Hard Scenario Evaluations

## Phase 1: Direct Training and Convergence Analysis

To evaluate learning performance and check for statistical significance in the larger environments, we need to train both the proposed `RS-DRL` model and the baseline `DQN` model on both the Medium and Hard maps.

1. **Train on the Medium Scenario:**
   Run the multi-seed training script using the pre-collected offline datasets filtered for `medium`.

   ```powershell
   python scripts/run_experiments.py --methods rs_drl baseline --seeds 42 43 44 --timesteps 20000 --rho-values 0.3 --output-base ./experiments/medium --offline-scenario medium
   ```

2. **Train on the Hard Scenario:**
   Run the multi-seed training script using the pre-collected offline datasets filtered for `hard`.

   ```powershell
   python scripts/run_experiments.py --methods rs_drl baseline --seeds 42 43 44 --timesteps 20000 --rho-values 0.3 --output-base ./experiments/hard --offline-scenario hard
   ```

3. **Perform Statistical Significance Analysis:**
   Write a script to read the TensorBoard event logs (or CSV files) generated in `./experiments/medium/` and `./experiments/hard/`.
   * Test for normality (Shapiro-Wilk) and variance equality (Levene's test).
   * Perform independent t-tests comparing final Q-values of `RS-DRL` versus `Baseline DQN`.
   * Calculate Cohen's $d$ effect sizes for both environments.

---

## Phase 2: Sensor Noise & Failure Robustness Evaluations

We must test the resilience of the agents trained in Medium and Hard environments against various levels of threat sensor noise.

1. **Evaluate Medium Models under Sensor Noise:**
   Evaluate the models trained on Medium across 10%, 30%, and 50% sensor noise levels over 50 episodes.

   ```powershell
   # 10% Noise
   python scripts/evaluate_rs_drl.py --model-path ./experiments/medium/rs_drl/seed_42/model.zip --offline-scenario medium --sensor-noise 0.1 --episodes 50
   # 30% Noise
   python scripts/evaluate_rs_drl.py --model-path ./experiments/medium/rs_drl/seed_42/model.zip --offline-scenario medium --sensor-noise 0.3 --episodes 50
   # 50% Noise
   python scripts/evaluate_rs_drl.py --model-path ./experiments/medium/rs_drl/seed_42/model.zip --offline-scenario medium --sensor-noise 0.5 --episodes 50
   ```

2. **Evaluate Hard Models under Sensor Noise:**
   Run evaluations on the models trained on Hard at the same noise levels:

   ```powershell
   # 10% Noise
   python scripts/evaluate_rs_drl.py --model-path ./experiments/hard/rs_drl/seed_42/model.zip --offline-scenario hard --sensor-noise 0.1 --episodes 50
   # 30% Noise
   python scripts/evaluate_rs_drl.py --model-path ./experiments/hard/rs_drl/seed_42/model.zip --offline-scenario hard --sensor-noise 0.3 --episodes 50
   # 50% Noise
   python scripts/evaluate_rs_drl.py --model-path ./experiments/hard/rs_drl/seed_42/model.zip --offline-scenario hard --sensor-noise 0.5 --episodes 50
   ```

3. **Compile Results & Plot:**
   Generate comparative plots (similar to the existing Figure 4) displaying noise levels vs. success rates/rewards for all three scenarios (Baseline, Medium, and Hard).

---

## Phase 3: Sensitivity & Ablation Analysis of the Shaping Parameter ($\rho$)

We need to conduct an ablation study to check if $\rho = 0.3$ is still the optimal setting for higher complexity levels.

1. **Ablation Grid on Medium Scenario:**
   Train the agent on the Medium map across different shaping rates ($\rho \in \{0.0, 0.1, 0.3, 0.5\}$):

   ```powershell
   python scripts/run_experiments.py --methods rs_drl --seeds 42 43 44 --timesteps 20000 --rho-values 0.0 0.1 0.3 0.5 --output-base ./experiments/medium_ablation --offline-scenario medium
   ```

2. **Ablation Grid on Hard Scenario:**
   Train the agent on the Hard map across different shaping rates:

   ```powershell
   python scripts/run_experiments.py --methods rs_drl --seeds 42 43 44 --timesteps 20000 --rho-values 0.0 0.1 0.3 0.5 --output-base ./experiments/hard_ablation --offline-scenario hard
   ```

3. **Compare Results:**
   Compare the learning speed and asymptotic rewards to confirm if $\rho = 0.3$ yields the best balance between risk-taking and learning speed in all maps.

---

## Phase 4: Bidirectional Generalization Matrix

We will test how well an agent trained in one scenario transfers to the other scenarios without fine-tuning.

1. **Test Medium-trained Agent on Baseline and Hard maps:**

   ```powershell
   python scripts/evaluate_rs_drl.py --model-path ./experiments/medium/rs_drl/seed_42/model.zip --offline-scenario baseline --episodes 50
   python scripts/evaluate_rs_drl.py --model-path ./experiments/medium/rs_drl/seed_42/model.zip --offline-scenario hard --episodes 50
   ```

2. **Test Hard-trained Agent on Baseline and Medium maps:**

   ```powershell
   python scripts/evaluate_rs_drl.py --model-path ./experiments/hard/rs_drl/seed_42/model.zip --offline-scenario baseline --episodes 50
   python scripts/evaluate_rs_drl.py --model-path ./experiments/hard/rs_drl/seed_42/model.zip --offline-scenario medium --episodes 50
   ```

3. **Build Generalization Matrix:**
   Create a $3 \times 3$ matrix mapping the training scenario (rows) to test scenarios (columns) for both success rate and average reward.

---

## Phase 5: Document and Figure Updates

* Save plots and tables generated from the steps above in `thesis_sources/figures/`.
* Modify [dartsim_detailed_guide_v2_fa.tex](file:///e:/projects/dart/thesis_sources/dartsim_detailed_guide_v2_fa.tex):
  * In Section 7.1, add the Direct Training results and statistical tests.
  * In Section 7.2, add the Bidirectional Generalization matrix.
  * In Section 7.3, add the comparative Sensor Noise evaluations for Medium and Hard.

---

### Summary of Work Done

* Created a complete step-by-step implementation plan covering direct training, noise robustness, parameter sensitivity, generalization matrices, and documentation updates.
* Mapped out the precise terminal commands using the project's existing [run_experiments.py](file:///e:/projects/dart/scripts/run_experiments.py) and [evaluate_rs_drl.py](file:///e:/projects/dart/scripts/evaluate_rs_drl.py) scripts.
