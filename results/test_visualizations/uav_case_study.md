# UAV Case Study in Many-objective Self-adaptation under Model Uncertainty

## Paper Introduction

**Title:** Many-objective Self-adaptation under Model Uncertainty  
**Authors:** Camilli et al.  
**Publication:** ACM Transactions on Autonomous and Adaptive Systems (TAAS), 2025

This paper presents **TUNE-II**, a self-adaptation framework that addresses model uncertainty through Bayesian Model Averaging (BMA). The paper evaluates TUNE-II on two case studies: a Rescue Robot (RR) system and a UAV team (UAV) system. This document provides a comprehensive analysis of how the UAV case study is used in the paper, including its setup, requirements, experimental methodology, and results.

---

## Case Studies Overview

The paper evaluates **TUNE-II** on **two case studies**:

1. **Rescue Robot (RR)**
   - A self-adaptive search-and-rescue robot (vision + LiDAR) used for emergency scenarios (fire/hurricane/earthquake)
   - Semantic space has **9 factors** (config + environment)
   - Study considers **5 probabilistic safety requirements** (e.g., human-detection, protective distance, contact, obstacle avoidance)

2. **UAV team (UAV)**
   - An autonomous team of unmanned aerial vehicles performing a surveying mission in a hostile environment (taken from Moreno et al.)
   - Semantic space has **7 factors**
   - Subject uses **12 requirements** derived from two requirement templates (target detection and threat-avoidance across altitude ranges)

**Additional Notes:**
- Both case studies are simulated using the **RUNE simulation framework** (parametric MDP mutators)
- Adaptations and requirement satisfaction are evaluated via simulation
- The paper reports per-requirement results (accuracy, adaptation success, cost, scalability) for both subjects

---

## UAV Case Study Overview

### Purpose

The UAV subject models an autonomous team of unmanned aerial vehicles that conduct a surveying mission in a hostile environment. The study was originally adapted from Moreno et al. [19], a classic benchmark for self-adaptive systems.

### Scenario Summary

- **Mission:** Reach ground-based targets (using downward-looking sensors) while avoiding airborne or ground threats (detected via forward-looking sensors)
- **Trade-off:** Flying lower increases target-detection likelihood but also increases exposure to threats; flying higher reduces threat risk but makes detection harder
- **Objective:** Maintain mission performance (detect targets, avoid threats) despite environmental uncertainty and changing conditions
- **Simulation Platform:** The system is simulated using RUNE, which models uncertainty via parametric Markov Decision Processes (pMDPs); factors in the "semantic space" influence stochastic transitions dynamically

---

## Semantic Space (Table 5)

Seven factors define the UAV system's semantic space — a mix of configuration and environment variables:

| Factor | Type | Domain / Range | Description |
|--------|------|----------------|-------------|
| formation | Configuration | {tight, loose} | Spatial formation of the UAV team during mission execution. |
| flying speed | Configuration | [5, 50] mph | Average airspeed of UAVs; influences both detection and avoidance probabilities. |
| electronic countermeasure (ECM) | Configuration | Boolean (Yes/No) | Whether ECM is active to reduce the likelihood of being detected by threats. |
| weather | Environment | {sun, clouds, rain, fog} | Meteorological condition affecting sensors and flight performance. |
| day time | Environment | 00:00–23:59 (discrete) | Time of day influencing visibility and detection probability. |
| threat range | Environment | [0.9, 3.7] km | Detection range of threats (affects UAV exposure). |
| # threats | Environment | Integer [1–10] | Number of active threats in the operational area. |

**Notes:**
- Configuration factors are controllable parameters adjusted by the UAV system during adaptation.
- Environmental factors model external uncertainty sources in RUNE's pMDP-based simulation.

These jointly define the system's operating context for adaptation and uncertainty modeling.

---

## System Diagram: UAV Team Dependability Factors

The following diagram illustrates the conceptual model for the UAV team case study, showing the interplay between configuration parameters, environmental factors, and the resulting dependability goals within a hostile operational context.

![UAV System Diagram](./uav_system_diagram.png)

**Description of the Diagram:**

This conceptual diagram illustrates the factors influencing the dependability goals of a UAV (Unmanned Aerial Vehicle) team operating in a hostile environment.

1. **Overall Structure:** The diagram uses boxes, a cloud, and arrows to show relationships between different categories of inputs and outputs. At its center are black silhouettes representing UAVs and a ground-based threat.

2. **Input Factors (Left Side):**
   - **"Configuration" Box (Blue Header):** This box lists controllable aspects of the UAV team.
     - `formation`
     - `flying speed`
     - `electronic countermeasure`
   - **"Environment" Box (Green Header):** This box lists external, often uncontrollable, environmental factors.
     - `weather`
     - `day time`
     - `threat range`
     - `# of threats`

3. **Central System and Hostile Environment:**
   - **UAVs and Threat Icons:** In the middle, there are three black silhouettes:
     - A large, fixed-wing UAV (resembling a Predator/Reaper drone).
     - A smaller, fixed-wing aircraft, possibly another UAV or a manned support aircraft.
     - A ground-based radar dish or sensor icon, representing a threat.
   - **"Hostile environment" Cloud:** Above the UAVs and threat, a light grey cloud shape is labeled "Hostile environment," indicating the operational context.

4. **Output/Goals (Right Side):**
   - **"Dependability goals" Box (Red Dotted Border):** This box outlines the desired outcomes or objectives for the UAV team.
     - `target detection`
     - `threat avoidance`

5. **Relationships (Arrows):**
   - An arrow points from the "Configuration" box to the central UAV/threat icons, indicating that configuration parameters directly affect the operation of the UAVs.
   - An arrow points from the "Environment" box to the central UAV/threat icons, showing that environmental factors also influence the UAVs' operation.
   - Both the "Configuration" and "Environment" boxes, through their influence on the central system, ultimately lead to an arrow pointing towards the "Dependability goals" box, suggesting that these factors determine the achievement of those goals.

In summary, the diagram illustrates a system where UAV "Configuration" and external "Environment" factors interact within a "Hostile environment" to influence the achievement of "Dependability goals" such as target detection and threat avoidance.

---

## Requirements and Goals (Table 6)

There are **12 requirements** derived from two templates:

### Template )₀ — Target Detection
"When the altitude is between ⟨θ₁⟩ and ⟨θ₂⟩ meters, the team of UAVs shall detect the target in more than ⟨p₁⟩ % of the cases."

### Template )₁ — Threat Avoidance
"When the altitude is between ⟨θ₁⟩ and ⟨θ₂⟩ meters, existing threats shall detect the UAVs in less than ⟨p₂⟩ % of the cases."

By instantiating these templates with six altitude ranges [θ₁, θ₂] and corresponding likelihood thresholds (p₁ and p₂), the study defines 12 concrete dependability requirements.

| Template ID | Template Description | Instantiated Variants | Goal Type |
|-------------|---------------------|----------------------|-----------|
| )₀ – Target Detection | When the altitude is between ⟨θ₁⟩ and ⟨θ₂⟩ meters, the UAV team shall detect the target in more than ⟨p₁⟩% of cases. | 6 instantiations for different altitude ranges and detection thresholds. | Maximize detection reliability. |
| )₁ – Threat Avoidance | When the altitude is between ⟨θ₁⟩ and ⟨θ₂⟩ meters, threats shall detect the UAVs in less than ⟨p₂⟩% of cases. | 6 instantiations for the same altitude ranges, with avoidance thresholds. | Minimize exposure to threats. |

**Total Requirements:** 12 (6 for detection, 6 for avoidance)

**Trade-off:** Lower altitude = better target detection but higher threat exposure; higher altitude = safer but reduced detection probability.

---

## Experimental Setup

### Simulations
- Each case study (including UAV) was simulated **1,000 times per method**
- Each simulation corresponds to a complete mission run, used to verify requirement satisfaction
- Experiments were run before and after "changes" in the environment — these represent model drift or non-steady mutation of parameters in the semantic space

### Compared Methods
1. **TUNE-II (BMA)** – Bayesian Model Averaging, the proposed approach
2. **MS (Model Selection)** – Chooses the single best logistic model
3. **RAND** – Random adaptation baseline

### Evaluation Metrics
- **Prediction accuracy (RQ1):** precision, recall, F1-score
- **Adaptation effectiveness (RQ2):** success frequency and adaptation cost
- **Computation cost (RQ3):** runtime to compute models
- **Scalability (RQ4):** performance under increasing features/observations

---

## Detailed Results

### RQ1 — Prediction Accuracy

**Before changes:**
- BMA and MS show similar precision/recall/F1 values (small/negligible effect size)
- MS sometimes has slightly higher medians, but differences are not significant

**After changes:**
- When the environment changes (non-steady functions), BMA clearly outperforms MS:
  - Large effect sizes in **66%** (precision), **41%** (recall), and **50%** (F1) of the UAV requirements
  - BMA shows smaller statistical dispersion (i.e., more consistent predictions)

**Interpretation:** BMA's averaging over multiple models makes it more robust when the model structure changes.

---

### RQ2 — Adaptation Effectiveness

Measured the percentage of requirements satisfied after TUNE-II's adaptation decisions, compared with MS and RAND.

**Success Frequency:**
- **BMA ≫ MS ≫ RAND**
- UAV's median success frequency under BMA ≈ **1.0**
- Statistical tests show BMA significantly better than both baselines (p < 10⁻⁵)
- Effect size (Vargha–Delaney Â) = 0.58 (vs MS) → small-to-medium effect

**Adaptation Cost:**
- No significant difference between BMA, MS, RAND
- BMA occasionally incurs slightly higher cost (optimal solutions farther from start states)
- For UAV, cost differences were not statistically significant

**Interpretation:** BMA improves adaptation success without adding noticeable operational cost.

---

### RQ3 — Computational Cost

Measured wall-clock time to compute models (analyze phase):

| Subject | Median MS time / req. | Median BMA time / req. | Relative Difference |
|---------|----------------------|------------------------|---------------------|
| RR | ≈ 43 s | < 5 s | ≈ × 7 |
| UAV | ≈ 22 s | < 5 s | ≈ × 3 |

- All p-values = 1.45 × 10⁻¹¹, effect size = 1.0 → highly significant
- **→ BMA consistently cheaper computationally** because it avoids exhaustive model search

---

### RQ4 — Scalability

- Tested varying numbers of features (2 – 64) and observations (200 – 25,600)
- Compared ENUM (exhaustive) vs MCMC sampling for model averaging
- BMA remained under 200 s timeout even for large configurations
- **→ Scalable for both RR and UAV**, with MCMC showing much lower runtime for larger model spaces

---

## Summary of Findings for the UAV Case Study

| Aspect | Observation | Significance |
|--------|-------------|--------------|
| Mission & Scenario | UAV team balances target detection vs threat avoidance | Demonstrates complex trade-offs under uncertainty |
| Semantic Space | 7 factors (3 config + 4 env.) | Moderate-sized parameter space |
| Requirements | 12 derived from 2 templates | Captures multi-objective dependability |
| Accuracy | BMA ≫ MS after environmental changes | TUNE-II robust to model drift |
| Adaptation Success | BMA significantly higher success freq (≈ 1.0) | Effective self-adaptation |
| Adaptation Cost | Similar across methods; BMA slightly higher but insignificant | Cost-neutral improvement |
| Computation Cost | BMA ≈ 3× faster than MS (22 s → < 5 s) | Efficient |
| Scalability | Handles > 6,000 observations; MCMC efficient | Practical for runtime use |

---

## Overall Interpretation

The UAV team case study demonstrates that:

1. **Model averaging (BMA) provides resilient prediction and adaptation** when model structures evolve
2. **TUNE-II achieves higher mission reliability** (nearly full requirement satisfaction) without notable performance cost
3. **Computational efficiency makes BMA suitable** for runtime deployment in real-world autonomous UAV systems

---

## References

- **Main Paper:** Camilli et al. "Many-objective Self-adaptation under Model Uncertainty." ACM Transactions on Autonomous and Adaptive Systems (TAAS), 2025.
- **Original UAV Benchmark:** Moreno et al. [19]
- **Simulation Framework:** RUNE (parametric MDP mutators)
