# SIRA — Learning the SIR Epidemic Model
### GSoC 2026 · HumanAI Foundation · Self-Initiated Evaluation

> **Project:** Learning the Susceptible-Infected-Removed Model  
> **Umbrella:** HumanAI Foundation  
> **Mentors:** Harrison Prosper (FSU) · Olivia Prosper (UTK) · Sergei Gleyzer (UA)  
> **Applicant:** Mohammed Saba · GGITS Jabalpur · Minor AI/ML @ IIT Guwahati  

---

## What This Project Is About

The SIR (Susceptible-Infected-Removed) model is one of the foundational models in epidemiology. It describes how a disease spreads through a population using three coupled ordinary differential equations (ODEs):

```
dS/dt = -β·S·I/N
dI/dt =  β·S·I/N - γ·I
dR/dt =  γ·I
```

Where:
- **S** = number of susceptible individuals
- **I** = number of infected individuals  
- **R** = number of removed (recovered/deceased) individuals
- **β** = infection rate, **γ** = recovery rate, **N** = total population
- **R₀ = β/γ** = basic reproduction number (epidemic spreads if R₀ > 1)

The **deterministic ODE** gives smooth, average trajectories. But real epidemics are **stochastic** — each infection and recovery event happens by chance. This project bridges that gap: simulate many stochastic epidemics, and train a machine learning model to learn the mean (deterministic) behaviour purely from the noisy stochastic data.

---

## Repository Structure

```
project_root/
├── notebook/
│   └── SIRA.ipynb                     ← main notebook (source)
│
├── model_output_pdf/                   
│   ├── SIRA.pdf                       ← final PDF
│
└── README.md                          ← documentation (ADD THIS)          # This file
```

---

## Pipeline Overview

The notebook is structured in 7 sections:

```
Section 1: SIR model definitions (ODE solver + Gillespie algorithm)
    ↓
Section 2: Sanity check — does stochastic mean → deterministic ODE?
    ↓
Section 3: Dataset generation — simulate across (β, γ) parameter grid
    ↓
Section 4: MLP architecture — (t, β, γ) → (S/N, I/N, R/N)
    ↓
Section 5: Training — 300 epochs, Adam + Cosine Annealing LR
    ↓
Section 6: Evaluation — generalization to unseen (β, γ) pairs
    ↓
Section 7: Quantitative error — MAE on unseen parameters
```

---

## Section 1 — The Gillespie Algorithm

The **Gillespie algorithm** (also called the Stochastic Simulation Algorithm, SSA) simulates the exact stochastic dynamics of the SIR model. At each step:

1. Compute two competing event rates:
   - Infection rate: `r_infect = β · S · I / N`
   - Recovery rate: `r_recover = γ · I`
2. Sample waiting time until next event: `dt ~ Exponential(1 / total_rate)`
3. Select which event occurs with probability proportional to its rate
4. Update S, I, R accordingly

This is an **exact** simulation — no approximation, no discretisation. Each run is a single possible realisation of the epidemic.

---

## Section 2 — Sanity Check

**The Law of Large Numbers** tells us that as the number of stochastic simulations grows, their average converges to the deterministic ODE solution.

We verify this empirically by running 200 independent Gillespie simulations at β=0.3, γ=0.1 (R₀=3.0, N=1000) and comparing the ensemble mean to the ODE solution.

**Result:** The stochastic mean (coloured lines) tracks the deterministic ODE (black dashed) with near-perfect agreement. Individual trajectories show realistic random variation around this mean.

![Sanity Check](sir_ensemble.png)

---

## Section 3 — Dataset Generation

To train a model that generalises across epidemic parameters, we simulate across a **6×6 parameter grid**:

| Parameter | Values |
|-----------|--------|
| β (infection rate) | 0.15, 0.20, 0.25, 0.30, 0.35, 0.40 |
| γ (recovery rate) | 0.05, 0.08, 0.10, 0.12, 0.15, 0.20 |

Only pairs where **R₀ = β/γ > 1** are kept (33 pairs) — below this threshold, the epidemic dies out without spreading and there is no interesting dynamics to learn.

For each (β, γ) pair:
- Run 150 stochastic Gillespie simulations
- Compute ensemble mean S(t), I(t), R(t) on a uniform time grid t = 0, 1, ..., 160
- Store one row per time step: `[t, β, γ, S_mean, I_mean, R_mean]`

**Dataset size:** ~5,300 rows × 6 columns

**Normalisation:**
- `t → t / 160` (maps to [0, 1])
- `S, I, R → S/N, I/N, R/N` (maps to [0, 1])
- β and γ are kept as raw floats (already small)

**Train/Val split:** 80/20, randomly shuffled.

---

## Section 4 — MLP Architecture

The model is a **Multilayer Perceptron (MLP)** with the following design:

```
Input:  (t_norm, β, γ)          — 3 features
         ↓
Linear(3 → 128) + Tanh
         ↓
Linear(128 → 128) + Tanh        ×3 (4 hidden layers total)
         ↓
Linear(128 → 3) + Sigmoid
         ↓
Output: (S/N, I/N, R/N)         — 3 values in [0,1]
```

**Architecture choices explained:**

- **Tanh activations** — smooth, unbounded, well-suited to learning continuous physical functions. ReLU would introduce kinks inappropriate for ODE trajectories.
- **Sigmoid output** — enforces outputs in [0, 1], consistent with normalised population fractions. Without this, the network could predict negative counts.
- **4 hidden layers, 128 neurons** — sufficient capacity to capture the nonlinear dynamics across the full (β, γ) parameter space without overfitting to ~5,300 training points.
- **Total parameters: 50,435** — lightweight, trains in under 5 minutes on CPU.

---

## Section 5 — Training

| Setting | Value |
|---------|-------|
| Loss function | MSELoss |
| Optimiser | Adam (lr=1e-3) |
| LR schedule | Cosine Annealing (T_max=300) |
| Batch size | 512 |
| Epochs | 300 |

**Results:**

| Epoch | Train MSE | Val MSE |
|-------|-----------|---------|
| 50 | 0.001420 | 0.001357 |
| 100 | 0.000439 | 0.000402 |
| 150 | 0.000147 | 0.000154 |
| 200 | 0.000085 | 0.000086 |
| 250 | 0.000065 | 0.000064 |
| 300 | 0.000062 | 0.000060 |

Train and validation loss track each other closely throughout — **no overfitting**. Final Val MSE: **0.000060**.

![Loss Curve](loss_curve.png)

---

## Section 6 & 7 — Evaluation on Unseen Parameters

The critical test: evaluate on **4 (β, γ) pairs that were never in the training grid**. The model must generalise — it cannot have memorised these curves.

| (β, γ) | R₀ | MAE\_S | MAE\_I | MAE\_R |
|--------|-----|--------|--------|--------|
| (0.28, 0.09) | 3.1 | 6.18 | 8.03 | 8.04 |
| (0.18, 0.07) | 2.6 | 7.89 | 7.69 | 8.86 |
| (0.38, 0.13) | 2.9 | 5.46 | 5.80 | 5.24 |
| (0.22, 0.11) | 2.0 | 8.67 | 6.06 | 9.07 |

All MAE values are **below 10 counts out of N=1000 (< 1% error)**. The I (Infected) compartment shows slightly higher error near the epidemic peak — expected, as the peak is the sharpest nonlinear feature in the trajectory.

![Evaluation](mlp_evaluation.png)

---

## How to Run

**Requirements:**
```
numpy
scipy
matplotlib
tqdm
torch
```

Install with:
```bash
pip install numpy scipy matplotlib tqdm torch
```

**Run the notebook:**
```bash
jupyter notebook SIRA_evaluation.ipynb
```

Run cells in order. Dataset generation takes ~2–3 minutes on CPU. Training takes ~3–5 minutes on CPU, under 1 minute on GPU.

**Expected runtime (CPU):** ~8 minutes end-to-end.

---

## Key Results Summary

| Metric | Value |
|--------|-------|
| Final Val MSE | 0.000060 |
| Max MAE on unseen params (any compartment) | 9.07 / 1000 = 0.9% |
| Overfitting | None (train ≈ val throughout) |
| Generalises to unseen (β, γ)? | Yes |
| Training time (CPU) | ~5 minutes |

---

## Connection to the GSoC Project

This evaluation covers the first two of the three expected project deliverables:

1. **Simulated SIR epidemic model** ✅ — Gillespie algorithm, ensemble simulation, ODE comparison
2. **Trained ML model to predict mean counts** ✅ — MLP achieving < 1% MAE on unseen parameters

The third deliverable — **symbolic ML / auto-differentiation to approximate S(t), I(t), R(t)** — is the planned summer work. The foundation built here (stochastic simulator + learned mean predictor) is the direct prerequisite for that phase.

---

*Mohammed Saba · GSoC 2026 · HumanAI Foundation · GGITS Jabalpur*
