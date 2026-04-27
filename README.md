# ISE619-End-to-end-Optimization-Project
# Data-Efficient DFL (Shortest Path) — Usage Guide

This repository runs stream-based(Pool-based are almost the same) active learning experiments for decision-focused learning (DFL).

---

## 1. Install Dependencies

pip install torch numpy matplotlib scipy

---

## 2. Run the Code

### Minimal run (default settings)

from your_file import run_experiment  
run_experiment()

or run it in google colab/jupyter notebook
---

## 3. Recommended Setup (standard experiment)

run_experiment(
    mode='5x5',
    methods=['supervise', 'uncertainty', 'margin', 'flip', 'regret'],
    stream_clustering=1.0,
    n_runs=5,
    online_mode=True,
)

---

## 4. Key Parameters

### (1) mode — experiment scale

quick   → fast debugging  
3x3     → default experiment  
5x5     → larger scale  
custom  → fully customizable  

---

### (2) methods — algorithms to run

['supervise', 'uncertainty', 'margin', 'flip', 'regret']

Run a subset:

methods=['supervise', 'margin']

---

### (3) stream_clustering — controls stream redundancy (important)

0.0 → i.i.d. (fully shuffled)  
1.0 → fully clustered (high redundancy)  

---

### (4) budget — labeling budget

budget=25

---

### (5) n_runs — number of repetitions

n_runs=25

---

### (6) online_mode — training style

True  → one SGD step per labeled sample (recommended)  
False → mini-batch retraining  

---

## 5. Outputs

The code automatically generates:

results_*.png        (regret curves)  
accuracy_*.png       (accuracy curves)  
scans_*.png          (number of scanned samples)  
flip_vs_margin_*.png  

Terminal output includes:

- final regret / accuracy
- method comparison
- average scanned samples

---

# Full `run_experiment` Parameter Reference

Below is the complete signature and description of all parameters.

---

## Function Signature

run_experiment(
    mode='3x3',
    grid_size=None, p=None, noise_level=None, deg=None,
    n_init=None, n_pool=None, n_test=None, budget=None,
    margin_threshold=0.1,
    hidden_dim=None, model_type='linear',
    init_lr=0.01, init_epochs=10,
    update_lr=0.05, update_steps=3,
    online_mode=True,
    n_runs=None, M=None, methods=None,
    q_tilde=0.5,
    prior_precision=1.0, laplace_noise_var=1.0,
    max_scan=None,
    stream_clustering=0.0,
    device='cpu', save_prefix='', verbose=True,
)

# run_experiment Parameter Explanation

---

## 1. Experiment Preset

### mode

Controls a predefined configuration.

- `quick`  → very small setup for debugging  
- `3x3`    → default experiment  
- `5x5`    → larger problem (slower)  
- `custom` → fully manual configuration  

If you do not override parameters, values are taken from this preset.

---

## 2. Problem / Data Generation

### grid_size

Size of the shortest-path grid.

- `3` → 3×3 grid  
- `5` → 5×5 grid  

Larger grid → harder optimization problem.

---

### p

Feature dimension of input `x`.

Higher `p` → harder prediction task.

---

### noise_level

Controls multiplicative noise in cost generation.

- `0.0`  → no noise  
- `0.25` → moderate noise (default)  

Higher noise → harder learning.

---

### deg

Degree of nonlinearity in the cost function:

\[
c(x) = 1 + (1 + x B^T)^{deg}
\]

- `1`  → nearly linear  
- `>1` → more nonlinear  

---

## 3. Dataset Split

### n_init

Number of initial labeled samples (warm start)

---

### n_pool

Number of unlabeled samples in the stream

---

### n_test

Number of test samples for evaluation

---

### budget

Maximum number of samples allowed to be labeled during active learning.

This is the main resource constraint.

---

## 4. Model

### model_type

- `linear` → linear model (default, fast and stable)  
- other    → MLP (2-layer neural network)

---

### hidden_dim

Hidden layer size (only used for MLP)

---

## 5. Training

### Initial training

- `init_lr` → learning rate  
- `init_epochs` → number of epochs  

Used on the initial labeled dataset.

---

### Update training

- `update_lr` → learning rate after each new label  
- `update_steps` → number of steps (used if `online_mode=False`)

---

### online_mode

- `True`  → one SGD step per new sample (recommended)  
- `False` → mini-batch retraining (slower but smoother)

---

## 6. Active Learning

### methods

List of methods to run:

- `supervise`   → no selection (baseline)  
- `uncertainty` → variance-based  
- `margin`      → decision boundary  
- `flip`        → decision instability (ours)  
- `regret`      → expected decision loss (ours)  

---

### q_tilde

Quantile used for thresholding selection.

- margin → select low-margin samples  
- flip/regret → select high-score samples  

Examples:

- `0.5` → median threshold  
- `0.2` → more selective  

---

### M

Number of Monte Carlo samples used in:

- flip probability  
- expected regret  

Higher `M` → more accurate but slower.

---

### margin_threshold

Used only in data generation to ensure class separability.

Not part of the learning algorithm.

---

## 7. Uncertainty (Laplace Approximation)

Used for `uncertainty`, `flip`, and `regret`.

### prior_precision

Strength of prior in Laplace approximation

---

### laplace_noise_var

Assumed observation noise variance

---

## 8. Stream Control (Important)

### stream_clustering

Controls redundancy in the stream:

- `0.0` → fully shuffled (i.i.d.)  
- `1.0` → fully clustered  
- between → Markov-chain clustering  

This is a key experimental knob.

---

### max_scan

Maximum number of samples scanned before forcing labeling.

- `None` → no limit  
- integer → prevents excessive scanning  

---

## 9. System

### device

- `cpu`  
- `cuda`  

---

### save_prefix

Prefix for output file names
