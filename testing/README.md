# NeMDO Operator Analysis Framework

This repository provides a lightweight numerical framework to evaluate and compare mesh-free discrete differential operators on irregular point sets. It supports classical kernels (SPH, LABFM) as well as Neural Mesh-Free Differential Operators (NeMDO) trained offline and loaded at runtime.

The code computes first-order derivatives and Laplacians of a known analytical test function and provides tools to analyse:

- Stability (spectral analysis)
- Convergence (relative L2 error)
- Resolving power (effective wavenumber behaviour)

All experiments are controlled through a single entry point: `main.py`.

---
## Repository Structure
```text
.
├── main.py                     # Entry point
├── classes/
│   └── simulation.py           # Core experiment driver
├── functions/
│   ├── [SPH, LABFM, and NeMDO operators].py 
│   ├── plot.py                 # Plotting utilities
│   └── nodes.py                # Point set generation and disorder
├── models/
│   ├── trained_models/         # Pretrained NeMDO models (used in paper)
│   └── [architectures].py      # Architectures to import trained models
└── README.md
```

---
## Running the Code
All experiments are executed by running:
```bash
python main.py
```

The script:
- Generates particle sets of varying resolutions

- Applies selected differential operators

- Computes numerical errors

- Optionally produces plots for stability, convergence, and resolving power

---

## Operator Selection
The operator type is selected via the `kernel_list` argument passed to `run(...)`:
```python
kernel_list = ['gnn', 2, 'q_s', 'wc2']
```

Supported options:

| Identifier | Description                               |
| ---------- | ----------------------------------------- |
| `'gnn'`    | NeMDO (learned operator)                  |
| `'q_s'`    | Quintic spline SPH kernel                 |
| `'wc2'`    | Wendland C2 SPH kernel                    |
| `2,4,6,8`  | LABFM with corresponding polynomial order |

---
## Spatial Resolution
The spatial resolution is controlled via:
```python
total_nodes_list = [10, 20, 50, 100]
```

Each value corresponds to the number of nodes per spatial dimension.
The total number of particles is therefore: $N = n^2$, where `n = total_nodes`.


--- 
## Test Function

The test function is

$\phi(\tilde{x}, \tilde{y}) = 1.0 + (\tilde{x}\tilde{y})^4 + \sum_{n=1}^{6} (\tilde{x}^n + \tilde{y}^n)$,

with shifted coordinates  
$\tilde{x} = x - 0.1453$ and $\tilde{y} = y - 0.16401$.

The framework computes:
- $x$-derivative
- Laplacian

---
## Enabled Analyses

All plots are controlled by boolean flags in `main.py`:
```python
plot_ls = [True, True, True]

bool_plot_stability   = plot_ls[0]
bool_plot_convergence = plot_ls[1]
bool_plot_resolving_p = plot_ls[2]
```

---
## Recommended Experiment Setups

The behaviour of each diagnostic depends on how resolutions and kernels are combined.  
Below are **recommended configurations** for typical use cases.


### Convergence Analysis

For convergence studies, it is recommended to run **multiple resolutions with a single kernel**.

**Single-kernel convergence (recommended):**
```python
total_nodes_list = [10, 20, 50, 100]
kernel_list      = ['gnn'] * 4
```

This setup produces clean relative L2 error trends across resolutions.

### Multi-kernel convergence comparison:
```python
total_nodes_list = [10, 20, 50, 100] * 2
kernel_list      = ['gnn'] * 4 + ['wc2'] * 4
```
This allows direct comparison of convergence behaviour between different operators.

### Stability Analysis
Stability is evaluated via spectral analysis and does not require multiple resolutions.
A single moderately fine resolution is sufficient.

Recommended setup:
```python
total_nodes_list = [50]
kernel_list      = ['gnn']
```
Multi-kernel comparison:
```python
total_nodes_list = [50] * 2
kernel_list      = ['gnn', 'wc2']
```

### Resolving Power Analysis
Resolving power depends on the operator’s effective wavenumber behaviour and similarly requires only one resolution.

Recommended setup:
```python
total_nodes_list = [50]
kernel_list      = ['gnn']
```
Multi-kernel comparison:
```python
total_nodes_list = [50] * 2
kernel_list      = ['gnn', 'wc2']
```

---
## Pretrained NeMDO Models

All pretrained NeMDO models used in the paper are located in:

```text
models/trained_models/
```

### Models used in main experiments

| Paper notation | File |
|---------------|------|
| NeMDO$_{p=2}^x$ | `nemdo_x.pth` |
| NeMDO$_{p=2}^\Delta$ | `nemdo_lap.pth` |

### Models used in computational cost analysis

| Paper notation | File |
|---------------|------|
| NeMDO$_{p=2}^1$ | `nemdo_1.pth` |
| NeMDO$_{p=2}^2$ | `nemdo_2.pth` |
| NeMDO$_{p=2}^3$ | `nemdo_x.pth` |

---
## Particle Disorder Control
Particle disorder is introduced in `functions/nodes.py`. Specifically, through the function: `random_matrix(...)`.
This function enables controlled perturbations of the particle layout, allowing the robustness of the operators to be assessed on irregular point sets.