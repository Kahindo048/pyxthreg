# PyXthreg: High-Performance Panel Threshold Regression in Python

[![PyPI version](https://badge.fury.io/py/pyxthreg.svg)](https://badge.fury.io/py/pyxthreg)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`pyxthreg` is a highly optimized Python package for estimating fixed-effects panel threshold models, originally pioneered by Hansen (1999). 

Built from the ground up for massive empirical datasets, it replicates the mathematical exactness of the historical Stata module `xthreg` (Wang, 2015) while delivering a **multifold speedup** by circumventing the Python Global Interpreter Lock (GIL) via JIT compilation (Numba) and multi-core parallelization.

Ideal for applied econometrics and macroeconomic research, this package modernizes regime-switching modeling within the Python data science ecosystem.

---
## 🌟 Key Features

* **Absolute Stata Parity:** Replicates point estimates, standard errors, and Hansen's sequential Likelihood Ratio (LR) bootstrap tests with exact mathematical precision.
* **Autonomous Regime Discovery:** Features an intelligent sequential algorithm (`thnum="auto"`) that dynamically searches for an arbitrary number of $K$ thresholds, strictly halting when additional structural breaks lose statistical significance.
* **Massive Speedup (Numba JIT):** Executes the computationally heavy residual-based bootstrap iterations concurrently across all available CPU cores, reducing execution times from hours to seconds.
* **Native Robust Inference:** Supports cluster-robust Sandwich variance-covariance estimators (`robust=True`) to seamlessly correct for heteroskedasticity and intra-group serial correlation.
* **Memory-Efficient Fixed Effects:** Natively applies a two-way partial within-transformation (`time_fe=True`) via the Frisch-Waugh-Lovell theorem, avoiding the creation of memory-heavy dummy variables.
* **Publication-Ready Visualizations:** Built-in methods to generate the classic Hansen LR V-shaped confidence intervals, SSR evolution plots, and dynamic regime transition charts.

---
## 📦 Installation

The stable release is available on the Python Package Index (PyPI). Install it using `pip`:

```bash
pip install pyxthreg
```

For export to Word/Excel and example datasets:

```bash
pip install "pyxtabond2[export]"
```

Development install from source:

```bash
git clone [https://github.com/Kahindo048/pyxthreg.git](https://github.com/Kahindo048/pyxthreg.git)
cd pyxthreg
pip install -e .
```
🚀 Quick Start
The API is designed to be intuitive and strictly requires a standard "long format" pandas DataFrame.

```python
import pandas as pd
from pyxthreg.estimator import ThresholdPanel
from pyxthreg.load_data import load_dataset

# ==========================================
# 1. DATA LOADING
# ==========================================
# Using the package's utility function to load the test panel.
# This dataset contains a strongly balanced panel.
try:
    df = load_dataset("model_1.dta")
    print(f"Data loaded successfully: {df.shape[0]} observations.")
except FileNotFoundError:
    print("Error: The file 'model_1.dta' could not be found.")

# ==========================================
# 2. ECONOMETRIC MODEL SPECIFICATION
# ==========================================
# Instantiating the model with the panel data structure.
model = ThresholdPanel(
    data=df, 
    dep='y',             # Dependent variable (Y)
    indep=['x1', 'x2'],  # Control variables (regime-independent)
    rx=['rx1'],          # Regime-dependent variable(s) (subject to structural break)
    qx='q',              # Endogenous threshold variable determining the transition
    entity_col='id',     # Cross-sectional identifier (e.g., countries, firms)
    time_col='year'      # Time-series identifier (e.g., years)
)

# ==========================================
# 3. MODEL ESTIMATION AND BOOTSTRAP
# ==========================================
# Executing the search engine and simulating the asymptotic distribution.
# 
# Parameters:
# - thnum=1   : Forces the estimation of a single threshold.
# - trim=0.05 : Trims 5% of observations at the extremes to ensure 
#               matrix invertibility within each regime.
# - grid=0    : Perfect exhaustive search over exact values 
#               (eliminates the grid interpolation error found in legacy software).
# - bs=300    : 300 replications for the residual bootstrap (computes P-values).
model.fit(thnum=1, trim=0.05, grid=0, bs=300)

# ==========================================
# 4. INFERENCE AND RESULTS
# ==========================================
# Display the full regression results table in standard academic format.
model.summary()
```

See `examples/example.py`

## Performance

Because non-dynamic threshold modeling relies on intensive grid searches and massive residual-based bootstrapping, computational speed is paramount. pyxthreg solves this via a hybrid Python/C architecture.

In standardized benchmarking (300 bootstrap replications, 300 grid points) against Stata's xthreg on a panel of 20,000 observations, pyxthreg completes the estimation in ~32 seconds, compared to over 125 seconds in legacy software (a nearly 4x speedup).

---
## 📖 References & Methodology
This package implements the algorithms and corrections outlined in the following seminal papers:

Hansen, B. E. (1999). Threshold effects in non-dynamic panels: Estimation, testing, and inference. Journal of Econometrics, 93(2), 345-368.

Hansen, B. E. (2000). Sample splitting and threshold estimation. Econometrica, 68(3), 575-603.

Davies, R. B. (1977). Hypothesis testing when a nuisance parameter is present only under the alternative. Biometrika, 64(2), 247-254.

Wang, Q. (2015). Fixed-effect panel threshold model using Stata. The Stata Journal, 15(1), 121-131.
---

## 🤝 Contributing
Contributions, issues, and feature requests are highly welcome! Because this package utilizes numba JIT compilation, please ensure that any modifications to the core engine inside _computation.py strictly adhere to nopython constraints. Feel free to check the issues page on the GitHub repository.

## License

MIT License. See `LICENSE` if included in the distribution.
