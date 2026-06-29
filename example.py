"""
PyXthreg Usage Example: 1-Threshold Model (Piecewise Linear Model)
This script demonstrates the estimation of a non-dynamic panel with fixed effects 
according to the Hansen (1999) methodology.
"""

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
model.fit(thnum=2, trim=0.05, grid=0, bs=300)

# ==========================================
# 4. INFERENCE AND RESULTS
# ==========================================
# Display the full regression results table in standard academic format.
model.summary()