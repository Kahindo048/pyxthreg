"""
PyXthreg: High-Performance Panel Threshold Regression
===================================================

PyXthreg is an optimized Python package for estimating fixed-effects panel data models 
with multiple thresholds, strictly adhering to the methodology of Hansen (1999).

Key Features
------------
- Ultra-fast Numba/C-compiled within-transformation and estimation engine.
- Sequential threshold search with Hansen's Refinement for asymptotic consistency.
- Autonomous selection of the optimal number of regimes via bootstrap (thnum="auto").
- Stata-matched statistical inference, degrees of freedom, and variance components.
- Publication-ready LaTeX exports (including multi-model Stargazer tables).
- Academic-quality diagnostic plotting (Elbow plots, regime distribution, and dynamics).

How to use
----------
>>> import pandas as pd
>>> from pyxthreg import ThresholdPanel, PanelStargazer
>>> df = pd.read_stata("panel_data.dta")
>>> model = ThresholdPanel(data=df, dep='y', indep=['x1'], rx=['rx'], qx='q', entity_col='id', time_col='year')
>>> model.fit(thnum="auto", trim=0.05, grid=300, bs=300)
>>> model.summary()
"""

__version__ = "0.1.0"
__author__ = "ThinkBit Edge Inc. / Rukara Kahindo"

# Expose the main classes to the package's top level
from .estimator import ThresholdPanel
from .stargazer import PanelStargazer
from .load_data import list_datasets, load_dataset

# Restrict what is imported when a user runs `from pyxthreg import *`
__all__ = [
    "ThresholdPanel",
    "PanelStargazer",
    "list_datasets",
    "load_dataset"
]