import numpy as np
import pandas as pd

def check_balanced_panel(df, entity_col, time_col):
    """
    Checks if the panel data is strictly balanced.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The panel dataset.
    entity_col : str
        The name of the column representing individual entities (e.g., ID).
    time_col : str
        The name of the column representing the time periods.
        
    Returns
    -------
    N : int
        The number of unique entities.
    T : int
        The number of time periods per entity.
        
    Raises
    ------
    ValueError
        If the panel is unbalanced (entities have different numbers of observations).
    """
    # Count the number of unique time periods for each entity
    obs_per_entity = df.groupby(entity_col)[time_col].nunique()
    
    # Take the T of the first individual as the reference benchmark
    T = obs_per_entity.iloc[0]
    
    # Verify that all entities have exactly T observations
    if not (obs_per_entity == T).all():
        raise ValueError(
            "The panel is unbalanced. "
            "Hansen's threshold algorithm requires a strongly balanced panel. "
            "Please clean and balance your dataset before estimation."
        )
    
    N = len(obs_per_entity)
    return N, T

def check_time_varying(df, cols, entity_col):
    """
    Checks that the explanatory variables exhibit time variation.
    
    Time-invariant variables are perfectly collinear with individual fixed effects
    and will cause a matrix singularity during the within-transformation.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The panel dataset.
    cols : list of str
        List of explanatory variables to check.
    entity_col : str
        The name of the column representing individual entities.
        
    Raises
    ------
    ValueError
        If any variable in `cols` is time-invariant for all entities.
    """
    if not cols:
        return
        
    # Check the number of unique values per individual for each variable.
    # If the maximum number of unique values for an individual is 1, the variable is invariant.
    nunique_per_entity = df.groupby(entity_col)[cols].nunique()
    max_unique = nunique_per_entity.max()
    
    invariant_cols = max_unique[max_unique == 1].index.tolist()
    
    if invariant_cols:
        raise ValueError(
            f"Time-invariant variables detected: {invariant_cols}. "
            "These cannot be estimated with individual fixed effects (Within estimator). "
            "Please remove them from the model specification."
        )

def extract_matrices(df, dep, indep, rx, qx, entity_col, time_col, precision=np.float64):
    """
    Extracts and prepares contiguous NumPy matrices for the Numba computation engine.
    
    This function is critical for memory performance. It sorts the panel and forces
    contiguous memory allocation, allowing maximum CPU cache utilization.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The panel dataset.
    dep : str
        The name of the dependent variable.
    indep : list of str
        List of regime-independent variables.
    rx : list of str
        List of regime-dependent variables.
    qx : str
        The name of the threshold (transition) variable.
    entity_col : str
        The name of the column representing individual entities.
    time_col : str
        The name of the column representing the time periods.
    precision : numpy.dtype, optional
        The numerical precision for the matrices, by default np.float64.
        
    Returns
    -------
    Y : numpy.ndarray
        Dependent variable matrix of shape (N*T, 1).
    X : numpy.ndarray
        Regime-independent variables matrix of shape (N*T, K_X).
    R : numpy.ndarray
        Regime-dependent variables matrix of shape (N*T, K_R).
    Q : numpy.ndarray
        Threshold variable array (flattened) of shape (N*T,).
    N : int
        The number of entities.
    T : int
        The number of time periods.
    """
    # 1. Strict sorting: Essential to implicitly maintain the (N x T) 
    # structure across 1D and 2D arrays without needing explicit multi-indexing.
    df_sorted = df.sort_values(by=[entity_col, time_col]).copy()
    
    # 2. Basic integrity checks
    N, T = check_balanced_panel(df_sorted, entity_col, time_col)
    check_time_varying(df_sorted, indep + rx, entity_col)
    
    # 3. Raw matrix extraction
    # np.ascontiguousarray forces the data to be stored in adjacent memory blocks.
    # This is a prerequisite for Numba to execute C-level loop optimizations efficiently.
    
    Y = np.ascontiguousarray(df_sorted[dep].values, dtype=precision).reshape(-1, 1)
    
    # If regime-independent variables are provided
    if indep:
        X = np.ascontiguousarray(df_sorted[indep].values, dtype=precision)
    else:
        # Fallback if only regime-dependent variables exist in the model
        X = np.empty((N * T, 0), dtype=precision)
        
    R = np.ascontiguousarray(df_sorted[rx].values, dtype=precision)
    Q = np.ascontiguousarray(df_sorted[qx].values, dtype=precision).flatten()
    
    return Y, X, R, Q, N, T