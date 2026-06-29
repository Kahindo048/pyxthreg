import numpy as np
from numba import njit
from .search import get_grid
from .core import transform_matrix_partial

@njit(fastmath=True, nogil=True)
def search_next_threshold(Y_within, X_within, R, Q, N, T, grid, existing_gammas, trim_percent, time_fe):
    """
    Searches for an additional threshold conditional on the existing thresholds.
    
    Parameters
    ----------
    Y_within : numpy.ndarray
        The within-transformed dependent variable.
    X_within : numpy.ndarray
        The within-transformed regime-independent variables.
    R : numpy.ndarray
        The raw regime-dependent variables.
    Q : numpy.ndarray
        The threshold variable array.
    N : int
        Number of entities.
    T : int
        Number of time periods.
    grid : numpy.ndarray
        The grid of candidate threshold values.
    existing_gammas : numpy.ndarray
        Array of previously identified thresholds.
    trim_percent : float
        The trimming percentage ensuring sufficient regime size.
    time_fe : bool
        Whether to apply two-way fixed effects transformations.
        
    Returns
    -------
    best_gamma : float
        The optimal candidate threshold.
    min_ssr : float
        The minimum Residual Sum of Squares (SSR).
    """
    n_obs = N * T
    min_obs = int(n_obs * trim_percent) 
    
    K_X = X_within.shape[1]
    K_R = R.shape[1]
    n_test_gammas = len(existing_gammas) + 1
    n_regimes = n_test_gammas + 1
    K_total = K_X + n_regimes * K_R
    
    # Pre-allocation for grid iterations
    Z = np.empty((n_obs, K_total), dtype=np.float64)
    ZtZ = np.zeros((K_total, K_total), dtype=np.float64)
    ZtY = np.zeros((K_total, 1), dtype=np.float64)
    
    best_gamma = grid[0]
    min_ssr = np.inf
    
    for idx in range(len(grid)):
        gamma_cand = grid[idx]
        
        # 1. Build and sort candidate thresholds
        test_gammas = np.empty(n_test_gammas, dtype=np.float64)
        for j in range(len(existing_gammas)):
            test_gammas[j] = existing_gammas[j]
        test_gammas[-1] = gamma_cand
        test_gammas = np.sort(test_gammas)
        
        # 2. Strict verification of regime sample sizes
        counts = np.zeros(n_regimes, dtype=np.int32)
        for i in range(n_obs):
            q_val = Q[i]
            r_idx = 0
            for g in range(n_test_gammas):
                if q_val > test_gammas[g]:
                    r_idx += 1
                else:
                    break
            counts[r_idx] += 1
            
        valid = True
        for c in counts:
            if c < min_obs:
                valid = False
                break
        
        if not valid:
            continue
            
        # 3. Matrix Z allocation (splitting regimes)
        for i in range(n_obs):
            for k in range(K_X):
                Z[i, k] = X_within[i, k]
            
            q_val = Q[i]
            r_idx = 0
            for g in range(n_test_gammas):
                if q_val > test_gammas[g]:
                    r_idx += 1
                else:
                    break
                    
            for reg in range(n_regimes):
                if reg == r_idx:
                    for k in range(K_R):
                        Z[i, K_X + reg * K_R + k] = R[i, k]
                else:
                    for k in range(K_R):
                        Z[i, K_X + reg * K_R + k] = 0.0
                        
        # 4. Ultra-fast partial transformation (One-Way or Two-Way)
        Z_within = transform_matrix_partial(Z, N, T, K_X, time_fe)
                    
        # 5. OLS and SSR computation
        for j in range(K_total):
            ZtY[j, 0] = 0.0
            for l in range(K_total):
                ZtZ[j, l] = 0.0
        
        for i in range(n_obs):
            y_val = Y_within[i, 0]
            for j in range(K_total):
                z_val = Z_within[i, j]
                ZtY[j, 0] += z_val * y_val
                for l in range(K_total):
                    ZtZ[j, l] += z_val * Z_within[i, l]
                    
        beta = np.linalg.solve(ZtZ, ZtY)
        
        ssr = 0.0
        for i in range(n_obs):
            pred = 0.0
            for k in range(K_total):
                pred += Z_within[i, k] * beta[k, 0]
            resid = Y_within[i, 0] - pred
            ssr += resid * resid
            
        if ssr < min_ssr:
            min_ssr = ssr
            best_gamma = gamma_cand
            
    return best_gamma, min_ssr

@njit(fastmath=True, nogil=True)
def fit_sequential_thresholds(Y_within, X_within, R, Q, N, T, trim_percent, grid_size, max_thresholds, time_fe):
    """
    Executes the complete Hansen sequential algorithm to identify N thresholds.
    """
    grid = get_grid(Q, trim_percent, grid_size)
    thresholds = np.zeros(max_thresholds, dtype=np.float64)
    ssrs = np.zeros(max_thresholds, dtype=np.float64)
    
    for k in range(max_thresholds):
        if k == 0:
            empty_gammas = np.zeros(0, dtype=np.float64)
            best_g, min_s = search_next_threshold(Y_within, X_within, R, Q, N, T, grid, empty_gammas, trim_percent, time_fe)
            thresholds[0] = best_g
            ssrs[0] = min_s
        else:
            existing = thresholds[:k]
            best_g, min_s = search_next_threshold(Y_within, X_within, R, Q, N, T, grid, existing, trim_percent, time_fe)
            thresholds[k] = best_g
            
            # Hansen's Refinement
            for r_idx in range(k):
                fixed_gammas = np.empty(k, dtype=np.float64)
                c = 0
                for j in range(k + 1):
                    if j != r_idx:
                        fixed_gammas[c] = thresholds[j]
                        c += 1
                        
                refined_g, refined_s = search_next_threshold(Y_within, X_within, R, Q, N, T, grid, fixed_gammas, trim_percent, time_fe)
                thresholds[r_idx] = refined_g
                min_s = refined_s 
                
            ssrs[k] = min_s
            
    return thresholds, ssrs