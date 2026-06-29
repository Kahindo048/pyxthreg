import numpy as np
from numba import njit
from joblib import Parallel, delayed
from .search import get_grid
from .sequential import search_next_threshold
from .core import transform_matrix_partial

# NOGIL=TRUE ALLOWS MULTI-THREADING BYPASSING PYTHON'S GIL
@njit(fastmath=True, nogil=True)
def compute_ssr_fixed(Y_within, X_within, R, Q, N, T, fixed_gammas, time_fe):
    """
    Estimates the model under the null hypothesis H0 (Fixed thresholds).
    
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
    fixed_gammas : numpy.ndarray
        Array of fixed threshold values to condition the estimation on.
    time_fe : bool
        If True, applies Two-Way fixed effects transformation.
        
    Returns
    -------
    ssr : float
        The Sum of Squared Residuals for the restricted model.
    Y_pred : numpy.ndarray
        The predicted values of the dependent variable under H0.
    residuals : numpy.ndarray
        The array of model residuals under H0.
    """
    n_obs = N * T
    K_X = X_within.shape[1]
    K_R = R.shape[1]
    n_regimes = len(fixed_gammas) + 1
    K_total = K_X + n_regimes * K_R

    Z = np.empty((n_obs, K_total), dtype=np.float64)
    ZtZ = np.zeros((K_total, K_total), dtype=np.float64)
    ZtY = np.zeros((K_total, 1), dtype=np.float64)

    test_gammas = np.sort(fixed_gammas)

    # Populate Z matrix with regime splitting
    for i in range(n_obs):
        for k in range(K_X):
            Z[i, k] = X_within[i, k]
        q_val = Q[i]
        r_idx = 0
        for g in range(len(test_gammas)):
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

    # Transformation Partielle ultra-rapide (One-Way ou Two-Way)
    Z_within = transform_matrix_partial(Z, N, T, K_X, time_fe)

    # Fast OLS computation
    for j in range(K_total):
        for l in range(K_total):
            ZtZ[j, l] = 0.0
        ZtY[j, 0] = 0.0

    for i in range(n_obs):
        y_val = Y_within[i, 0]
        for j in range(K_total):
            z_val = Z_within[i, j]
            ZtY[j, 0] += z_val * y_val
            for l in range(K_total):
                ZtZ[j, l] += z_val * Z_within[i, l]

    beta = np.linalg.solve(ZtZ, ZtY)

    ssr = 0.0
    Y_pred = np.empty((n_obs, 1), dtype=np.float64)
    residuals = np.empty((n_obs, 1), dtype=np.float64)

    for i in range(n_obs):
        pred = 0.0
        for k in range(K_total):
            pred += Z_within[i, k] * beta[k, 0]
        Y_pred[i, 0] = pred
        resid = Y_within[i, 0] - pred
        residuals[i, 0] = resid
        ssr += resid * resid

    return ssr, Y_pred, residuals

@njit(fastmath=True, nogil=True)
def _boot_iteration(Y_pred_H0, resid_H0, X_within, R, Q, N, T, grid, fixed_gammas, trim_percent, eta, time_fe):
    """
    Compiled iteration for the bootstrap procedure, mimicking the Stata DGP process (Wang, 2015).
    """
    n_obs = N * T
    Y_boot_raw = np.empty((n_obs, 1), dtype=np.float64)

    # 1. Stata Method: Y_boot = residuals * N(0,1) noise, observation by observation
    for t in range(n_obs):
        Y_boot_raw[t, 0] = resid_H0[t, 0] * eta[t]
        
    # 2. IMPERATIVE: Recenter Y_boot (Within Transformation)
    # The random noise breaks the zero-mean property of residuals per group.
    # En appliquant start_col=0, la transformation cible toute la matrice (l'unique colonne Y)
    Y_boot = transform_matrix_partial(Y_boot_raw, N, T, 0, time_fe)

    # 3. Estimations on the Bootstrap sample
    ssr_H0, _, _ = compute_ssr_fixed(Y_boot, X_within, R, Q, N, T, fixed_gammas, time_fe)
    _, ssr_H1 = search_next_threshold(Y_boot, X_within, R, Q, N, T, grid, fixed_gammas, trim_percent, time_fe)

    return ssr_H0, ssr_H1

def test_threshold_effect(Y_within, X_within, R, Q, N, T, grid, fixed_gammas, trim_percent, n_boot=300, n_jobs=-1, time_fe=False):
    """
    Performs the Hansen (1999) bootstrap test for threshold effect significance.
    """
    ssr_H0_true, Y_pred_H0, resid_H0 = compute_ssr_fixed(Y_within, X_within, R, Q, N, T, fixed_gammas, time_fe)
    _, ssr_H1_true = search_next_threshold(Y_within, X_within, R, Q, N, T, grid, fixed_gammas, trim_percent, time_fe)

    # --- HISTORICAL STATA ADJUSTMENT ---
    df_penalty = (T - 1) if time_fe else 0
    df_boot = (N * T) - T - df_penalty
    
    sigma2_H1_true = ssr_H1_true / df_boot
    F_true = (ssr_H0_true - ssr_H1_true) / sigma2_H1_true

    np.random.seed(42)
    # Generate N(0,1) noise for the entire matrix across all bootstrap iterations
    etas = np.random.normal(0, 1, size=(n_boot, N * T))

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_boot_iteration)(
            Y_pred_H0, resid_H0, X_within, R, Q, N, T, grid, fixed_gammas, trim_percent, etas[m], time_fe
        ) for m in range(n_boot)
    )

    F_boots = np.zeros(n_boot)
    for m in range(n_boot):
        s0_m, s1_m = results[m]
        sig2_1_m = s1_m / df_boot
        F_boots[m] = (s0_m - s1_m) / sig2_1_m

    p_value = np.mean(F_boots >= F_true)
    crit10, crit5, crit1 = np.percentile(F_boots, [90, 95, 99])
    
    return F_true, p_value, crit10, crit5, crit1

def bootstrap_all_thresholds(Y_within, X_within, R, Q, N, T, thresholds, trim_percent, grid_size=300, n_boot=300, n_jobs=-1, time_fe=False):
    """
    Executes the sequential bootstrap test for all identified thresholds.
    """
    grid = get_grid(Q, trim_percent, grid_size)
    results = []

    for k in range(len(thresholds)):
        fixed = thresholds[:k]
        F_true, p_val, c10, c5, c1 = test_threshold_effect(
            Y_within, X_within, R, Q, N, T, grid, fixed, trim_percent, n_boot, n_jobs, time_fe
        )
        
        results.append({
            "test": f"{k} vs {k+1}",
            "F_stat": F_true,
            "p_value": p_val,
            "crit10": c10,  
            "crit5": c5,    
            "crit1": c1     
        })

    return results