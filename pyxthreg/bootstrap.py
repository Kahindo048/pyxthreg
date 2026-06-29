import numpy as np
from numba import njit
from joblib import Parallel, delayed
from .search import get_grid
from .sequential import search_next_threshold
from .core import transform_matrix_partial
import contextlib
import joblib
from tqdm.auto import tqdm

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report into tqdm progress bar.
    This allows parallel multi-core processes to update a single console UI.
    """
    # CORRECTION ICI : on utilise joblib.parallel au lieu de joblib.callbacks
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

# NOGIL=TRUE ALLOWS MULTI-THREADING BYPASSING PYTHON'S GIL
@njit(fastmath=True, nogil=True)
def compute_ssr_fixed(Y_within, X_within, R, Q, N, T, fixed_gammas, time_fe):
    """
    Estimates the fixed-effects panel threshold model under the null hypothesis (H0)
    of no additional threshold effect, imposing the exogenously fixed thresholds.

    Parameters
    ----------
    Y_within : numpy.ndarray
        The within-transformed dependent variable column vector (N*T x 1).
    X_within : numpy.ndarray
        The within-transformed regime-independent design matrix.
    R : numpy.ndarray
        The raw (untransformed) regime-dependent variable matrix.
    Q : numpy.ndarray
        The threshold variable vector (N*T x 1).
    N : int
        Number of cross-sectional entities (groups).
    T : int
        Number of time periods per entity.
    fixed_gammas : numpy.ndarray
        Array of fixed threshold parameters defining the restricted model's regimes.
    time_fe : bool
        If True, executes a concurrent Two-Way partial within-transformation.

    Returns
    -------
    ssr : float
        The Sum of Squared Residuals (SSR) for the restricted model.
    Y_pred : numpy.ndarray
        The fitted (predicted) values of the dependent variable under H0.
    residuals : numpy.ndarray
        The estimated structural residuals under H0.
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

    # Fast partial within-transformation (One-Way or Two-Way)
    Z_within = transform_matrix_partial(Z, N, T, K_X, time_fe)

    # Fast OLS computation (Matrix Inversion)
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
    Compiled JIT execution for a single residual-based bootstrap iteration.
    Simulates a new asymptotic distribution mimicking the Stata DGP process (Wang, 2015).

    Parameters
    ----------
    Y_pred_H0 : numpy.ndarray
        Fitted values from the restricted model.
    resid_H0 : numpy.ndarray
        Residuals from the restricted model.
    eta : numpy.ndarray
        A drawn vector of N(0,1) i.i.d. standard normal random variables.
    
    Returns
    -------
    ssr_H0 : float
        Simulated SSR under the null hypothesis (k thresholds).
    ssr_H1 : float
        Simulated SSR under the alternative hypothesis (k+1 thresholds).
    """
    n_obs = N * T
    Y_boot_raw = np.empty((n_obs, 1), dtype=np.float64)

    # 1. Generate Bootstrap Sample: Y_boot = residuals * N(0,1) noise
    for t in range(n_obs):
        Y_boot_raw[t, 0] = resid_H0[t, 0] * eta[t]
        
    # 2. IMPERATIVE: Recenter Y_boot (Within Transformation)
    # The random noise breaks the zero-mean property of residuals per group.
    Y_boot = transform_matrix_partial(Y_boot_raw, N, T, 0, time_fe)

    # 3. Estimations on the Simulated Bootstrap sample
    ssr_H0, _, _ = compute_ssr_fixed(Y_boot, X_within, R, Q, N, T, fixed_gammas, time_fe)
    _, ssr_H1 = search_next_threshold(Y_boot, X_within, R, Q, N, T, grid, fixed_gammas, trim_percent, time_fe)

    return ssr_H0, ssr_H1


def test_threshold_effect(Y_within, X_within, R, Q, N, T, grid, fixed_gammas, trim_percent, n_boot=300, n_jobs=-1, time_fe=False, nobslog=False, test_name="Single"):
    """
    Performs the Hansen (1999) residual-based bootstrap test to evaluate the 
    statistical significance of an additional threshold effect.

    Since the threshold parameter is unidentified under the null hypothesis 
    (the "Davies Problem"), a simulated Likelihood Ratio (F-statistic) is computed.

    Parameters
    ----------
    Y_within, X_within, R, Q : numpy.ndarray
        Data matrices prepared by the estimator.
    grid : numpy.ndarray
        The valid search grid of the threshold variable after trimming.
    fixed_gammas : numpy.ndarray
        The current identified thresholds (used as the null hypothesis constraints).
    n_boot : int, default=300
        Number of bootstrap replications to simulate the asymptotic distribution.
    n_jobs : int, default=-1
        Number of CPU cores to utilize during the parallel bootstrap loop.
    nobslog : bool, default=False
        If True, suppresses the tqdm progress bar output.

    Returns
    -------
    F_true : float
        The empirical pseudo Likelihood-Ratio (F-statistic) of the true sample.
    p_value : float
        The simulated probability of observing an F-stat greater than F_true under H0.
    crit10, crit5, crit1 : float
        The simulated critical values at the 90%, 95%, and 99% confidence levels.
    """
    ssr_H0_true, Y_pred_H0, resid_H0 = compute_ssr_fixed(Y_within, X_within, R, Q, N, T, fixed_gammas, time_fe)
    _, ssr_H1_true = search_next_threshold(Y_within, X_within, R, Q, N, T, grid, fixed_gammas, trim_percent, time_fe)

    # --- HISTORICAL STATA ADJUSTMENT ---
    # Matches the exact degrees of freedom calculation from Wang (2015)
    df_penalty = (T - 1) if time_fe else 0
    df_boot = (N * T) - T - df_penalty
    
    sigma2_H1_true = ssr_H1_true / df_boot
    F_true = (ssr_H0_true - ssr_H1_true) / sigma2_H1_true

    np.random.seed(42)
    # Generate N(0,1) noise for the entire matrix across all bootstrap iterations
    etas = np.random.normal(0, 1, size=(n_boot, N * T))

    # --- PROGRESS BAR WITH TQDM & JOBLIB ---
    desc_text = f"Bootstrap ({test_name})"
    with tqdm_joblib(tqdm(desc=desc_text, total=n_boot, unit="rep", ncols=85, colour='blue', disable=nobslog)):
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


def bootstrap_all_thresholds(Y_within, X_within, R, Q, N, T, thresholds, trim_percent, grid_size=300, n_boot=300, n_jobs=-1, time_fe=False, nobslog=False):
    """
    Executes the sequential Hansen bootstrap test for all identified thresholds 
    in a multiple-threshold model.

    Parameters
    ----------
    thresholds : numpy.ndarray
        Array containing the optimal threshold estimates to be tested sequentially.
    
    Returns
    -------
    results : list of dicts
        A list of dictionaries containing the F-statistic, p-value, and critical 
        values for each sequential structural break test (e.g., 0 vs 1, 1 vs 2).
    """
    grid = get_grid(Q, trim_percent, grid_size)
    results = []

    for k in range(len(thresholds)):
        fixed = thresholds[:k]
        test_lbl = "Single" if k == 0 else f"{k} vs {k+1}"
        
        F_true, p_val, c10, c5, c1 = test_threshold_effect(
            Y_within, X_within, R, Q, N, T, grid, fixed, trim_percent, n_boot, n_jobs, time_fe,
            nobslog=nobslog, test_name=test_lbl
        )
        
        results.append({
            "test": test_lbl,
            "F_stat": F_true,
            "p_value": p_val,
            "crit10": c10,  
            "crit5": c5,    
            "crit1": c1     
        })

    return results