import numpy as np
from scipy.stats import t as t_dist
from numba import njit
from .core import transform_matrix_partial

@njit(fastmath=True, nogil=True)
def build_final_model(Y_within, X_within, R, Q, N, T_periods, thresholds, time_fe):
    """
    Reconstructs the full design matrix Z and computes the raw OLS parameters.
    """
    n_obs = N * T_periods
    K_X = X_within.shape[1]
    K_R = R.shape[1]
    
    gammas = np.sort(thresholds)
    n_regimes = len(gammas) + 1
    K_total = K_X + n_regimes * K_R
    
    Z = np.empty((n_obs, K_total), dtype=np.float64)
    
    for i in range(n_obs):
        for k in range(K_X):
            Z[i, k] = X_within[i, k]
            
        q_val = Q[i]
        r_idx = 0
        for g in range(len(gammas)):
            if q_val > gammas[g]:
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
                    
    Z_within = transform_matrix_partial(Z, N, T_periods, K_X, time_fe)
                
    ZtZ = Z_within.T @ Z_within
    ZtY = Z_within.T @ Y_within
    beta = np.linalg.solve(ZtZ, ZtY)
    
    residuals = Y_within - Z_within @ beta
    ssr = 0.0
    for i in range(n_obs):
        ssr += residuals[i, 0] ** 2
        
    return beta, ZtZ, residuals, ssr, K_total, Z_within

@njit(fastmath=True, nogil=True)
def compute_cluster_meat(Z_within, residuals, N, T, K):
    """
    Computes the core 'Meat' of the Sandwich variance estimator.
    Clusters by panel entity (id) to correct for heteroskedasticity AND serial correlation.
    """
    meat = np.zeros((K, K), dtype=np.float64)
    
    for i in range(N):
        u_i = np.zeros(K, dtype=np.float64)
        start = i * T
        end = start + T
        
        for t in range(start, end):
            e_it = residuals[t, 0]
            for k in range(K):
                u_i[k] += Z_within[t, k] * e_it
        
        for k1 in range(K):
            for k2 in range(K):
                meat[k1, k2] += u_i[k1] * u_i[k2]
                
    return meat

def get_standard_errors(Y_within, X_within, R, Q, N, T, thresholds, time_fe, robust=False):
    """
    Calculates standard errors, t-statistics, and p-values for the final model.
    If robust=True, applies the Cluster-Robust Sandwich estimator matching Stata.
    
    Parameters
    ----------
    robust : bool, optional
        Whether to compute cluster-robust standard errors, by default False.
    """
    beta, ZtZ, residuals, ssr, K_total, Z_within = build_final_model(Y_within, X_within, R, Q, N, T, thresholds, time_fe)
    
    n_obs = N * T
    df_resid = n_obs - N - K_total
    sigma2 = ssr / df_resid
    ZtZ_inv = np.linalg.inv(ZtZ)
    
    if robust:
        meat = compute_cluster_meat(Z_within, residuals, N, T, K_total)
        # Small sample adjustment matching Stata's xtreg, fe vce(robust)
        c = N / (N - 1)
        V = c * (ZtZ_inv @ meat @ ZtZ_inv)
        df_tests = N - 1 
    else:
        V = sigma2 * ZtZ_inv
        df_tests = df_resid
        
    se = np.sqrt(np.diag(V)).reshape(-1, 1)
    t_stats = beta / se
    p_values = 2 * t_dist.sf(np.abs(t_stats), df=df_tests)
    
    return beta, se, t_stats, p_values, df_resid, V, df_tests