import numpy as np
from numba import njit

@njit(fastmath=True, nogil=True)
def within_transform(A, N, T):
    """
    Applies the ultra-fast within-transformation (One-Way Fixed Effects).
    
    Subtracts the temporal mean of each individual to eliminate fixed effects.
    
    Parameters
    ----------
    A : numpy.ndarray
        A contiguous matrix of shape (N*T, K).
    N : int
        Number of entities.
    T : int
        Number of time periods.
        
    Returns
    -------
    numpy.ndarray
        The transformed matrix of shape (N*T, K).
    """
    K = A.shape[1]
    A_within = np.empty_like(A)
    
    for i in range(N):
        start = i * T
        end = start + T
        
        for k in range(K):
            mean_ik = 0.0
            for t in range(start, end):
                mean_ik += A[t, k]
            mean_ik /= T
            
            for t in range(start, end):
                A_within[t, k] = A[t, k] - mean_ik
                
    return A_within

@njit(fastmath=True, nogil=True)
def twoway_within_transform(A, N, T):
    """
    Applies the double within-transformation (Two-Way Fixed Effects).
    
    Eliminates both individual and time fixed effects using the Frisch-Waugh-Lovell
    theorem approximation: x_it* = x_it - mean(x_i) - mean(x_t) + mean(x_overall).
    
    Parameters
    ----------
    A : numpy.ndarray
        A contiguous matrix of shape (N*T, K).
    N : int
        Number of entities.
    T : int
        Number of time periods.
        
    Returns
    -------
    numpy.ndarray
        The transformed matrix of shape (N*T, K).
    """
    K = A.shape[1]
    A_tw = np.empty_like(A)
    
    for k in range(K):
        # 1. Overall mean
        mean_overall = 0.0
        for i in range(N * T):
            mean_overall += A[i, k]
        mean_overall /= (N * T)
        
        # 2. Individual means (by group i)
        mean_i = np.zeros(N, dtype=np.float64)
        for i in range(N):
            start = i * T
            end = start + T
            for t in range(start, end):
                mean_i[i] += A[t, k]
            mean_i[i] /= T
            
        # 3. Temporal means (by period t)
        mean_t = np.zeros(T, dtype=np.float64)
        for t in range(T):
            for i in range(N):
                idx = i * T + t
                mean_t[t] += A[idx, k]
            mean_t[t] /= N
            
        # 4. Double Transformation
        for i in range(N):
            for t in range(T):
                idx = i * T + t
                A_tw[idx, k] = A[idx, k] - mean_i[i] - mean_t[t] + mean_overall
                
    return A_tw

@njit(fastmath=True, nogil=True)
def get_ssr_for_threshold(Y_within, X_within, R, Q, gamma, N, T):
    """
    Calculates the Residual Sum of Squares (SSR) for a single candidate threshold.
    """
    n_obs = N * T
    K_X = X_within.shape[1]
    K_R = R.shape[1]
    
    K_total = K_X + 2 * K_R
    Z = np.empty((n_obs, K_total), dtype=Y_within.dtype)
    
    for i in range(n_obs):
        for k in range(K_X):
            Z[i, k] = X_within[i, k]
            
        if Q[i] <= gamma:
            for k in range(K_R):
                Z[i, K_X + k] = R[i, k]           
                Z[i, K_X + K_R + k] = 0.0         
        else:
            for k in range(K_R):
                Z[i, K_X + k] = 0.0               
                Z[i, K_X + K_R + k] = R[i, k]     
                
    Z_within = within_transform(Z, N, T)
    
    ZtZ = Z_within.T @ Z_within
    ZtY = Z_within.T @ Y_within
    beta = np.linalg.solve(ZtZ, ZtY)
    
    residuals = Y_within - Z_within @ beta
    ssr = 0.0
    for i in range(n_obs):
        ssr += residuals[i, 0] ** 2
        
    return ssr

@njit(fastmath=True, nogil=True)
def transform_matrix_partial(A, N, T, start_col, time_fe=False):
    """
    Optimized partial within-transformation for dynamically generated regime variables.
    
    Applies the within-transformation (One-Way or Two-Way) ONLY to columns 
    from 'start_col' onwards, preserving the pre-transformed control variables (X).
    
    Parameters
    ----------
    A : numpy.ndarray
        The full design matrix Z.
    N : int
        Number of entities.
    T : int
        Number of time periods.
    start_col : int
        The index of the first regime-dependent column to transform.
    time_fe : bool, optional
        If True, applies the Two-Way fixed effects transformation, by default False.
        
    Returns
    -------
    numpy.ndarray
        The partially transformed matrix.
    """
    K = A.shape[1]
    A_out = np.empty_like(A)

    # 1. Direct copy of previously transformed independent variables
    for k in range(start_col):
        for i in range(N * T):
            A_out[i, k] = A[i, k]

    # 2. Individual Fixed Effects purge (One-Way) on regime columns
    for k in range(start_col, K):
        for i in range(N):
            start = i * T
            end = start + T
            mean_i = 0.0
            for t in range(start, end):
                mean_i += A[t, k]
            mean_i /= T
            for t in range(start, end):
                A_out[t, k] = A[t, k] - mean_i

    # 3. Temporal Fixed Effects purge (Two-Way) on regime columns
    if time_fe:
        for k in range(start_col, K):
            mean_t = np.zeros(T, dtype=np.float64)
            for t in range(T):
                for i in range(N):
                    mean_t[t] += A_out[i * T + t, k]
                mean_t[t] /= N
            
            mean_overall = 0.0
            for t in range(T):
                mean_overall += mean_t[t]
            mean_overall /= T
            
            for i in range(N):
                for t in range(T):
                    A_out[i * T + t, k] = A_out[i * T + t, k] - mean_t[t] + mean_overall

    return A_out