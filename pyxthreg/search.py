import numpy as np
from numba import njit
from .core import get_ssr_for_threshold

@njit(fastmath=True, nogil=True)
def get_grid(Q, trim_percent, grid_size=300):
    """
    Creates the search grid for the threshold variable gamma.
    
    Applies symmetric trimming to remove extreme values and then selects 
    'grid_size' equidistant points (percentiles) to accelerate the search.
    
    Parameters
    ----------
    Q : numpy.ndarray
        The threshold variable array (flattened) of shape (N*T,).
    trim_percent : float
        The trimming percentage (e.g., 0.05 for 5%) to remove from both ends.
    grid_size : int, optional
        The maximum number of grid points to evaluate, by default 300.
        If set to 0, an exhaustive search over all valid unique points is performed.
        
    Returns
    -------
    numpy.ndarray
        The refined grid of candidate threshold values.
    """
    # 1. Extract unique values and sort them
    unique_q = np.unique(Q)
    unique_q.sort()
    n_unique = len(unique_q)
    
    # 2. Apply trimming (remove extreme tails)
    trim_count = int(np.floor(n_unique * trim_percent))
    
    # Fallback if the trimming is too aggressive for the data size
    if trim_count * 2 >= n_unique:
        valid_q = unique_q
    else:
        valid_q = unique_q[trim_count : n_unique - trim_count]
        
    n_valid = len(valid_q)
    
    # 3. Reduction to grid_size (The secret to computational speed)
    # If exhaustive search is requested or if there are fewer points than the grid size
    if grid_size <= 0 or n_valid <= grid_size:
        return valid_q
        
    # Allocate the new grid
    grid = np.empty(grid_size, dtype=valid_q.dtype)
    
    # Select equidistant points (percentile approximation)
    for i in range(grid_size):
        # Calculate the exact index to cover the entire valid_q space
        idx = int(np.round(i * (n_valid - 1) / (grid_size - 1)))
        grid[i] = valid_q[idx]
        
    # Apply a final np.unique in case rounding creates duplicates
    return np.unique(grid)

@njit(fastmath=True, nogil=True)
def grid_search_1_threshold(Y_within, X_within, R, Q, N, T, trim_percent, grid_size=300):
    """
    Searches for the optimal threshold (single threshold model) over a reduced grid.
    
    Evaluates the Residual Sum of Squares (SSR) for each candidate threshold
    and identifies the one that minimizes the SSR.
    
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
    trim_percent : float
        Trimming percentage for regime sizes.
    grid_size : int, optional
        Number of points in the search grid, by default 300.
        
    Returns
    -------
    best_gamma : float
        The candidate threshold value that minimizes the SSR.
    min_ssr : float
        The minimum Residual Sum of Squares achieved.
    grid : numpy.ndarray
        The grid of evaluated threshold candidates.
    ssr_sequence : numpy.ndarray
        The sequence of SSR values corresponding to each grid point 
        (useful for plotting the Likelihood Ratio statistic).
    """
    # Obtain candidates on the grid
    grid = get_grid(Q, trim_percent, grid_size)
    n_grid = len(grid)
    
    best_gamma = grid[0]
    min_ssr = np.inf
    
    # Store the entire sequence (for the LR stat plot)
    ssr_sequence = np.empty(n_grid, dtype=np.float64)
    
    # Search loop
    for idx in range(n_grid):
        gamma_candidate = grid[idx]
        
        current_ssr = get_ssr_for_threshold(
            Y_within, X_within, R, Q, gamma_candidate, N, T
        )
        
        ssr_sequence[idx] = current_ssr
        
        if current_ssr < min_ssr:
            min_ssr = current_ssr
            best_gamma = gamma_candidate
            
    return best_gamma, min_ssr, grid, ssr_sequence