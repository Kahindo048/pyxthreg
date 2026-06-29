import numpy as np
import math
import matplotlib.pyplot as plt
from numba import njit
from matplotlib.colors import LinearSegmentedColormap

@njit(fastmath=True, nogil=True)
def sweep_threshold_for_lr(Y_within, X_within, R, Q, N, T, grid, fixed_gammas, trim_percent):
    """
    Sweeps the grid for a specific threshold while keeping other thresholds fixed.
    
    This is an optimized version of the core engine modified to return the 
    complete sequence of Residual Sum of Squares (SSR) for the entire grid, 
    which is necessary to plot the Likelihood Ratio (LR) curve.
    
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
    fixed_gammas : numpy.ndarray
        The previously identified thresholds kept fixed during this sweep.
    trim_percent : float
        Trimming percentage for regime sizes.
        
    Returns
    -------
    ssr_sequence : numpy.ndarray
        The sequence of SSR values evaluated at each point on the grid.
    """
    n_obs = N * T
    min_obs = int(n_obs * trim_percent) 
    
    K_X = X_within.shape[1]
    K_R = R.shape[1]
    n_test_gammas = len(fixed_gammas) + 1
    n_regimes = n_test_gammas + 1
    K_total = K_X + n_regimes * K_R
    
    Z = np.empty((n_obs, K_total), dtype=np.float64)
    Z_within = np.empty((n_obs, K_total), dtype=np.float64)
    ZtZ = np.zeros((K_total, K_total), dtype=np.float64)
    ZtY = np.zeros((K_total, 1), dtype=np.float64)
    
    n_grid = len(grid)
    ssr_sequence = np.empty(n_grid, dtype=np.float64)
    
    for idx in range(n_grid):
        gamma_cand = grid[idx]
        
        # Combine fixed thresholds with the current candidate
        test_gammas = np.empty(n_test_gammas, dtype=np.float64)
        for j in range(len(fixed_gammas)):
            test_gammas[j] = fixed_gammas[j]
        test_gammas[-1] = gamma_cand
        test_gammas = np.sort(test_gammas)
        
        # Verify observation counts per regime
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
            ssr_sequence[idx] = np.inf  # Regime too small, value ignored
            continue 
            
        # Populate matrix Z
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
                        
        # Within transformation
        for i in range(n_obs):
            for k in range(K_X):
                Z_within[i, k] = Z[i, k]
                
        for i in range(N):
            start = i * T
            end = start + T
            for k in range(K_X, K_total):
                mean_ik = 0.0
                for t in range(start, end):
                    mean_ik += Z[t, k]
                mean_ik /= T
                for t in range(start, end):
                    Z_within[t, k] = Z[t, k] - mean_ik
                    
        # Fast OLS and SSR computation
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
            
        ssr_sequence[idx] = ssr
        
    return ssr_sequence

def compute_confidence_intervals(Y_within, X_within, R, Q, N, T, grid, thresholds, min_ssr_global, df_resid, trim_percent, alpha=0.05):
    """
    Computes Hansen's confidence intervals by inverting the LR statistic.
    
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
        The search grid of candidate threshold values.
    thresholds : numpy.ndarray
        The identified optimal threshold values.
    min_ssr_global : float
        The minimum global SSR of the final model.
    df_resid : int
        The residual degrees of freedom.
    trim_percent : float
        Trimming percentage for regime sizes.
    alpha : float, optional
        Significance level, by default 0.05 (for 95% confidence intervals).
        
    Returns
    -------
    ci_results : list of tuples
        List containing the lower and upper bounds for each threshold's confidence interval.
    lr_sequences : list of numpy.ndarray
        List of LR statistic sequences corresponding to each threshold search.
    c_alpha : float
        The asymptotic critical value from Hansen (1999).
    """
    # Hansen's asymptotic critical value
    c_alpha = -2 * np.log(1 - np.sqrt(1 - alpha))
    sigma2 = min_ssr_global / df_resid
    
    ci_results = []
    lr_sequences = []
    
    for k in range(len(thresholds)):
        # To evaluate the interval of threshold K, we keep the other thresholds fixed
        fixed = np.delete(thresholds, k)
        
        # Get the SSR sequence for the entire grid
        ssr_seq = sweep_threshold_for_lr(Y_within, X_within, R, Q, N, T, grid, fixed, trim_percent)
        
        # Ignore "inf" values generated by trimming restrictions
        ssr_seq_clean = np.where(ssr_seq == np.inf, np.nan, ssr_seq)
        
        # Likelihood Ratio (LR) statistic formula
        lr_seq = (ssr_seq_clean - min_ssr_global) / sigma2
        
        # Find the interval where LR <= critical value
        valid_mask = (lr_seq <= c_alpha)
        if np.any(valid_mask):
            valid_grid = grid[valid_mask]
            ci_lower = valid_grid[0]
            ci_upper = valid_grid[-1]
        else:
            # Fallback (very rare) if the curve is too steep
            idx_min = np.nanargmin(lr_seq)
            ci_lower = grid[idx_min]
            ci_upper = grid[idx_min]
            
        ci_results.append((ci_lower, ci_upper))
        lr_sequences.append(lr_seq)
        
    return ci_results, lr_sequences, c_alpha

def plot_hansen_lr(grid, lr_sequences, thresholds, ci_results, c_alpha, title="LR Statistic and Confidence Intervals"):
    """
    Generates academic-quality plots of Hansen's Likelihood Ratio (LR) statistics.
    Plots one subplot per identified threshold, arranged in a grid (max 3 columns).
    Automatically zooms around the relevant confidence region.
    
    Parameters
    ----------
    grid : numpy.ndarray
        The grid of threshold candidate values.
    lr_sequences : list of numpy.ndarray
        The sequences of LR statistics for each threshold.
    thresholds : numpy.ndarray
        The identified optimal threshold values.
    ci_results : list of tuples
        The lower and upper bounds of the confidence intervals.
    c_alpha : float
        The critical value.
    title : str, optional
        The overall figure title.
    """
    n_thresholds = len(thresholds)
    
    # Calculate grid layout (max 3 columns)
    n_cols = min(n_thresholds, 3)
    n_rows = math.ceil(n_thresholds / 3)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.5 * n_rows))
    
    # Flatten the axes array for easy iteration
    if n_thresholds == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten()
        
    for k in range(n_thresholds):
        ax = axes_flat[k]
        lr_seq = lr_sequences[k]
        est_gamma = thresholds[k]
        ci_low, ci_high = ci_results[k]
        
        # LR Stat curve
        ax.plot(grid, lr_seq, color='#1f77b4', lw=2.5, label='LR Statistic')
        
        # 95% Critical line
        ax.axhline(y=c_alpha, color='red', linestyle='--', lw=1.5, 
                   label=f'95% Critical Value ({c_alpha:.2f})')
        
        # Estimated point (inverted peak of the "V")
        ax.scatter([est_gamma], [0], color='black', s=80, zorder=5, 
                   label=f'Estimated Threshold: {est_gamma:.4f}')
        
        # Shaded confidence interval area
        valid_mask = lr_seq <= c_alpha
        ax.fill_between(grid, 0, c_alpha, where=valid_mask, color='gray', alpha=0.3, 
                        label=f'95% CI [{ci_low:.4f}, {ci_high:.4f}]')
        
        ax.set_title(f'Break #{k+1}', fontweight='bold', fontsize=12)
        ax.set_xlabel(r'Threshold variable ($\gamma$)', fontsize=11)
        ax.set_ylabel('LR Statistic', fontsize=11)
        
        # ==========================================
        # AUTOMATIC ZOOM LOGIC
        # ==========================================
        # Target only the region where LR < 40 to focus on the confidence valley
        zoom_mask = lr_seq <= 40
        if np.any(zoom_mask):
            x_min = np.min(grid[zoom_mask])
            x_max = np.max(grid[zoom_mask])
            
            # Add visual padding
            padding = (x_max - x_min) * 0.15
            if padding == 0:  # Failsafe for vertical line curves
                padding = 0.1
                
            ax.set_xlim(x_min - padding, x_max + padding)
            
        # Set visual Y limit to 30 (Standard practice for Hansen plots)
        ax.set_ylim(0, 30)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        ax.legend(loc='upper right', fancybox=True, framealpha=0.9, fontsize=9)
        
    # Remove empty subplots
    for k in range(n_thresholds, len(axes_flat)):
        fig.delaxes(axes_flat[k])
        
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout() 
    plt.show()
    
def plot_diagnostics_panel(ssr_linear, ssrs_thresholds, Q, thresholds, title="Regime Selection Diagnostics"):
    """
    Generates two publication-quality diagnostic plots to validate the chosen number of thresholds:
    1. Elbow Plot (Decrease of unexplained variance/SSR)
    2. Histogram of the sample distribution across Economic Regimes
    
    Parameters
    ----------
    ssr_linear : float
        The SSR of the linear model (0 thresholds).
    ssrs_thresholds : numpy.ndarray
        Array containing the SSR for each cumulative threshold level.
    Q : numpy.ndarray
        The threshold variable array (flattened).
    thresholds : numpy.ndarray
        The identified optimal threshold values.
    title : str, optional
        The overall figure title.
    """
    # Academic color palette (Readable in Black & White printing)
    academic_navy = '#1f497d'  
    academic_slate = '#7f8c8d' 
    text_dark = '#333333'      
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    
    # ---------------------------------------------------------
    # PLOT 1 : ELBOW PLOT (SSR vs Number of Thresholds)
    # ---------------------------------------------------------
    all_ssrs = [ssr_linear] + list(ssrs_thresholds)
    n_seuils_axis = list(range(len(all_ssrs)))
    
    # Square marker 's' and dark line (classic academic style)
    ax1.plot(n_seuils_axis, all_ssrs, marker='s', color=academic_navy, lw=2.0, ms=7, 
             mec='black', label='Model SSR')
             
    ax1.set_title("Parsimony Criterion: SSR Evolution", fontsize=12)
    ax1.set_xlabel("Number of Estimated Thresholds", fontsize=11)
    ax1.set_ylabel("Sum of Squared Residuals (SSR)", fontsize=11)
    ax1.set_xticks(n_seuils_axis)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Add percentage gain annotations
    for i in range(1, len(all_ssrs)):
        gain = ((all_ssrs[i-1] - all_ssrs[i]) / all_ssrs[i-1]) * 100
        # Marginal gains (< 1%) are greyed out to highlight insignificance
        color_annotation = text_dark if gain > 1.0 else '#95a5a6'
        weight = 'bold' if gain > 1.0 else 'normal'
        
        ax1.annotate(f"-{gain:.1f}%", (n_seuils_axis[i], all_ssrs[i]), 
                     textcoords="offset points", xytext=(0, 12), ha='center', 
                     fontsize=10, fontweight=weight, color=color_annotation)
                     
    ax1.legend(frameon=False, fontsize=10)

    # ---------------------------------------------------------
    # PLOT 2 : OBSERVATION DISTRIBUTION PER REGIME
    # ---------------------------------------------------------
    n_obs = len(Q)
    regime_indices = np.zeros(n_obs, dtype=np.int32)
    for g in thresholds:
        regime_indices += (Q > g).astype(np.int32)
        
    unique_regimes, counts = np.unique(regime_indices, return_counts=True)
    pct_counts = (counts / n_obs) * 100
    
    regime_labels = [f"Regime {r+1}" for r in unique_regimes]
    
    bars = ax2.bar(regime_labels, pct_counts, color=academic_slate, edgecolor='black', 
                   linewidth=1.2, width=0.45)
                   
    ax2.set_title("Sample Distribution by Regime", fontsize=12)
    ax2.set_ylabel("Percentage of observations (%)", fontsize=11)
    
    # Leave 25% empty space above the tallest bar for annotations
    ax2.set_ylim(0, max(pct_counts) * 1.25) 
    ax2.grid(True, linestyle='--', alpha=0.4, axis='y')
    
    # Display the percentage and exact N count above each bar
    for bar, pct, raw_count in zip(bars, pct_counts, counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.0, 
                 f"{pct:.1f}%\n($N={raw_count}$)", # Math notation for N
                 ha='center', va='bottom', fontsize=10, color=text_dark)
                 
    # ---------------------------------------------------------
    # LAYOUT FORMATTING
    # ---------------------------------------------------------
    plt.suptitle(title, fontsize=14, y=0.97)
    
    # Secret trick: force tight_layout to leave the top 8% empty for the title
    plt.tight_layout(rect=[0, 0, 1, 0.92]) 
    fig.subplots_adjust(top=0.85)
    plt.show()
    
def plot_regime_dynamics(Q, thresholds, N, T, title="Temporal Dynamics of Regimes"):
    """
    Generates a stacked area chart depicting regime transitions over time.
    Allows researchers to visually assess the temporal stability of the specification.
    The color palette automatically adapts to any number of regimes.
    
    Parameters
    ----------
    Q : numpy.ndarray
        The threshold variable array (flattened).
    thresholds : numpy.ndarray
        The identified optimal threshold values.
    N : int
        Number of entities.
    T : int
        Number of time periods.
    title : str, optional
        The overall figure title.
    """
    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    # Reshape the Q matrix to isolate the time dimension (N x T)
    Q_time = Q.reshape(N, T)
    regime_time = np.zeros((N, T), dtype=np.int32)
    for g in thresholds:
        regime_time += (Q_time > g).astype(np.int32)
        
    unique_regimes = np.unique(regime_time)
    n_regimes = len(unique_regimes)
    regime_labels = [f"Regime {r+1}" for r in unique_regimes]
    
    # =========================================================
    # DYNAMIC COLOR PALETTE GENERATION
    # =========================================================
    base_colors = ['#1f497d', '#4b77a9', '#7f8c8d', '#b3bccc', '#e0e4ec']
    
    if n_regimes <= len(base_colors):
        # If 5 regimes or fewer, simply map the base colors
        area_colors = base_colors[:n_regimes]
    else:
        # If >5 regimes, interpolate a perfect continuous gradient
        cmap = LinearSegmentedColormap.from_list("academic_cmap", base_colors)
        area_colors = [cmap(i) for i in np.linspace(0, 1, n_regimes)]
        
    time_axis = np.arange(1, T + 1)
    regime_proportions = np.zeros((n_regimes, T))
    
    for t_idx in range(T):
        for r_idx, r in enumerate(unique_regimes):
            regime_proportions[r_idx, t_idx] = (np.sum(regime_time[:, t_idx] == r) / N) * 100
            
    # Plotting with the dynamic color gradient
    ax.stackplot(time_axis, regime_proportions, labels=regime_labels, 
                  colors=area_colors, alpha=0.85, edgecolor='white', linewidth=0.5)
                  
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.set_xlabel("Time Periods (T)", fontsize=11)
    ax.set_ylabel("Percentage of panel (%)", fontsize=11)
    ax.set_xlim(1, T)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle='--', alpha=0.4)
    
    # Reverse the legend order so it matches the visual top-to-bottom stacking
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='center left', bbox_to_anchor=(1.02, 0.5), 
               frameon=False, fontsize=10)

    plt.tight_layout() 
    fig.subplots_adjust(right=0.85)
    plt.show()