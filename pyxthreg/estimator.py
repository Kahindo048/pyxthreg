import numpy as np
import pandas as pd
from scipy.stats import f as f_dist, t as t_dist

from .utils import extract_matrices
from .core import within_transform, twoway_within_transform
from .sequential import fit_sequential_thresholds
from .inference import get_standard_errors
from .bootstrap import bootstrap_all_thresholds, test_threshold_effect
from .search import get_grid
from .plotting import compute_confidence_intervals, plot_hansen_lr

class ThresholdPanel:
    """
    Estimator for fixed-effects panel models with multiple thresholds.
    
    Exclusive feature: Automatic selection of the optimal number of regimes 
    using a sequential bootstrap algorithm (thnum="auto").
    """
    def __init__(self, data, dep, rx, qx, indep=None, entity_col='id', time_col='year'):
        self.data = data.copy()
        self.dep = dep
        self.rx = rx if isinstance(rx, list) else [rx]
        self.qx = qx
        self.indep = indep if indep is not None else []
        self.entity_col = entity_col
        self.time_col = time_col
        self.is_fitted = False
        self.results = {}

    def fit(self, thnum="auto", max_thresholds=5, trim=0.05, grid=300, bs=300, thlevel=95, 
            thgiven=None, gen=None, nobslog=False, time_fe=False, robust=False, n_jobs=-1):
        """
        Estimates the threshold panel model.
        
        Parameters
        ----------
        thnum : int or str, optional
            Number of thresholds to estimate. If "auto", determines optimal number, by default "auto".
        time_fe : bool, optional
            If True, applies two-way fixed effects (individual and time), by default False.
        robust : bool, optional
            If True, computes cluster-robust standard errors (clustered by entity), by default False.
        """
        alpha = 1.0 - (thlevel / 100.0)
        self.time_fe = time_fe 
        self.robust = robust 

        Y_mat, X_mat, R_mat, Q_mat, N, T = extract_matrices(
            df=self.data, dep=self.dep, indep=self.indep, rx=self.rx, qx=self.qx,
            entity_col=self.entity_col, time_col=self.time_col
        )
        
        if self.time_fe:
            Y_w = twoway_within_transform(Y_mat, N, T)
            X_w = twoway_within_transform(X_mat, N, T)
        else:
            Y_w = within_transform(Y_mat, N, T)
            X_w = within_transform(X_mat, N, T)
        
        K_X = X_mat.shape[1] if X_mat.size > 0 else 0
        K_R = R_mat.shape[1]
        Z_linear = np.empty((N * T, K_X + K_R))
        if K_X > 0: Z_linear[:, :K_X] = X_w
        Z_linear[:, K_X:] = R_mat 
        
        from .core import transform_matrix_partial
        Z_linear = transform_matrix_partial(Z_linear, N, T, K_X, self.time_fe)
        
        beta_lin = np.linalg.solve(Z_linear.T @ Z_linear, Z_linear.T @ Y_w)
        res_lin = Y_w - Z_linear @ beta_lin
        ssr_linear = np.sum(res_lin ** 2)
        
        grid_vals = get_grid(Q_mat, trim, grid)
        
        if thgiven is not None:
            if not isinstance(thgiven, (list, np.ndarray)): thgiven = [thgiven]
            thresholds = np.sort(np.array(thgiven, dtype=np.float64))
            if not nobslog: print(f"thgiven option: Imposed thresholds {thresholds}")
            ssrs = [np.nan] * len(thresholds)
            boot_results = []
            ci_results = [(np.nan, np.nan)] * len(thresholds)
            ssr_final = ssr_linear
            
        elif str(thnum).lower() == "auto":
            if not nobslog: 
                print(f"Auto-Selection enabled (Level = {thlevel}%): Sequential search for the optimal model...")
            
            thresholds_accepted = []
            ssrs_accepted = []
            boot_results = []
            
            for k in range(1, max_thresholds + 1):
                if not nobslog: print(f"  Testing {k-1} vs {k} threshold(s)...", end="", flush=True)
                
                temp_thresh, temp_ssrs = fit_sequential_thresholds(Y_w, X_w, R_mat, Q_mat, N, T, trim, grid, k, self.time_fe)
                fixed_for_test = temp_thresh[:k-1]
                
                F_true, p_val, c10, c5, c1 = test_threshold_effect(
                    Y_w, X_w, R_mat, Q_mat, N, T, grid_vals, fixed_for_test, trim, bs, n_jobs, self.time_fe
                )
                
                boot_dict = {
                    "test": "Single" if k == 1 else f"{k-1} vs {k}",
                    "F_stat": F_true, "p_value": p_val,
                    "crit10": c10, "crit5": c5, "crit1": c1,
                    "rss": temp_ssrs[-1]
                }
                boot_results.append(boot_dict)
                
                if p_val <= alpha:
                    if not nobslog: print(f" Rejected! (p-value = {p_val:.4f}). Threshold validated.")
                    thresholds_accepted = temp_thresh
                    ssrs_accepted = temp_ssrs
                else:
                    if not nobslog: print(f" Not significant (p-value = {p_val:.4f}). Stopping search loop.")
                    break
                    
            thresholds = np.array(thresholds_accepted)
            ssrs = np.array(ssrs_accepted)
            ssr_final = ssrs[-1] if len(ssrs) > 0 else ssr_linear
            
            ci_results, lr_sequences, c_alpha = compute_confidence_intervals(
                Y_w, X_w, R_mat, Q_mat, N, T, grid_vals, thresholds, ssr_final, 
                (N*T - N - (X_mat.shape[1] + (len(thresholds)+1)*R_mat.shape[1]) - ((T-1) if self.time_fe else 0)), trim, alpha
            )
            
        else:
            thnum = int(thnum)
            if not nobslog: print(f"Fixed estimation of {thnum} threshold(s)...", flush=True)
            
            thresholds, ssrs = fit_sequential_thresholds(Y_w, X_w, R_mat, Q_mat, N, T, trim, grid, thnum, self.time_fe)
            if not nobslog: print(f"Executing Bootstrap ({bs} replications)...", flush=True)
            
            boot_results = bootstrap_all_thresholds(Y_w, X_w, R_mat, Q_mat, N, T, thresholds, trim, grid, bs, n_jobs, self.time_fe)
            
            for i, boot in enumerate(boot_results):
                boot['rss'] = ssrs[i]
                boot['test'] = "Single" if i == 0 else f"{i} vs {i+1}"
                
            ssr_final = ssrs[-1] if len(ssrs) > 0 else ssr_linear
            df_penalty = (T - 1) if self.time_fe else 0
            df_for_ci = (N * T) - N - (X_mat.shape[1] + (len(thresholds) + 1) * R_mat.shape[1]) - df_penalty
            
            ci_results, lr_sequences, c_alpha = compute_confidence_intervals(
                Y_w, X_w, R_mat, Q_mat, N, T, grid_vals, thresholds, ssr_final, df_for_ci, trim, alpha
            )

        sort_idx = np.argsort(thresholds)
        thresholds = thresholds[sort_idx]
        
        if thgiven is None:
            ci_results = [ci_results[i] for i in sort_idx]
            lr_sequences = [lr_sequences[i] for i in sort_idx]
            self.results['ci_results'] = ci_results
            self.results['lr_sequences'] = lr_sequences
            
        beta, se, t_stats, p_values, df_resid, V_beta, df_tests = get_standard_errors(
            Y_w, X_w, R_mat, Q_mat, N, T, thresholds, self.time_fe, self.robust
        )
        
        df_penalty = (T - 1) if self.time_fe else 0
        df_resid -= df_penalty
        if not self.robust:
            df_tests -= df_penalty
        
        self.results['ssr_linear'] = ssr_linear
        self.results['Q_mat'] = Q_mat
        self.results['trim'] = trim
        
        if gen is not None:
            regime_array = np.zeros(len(self.data), dtype=int)
            q_series = self.data[self.qx].values
            for g in thresholds: regime_array += (q_series > g).astype(int)
            self.data[gen] = regime_array + 1

        n_regimes = len(thresholds) + 1
        K_total = K_X + n_regimes * K_R
        n_obs = N * T
        
        Z_raw = np.empty((n_obs, K_total))
        if K_X > 0: Z_raw[:, :K_X] = X_mat
        for i in range(n_obs):
            r_idx = sum(Q_mat[i] > g for g in thresholds)
            for reg in range(n_regimes):
                Z_raw[i, K_X + reg*K_R : K_X + (reg+1)*K_R] = R_mat[i, :] if reg == r_idx else 0.0
                    
        means_Z = np.mean(Z_raw, axis=0).reshape(-1, 1)
        _cons = np.mean(Y_mat) - (means_Z.T @ beta)[0, 0]
        
        sigma2 = ssr_final / df_resid if not np.isnan(ssr_final) else np.nan
        t_crit = t_dist.ppf(1 - alpha / 2, df_tests)
        
        if not np.isnan(sigma2):
            var_cons = (sigma2 / n_obs) + (means_Z.T @ V_beta @ means_Z)[0, 0]
            se_cons = np.sqrt(var_cons)
            t_cons = _cons / se_cons
            p_cons = 2 * t_dist.sf(np.abs(t_cons), df=df_tests)
            ci_l_cons = _cons - t_crit * se_cons
            ci_u_cons = _cons + t_crit * se_cons
        else:
            se_cons, t_cons, p_cons, ci_l_cons, ci_u_cons = np.nan, np.nan, np.nan, np.nan, np.nan

        ci_l = beta.flatten() - t_crit * se.flatten()
        ci_u = beta.flatten() + t_crit * se.flatten()

        Xb = Z_raw @ beta
        Y_bar_i = np.mean(Y_mat.reshape(N, T), axis=1)
        Z_bar_i = np.mean(Z_raw.reshape(N, T, K_total), axis=1)
        Xb_bar_i = Z_bar_i @ beta
        u_i = Y_bar_i - Xb_bar_i.flatten() - _cons
        
        sigma_u = np.std(u_i, ddof=1)
        sigma_e = np.sqrt(sigma2) if not np.isnan(sigma2) else np.nan
        rho = sigma_u**2 / (sigma_u**2 + sigma_e**2) if not np.isnan(sigma_e) else np.nan
        
        u_i_expanded = np.repeat(u_i, T)
        corr_u_xb = np.corrcoef(u_i_expanded, Xb.flatten())[0, 1]
        
        SST_within = np.sum(Y_w ** 2)
        r2_within = 1.0 - (ssr_final / SST_within) if not np.isnan(ssr_final) else np.nan
        try: r2_between = np.corrcoef(Y_bar_i, Xb_bar_i.flatten())[0, 1]**2
        except: r2_between = np.nan
        try: r2_overall = np.corrcoef(Y_mat.flatten(), Xb.flatten())[0, 1]**2
        except: r2_overall = np.nan
        
        if self.robust:
            F_overall = float((beta.T @ np.linalg.inv(V_beta) @ beta)[0, 0] / K_total)
        else:
            F_overall = (r2_within / K_total) / ((1.0 - r2_within) / df_resid) if not np.isnan(r2_within) else np.nan
            
        F_pval = f_dist.sf(F_overall, K_total, df_tests) if not np.isnan(F_overall) else np.nan
        
        Z_pool = np.hstack([np.ones((n_obs, 1)), Z_raw])
        beta_pool = np.linalg.solve(Z_pool.T @ Z_pool, Z_pool.T @ Y_mat)
        ssr_pool = np.sum((Y_mat - Z_pool @ beta_pool)**2)
        F_ui = ((ssr_pool - ssr_final) / (N - 1)) / (ssr_final / df_resid) if not np.isnan(ssr_final) else np.nan
        P_ui = f_dist.sf(F_ui, N - 1, df_resid) if not np.isnan(F_ui) else np.nan
        
        self.results.update({
            'N': N, 'T': T, 'n_obs': n_obs, 'df_resid': df_resid, 'df_tests': df_tests, 'K_total': K_total,
            'thresholds': thresholds, 'ci_results': ci_results, 'boot_results': boot_results,
            'beta': beta, 'se': se, 't_stats': t_stats, 'p_values': p_values, 'ci_l': ci_l, 'ci_u': ci_u,
            '_cons': _cons, 'se_cons': se_cons, 't_cons': t_cons, 'p_cons': p_cons, 'ci_l_cons': ci_l_cons, 'ci_u_cons': ci_u_cons,
            'r2_w': r2_within, 'r2_b': r2_between, 'r2_o': r2_overall, 
            'F_overall': F_overall, 'F_pval': F_pval, 'sigma_u': sigma_u, 'sigma_e': sigma_e, 
            'rho': rho, 'corr_u_xb': corr_u_xb, 'F_ui': F_ui, 'P_ui': P_ui,
            'thlevel': thlevel, 'thgiven': thgiven is not None, 'ssrs': ssrs, 'grid': grid_vals
        })
        if thgiven is None:
            self.results['lr_sequences'] = lr_sequences
            self.results['c_alpha'] = c_alpha
            
        self.is_fitted = True
        return self

    def plot_diagnostics(self):
        if not self.is_fitted: raise ValueError("You must call .fit() before running diagnostics.")
        if self.results['thgiven']: raise ValueError("Diagnostics are not available when thgiven is used.")
        from .plotting import plot_diagnostics_panel
        plot_diagnostics_panel(self.results['ssr_linear'], self.results['ssrs'], self.results['Q_mat'].flatten(), self.results['thresholds'])

    def plot_dynamics(self):
        if not self.is_fitted: raise ValueError("You must call .fit() before running diagnostics.")
        if self.results['thgiven']: raise ValueError("Temporal dynamics are not available when thgiven is used.")
        from .plotting import plot_regime_dynamics
        plot_regime_dynamics(self.results['Q_mat'].flatten(), self.results['thresholds'], self.results['N'], self.results['T'])
        
    def plot(self):
        if not self.is_fitted: raise ValueError("You must call .fit() before plotting.")
        if self.results['thgiven']: raise ValueError("Plot not available when thgiven is used.")
        from .plotting import plot_hansen_lr
        if len(self.results['thresholds']) == 0:
            print("No thresholds to plot (Optimal model is linear).")
            return
        plot_hansen_lr(self.results['grid'], self.results['lr_sequences'], self.results['thresholds'], 
                       self.results['ci_results'], self.results['c_alpha'])

    def to_latex(self, filepath=None):
        import io
        import sys
        if not self.is_fitted: 
            raise ValueError("You must call .fit() before exporting to LaTeX.")
        capture = io.StringIO()
        sys.stdout = capture
        self.summary()
        sys.stdout = sys.__stdout__
        full_text = capture.getvalue()
        latex_str = "% Exact raw output from PyXthreg estimation\n\\begin{verbatim}\n" + full_text + "\\end{verbatim}\n"
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(latex_str)
            print(f"LaTeX output (verbatim) successfully saved to: {filepath}")
        return latex_str
    
    def summary(self, noreg=False):
        if not self.is_fitted: raise ValueError("You must call .fit() before generating a summary.")
        res = self.results
        W = 78 
        def fmt_p(val): return "0.0000" if val < 0.0001 else f"{val:.4f}"
        
        print(f"\n{'PANEL THRESHOLD REGRESSION RESULTS':^{W}}")
        print("="*W)
        
        est_name = "Hansen FE (Robust)" if getattr(self, 'robust', False) else "Hansen Threshold FE"
        
        header_rows = [
            ("Dep. Variable:", self.dep, "R-squared (Within):", f"{res['r2_w']:.4f}"),
            ("Estimator:", est_name, "R-squared (Between):", f"{res['r2_b']:.4f}"),
            ("No. Observations:", f"{res['n_obs']}", "R-squared (Overall):", f"{res['r2_o']:.4f}"),
            ("No. Groups (N):", f"{res['N']}", f"F-stat ({res['K_total']},{res['df_tests']}):", f"{res['F_overall']:.2f}"),
            ("Obs per Group (T):", f"{res['T']}", "Prob (F-stat):", f"{fmt_p(res['F_pval'])}")
        ]
        for r in header_rows: print(f"{r[0]:<18}{r[1]:<20}{r[2]:<22}{r[3]:>18}")
        
        if not res['thgiven']:
            print(f"\n{'Threshold Estimates (Level = ' + str(res['thlevel']) + '%)':^{W}}")
            print("="*W)
            if len(res['thresholds']) == 0:
                print(f"{'0 thresholds validated (Linear model is optimal)':^{W}}")
            else:
                print(f"{'Model':<15}{'Threshold':>20}{'Lower CI':>21}{'Upper CI':>22}")
                print("-" * W)
                for i, thresh in enumerate(res['thresholds']):
                    ci_l, ci_u = res['ci_results'][i]
                    print(f"Th-{i+1:<12}{thresh:>20.4f}{ci_l:>21.4f}{ci_u:>22.4f}")
                
            print(f"\n{'Threshold Effect Test (Bootstrap)':^{W}}")
            print("="*W)
            print(f"{'Test':<10}{'RSS':>11}{'MSE':>10}{'F-stat':>10}{'Prob':>9}{'Crit10':>10}{'Crit5':>9}{'Crit1':>9}")
            print("-" * W)
            for boot in res['boot_results']:
                rss = boot.get('rss', np.nan)
                df_boot = res['N'] * res['T'] - res['T'] - ((res['T'] - 1) if getattr(self, 'time_fe', False) else 0)
                mse = rss / df_boot if not np.isnan(rss) else np.nan
                print(f"{boot['test']:<10}{rss:>11.4f}{mse:>10.4f}{boot['F_stat']:>10.2f}{fmt_p(boot['p_value']):>9}{boot['crit10']:>10.2f}{boot['crit5']:>9.2f}{boot['crit1']:>9.2f}")
        
        if noreg: return print("="*W + "\n")
            
        print(f"\n{'Fixed-Effects (Within) Regression Coefficients':^{W}}")
        print("="*W)
        ci_left_lbl, ci_right_lbl = f"[{ (100 - res['thlevel'])/2/100 :.3f}", f"{ 1 - (100 - res['thlevel'])/2/100 :.3f}]"
        
        se_label = "Robust SE" if getattr(self, 'robust', False) else "std err"
        print(f"{'Variable':<16}{'coef':>12}{se_label:>12}{'t':>9}{'P>|t|':>9}{ci_left_lbl:>10}{ci_right_lbl:>10}")
        print("-" * W)
        
        idx = 0
        for name in self.indep:
            coef, se, t_val, p_val = res['beta'][idx,0], res['se'][idx,0], res['t_stats'][idx,0], res['p_values'][idx,0]
            print(f"{name:<16}{coef:>12.4f}{se:>12.4f}{t_val:>9.2f}{fmt_p(p_val):>9}{res['ci_l'][idx]:>10.4f}{res['ci_u'][idx]:>10.4f}")
            idx += 1
        for reg in range(len(res['thresholds']) + 1):
            for var in self.rx:
                name = f"{var} (Regime {reg+1})"
                coef, se, t_val, p_val = res['beta'][idx,0], res['se'][idx,0], res['t_stats'][idx,0], res['p_values'][idx,0]
                print(f"{name:<16}{coef:>12.4f}{se:>12.4f}{t_val:>9.2f}{fmt_p(p_val):>9}{res['ci_l'][idx]:>10.4f}{res['ci_u'][idx]:>10.4f}")
                idx += 1
                
        print(f"{'_cons':<16}{res['_cons']:>12.4f}{res['se_cons']:>12.4f}{res['t_cons']:>9.2f}{fmt_p(res['p_cons']):>9}{res['ci_l_cons']:>10.4f}{res['ci_u_cons']:>10.4f}")

        print(f"\n{'Variance Components':^{W}}")
        print("="*W)
        var_rows = [
            ("sigma_u", f"{res['sigma_u']:.4f}", "Corr(u_i, Xb):", f"{res['corr_u_xb']:.4f}"),
            ("sigma_e", f"{res['sigma_e']:.4f}", f"F-test (u_i=0) F({res['N']-1},{res['df_resid']}):", f"{res['F_ui']:.2f}"),
            ("rho (variance due to u_i)", f"{res['rho']:.4f}", "Prob > F:", fmt_p(res['P_ui']))
        ]
        for r in var_rows: print(f"{r[0]:<32}{r[1]:>8}    {r[2]:<28}{r[3]:>6}")
        print("="*W + "\n")