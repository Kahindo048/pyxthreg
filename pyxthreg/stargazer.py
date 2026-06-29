import numpy as np

class PanelStargazer:
    """
    Generator for multi-specification academic publication tables.
    
    Aligns multiple estimated models in columns, manages strict variable alignment, 
    and automatically appends standard econometric significance stars 
    (using safe LaTeX math mode typography).
    
    Parameters
    ----------
    models : ThresholdPanel or list of ThresholdPanel
        A single fitted model or a list of fitted ThresholdPanel models to be 
        compared side-by-side.
        
    Raises
    ------
    ValueError
        If any of the provided models have not been fitted prior to initialization.
    """
    def __init__(self, models):
        if not isinstance(models, list):
            models = [models]
        self.models = models
        self.var_names = []
        
        # Unique mapping of all variables across all provided models
        for m in self.models:
            if not m.is_fitted:
                raise ValueError("All models must be fitted using .fit() prior to export.")
            
            # Reconstruct the list of variables for the current model
            m_vars = list(m.indep)
            n_regimes = len(m.results['thresholds']) + 1
            for reg in range(n_regimes):
                for var in m.rx:
                    m_vars.append(f"{var} (Regime {reg+1})")
            m_vars.append("_cons")
            
            # Add to the global list without duplicates (to preserve logical ordering)
            for v in m_vars:
                if v not in self.var_names:
                    self.var_names.append(v)

    def _get_stars(self, p_value):
        """
        Assigns econometric significance stars formatted for LaTeX math mode.
        
        Parameters
        ----------
        p_value : float
            The p-value associated with a coefficient.
            
        Returns
        -------
        str
            LaTeX formatted string representing the significance level.
        """
        if np.isnan(p_value): return ""
        if p_value < 0.01: return "^{***}"
        if p_value < 0.05: return "^{**}"
        if p_value < 0.10: return "^{*}"
        return ""

    def to_latex(self, filepath=None, label="tab:stargazer", caption="Comparison of Multiple Threshold Specifications"):
        """
        Generates the LaTeX source code for the comparative Stargazer table.
        
        Parameters
        ----------
        filepath : str, optional
            File path to save the generated LaTeX code (e.g., 'table.tex').
        label : str, optional
            LaTeX label for cross-referencing in the document, by default "tab:stargazer".
        caption : str, optional
            Title of the LaTeX table.
            
        Returns
        -------
        str
            The complete LaTeX table code as a string.
        """
        n_models = len(self.models)
        col_spec = "l" + "c" * n_models
        
        tex = []
        tex.append("% Remember to include \\usepackage{booktabs} in your LaTeX preamble")
        tex.append("\\begin{table}[htbp]")
        tex.append("\\centering")
        tex.append("\\small")
        tex.append(f"\\caption{{{caption}}}")
        tex.append(f"\\label{{{label}}}")
        tex.append(f"\\begin{{tabular}}{{{col_spec}}}")
        tex.append("\\toprule")
        
        # Header: Specification numbering (1), (2), etc.
        headers = ["Variable"] + [f"({i+1})" for i in range(n_models)]
        tex.append(" & ".join(headers) + " \\\\")
        tex.append("\\midrule")
        
        # Body filling: Alignment of coefficients and standard errors
        for var in self.var_names:
            safe_var = var.replace("_", "\\_") # Protection for LaTeX compilation
            coef_row = [safe_var]
            se_row = [""]
            
            for m in self.models:
                res = m.results
                
                # Recreate the model's index map to find the target variable
                m_vars = list(m.indep)
                n_regimes = len(res['thresholds']) + 1
                for reg in range(n_regimes):
                    for v in m.rx:
                        m_vars.append(f"{v} (Regime {reg+1})")
                m_vars.append("_cons")
                
                if var in m_vars:
                    idx = m_vars.index(var)
                    if var == "_cons":
                        coef, se, p_val = res['_cons'], res['se_cons'], res['p_cons']
                    else:
                        coef, se, p_val = res['beta'][idx, 0], res['se'][idx, 0], res['p_values'][idx, 0]
                    
                    stars = self._get_stars(p_val)
                    # Strict encapsulation in math mode $ ... $
                    coef_row.append(f"${coef:.4f}{stars}$")
                    se_row.append(f"$({se:.4f})$" if not np.isnan(se) else "")
                else:
                    # The variable does not exist in this specific model specification
                    coef_row.append("")
                    se_row.append("")
            
            tex.append(" & ".join(coef_row) + " \\\\")
            tex.append(" & ".join(se_row) + " \\\\[0.15cm]") 
            
        tex.append("\\midrule")
        
        # Global diagnostic statistics
        obs_row = ["Observations"]
        groups_row = ["Groups (N)"]
        r2_row = ["$R^2$ (Within)"]
        th_row = ["Thresholds"]
        
        for m in self.models:
            res = m.results
            # Positive integers remain as text, but R2 and Thresholds use math mode for proper minus signs
            obs_row.append(f"{res['n_obs']}")
            groups_row.append(f"{res['N']}")
            r2_row.append(f"${res['r2_w']:.4f}$")
            
            if len(res['thresholds']) > 0:
                th_str = ", ".join([f"{t:.3f}" for t in res['thresholds']])
                th_row.append(f"${th_str}$") 
            else:
                th_row.append("None")
            
        tex.append(" & ".join(obs_row) + " \\\\")
        tex.append(" & ".join(groups_row) + " \\\\")
        tex.append(" & ".join(r2_row) + " \\\\")
        tex.append(" & ".join(th_row) + " \\\\")
        
        tex.append("\\bottomrule")
        tex.append("\\multicolumn{" + str(n_models + 1) + "}{l}{\\footnotesize $^{*}$ $p<0.1$; $^{**}$ $p<0.05$; $^{***}$ $p<0.01$} \\\\")
        tex.append("\\end{tabular}")
        tex.append("\\end{table}")
        
        latex_str = "\n".join(tex)
        
        # Saving logic
        if filepath is not None:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(latex_str)
            print(f"Stargazer table successfully saved to: {filepath}")
            
        return latex_str