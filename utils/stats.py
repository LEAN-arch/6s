# six_sigma/utils/stats.py
"""
Utility functions for performing standardized, expert-level statistical calculations.

This module houses reusable functions for quality engineering statistics, such as
process capability analysis, Gage R&R, and hypothesis testing. Centralizing these
calculations ensures consistency, testability, and clarity.

SME Overhaul:
- Added `calculate_gage_rr` to perform full ANOVA-based Gage R&R.
- Added `perform_t_test` and `perform_anova` for hypothesis testing.
- Hardened `calculate_process_capability` with more robust checks.
- Functions now return both results and plots for better integration.
"""

import logging
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

def calculate_process_capability(data: pd.Series, lsl: float, usl: float) -> Tuple[float, float, float, float]:
    """Calculates key process capability and performance indices."""
    if data.empty or len(data) < 2: return 0.0, 0.0, 0.0, 0.0
    
    try:
        std_dev = data.std()
        mean = data.mean()
        if std_dev == 0 or pd.isna(std_dev): return float('inf'), float('inf'), float('inf'), float('inf')
        
        pp = (usl - lsl) / (6 * std_dev)
        ppk = min((usl - mean) / (3 * std_dev), (mean - lsl) / (3 * std_dev))
        # For this tool, Cp/Cpk use the same overall stdev as Pp/Ppk
        return pp, ppk, pp, ppk
    except Exception as e:
        logger.error(f"Error in process capability calculation: {e}", exc_info=True)
        return 0.0, 0.0, 0.0, 0.0

def calculate_gage_rr(df: pd.DataFrame) -> Tuple[pd.DataFrame, go.Figure, go.Figure]:
    """Performs Gage R&R analysis using ANOVA method and returns a results table and plots."""
    from six_sigma.utils.plotting import create_gage_rr_plots
    
    try:
        formula = 'measurement ~ C(operator) + C(part_id) + C(operator):C(part_id)'
        model = ols(formula, data=df).fit()
        anova_table = anova_lm(model, typ=2)
        
        n_parts = df['part_id'].nunique()
        n_ops = df['operator'].nunique()
        n_reps = df.groupby(['part_id', 'operator']).size().iloc[0]
        
        ms_op = anova_table.loc['C(operator)', 'mean_sq']
        ms_part = anova_table.loc['C(part_id)', 'mean_sq']
        ms_interact = anova_table.loc['C(operator):C(part_id)', 'mean_sq']
        ms_error = anova_table.loc['Residual', 'mean_sq']
        
        var_repeat = ms_error
        var_repro_op = max(0, (ms_op - ms_interact) / (n_parts * n_reps))
        var_repro_int = max(0, (ms_interact - ms_error) / n_reps)
        var_repro = var_repro_op + var_repro_int
        var_part = max(0, (ms_part - ms_interact) / (n_ops * n_reps))
        var_grr = var_repeat + var_repro
        total_var = var_grr + var_part
        
        results = {
            'Source': ['Total Gage R&R', '  Repeatability (EV)', '  Reproducibility (AV)', 'Part-to-Part', 'Total Variation'],
            'Variance': [var_grr, var_repeat, var_repro, var_part, total_var]
        }
        results_df = pd.DataFrame(results).set_index('Source')
        results_df['% Contribution'] = (results_df['Variance'] / total_var) * 100 if total_var > 0 else 0
        
        fig1, fig2 = create_gage_rr_plots(df)
        return results_df, fig1, fig2
    except Exception as e:
        logger.error(f"Gage R&R calculation failed: {e}", exc_info=True)
        return pd.DataFrame(), go.Figure(), go.Figure()

def perform_t_test(sample1: pd.Series, sample2: pd.Series, name1: str, name2: str) -> Tuple[go.Figure, Dict[str, float]]:
    """Performs an independent 2-sample t-test and returns a plot and results."""
    ttest_res = stats.ttest_ind(sample1, sample2, equal_var=False) # Welch's t-test
    
    df1 = pd.DataFrame({'Group': name1, 'Value': sample1})
    df2 = pd.DataFrame({'Group': name2, 'Value': sample2})
    df_plot = pd.concat([df1, df2])
    
    fig = px.box(df_plot, x='Group', y='Value', points='all', title=f"Comparison: {name1} vs. {name2}", color='Group')
    
    result = {'p_value': ttest_res.pvalue, 't_statistic': ttest_res.statistic}
    return fig, result

def perform_anova(df: pd.DataFrame, value_col: str, group_col: str, title: str) -> Tuple[go.Figure, Dict[str, float]]:
    """Performs a one-way ANOVA from a dataframe and returns a plot and results."""
    groups = [df[value_col][df[group_col] == g] for g in df[group_col].unique()]
    anova_res = stats.f_oneway(*groups)
    
    fig = px.box(df, x=group_col, y=value_col, points='all', title=title, color=group_col)
    
    result = {'p_value': anova_res.pvalue, 'f_statistic': anova_res.statistic}
    return fig, result
