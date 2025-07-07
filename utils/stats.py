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
        if std_dev == 0: return float('inf'), float('inf'), float('inf'), float('inf')
        
        pp = (usl - lsl) / (6 * std_dev)
        ppk = min((usl - mean) / (3 * std_dev), (mean - lsl) / (3 * std_dev))
        # For this tool, Cp/Cpk use the same overall stdev as Pp/Ppk
        return pp, ppk, pp, ppk
    except Exception as e:
        logger.error(f"Error in process capability calculation: {e}", exc_info=True)
        return 0.0, 0.0, 0.0, 0.0

def calculate_gage_rr(df: pd.DataFrame) -> Tuple[pd.DataFrame, go.Figure, go.Figure]:
    """Performs Gage R&R analysis and returns a results table and plots."""
    from six_sigma.utils.plotting import create_gage_rr_plots
    
    formula = 'measurement ~ C(operator) + C(part_id) + C(operator):C(part_id)'
    model = ols(formula, data=df).fit()
    anova_table = anova_lm(model, typ=2)
    
    n_parts, n_ops, n_reps = df['part_id'].nunique(), df['operator'].nunique(), df.groupby(['part_id', 'operator']).size().iloc[0]
    
    ms_op, ms_part, ms_interact, ms_error = anova_table.loc[anova_table.index.str.contains('operator'), 'mean_sq'].iloc[0], anova_table.loc['C(part_id)', 'mean_sq'], anova_table.loc[anova_table.index.str.contains('operator:part_id'), 'mean_sq'].iloc[0], anova_table.loc['Residual', 'mean_sq']
    
    var_repeat = ms_error
    var_repro_op = max(0, (ms_op - ms_interact) / (n_parts * n_reps))
    var_repro_int = max(0, (ms_interact - ms_error) / n_reps)
    var_repro = var_repro_op + var_repro_int
    var_part = max(0, (ms_part - ms_interact) / (n_ops * n_reps))
    var_grr = var_repeat + var_repro
    total_var = var_grr + var_part
    
    results = {
        'Source': ['Total Gage R&R', '  Repeatability', '  Reproducibility', 'Part-to-Part', 'Total Variation'],
        'Variance': [var_grr, var_repeat, var_repro, var_part, total_var]
    }
    results_df = pd.DataFrame(results).set_index('Source')
    results_df['% Contribution'] = (results_df['Variance'] / total_var) * 100 if total_var > 0 else 0
    
    main_plot, components_plot = create_gage_rr_plots(df)
    return results_df, main_plot, components_plot

def perform_t_test(sample1: pd.Series, sample2: pd.Series, name1: str, name2: str) -> Tuple[go.Figure, Dict[str, float]]:
    """Performs an independent 2-sample t-test and returns a plot and results."""
    ttest_res = stats.ttest_ind(sample1, sample2, equal_var=False) # Welch's t-test
    
    df_plot = pd.DataFrame({name1: sample1, name2: sample2})
    fig = px.box(df_plot, points='all', title=f"Comparison: {name1} vs. {name2}")
    
    result = {'p_value': ttest_res.pvalue, 't_statistic': ttest_res.statistic}
    return fig, result

def perform_anova(*series: pd.Series, series_names: List[str]) -> Tuple[go.Figure, Dict[str, float]]:
    """Performs a one-way ANOVA and returns a plot and results."""
    anova_res = stats.f_oneway(*series)
    
    df_plot = pd.DataFrame({name: s for name, s in zip(series_names, series)})
    fig = px.box(df_plot, points='all', title=f"ANOVA Comparison: {', '.join(series_names)}")
    
    result = {'p_value': anova_res.pvalue, 'f_statistic': anova_res.statistic}
    return fig, result
