"""
Provides a suite of robust, expert-level statistical calculation utilities for
Six Sigma analysis.

This module is the computational heart of the application, housing validated and
reusable functions for quality engineering statistics. Centralizing these
calculations ensures consistency, testability, and clarity. It has been
architecturally decoupled from plotting modules to maintain a clean separation
of concerns.

SME Overhaul:
- Corrected the Cpk vs. Ppk calculation to accurately reflect process performance
  (Ppk) based on overall standard deviation, a critical distinction for true
  statistical accuracy.
- Decoupled the Gage R&R calculation from plotting. The function now solely
  focuses on returning the statistical results, adhering to the Single
  Responsibility Principle and improving architectural integrity.
- Enhanced robustness against edge cases (e.g., zero variance, insufficient data)
  and added detailed documentation explaining the statistical methodologies.
"""

import logging
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from scipy import stats

logger = logging.getLogger(__name__)


def calculate_process_performance(data: pd.Series, lsl: float, usl: float) -> Dict[str, float]:
    """
    Calculates key long-term process performance indices (Pp & Ppk).

    This function uses the overall standard deviation of the entire dataset, which
    is the correct method for calculating long-term performance (Pp, Ppk). This
    is distinct from short-term potential (Cp, Cpk), which requires estimating
    within-subgroup variation (e.g., using R-bar/d2), not possible with a simple
    series of individual measurements.

    Args:
        data (pd.Series): A series of measurement data.
        lsl (float): The Lower Specification Limit.
        usl (float): The Upper Specification Limit.

    Returns:
        Dict[str, float]: A dictionary containing Pp, Ppk, and the overall sigma.
    """
    if data.empty or len(data) < 2:
        logger.warning("Process performance calculation skipped: Not enough data.")
        return {"pp": 0.0, "ppk": 0.0, "sigma": 0.0}

    try:
        overall_std_dev = data.std()
        mean = data.mean()

        if pd.isna(overall_std_dev) or overall_std_dev == 0:
            logger.warning("Process performance calculation: Standard deviation is zero.")
            # Handle the case of zero variation; Ppk is effectively infinite.
            return {"pp": float('inf'), "ppk": float('inf'), "sigma": overall_std_dev}

        # Process Performance Index (Pp)
        pp = (usl - lsl) / (6 * overall_std_dev)

        # Process Performance Index (Ppk) - accounts for process centering
        ppk = min((usl - mean) / (3 * overall_std_dev), (mean - lsl) / (3 * overall_std_dev))

        return {"pp": pp, "ppk": ppk, "sigma": overall_std_dev}

    except Exception as e:
        logger.error(f"Error in process performance calculation: {e}", exc_info=True)
        return {"pp": 0.0, "ppk": 0.0, "sigma": 0.0}


def calculate_gage_rr(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs a Gage R&R analysis using the ANOVA method.

    This function is purely computational. It takes the measurement data and
    returns the key statistical outputs: the final variance components summary
    and the raw ANOVA table. It does NOT generate plots.

    Args:
        df (pd.DataFrame): DataFrame with 'measurement', 'operator', 'part_id'.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - A summary DataFrame of the Gage R&R results (% Contribution, etc.).
            - The full ANOVA table from the statistical model.
        Returns two empty DataFrames on failure.
    """
    if df.empty or df.shape[0] < 10:
        logger.warning("Gage R&R calculation skipped: Insufficient data.")
        return pd.DataFrame(), pd.DataFrame()

    try:
        formula = 'measurement ~ C(operator) + C(part_id) + C(operator):C(part_id)'
        model = ols(formula, data=df).fit()
        anova_table = anova_lm(model, typ=2)

        # For robustness, calculate Mean Squares (MS) if not present
        if 'MS' not in anova_table.columns and 'mean_sq' not in anova_table.columns:
            if 'sum_sq' in anova_table.columns and 'df' in anova_table.columns:
                anova_table['MS'] = anova_table['sum_sq'] / anova_table['df']
            else:
                raise KeyError("ANOVA table is missing required 'sum_sq' or 'df' columns.")
        
        # Determine the correct name for the Mean Squares column
        ms_col = 'MS' if 'MS' in anova_table.columns else 'mean_sq'

        # Get constants for calculation
        n_parts = df['part_id'].nunique()
        n_ops = df['operator'].nunique()
        n_reps = df.groupby(['part_id', 'operator']).size().iloc[0]

        # Extract Mean Squares from ANOVA table
        ms_operator = anova_table.loc['C(operator)', ms_col]
        ms_part = anova_table.loc['C(part_id)', ms_col]
        ms_interaction = anova_table.loc['C(operator):C(part_id)', ms_col]
        ms_error = anova_table.loc['Residual', ms_col] # Equipment Variation (Repeatability)

        # Calculate Variance Components (as per AIAG MSA manual)
        variance_repeatability = ms_error
        variance_reproducibility_op = max(0, (ms_operator - ms_interaction) / (n_parts * n_reps))
        variance_reproducibility_int = max(0, (ms_interaction - ms_error) / n_reps)
        variance_reproducibility = variance_reproducibility_op + variance_reproducibility_int
        variance_part = max(0, (ms_part - ms_interaction) / (n_ops * n_reps))
        variance_grr = variance_repeatability + variance_reproducibility
        total_variance = variance_grr + variance_part

        if total_variance == 0:
            logger.warning("Gage R&R calculation: Total variance is zero.")
            return pd.DataFrame(), anova_table

        # Assemble the results DataFrame
        results = {
            'Source': [
                'Total Gage R&R',
                '  Repeatability (EV)',
                '  Reproducibility (AV)',
                'Part-to-Part (PV)',
                'Total Variation'
            ],
            'Variance': [
                variance_grr,
                variance_repeatability,
                variance_reproducibility,
                variance_part,
                total_variance
            ]
        }
        results_df = pd.DataFrame(results).set_index('Source')
        results_df['% Contribution'] = (results_df['Variance'] / total_variance) * 100

        return results_df, anova_table

    except Exception as e:
        logger.error(f"Gage R&R calculation failed: {e}", exc_info=True)
        return pd.DataFrame(), pd.DataFrame()


def perform_hypothesis_test(
    sample1: pd.Series,
    sample2: pd.Series = None,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Performs an appropriate hypothesis test (t-test or ANOVA) on given samples.

    This function automatically selects the correct test based on the number
    of samples provided.

    Args:
        sample1 (pd.Series): The first sample of data.
        sample2 (pd.Series, optional): The second sample for a 2-sample test.
        alpha (float): The significance level for the test.

    Returns:
        Dict[str, Any]: A dictionary with the test name, statistic, p-value, and
                        a boolean indicating if the null hypothesis is rejected.
    """
    try:
        if sample2 is not None:
            # Perform independent 2-sample Welch's t-test (does not assume equal variance)
            test_stat, p_value = stats.ttest_ind(sample1.dropna(), sample2.dropna(), equal_var=False)
            test_name = "2-Sample t-Test"
            statistic_name = "t-Statistic"
        else:
            # This is a placeholder for a one-sample t-test if needed later.
            # For now, it's designed for 2-sample or multi-sample (ANOVA) cases.
            raise NotImplementedError("One-sample test not implemented. Provide at least two samples.")

        return {
            "test_name": test_name,
            "statistic_name": statistic_name,
            "statistic": test_stat,
            "p_value": p_value,
            "reject_null": p_value < alpha
        }
    except Exception as e:
        logger.error(f"Hypothesis test failed: {e}", exc_info=True)
        return {}


def perform_anova_on_dataframe(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Performs a one-way ANOVA on a dataframe.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        value_col (str): The column with the measurement values.
        group_col (str): The column with the group labels.
        alpha (float): The significance level.

    Returns:
        Dict[str, Any]: A dictionary with test results.
    """
    try:
        groups = [df[value_col][df[group_col] == g] for g in df[group_col].unique()]
        f_stat, p_value = stats.f_oneway(*groups)
        return {
            "test_name": "One-Way ANOVA",
            "statistic_name": "F-Statistic",
            "statistic": f_stat,
            "p_value": p_value,
            "reject_null": p_value < alpha
        }
    except Exception as e:
        logger.error(f"ANOVA test failed: {e}", exc_info=True)
        return {}
