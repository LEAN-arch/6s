# six_sigma/utils/stats.py
"""
Utility functions for performing standardized statistical calculations.

This module houses reusable functions for quality engineering statistics,
such as process capability analysis. Centralizing these calculations ensures
consistency, testability, and clarity.
"""

import logging
import pandas as pd
import numpy as np
from typing import Tuple

logger = logging.getLogger(__name__)

def calculate_process_capability(
    data_series: pd.Series,
    lsl: float,
    usl: float
) -> Tuple[float, float, float, float]:
    """
    Calculates key process capability and performance indices.

    - Cpk/Cp are based on the short-term standard deviation (within subgroups).
    - Ppk/Pp are based on the overall, long-term standard deviation.
    - For this implementation, we use a simplified approach where the overall
      standard deviation is used for both, which is common when rational
      subgrouping is not available.

    Args:
        data_series (pd.Series): A pandas Series of measurement data.
        lsl (float): The Lower Specification Limit for the process.
        usl (float): The Upper Specification Limit for the process.

    Returns:
        Tuple[float, float, float, float]: A tuple containing Cp, Cpk, Pp, Ppk.
    """
    if data_series.empty or len(data_series) < 2:
        logger.warning("Process capability calculation skipped: data series is empty or too small.")
        return 0.0, 0.0, 0.0, 0.0
        
    try:
        # For simplicity in this tool, we will use the overall standard deviation
        # as the estimator for both short-term and long-term variation. In a more
        # advanced setting, a moving range or pooled standard deviation from
        # subgroups would be used for the Cp/Cpk calculation.
        overall_std = data_series.std()
        mean = data_series.mean()

        if overall_std == 0:
            logger.warning("Process capability calculation skipped: standard deviation is zero.")
            return float('inf'), float('inf'), float('inf'), float('inf')

        # --- Performance Indices (Pp, Ppk) ---
        # These always use the overall standard deviation.
        pp = (usl - lsl) / (6 * overall_std)
        ppk = min((usl - mean) / (3 * overall_std), (mean - lsl) / (3 * overall_std))

        # --- Capability Indices (Cp, Cpk) ---
        # As noted above, we use overall_std as the estimator.
        # In a real-world SPC application, this would be `short_term_std`.
        short_term_std_estimator = overall_std
        cp = (usl - lsl) / (6 * short_term_std_estimator)
        cpk = min((usl - mean) / (3 * short_term_std_estimator), (mean - lsl) / (3 * short_term_std_estimator))

        return cp, cpk, pp, ppk

    except (TypeError, ValueError, ZeroDivisionError) as e:
        logger.error(f"Error during process capability calculation: {e}", exc_info=True)
        return 0.0, 0.0, 0.0, 0.0
