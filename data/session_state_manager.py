# six_sigma/data/session_state_manager.py
"""
Manages the application's session state, acting as an in-memory data source for
a Master Black Belt's quality improvement program.

SME Overhaul: This module generates a highly sophisticated and interconnected
dataset tailored for advanced Six Sigma analysis. It includes multi-step process
yields for RTY, granular internal/external COPQ data, and structured datasets for
Gage R&R, DOE, and hypothesis testing, providing a realistic foundation for an
expert-level toolkit.
"""

import logging
import random
from datetime import date, timedelta
from typing import Any, Dict, List, Optional
import streamlit as st
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def _create_mbb_model(version: int) -> Dict[str, Any]:
    """Generates the complete dataset for the Six Sigma DMAIC Command Center."""
    np.random.seed(42)
    random.seed(42)
    base_date = date.today() - timedelta(days=365)
    sites = ["Andover, US", "Eindhoven, NL", "Shanghai, CN"]

    # --- 1. Multi-Step Process Yield Data (for FTY/RTY) ---
    yield_data = []
    process_steps = ["Component Kitting", "Sub-Assembly", "Main Assembly", "Final QC Test"]
    for i in range(365):
        current_date = base_date + timedelta(days=i)
        for site in sites:
            base_yields = [0.995, 0.97, 0.95, 0.98] if "Eindhoven" in site else [0.99, 0.95, 0.92, 0.96]
            units_in = 1000
            for j, step in enumerate(process_steps):
                daily_yield = base_yields[j] * (1 + (i / 3650)) + np.random.uniform(-0.005, 0.005)
                daily_yield = min(daily_yield, 0.999)
                units_out = int(units_in * daily_yield)
                yield_data.append({
                    "date": current_date, "site": site, "step_name": step,
                    "units_in": units_in, "units_out": units_out, "step_yield": daily_yield
                })
                units_in = units_out

    # --- 2. Granular COPQ Data ---
    copq_data = []
    internal_failures = {"Scrap": 500, "Rework": 150, "Re-test": 75}
    external_failures = {"Warranty Claim": 2500, "Customer Complaint": 50, "Field Service": 800}
    for i in range(500):
        is_internal = random.random() > 0.3
        failure_type = "Internal" if is_internal else "External"
        category = random.choice(list(internal_failures.keys())) if is_internal else random.choice(list(external_failures.keys()))
        cost = internal_failures.get(category) or external_failures.get(category)
        copq_data.append({
            "date": base_date + timedelta(days=random.randint(0, 364)),
            "site": random.choice(sites),
            "failure_type": failure_type, "category": category,
            "cost": int(cost * (1 + np.random.uniform(-0.1, 0.1)))
        })

    # --- 3. Detailed DMAIC Project Charters ---
    dmaic_projects = [
        {"id": "DMAIC-001", "site": "Andover, US", "product_line": "HeartStart Defibrillator", "title": "Reduce Sub-Assembly Step Defects", "phase": "Analyze",
         "problem_statement": "The 'Sub-Assembly' step has a First Time Yield (FTY) of 95%, causing significant downstream rework and scrap. This is the primary bottleneck in the production line.",
         "goal_statement": "Increase the FTY of the 'Sub-Assembly' step from 95% to >99% by Q4, improving overall Rolled Throughput Yield (RTY) by 4 percentage points and reducing associated COPQ.",
         "team": ["John Smith (MBB Lead)", "Jane Doe (Engineer)", "Mike Ross (Ops)"], "start_date": str(base_date + timedelta(days=180)),
         "baseline_cpk": 0.85, "baseline_dpmo": 22750},
        {"id": "DMAIC-002", "site": "Eindhoven, NL", "product_line": "IntelliVue Patient Monitor", "title": "Optimize Display Module Bonding Process", "phase": "Improve",
         "problem_statement": "The display module bonding process has high variability in bond strength, leading to a Ppk of 0.7. Failures are caught in final QC, resulting in $150k/yr in internal scrap costs.",
         "goal_statement": "Identify significant factors in the bonding process via DOE and implement controls to increase Ppk to >1.33, reducing scrap costs by 80%.",
         "team": ["Sofia Chen (BB Lead)", "David Lee (Engineer)"], "start_date": str(base_date + timedelta(days=200)),
         "baseline_cpk": 0.72, "baseline_dpmo": 45500}
    ]

    # --- 4. Structured Datasets for Statistical Tools ---
    gage_data = []
    parts = range(1, 11)
    operators = ["Operator A", "Operator B", "Operator C"]
    for part in parts:
        true_value = 10 + part * 0.1
        for operator in operators:
            op_bias = 0.01 if operator == "Operator B" else -0.01 if operator == "Operator C" else 0
            for replicate in range(3):
                measurement = true_value + op_bias + np.random.normal(0, 0.05)
                gage_data.append({"part_id": part, "operator": operator, "replicate": replicate + 1, "measurement": measurement})

    doe_df = pd.DataFrame([
        {'temp': -1, 'time': -1, 'pressure': -1, 'strength': 75.3}, {'temp': 1, 'time': -1, 'pressure': -1, 'strength': 78.9},
        {'temp': -1, 'time': 1, 'pressure': -1, 'strength': 85.1}, {'temp': 1, 'time': 1, 'pressure': -1, 'strength': 95.2},
        {'temp': -1, 'time': -1, 'pressure': 1, 'strength': 76.1}, {'temp': 1, 'time': -1, 'pressure': 1, 'strength': 79.5},
        {'temp': -1, 'time': 1, 'pressure': 1, 'strength': 84.8}, {'temp': 1, 'time': 1, 'pressure': 1, 'strength': 96.0},
        {'temp': 0, 'time': 0, 'pressure': 0, 'strength': 92.5}, {'temp': 0, 'time': 0, 'pressure': 0, 'strength': 91.8},
    ])
    
    ht_data = pd.DataFrame({
        'before_change': np.random.normal(loc=25.5, scale=1.2, size=30), 'after_change': np.random.normal(loc=26.5, scale=1.1, size=30),
        'supplier_a': np.random.normal(loc=50.1, scale=0.5, size=50), 'supplier_b': np.random.normal(loc=50.3, scale=0.5, size=50),
        'supplier_c': np.random.normal(loc=50.1, scale=0.5, size=50),
    })
    
    process_data_df = pd.DataFrame({'timestamp': pd.to_datetime(pd.date_range(start=base_date, periods=200, freq='D')), 'seal_strength': np.concatenate([np.random.normal(85, 2.5, 100), np.random.normal(82, 2.8, 100)])})
    process_specs = {"seal_strength": {"lsl": 78, "usl": 92, "target": 85}}

    # --- Assemble Final Data Model ---
    return {
        "data_version": version, "yield_data": yield_data, "copq_data": copq_data,
        "dmaic_projects": dmaic_projects,
        "gage_rr_data": pd.DataFrame(gage_data), "doe_data": doe_df, "hypothesis_testing_data": ht_data,
        "process_data": process_data_df, "process_specs": process_specs
    }

class SessionStateManager:
    _DATA_KEY = "six_sigma_mbb_data"
    _CURRENT_DATA_VERSION = 5 # Incremented for final MBB overhaul

    def __init__(self):
        session_data = st.session_state.get(self._DATA_KEY)
        if not session_data or session_data.get("data_version") != self._CURRENT_DATA_VERSION:
            logger.info(f"Initializing session state with MBB data model v{self._CURRENT_DATA_VERSION}.")
            try:
                st.session_state[self._DATA_KEY] = _create_mbb_model(self._CURRENT_DATA_VERSION)
                logger.info("Session state initialized successfully.")
            except Exception as e:
                logger.critical(f"FATAL: Data generation failed: {e}", exc_info=True)
                st.error(f"A critical error occurred during application startup: {e}.", icon="🚨")
                st.stop()

    def get_data(self, key: str) -> Any:
        return st.session_state.get(self._DATA_KEY, {}).get(key)
