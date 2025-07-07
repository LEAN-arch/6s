# six_sigma/data/session_state_manager.py
"""
Manages the application's session state, acting as an in-memory data source for
a Master Black Belt's quality improvement program.

SME Overhaul: This module generates a highly sophisticated and interconnected
dataset tailored for advanced Six Sigma analysis. It includes multi-step process
yields for RTY, granular internal/external COPQ data, and structured datasets for
Gage R&R, DOE, and modern ML model comparisons.
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
    products_by_site = {
        "Andover, US": ["HeartStart Defibrillator"],
        "Eindhoven, NL": ["IntelliVue Patient Monitor", "Zenition C-arm"],
        "Shanghai, CN": ["Affiniti Ultrasound"]
    }

    # 1. Multi-Step Process Yield Data
    yield_data, process_steps = [], ["Component Kitting", "Sub-Assembly", "Main Assembly", "Final QC Test"]
    rty_by_date = []
    for i in range(365):
        current_date = base_date + timedelta(days=i)
        for site in sites:
            base_yields = [0.995, 0.97, 0.95, 0.98] if "Eindhoven" in site else [0.99, 0.95, 0.92, 0.96]
            units_in, rty = 1000, 1.0
            for j, step in enumerate(process_steps):
                daily_yield = min(base_yields[j] * (1 + (i / 3650)) + np.random.uniform(-0.005, 0.005), 0.999)
                units_out = int(units_in * daily_yield)
                yield_data.append({"date": current_date, "site": site, "step_name": step, "units_in": units_in, "units_out": units_out, "step_yield": daily_yield})
                units_in, rty = units_out, rty * daily_yield
            rty_by_date.append({"date": current_date, "site": site, "rty": rty})

    # 2. Granular COPQ Data
    copq_data, internal_failures, external_failures = [], {"Scrap": 500, "Rework": 150, "Re-test": 75}, {"Warranty Claim": 2500, "Customer Complaint": 50, "Field Service": 800}
    for _ in range(500):
        is_internal, site = random.random() > 0.3, random.choice(sites)
        failure_type = "Internal" if is_internal else "External"
        category = random.choice(list(internal_failures.keys())) if is_internal else random.choice(list(external_failures.keys()))
        cost = (internal_failures.get(category) or external_failures.get(category)) * (1 + np.random.uniform(-0.1, 0.1))
        copq_data.append({"date": base_date + timedelta(days=random.randint(0, 364)), "site": site, "product_line": random.choice(products_by_site[site]), "failure_type": failure_type, "category": category, "cost": int(cost)})

    # 3. Detailed DMAIC Project Charters
    dmaic_projects = [
        {"id": "DMAIC-001", "site": "Andover, US", "product_line": "HeartStart Defibrillator", "title": "Reduce Sub-Assembly Step Defects", "phase": "Analyze", "problem_statement": "The 'Sub-Assembly' step has a First Time Yield (FTY) of 95%, causing significant downstream rework and scrap. This is the primary bottleneck in the production line.", "goal_statement": "Increase the FTY of the 'Sub-Assembly' step from 95% to >99% by Q4, improving overall Rolled Throughput Yield (RTY) by 4 percentage points and reducing associated COPQ.", "team": ["John Smith (MBB Lead)", "Jane Doe (Engineer)", "Mike Ross (Ops)"], "start_date": str(base_date + timedelta(days=180)), "baseline_cpk": 0.85, "baseline_dpmo": 22750},
        {"id": "DMAIC-002", "site": "Eindhoven, NL", "product_line": "IntelliVue Patient Monitor", "title": "Optimize Display Module Bonding Process", "phase": "Improve", "problem_statement": "The display module bonding process has high variability in bond strength, leading to a Ppk of 0.7. Failures are caught in final QC, resulting in $150k/yr in internal scrap costs.", "goal_statement": "Identify significant factors in the bonding process via DOE and implement controls to increase Ppk to >1.33, reducing scrap costs by 80%.", "team": ["Sofia Chen (BB Lead)", "David Lee (Engineer)"], "start_date": str(base_date + timedelta(days=200)), "baseline_cpk": 0.72, "baseline_dpmo": 45500}
    ]

    # 4. Structured Datasets for Statistical Tools
    gage_data = []
    for part in range(1, 11):
        true_value = 10 + part * 0.1
        for operator in ["Operator A", "Operator B", "Operator C"]:
            op_bias = 0.01 if operator == "Operator B" else -0.01 if operator == "Operator C" else 0
            for _ in range(3): gage_data.append({"part_id": part, "operator": operator, "measurement": true_value + op_bias + np.random.normal(0, 0.05)})

    doe_df = pd.DataFrame([{'temp': -1, 'time': -1, 'pressure': -1, 'strength': 75.3}, {'temp': 1, 'time': -1, 'pressure': -1, 'strength': 78.9}, {'temp': -1, 'time': 1, 'pressure': -1, 'strength': 85.1}, {'temp': 1, 'time': 1, 'pressure': -1, 'strength': 95.2}, {'temp': -1, 'time': -1, 'pressure': 1, 'strength': 76.1}, {'temp': 1, 'time': -1, 'pressure': 1, 'strength': 79.5}, {'temp': -1, 'time': 1, 'pressure': 1, 'strength': 84.8}, {'temp': 1, 'time': 1, 'pressure': 1, 'strength': 96.0}, {'temp': 0, 'time': 0, 'pressure': 0, 'strength': 92.5}, {'temp': 0, 'time': 0, 'pressure': 0, 'strength': 91.8}])
    
    ht_data = pd.DataFrame({'before_change': np.random.normal(25.5, 1.2, 50), 'after_change': np.random.normal(26.5, 1.1, 50), 'supplier_a': np.random.normal(50.1, 0.5, 50), 'supplier_b': np.random.normal(50.3, 0.5, 50), 'supplier_c': np.random.normal(50.1, 0.5, 50)})
    
    process_data_df = pd.DataFrame({'timestamp': pd.to_datetime(pd.date_range(start=base_date, periods=200, freq='D')), 'seal_strength': np.concatenate([np.random.normal(85, 2.5, 100), np.random.normal(82, 2.8, 100)])})
    process_specs = {"seal_strength": {"lsl": 78, "usl": 92, "target": 85}}
    
    release_batches = [{"batch_id": f"BATCH-{202400 + i}", "test_measurement": np.random.normal(10.8, 0.8) if random.random() < 0.08 else np.random.normal(10.0, 0.2), "true_status": "Fail" if random.random() < 0.08 else "Pass"} for i in range(150)]
    
    # 5. Data for Modern ML vs. Classical Comparison & Predictive Quality
    ml_data = []
    for _ in range(500):
        temp = np.random.normal(200, 10); pressure = np.random.normal(50, 5); vibration = np.random.normal(1.5, 0.5)
        fail_prob = 1 / (1 + np.exp(-(0.1 * (temp - 210) + 0.5 * (pressure - 52))))
        ml_data.append({"in_process_temp": temp, "in_process_pressure": pressure, "in_process_vibration": vibration, "final_qc_outcome": "Fail" if random.random() < fail_prob else "Pass"})
    
    x_rsm = np.linspace(-3, 3, 50); y_rsm = np.linspace(-3, 3, 50)
    xx, yy = np.meshgrid(x_rsm, y_rsm)
    z_rsm = np.sin(np.sqrt(xx**2 + yy**2)) - (xx**2 + yy**2) * 0.05 + np.random.rand(50, 50) * 0.2
    ml_vs_classical_data = pd.DataFrame({'x': xx.flatten(), 'y': yy.flatten(), 'z': z_rsm.flatten()})

    # 6. Kaizen & Training Data
    kaizen_events = [{"id": "KZN-01", "site": "Eindhoven, NL", "date": base_date + timedelta(days=250), "title": "5S Implementation on Monitor Assembly Line 3", "outcome": "Reduced tool search time by 60%; cleared 25 sq. meters of floor space.", "team": ["Maria Rodriguez", "David Lee"]}, {"id": "KZN-02", "site": "Andover, US", "date": base_date + timedelta(days=300), "title": "Value Stream Mapping of Defibrillator Sub-assembly", "outcome": "Identified 3 non-value-add steps, reducing process cycle time by 15%.", "team": ["John Smith", "Jane Doe"]}]
    training_materials = [{"id": "TRN-001", "title": "Introduction to DMAIC Methodology", "type": "eLearning", "duration_hr": 4, "link": "#", "target_audience": "All"}, {"id": "TRN-002", "title": "Statistical Process Control (SPC) Fundamentals", "type": "PDF Guide", "duration_hr": 2, "link": "#", "target_audience": "Engineers, Technicians"}, {"id": "TRN-003", "title": "Design of Experiments (DOE) Workshop", "type": "Workshop Slides", "duration_hr": 8, "link": "#", "target_audience": "Engineers"}, {"id": "TRN-004", "title": "Root Cause Analysis (Fishbone & 5 Whys)", "type": "eLearning", "duration_hr": 2, "link": "#", "target_audience": "All"}]

    return {
        "data_version": version, "yield_data": yield_data, "copq_data": copq_data,
        "dmaic_projects": dmaic_projects, "global_kpis": rty_by_date,
        "gage_rr_data": pd.DataFrame(gage_data), "doe_data": doe_df, "hypothesis_testing_data": ht_data,
        "process_data": process_data_df, "process_specs": process_specs,
        "release_data": pd.DataFrame(release_batches), "predictive_quality_data": pd.DataFrame(ml_data),
        "ml_vs_classical_data": ml_vs_classical_data,
        "kaizen_events": kaizen_events, "training_materials": training_materials
    }

class SessionStateManager:
    _DATA_KEY = "six_sigma_mbb_data"
    _CURRENT_DATA_VERSION = 7 # Final MBB full overhaul version

    def __init__(self):
        if self._DATA_KEY not in st.session_state or st.session_state[self._DATA_KEY].get("data_version") != self._CURRENT_DATA_VERSION:
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
