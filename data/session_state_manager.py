"""
Manages the application's session state, acting as a sophisticated, in-memory
data source that simulates a real-world quality improvement program for a
Master Black Belt (MBB).

SME Overhaul:
This module has been significantly re-architected to generate a coherent and
interconnected dataset. Instead of random, disconnected data points, it now
creates a compelling narrative. For example, a specific process bottleneck at a
manufacturing site is programmatically linked to increased internal failure costs
(COPQ) and serves as the data-driven justification for a specific DMAIC project
in the pipeline. This creates a realistic, commercial-grade experience for the user.
It has been further extended to generate dedicated datasets for each DMAIC project,
enabling a fully dynamic and context-aware toolkit experience.
"""

import logging
import random
from datetime import date, timedelta
from typing import Any, Dict, List
import streamlit as st
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --- Data Generation Helper Functions ---
# ... (All helper functions, including the new _generate_optimization_data
#      and _generate_failure_clustering_data from the previous version, are correct and remain here) ...
def _generate_yield_data() -> (List[Dict[str, Any]], List[Dict[str, Any]], List[str]):
    # ... code from previous version ...
    yield_data, rty_by_date = [], []
    base_date = date.today() - timedelta(days=365)
    sites = ["Andover, US", "Eindhoven, NL", "Shanghai, CN"]
    process_steps = ["Component Kitting", "Sub-Assembly", "Main Assembly", "Final QC Test"]
    for i in range(365):
        current_date = base_date + timedelta(days=i)
        for site in sites:
            if "Eindhoven" in site: base_yields = [0.995, 0.97, 0.95, 0.98]
            elif "Shanghai" in site: base_yields = [0.99, 0.98, 0.96, 0.97]
            else: base_yields = [0.99, 0.92, 0.95, 0.96]
            units_in, rty = 1000, 1.0
            for j, step in enumerate(process_steps):
                daily_yield = min(base_yields[j] * (1 + (i / 3650)) + np.random.uniform(-0.005, 0.005), 0.999)
                units_out = int(units_in * daily_yield)
                yield_data.append({"date": current_date, "site": site, "step_name": step, "units_in": units_in, "units_out": units_out, "step_yield": daily_yield})
                units_in, rty = units_out, rty * daily_yield
            rty_by_date.append({"date": current_date, "site": site, "rty": rty})
    return yield_data, rty_by_date, process_steps

def _generate_copq_data(sites: List[str], products_by_site: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    # ... code from previous version ...
    copq_data = []
    base_date = date.today() - timedelta(days=365)
    internal_failures = {"Scrap": 500, "Rework": 150, "Re-test": 75}
    external_failures = {"Warranty Claim": 2500, "Customer Complaint": 50, "Field Service": 800}
    for _ in range(700):
        is_internal, site = random.random() > 0.3, random.choice(sites)
        if site == "Andover, US" and random.random() < 0.6:
            failure_type = "Internal"; category = random.choices(list(internal_failures.keys()), weights=[0.5, 0.4, 0.1], k=1)[0]
        else:
            failure_type = "Internal" if is_internal else "External"; category = random.choice(list(internal_failures.keys())) if is_internal else random.choice(list(external_failures.keys()))
        cost = (internal_failures.get(category) or external_failures.get(category)) * (1 + np.random.uniform(-0.1, 0.1))
        copq_data.append({"date": base_date + timedelta(days=random.randint(0, 364)), "site": site, "product_line": random.choice(products_by_site[site]), "failure_type": failure_type, "category": category, "cost": int(cost)})
    return copq_data

def _generate_dmaic_projects(base_date: date) -> List[Dict[str, Any]]:
    # ... code from previous version ...
    return [
        {"id": "DMAIC-001", "site": "Andover, US", "product_line": "HeartStart Defibrillator", "title": "Reduce Sub-Assembly Step Defects", "phase": "Analyze", "problem_statement": "The 'Sub-Assembly' step has a First Time Yield (FTY) of ~92%, well below the target of 98%. This is causing significant downstream rework and scrap, identified as the primary driver of internal failure costs for the Defibrillator line.", "goal_statement": "Increase the FTY of the 'Sub-Assembly' step from 92% to >98% by Q4. This will improve overall Rolled Throughput Yield (RTY) by 6 percentage points and reduce associated rework/scrap COPQ by over $200k annually.", "team": ["John Smith (MBB Lead)", "Jane Doe (Engineer)", "Mike Ross (Ops)"], "start_date": str(base_date + timedelta(days=180)), },
        {"id": "DMAIC-002", "site": "Eindhoven, NL", "product_line": "IntelliVue Patient Monitor", "title": "Optimize Display Module Bonding Process", "phase": "Improve", "problem_statement": "The display module bonding process within Main Assembly has high variability in bond strength, leading to a Ppk of 0.7. Failures are caught in final QC, resulting in $150k/yr in internal scrap costs.", "goal_statement": "Identify significant factors in the bonding process via DOE and implement controls to increase Ppk to >1.33, reducing scrap costs by 80%.", "team": ["Sofia Chen (BB Lead)", "David Lee (Engineer)"], "start_date": str(base_date + timedelta(days=200)), }
    ]

def _generate_dmaic_data() -> Dict[str, Any]:
    # ... code from previous version ...
    project_data_001 = {"baseline": pd.DataFrame({'measurement': np.random.normal(10.2, 0.5, 200)}), "shifts": pd.DataFrame({'shift_1': np.random.normal(10.25, 0.55, 50), 'shift_2': np.random.normal(10.15, 0.45, 50)}), "specs": {"lsl": 9.0, "usl": 11.0, "target": 10.0}}
    project_data_002 = {"baseline": pd.DataFrame({'measurement': np.random.normal(85, 2.8, 200)}), "shifts": pd.DataFrame({'before': np.random.normal(85, 2.8, 50), 'after': np.random.normal(88, 1.2, 50)}), "specs": {"lsl": 78, "usl": 92, "target": 85}}
    return {"DMAIC-001": project_data_001, "DMAIC-002": project_data_002}

def _generate_statistical_tool_data() -> Dict[str, Any]:
    # ... code from previous version ...
    gage_data = [];
    for part in range(1, 11):
        true_value = 10 + part * 0.1
        for operator in ["Operator A", "Operator B", "Operator C"]:
            op_bias = 0.015 if operator == "Operator B" else -0.01 if operator == "Operator C" else 0
            for _ in range(3): gage_data.append({"part_id": part, "operator": operator, "measurement": true_value + op_bias + np.random.normal(0, 0.05)})
    doe_df = pd.DataFrame([{'temp': -1, 'time': -1, 'pressure': -1, 'strength': 75.3}, {'temp': 1, 'time': -1, 'pressure': -1, 'strength': 78.9}, {'temp': -1, 'time': 1, 'pressure': -1, 'strength': 85.1}, {'temp': 1, 'time': 1, 'pressure': -1, 'strength': 95.2}, {'temp': -1, 'time': -1, 'pressure': 1, 'strength': 76.1}, {'temp': 1, 'time': -1, 'pressure': 1, 'strength': 79.5}, {'temp': -1, 'time': 1, 'pressure': 1, 'strength': 84.8}, {'temp': 1, 'time': 1, 'pressure': 1, 'strength': 96.0}, {'temp': 0, 'time': 0, 'pressure': 0, 'strength': 92.5}, {'temp': 0, 'time': 0, 'pressure': 0, 'strength': 91.8}])
    return {"gage_rr_data": pd.DataFrame(gage_data), "doe_data": doe_df}

def _generate_ml_lab_data() -> Dict[str, Any]:
    # ... code from previous version ...
    ml_data = []
    for _ in range(500):
        temp = np.random.normal(200, 10); pressure = np.random.normal(50, 5); vibration = np.random.normal(1.5, 0.5)
        fail_prob = 1 / (1 + np.exp(-(0.1 * (temp - 215) + 0.5 * (pressure - 52) - 0.05 * vibration)))
        ml_data.append({"in_process_temp": temp, "in_process_pressure": pressure, "in_process_vibration": vibration, "final_qc_outcome": "Fail" if random.random() < fail_prob else "Pass"})
    release_batches = [{"batch_id": f"BATCH-{202400 + i}", "test_measurement": np.random.normal(10.8, 0.8) if random.random() < 0.10 else np.random.normal(10.0, 0.2), "true_status": "Fail" if random.random() < 0.10 else "Pass"} for i in range(200)]
    return {"predictive_quality_data": pd.DataFrame(ml_data), "release_data": pd.DataFrame(release_batches)}

def _generate_optimization_data() -> pd.DataFrame:
    # ... code from previous version ...
    x = np.linspace(-5, 5, 50); y = np.linspace(-5, 5, 50); xx, yy = np.meshgrid(x, y)
    z = (np.sin(np.sqrt(xx**2 + yy**2)) - (xx**2 + yy**2) * 0.05 + np.cos(xx) * 2 - np.sin(yy) * 1.5)
    return pd.DataFrame({'x': xx.flatten(), 'y': yy.flatten(), 'z': z.flatten()})

def _generate_failure_clustering_data() -> pd.DataFrame:
    # ... code from previous version ...
    c1 = pd.DataFrame({'temperature': np.random.normal(250, 5, 50), 'pressure': np.random.normal(40, 3, 50), 'failure_mode': 'Seal Delamination'})
    c2 = pd.DataFrame({'temperature': np.random.normal(180, 5, 50), 'pressure': np.random.normal(80, 3, 50), 'failure_mode': 'Material Warping'})
    c3 = pd.DataFrame({'temperature': np.random.normal(215, 8, 50), 'pressure': np.random.normal(60, 5, 50), 'failure_mode': 'Component Fracture'})
    return pd.concat([c1, c2, c3], ignore_index=True)

def _create_mbb_model(version: int) -> Dict[str, Any]:
    """Orchestrates all helper functions to build the complete, interconnected data model."""
    np.random.seed(42); random.seed(42)
    base_date = date.today() - timedelta(days=365)
    sites = ["Andover, US", "Eindhoven, NL", "Shanghai, CN"]
    products_by_site = {"Andover, US": ["HeartStart Defibrillator"], "Eindhoven, NL": ["IntelliVue Patient Monitor", "Zenition C-arm"], "Shanghai, CN": ["Affiniti Ultrasound"]}

    yield_data, rty_by_date, _ = _generate_yield_data()
    copq_data = _generate_copq_data(sites, products_by_site)
    dmaic_projects = _generate_dmaic_projects(base_date)
    dmaic_project_datasets = _generate_dmaic_data()
    stat_tool_data = _generate_statistical_tool_data()
    ml_lab_data = _generate_ml_lab_data()
    
    # *** DEFINITIVE FIX: Ensure all generated datasets are added to the final model ***
    optimization_data = _generate_optimization_data()
    clustering_data = _generate_failure_clustering_data()

    kaizen_events = [{"id": "KZN-01", "site": "Eindhoven, NL", "date": base_date + timedelta(days=250), "title": "5S Implementation on Monitor Assembly Line 3", "outcome": "Reduced tool search time by 60%; cleared 25 sq. meters of floor space.", "team": ["Maria Rodriguez", "David Lee"]}, {"id": "KZN-02", "site": "Andover, US", "date": base_date + timedelta(days=300), "title": "Value Stream Mapping of Defibrillator Sub-assembly", "outcome": "Identified 3 non-value-add steps, reducing process cycle time by 15%.", "team": ["John Smith", "Jane Doe"]}]
    training_materials = [{"id": "TRN-001", "title": "Introduction to DMAIC Methodology", "type": "eLearning", "duration_hr": 4, "link": "#", "target_audience": "All"}, {"id": "TRN-002", "title": "Statistical Process Control (SPC) Fundamentals", "type": "PDF Guide", "duration_hr": 2, "link": "#", "target_audience": "Engineers, Technicians"}, {"id": "TRN-003", "title": "Design of Experiments (DOE) Workshop", "type": "Workshop Slides", "duration_hr": 8, "link": "#", "target_audience": "Engineers"}, {"id": "TRN-004", "title": "Root Cause Analysis (Fishbone & 5 Whys)", "type": "eLearning", "duration_hr": 2, "link": "#", "target_audience": "All"}]

    model_data = {
        "data_version": version,
        "yield_data": yield_data,
        "copq_data": copq_data,
        "dmaic_projects": dmaic_projects,
        "dmaic_project_data": dmaic_project_datasets,
        "global_kpis": rty_by_date,
        "kaizen_events": kaizen_events,
        "training_materials": training_materials,
        # *** DEFINITIVE FIX: Add the missing datasets here ***
        "optimization_data": optimization_data,
        "failure_clustering_data": clustering_data,
    }
    model_data.update(stat_tool_data)
    model_data.update(ml_lab_data)
    
    return model_data

class SessionStateManager:
    # ... (class remains the same) ...
    _DATA_KEY = "six_sigma_mbb_data_v4_final"
    _CURRENT_DATA_VERSION = 12.0
    def __init__(self):
        if self._DATA_KEY not in st.session_state or st.session_state[self._DATA_KEY].get("data_version") != self._CURRENT_DATA_VERSION:
            logger.info(f"Initializing session state with MBB data model v{self._CURRENT_DATA_VERSION}.")
            try:
                with st.spinner("Generating sophisticated, interconnected datasets... Please wait."): st.session_state[self._DATA_KEY] = _create_mbb_model(self._CURRENT_DATA_VERSION)
                logger.info("Session state initialized successfully with narrative-driven data.")
            except Exception as e:
                logger.critical(f"FATAL: Data generation failed: {e}", exc_info=True)
                st.error(f"A critical error occurred during application data model startup: {e}.", icon="🚨")
                st.stop()
    def get_data(self, key: str) -> Any:
        return st.session_state.get(self._DATA_KEY, {}).get(key)
