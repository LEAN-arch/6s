# six_sigma/app.py
"""
Main application entry point for the Six Sigma Master Black Belt (MBB) Command Center.

This Streamlit application is the primary digital workspace for a Six Sigma MBB,
focused on executing high-impact improvement projects. The architecture is
centered on the DMAIC methodology, supported by deep-dive analytical modules
for identifying and quantifying opportunities (COPQ, FTY) and for advanced
statistical analysis.

SME Overhaul:
- Complete re-architecture for a commercial-grade, professional feel.
- A sophisticated sidebar navigation organizes the platform into logical workspaces.
- All modules, including new ones for ML, are fully integrated.
"""

import logging
import os
import sys
import streamlit as st

# --- Robust Path Correction Block ---
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path: sys.path.insert(0, project_root)
except Exception as e:
    project_root = os.path.abspath(os.path.join(os.getcwd(), "."))
    if project_root not in sys.path: sys.path.insert(0, project_root)
    st.warning(f"Path correction failed: {e}. Assuming execution from project root.")

# --- Local Application Imports ---
try:
    from six_sigma.data.session_state_manager import SessionStateManager
    from six_sigma.dashboards.global_operations_dashboard import render_global_dashboard
    from six_sigma.dashboards.project_pipeline import render_project_pipeline
    from six_sigma.dashboards.copq_dashboard import render_copq_dashboard
    from six_sigma.dashboards.fty_dashboard import render_fty_dashboard
    from six_sigma.dashboards.dmaic_toolkit import render_dmaic_toolkit
    from six_sigma.dashboards.advanced_tools_suite import render_advanced_tools_suite
    from six_sigma.dashboards.ml_analytics_lab import render_ml_analytics_lab
    from six_sigma.dashboards.kaizen_training_hub import render_kaizen_training_hub
except ImportError as e:
    st.error(f"Fatal Error: A required module could not be imported: {e}. "
             "Please ensure the project structure is correct and all "
             "subdirectories contain an `__init__.py` file.")
    logging.critical(f"Fatal module import error: {e}", exc_info=True)
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Six Sigma DMAIC Command Center",
    page_icon="📈"
)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

# ==============================================================================
# --- Main Application Logic ---
# ==============================================================================

def main() -> None:
    """Main function to initialize the Session State and render the Streamlit app."""
    st.set_option('deprecation.showPyplotGlobalUse', False) # Suppress Pyplot global use warning for SHAP plots
    
    st.title("📈 Six Sigma Command Center")
    st.caption("A Commercial-Grade Platform for Data-Driven Process Excellence")

    try:
        ssm = SessionStateManager()
        logger.info("Application initialized. Session State Manager loaded.")
    except Exception as e:
        st.error(f"Fatal Error: Could not initialize the application's data model: {e}")
        logger.critical(f"Failed to instantiate SessionStateManager: {e}", exc_info=True)
        st.stop()

    # --- Sidebar Navigation ---
    st.sidebar.title("Workspaces")
    
    st.sidebar.markdown("### Strategic Dashboards")
    strategic_choice = st.sidebar.radio("Overview & Planning:", 
                                        ["Global Operations", "Improvement Pipeline"], label_visibility="collapsed")

    st.sidebar.markdown("### Analytical Workbenches")
    analytical_choice = st.sidebar.radio("Analysis & Execution:", 
                                         ["DMAIC Project Toolkit", "First Time Yield (FTY)", "Cost of Poor Quality (COPQ)"], label_visibility="collapsed")

    st.sidebar.markdown("### Advanced & Support Tools")
    advanced_choice = st.sidebar.radio("Specialized Tools & Resources:", 
                                       ["Advanced Statistical Tools", "ML & Analytics Lab", "Kaizen & Training Hub"], label_visibility="collapsed")

    # This logic block ensures only one "active" choice across all radio button groups
    # For simplicity, we can just use a single radio button with headers.
    st.sidebar.empty() # Clear the previous radio buttons
    st.sidebar.title("Workspaces")
    app_mode = st.sidebar.radio(
        "Navigation",
        [
            "Global Operations",
            "Improvement Pipeline",
            "DMAIC Project Toolkit",
            "First Time Yield (FTY)",
            "Cost of Poor Quality (COPQ)",
            "Advanced Statistical Tools",
            "ML & Analytics Lab",
            "Kaizen & Training Hub"
        ],
        index=2 # Default to the DMAIC toolkit
    )

    # --- Main Panel Rendering ---
    if app_mode == "Global Operations":
        render_global_dashboard(ssm)
    elif app_mode == "Improvement Pipeline":
        render_project_pipeline(ssm)
    elif app_mode == "First Time Yield (FTY)":
        render_fty_dashboard(ssm)
    elif app_mode == "Cost of Poor Quality (COPQ)":
        render_copq_dashboard(ssm)
    elif app_mode == "DMAIC Project Toolkit":
        render_dmaic_toolkit(ssm)
    elif app_mode == "Advanced Statistical Tools":
        render_advanced_tools_suite(ssm)
    elif app_mode == "ML & Analytics Lab":
        render_ml_analytics_lab(ssm)
    elif app_mode == "Kaizen & Training Hub":
        render_kaizen_training_hub(ssm)

# ==============================================================================
# --- SCRIPT EXECUTION ---
# ==============================================================================
if __name__ == "__main__":
    main()
