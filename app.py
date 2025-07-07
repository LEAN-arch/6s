"""
Main application entry point for the Six Sigma Master Black Belt (MBB) Command Center.

This Streamlit application serves as the primary digital workspace for a Six Sigma
MBB, designed for managing and executing a portfolio of high-impact process
improvement projects. The architecture is centered on the DMAIC methodology and
is supported by a suite of sophisticated dashboards for strategic oversight,
deep-dive analysis, and advanced statistical modeling.

SME Overhaul:
- Complete re-architecture for a commercial-grade, professional feel.
- A sophisticated sidebar navigation organizes the platform into logical, icon-driven
  workspaces, significantly improving user experience.
- All modules are fully integrated, and redundant/deprecated modules have been removed,
  resulting in a clean and maintainable codebase.
- Enhanced error handling and user guidance.
"""

import logging
import os
import sys
import streamlit as st

# --- Robust Path Correction Block ---
# Ensures that the application can be run from different directories by adding the
# project root to the system path, allowing for stable local imports.
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except Exception as e:
    # Fallback for environments where __file__ is not defined
    project_root = os.path.abspath(os.path.join(os.getcwd(), "."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    st.warning(f"Path correction failed with error: {e}. Assuming execution from project root.")

# --- Local Application Imports ---
# This block centrally imports all necessary dashboard rendering functions.
# A failure here indicates a critical problem with the project structure.
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
    st.error(
        f"Fatal Error: A required application module could not be imported: {e}. "
        "Please ensure the project structure is correct and all subdirectories "
        "contain an `__init__.py` file. The application cannot continue."
    )
    logging.critical(f"Fatal module import error: {e}", exc_info=True)
    st.stop()

# --- Page Configuration ---
# Must be the first Streamlit command.
st.set_page_config(
    layout="wide",
    page_title="Six Sigma Command Center",
    page_icon="📈"
)

# --- Global Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

# ==============================================================================
# --- Main Application Logic ---
# ==============================================================================

def main() -> None:
    """
    Main function to initialize the application, manage session state, and
    render the selected dashboard.
    """
    st.title("📈 Six Sigma Command Center")
    st.caption("A Commercial-Grade Platform for Data-Driven Process Excellence")

    # Initialize the session state manager, which generates and holds all app data.
    try:
        ssm = SessionStateManager()
        logger.info("Application initialized. Session State Manager loaded.")
    except Exception as e:
        st.error(f"Fatal Error: Could not initialize the application's data model: {e}")
        logger.critical(f"Failed to instantiate SessionStateManager: {e}", exc_info=True)
        st.stop()

    # --- Sidebar Navigation ---
    # A single, icon-driven radio button group provides clean and intuitive navigation.
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
        captions=[
            "Executive-level KPI dashboard",
            "Portfolio view of all projects",
            "Execute a specific DMAIC project",
            "Analyze process step efficiency",
            "Analyze cost of failures",
            "On-demand statistical analysis",
            "Predictive & advanced modeling",
            "Knowledge management & training"
        ],
        index=2  # Default to the DMAIC toolkit, the primary workspace.
    )

    # --- Main Panel Rendering ---
    # This block routes to the correct rendering function based on the sidebar selection.
    if app_mode == "Global Operations":
        render_global_dashboard(ssm)
    elif app_mode == "Improvement Pipeline":
        render_project_pipeline(ssm)
    elif app_mode == "DMAIC Project Toolkit":
        render_dmaic_toolkit(ssm)
    elif app_mode == "First Time Yield (FTY)":
        render_fty_dashboard(ssm)
    elif app_mode == "Cost of Poor Quality (COPQ)":
        render_copq_dashboard(ssm)
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
