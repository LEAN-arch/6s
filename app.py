# six_sigma/app.py
"""
Main application entry point for the Six Sigma Master Black Belt (MBB) Command Center.

This Streamlit application is the primary digital workspace for a Six Sigma MBB,
focused on executing high-impact improvement projects. The architecture is
centered on the DMAIC methodology, supported by deep-dive analytical modules
for identifying and quantifying opportunities (COPQ, FTY) and for advanced
statistical analysis.

SME Overhaul:
- The entire application architecture is now centered on the DMAIC workflow.
- A professional sidebar navigation is used instead of tabs.
- Imports are updated to reflect new, dedicated modules for FTY, COPQ, and advanced tools.
- Branding and titles are elevated to reflect an expert-level, MBB-focused toolkit.
"""

# --- Standard Library Imports ---
import logging
import os
import sys

# --- Third-party Imports ---
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
    from six_sigma.dashboards.dmaic_toolkit import render_dmaic_toolkit
    from six_sigma.dashboards.copq_dashboard import render_copq_dashboard
    from six_sigma.dashboards.fty_dashboard import render_fty_dashboard
    from six_sigma.dashboards.advanced_tools_suite import render_advanced_tools_suite
    from six_sigma.dashboards.project_pipeline import render_project_pipeline
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
# --- MAIN APPLICATION LOGIC ---
# ==============================================================================

def main() -> None:
    """Main function to initialize the Session State and render the Streamlit app."""
    st.title("📈 Six Sigma DMAIC Command Center")
    st.caption("A Master Black Belt's Toolkit for Data-Driven Process Improvement")

    try:
        ssm = SessionStateManager()
        logger.info("Application initialized. Session State Manager loaded.")
    except Exception as e:
        st.error(f"Fatal Error: Could not initialize the application's data model: {e}")
        logger.critical(f"Failed to instantiate SessionStateManager: {e}", exc_info=True)
        st.stop()

    # --- Sidebar for Navigation ---
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio(
        "Select a Workspace",
        [
            "DMAIC Project Workspace",
            "Improvement Project Pipeline",
            "First Time Yield (FTY) Analysis",
            "Cost of Poor Quality (COPQ) Analysis",
            "Advanced Statistical Tools"
        ],
        help="Select a workspace. The DMAIC Toolkit is the primary module for executing projects."
    )
    st.sidebar.markdown("---")
    st.sidebar.info("This application is a dedicated workspace for executing and managing Six Sigma improvement projects.")

    # --- Main Panel Rendering ---
    if app_mode == "DMAIC Project Workspace":
        render_dmaic_toolkit(ssm)
    elif app_mode == "Improvement Project Pipeline":
        render_project_pipeline(ssm)
    elif app_mode == "First Time Yield (FTY) Analysis":
        render_fty_dashboard(ssm)
    elif app_mode == "Cost of Poor Quality (COPQ) Analysis":
        render_copq_dashboard(ssm)
    elif app_mode == "Advanced Statistical Tools":
        render_advanced_tools_suite(ssm)


# ==============================================================================
# --- SCRIPT EXECUTION ---
# ==============================================================================
if __name__ == "__main__":
    main()
