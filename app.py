# six_sigma/app.py
"""
Main application entry point for the Six Sigma Quality Command Center.

This Streamlit application is the primary digital toolkit for a Quality and
Process Improvement Engineer. It is designed to drive continuous improvement,
cost savings, and process efficiency across global manufacturing sites.

The Command Center provides tools for monitoring Key Performance Indicators (KPIs),
analyzing the Cost of Poor Quality (COPQ) and First Time Right (FTR), executing
DMAIC improvement projects, and leveraging predictive analytics to foster a
culture of "Quality at the Source."

SME Overhaul:
- Re-architected tab structure for a more intuitive Quality Engineering workflow.
- Added imports for new, dedicated FTR and COPQ analysis dashboards.
- Refined branding and titles for a more professional feel.
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
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except Exception as e:
    project_root = os.path.abspath(os.path.join(os.getcwd(), "."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    st.warning(f"Could not precisely determine project root. Assuming '{project_root}'. Module imports may fail. Error: {e}")

# --- Local Application Imports ---
try:
    from six_sigma.data.session_state_manager import SessionStateManager
    from six_sigma.dashboards.global_operations_dashboard import render_global_dashboard
    from six_sigma.dashboards.copq_dashboard import render_copq_dashboard
    from six_sigma.dashboards.ftr_dashboard import render_ftr_dashboard
    from six_sigma.dashboards.dmaic_toolkit import render_dmaic_toolkit
    from six_sigma.dashboards.advanced_tools_suite import render_advanced_tools_suite
    from six_sigma.dashboards.kaizen_training_hub import render_kaizen_training_hub
except ImportError as e:
    st.error(f"Fatal Error: A required local module could not be imported: {e}. "
             "Please ensure the application's directory structure is correct and all "
             "subdirectories contain an `__init__.py` file.")
    logging.critical(f"Fatal module import error: {e}", exc_info=True)
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Six Sigma Quality Command Center",
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
    st.title("📈 Six Sigma Quality Command Center")
    st.caption("A Data-Driven Toolkit for Process Excellence and Continuous Improvement")

    try:
        ssm = SessionStateManager()
        logger.info("Application initialized. Session State Manager loaded.")
    except Exception as e:
        st.error(f"Fatal Error: Could not initialize the application's data model: {e}")
        logger.critical(f"Failed to instantiate SessionStateManager: {e}", exc_info=True)
        st.stop()

    # SME Overhaul: New tab structure for a more logical engineering workflow
    tab_titles = [
        "🌍 **Global KPI Dashboard**",
        "💰 **COPQ Analysis Center**",
        "✅ **First Time Right (FTR) Analysis**",
        "🛠️ **DMAIC Improvement Toolkit**",
        "🔮 **Advanced Tools Suite**",
        "🎓 **Kaizen & Training Hub**"
    ]
    tabs = st.tabs(tab_titles)

    with tabs[0]:
        render_global_dashboard(ssm)

    with tabs[1]:
        render_copq_dashboard(ssm)

    with tabs[2]:
        render_ftr_dashboard(ssm)

    with tabs[3]:
        render_dmaic_toolkit(ssm)
        
    with tabs[4]:
        render_advanced_tools_suite(ssm)

    with tabs[5]:
        render_kaizen_training_hub(ssm)

# ==============================================================================
# --- SCRIPT EXECUTION ---
# ==============================================================================
if __name__ == "__main__":
    main()
