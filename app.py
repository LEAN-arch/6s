# philips_quality_optimizer/app.py
"""
Main application entry point for the Philips Quality Optimization Command Center.

This Streamlit application is the primary digital toolkit for a Quality Optimization
Engineer Lead. It is designed to drive continuous improvement, cost savings, and
process efficiency across global medical device manufacturing sites.

The Command Center provides tools for monitoring Key Performance Indicators (KPIs),
executing DMAIC-based improvement projects, optimizing product release strategies,
and leveraging predictive analytics to foster a culture of "Quality at the Source."
It supports the key objectives of reducing scrap, improving First Time Right (FTR),
lowering Cost of Poor Quality (COPQ), and shortening cycle times, all within the
rigorous compliance framework of the medical device industry (ISO 13485, 21 CFR 820).
"""

# --- Standard Library Imports ---
import logging
import os
import sys

# --- Third-party Imports ---
import streamlit as st

# --- Robust Path Correction Block ---
# Ensures that local modules can be found regardless of execution directory.
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
    st.warning(f"Could not precisely determine project root. Assuming '{project_root}'. Module imports may fail. Error: {e}")

# --- Local Application Imports ---
# These imports reflect the new, redesigned architecture.
try:
    from philips_quality_optimizer.data.session_state_manager import SessionStateManager
    from philips_quality_optimizer.dashboards.global_operations_dashboard import render_global_dashboard
    from philips_quality_optimizer.dashboards.dmaic_toolkit import render_dmaic_toolkit
    from philips_quality_optimizer.dashboards.release_optimization_suite import render_release_optimization_suite
    from philips_quality_optimizer.dashboards.kaizen_training_hub import render_kaizen_training_hub
    from philips_quality_optimizer.dashboards.predictive_quality_lab import render_predictive_quality_lab
except ImportError as e:
    st.error(f"Fatal Error: A required local module could not be imported: {e}. "
             "Please ensure the application's directory structure is correct and all "
             "subdirectories contain an `__init__.py` file.")
    logging.critical(f"Fatal module import error: {e}", exc_info=True)
    st.stop()

# --- Page Configuration ---
# Must be the first Streamlit command.
st.set_page_config(
    layout="wide",
    page_title="Philips Quality Optimization Command Center",
    page_icon="🅿️" # Using a generic 'P' for Philips
)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

# ==============================================================================
# --- MAIN APPLICATION LOGIC ---
# ==============================================================================

def main() -> None:
    """
    Main function to initialize the Session State and render the Streamlit app.
    """
    st.title("🅿️ Quality Optimization Command Center")
    st.caption("Driving Continuous Improvement and Cost Savings Across Global Operations")

    # Initialize the session state. This class now generates a data model
    # focused on manufacturing sites, KPIs, and improvement projects.
    try:
        ssm = SessionStateManager()
        logger.info("Application initialized. Session State Manager loaded.")
    except Exception as e:
        st.error(f"Fatal Error: Could not initialize the application's data model: {e}")
        logger.critical(f"Failed to instantiate SessionStateManager: {e}", exc_info=True)
        st.stop()

    # Define the new, role-specific tabs for the application.
    tab_titles = [
        "🌍 **Global Operations Dashboard**",
        "🛠️ **DMAIC Improvement Toolkit**",
        "✅ **Product Release Optimization**",
        "🎓 **Kaizen & Training Hub**",
        "🔮 **Predictive Quality Lab**"
    ]
    (
        tab_global_dashboard,
        tab_dmaic_toolkit,
        tab_release_optimization,
        tab_kaizen_hub,
        tab_predictive_lab
    ) = st.tabs(tab_titles)

    # Render each tab by calling its dedicated function from the `dashboards` module.
    with tab_global_dashboard:
        render_global_dashboard(ssm)

    with tab_dmaic_toolkit:
        render_dmaic_toolkit(ssm)

    with tab_release_optimization:
        render_release_optimization_suite(ssm)

    with tab_kaizen_hub:
        render_kaizen_training_hub(ssm)

    with tab_predictive_lab:
        render_predictive_quality_lab(ssm)


# ==============================================================================
# --- SCRIPT EXECUTION ---
# ==============================================================================
if __name__ == "__main__":
    main()
