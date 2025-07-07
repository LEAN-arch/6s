# 6s/dashboards/dmaic_toolkit.py
"""
Renders the DMAIC Improvement Toolkit.

This module provides an interactive workspace for managing and executing
data-driven quality improvement projects using the Define, Measure, Analyze,
Improve, and Control (DMAIC) methodology. It serves as the central hub for
project charters, data analysis, and control planning.
"""

import logging
import pandas as pd
import streamlit as st
import plotly.express as px

# Import utilities from the new, redesigned project structure
from 6s.data.session_state_manager import SessionStateManager
from 6s.utils.plotting import create_control_chart, create_histogram_with_specs
from 6s.utils.stats import calculate_process_capability

logger = logging.getLogger(__name__)

def render_dmaic_toolkit(ssm: SessionStateManager) -> None:
    """
    Creates the UI for the DMAIC Improvement Toolkit tab.

    Args:
        ssm (SessionStateManager): The session state manager to access project and process data.
    """
    st.header("🛠️ DMAIC Improvement Project Toolkit")
    st.markdown("Select an active improvement project below to access the full suite of DMAIC tools and analysis capabilities.")

    try:
        # --- 1. Load Data and Select a Project ---
        projects = ssm.get_data("dmaic_projects")
        process_data_dict = ssm.get_data("process_data")
        process_df = process_data_dict.get('data')
        specs = process_data_dict.get('specs')

        if not projects:
            st.warning("No DMAIC projects have been defined. Please contact the program administrator.")
            return

        project_titles = {p['id']: f"{p['id']}: {p['title']}" for p in projects}
        selected_project_id = st.selectbox(
            "**Select Project:**",
            options=project_titles.keys(),
            format_func=lambda x: project_titles[x]
        )

        project = next((p for p in projects if p['id'] == selected_project_id), None)
        if not project:
            st.error("Selected project could not be found.")
            return

        st.divider()

        # --- 2. Define Tabs for each DMAIC phase ---
        phase_tabs = st.tabs(["**Define**", "**Measure**", "**Analyze**", "**Improve**", "**Control**"])

        # --- DEFINE PHASE ---
        with phase_tabs[0]:
            st.subheader(f"Define Phase: Project Charter")
            st.markdown(f"**Project Title:** {project['title']}")

            charter_cols = st.columns(2)
            with charter_cols[0]:
                st.info("Problem Statement", icon="❓")
                st.markdown(project['problem_statement'])
            with charter_cols[1]:
                st.success("Goal Statement", icon="🎯")
                st.markdown(project['goal_statement'])

            st.write("**Project Team:**")
            st.write(", ".join(project['team']))

        # --- MEASURE PHASE ---
        with phase_tabs[1]:
            st.subheader("Measure Phase: Process Baseline & Capability")
            st.markdown("This phase establishes a baseline for the current process performance and quantifies the problem.")
            
            if process_df is None or specs is None:
                st.warning("Process data for this product line is not available.")
                return
            
            # Allow user to select which metric to analyze from the process data
            metric_to_analyze = st.selectbox(
                "Select a Critical Process Parameter to Analyze:",
                options=specs.keys()
            )

            usl = specs[metric_to_analyze]['usl']
            lsl = specs[metric_to_analyze]['lsl']
            
            # Calculate Capability
            cp, cpk, pp, ppk = calculate_process_capability(process_df[metric_to_analyze], lsl, usl)

            st.markdown("##### Process Capability Analysis")
            cap_cols = st.columns(4)
            cap_cols[0].metric("Cp", f"{cp:.2f}", help="Process Potential (short-term). Target > 1.33")
            cap_cols[1].metric("Cpk", f"{cpk:.2f}", help="Process Capability (short-term, centered). Target > 1.33",
                               delta=f"{cpk-1.33:.2f} vs Target", delta_color="inverse" if cpk < 1.33 else "normal")
            cap_cols[2].metric("Pp", f"{pp:.2f}", help="Process Performance (long-term).")
            cap_cols[3].metric("Ppk", f"{ppk:.2f}", help="Process Performance (long-term, centered). Target > 1.33",
                               delta=f"{ppk-1.33:.2f} vs Target", delta_color="inverse" if ppk < 1.33 else "normal")

            # Visualize Baseline
            hist_fig = create_histogram_with_specs(process_df[metric_to_analyze], lsl, usl, metric_to_analyze)
            st.plotly_chart(hist_fig, use_container_width=True)


        # --- ANALYZE PHASE ---
        with phase_tabs[2]:
            st.subheader("Analyze Phase: Root Cause Identification")
            st.markdown("Analyze the data to identify the root cause(s) of defects and process variation.")
            
            st.info("In a real scenario, this section would contain advanced statistical tools like ANOVA, Regression, and Hypothesis Testing to investigate relationships between process inputs (X's) and outputs (Y's).")
            
            # Example: Correlation Matrix
            st.markdown("##### Correlation Analysis of Process Parameters")
            corr_matrix = process_df[['voltage_a', 'pressure_b', 'seal_temperature']].corr()
            fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto",
                                 title="Correlation Between Process Parameters",
                                 color_continuous_scale='RdBu_r')
            st.plotly_chart(fig_corr, use_container_width=True)
            st.caption("This heatmap shows the statistical correlation between different process inputs. Strong correlations can suggest relationships to investigate further.")

        # --- IMPROVE PHASE ---
        with phase_tabs[3]:
            st.subheader("Improve Phase: Develop and Verify Solutions")
            st.markdown("Develop, test, and implement solutions that address the identified root causes.")
            st.info("This phase typically involves Design of Experiments (DOE) to find optimal process settings. The results would be documented here.")
            
            # Placeholder for DOE results
            st.markdown("##### Example: Solution Verification")
            st.write("""
            **Proposed Solution:** Implement new controller for seal temperature to reduce variability.
            **Verification Plan:** Run 20 units with the new controller and confirm Cpk for 'seal_temperature' improves to > 1.67.
            **Status:** Completed.
            **Result:** New Cpk achieved **1.72**. Solution verified.
            """)

        # --- CONTROL PHASE ---
        with phase_tabs[4]:
            st.subheader("Control Phase: Sustain the Gains")
            st.markdown("Establish process controls and monitoring plans to ensure the improvements are sustained over the long term.")
            st.markdown("##### Control Plan: Statistical Process Control (SPC)")
            
            st.info("The control chart below simulates real-time monitoring of the improved process. It helps detect any new special cause variation before it leads to defects.")
            
            # Use the same metric selected in the Measure phase for consistency
            control_metric = metric_to_analyze
            spc_fig = create_control_chart(process_df[control_metric], control_metric, lsl, usl)
            st.plotly_chart(spc_fig, use_container_width=True)

    except Exception as e:
        st.error("An error occurred while rendering the DMAIC Toolkit.")
        logger.error(f"Failed to render DMAIC toolkit: {e}", exc_info=True)
