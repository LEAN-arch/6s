# six_sigma/dashboards/dmaic_toolkit.py
"""
Renders the expert-level DMAIC Improvement Toolkit.
This module provides an interactive, end-to-end workspace for executing
complex Six Sigma projects. It guides an MBB through each phase of the DMAIC
methodology, embedding advanced statistical tools, Root Cause Analysis (RCA)
methodologies, and formal phase-gate approvals.
"""

import logging
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm
import numpy as np  # <--- FIX: Added missing import

from six_sigma.data.session_state_manager import SessionStateManager
from six_sigma.utils.plotting import create_control_chart, create_histogram_with_specs, create_doe_plots, create_gage_rr_plots
from six_sigma.utils.stats import calculate_process_capability, perform_t_test, perform_anova, calculate_gage_rr

logger = logging.getLogger(__name__)

def render_fishbone_diagram(effect: str):
    """Renders a Fishbone (Ishikawa) diagram using Streamlit columns and markdown."""
    st.markdown("##### Fishbone (Ishikawa) Diagram")
    
    causes = {
        "Measurement": ["Gage not calibrated", "Incorrect test procedure", "Subjective visual inspection"],
        "Material": ["Inconsistent raw material", "Supplier quality issues", "Improper storage"],
        "Personnel": ["Inadequate training", "High operator fatigue", "SOP not followed"],
        "Environment": ["Poor lighting", "Temperature fluctuations", "ESD contamination"],
        "Machine": ["Fixture wear & tear", "Incorrect machine settings", "Lack of preventative maintenance"],
        "Method": ["Outdated SOP", "Inefficient assembly sequence", "Poor handling instructions"]
    }
    
    st.markdown("<hr>", unsafe_allow_html=True)
    cols = st.columns(6)
    categories = list(causes.keys())

    for i, col in enumerate(cols):
        with col:
            st.markdown(f"**{categories[i]}**")
            for cause in causes[categories[i]]:
                st.markdown(f"- {cause}")

    st.info(f"**Effect:** {effect}")

def render_dmaic_toolkit(ssm: SessionStateManager) -> None:
    """Creates the UI for the DMAIC Improvement Toolkit workspace."""
    st.header("🛠️ DMAIC Project Workspace")
    st.markdown("Select an active improvement project below to access the full suite of DMAIC tools and analysis capabilities. This is your primary workspace for project execution.")

    try:
        projects = ssm.get_data("dmaic_projects")
        if not projects:
            st.warning("No DMAIC projects have been defined in the Project Pipeline.")
            return

        project_titles = {p['id']: f"{p['id']}: {p['title']} ({p['site']})" for p in projects}
        
        if 'selected_project_id' not in st.session_state or st.session_state.selected_project_id not in project_titles:
            st.session_state.selected_project_id = list(project_titles.keys())[0]

        st.session_state.selected_project_id = st.selectbox(
            "**Select Active Project:**", options=project_titles.keys(),
            format_func=lambda x: project_titles[x], key="project_selector"
        )
        
        project = next((p for p in projects if p['id'] == st.session_state.selected_project_id), None)
        
        phase_tabs = st.tabs(["**Define**", "**Measure**", "**Analyze (RCA)**", "**Improve (DOE)**", "**Control**"])

        # --- DEFINE PHASE ---
        with phase_tabs[0]:
            st.subheader(f"Define Phase: Project Charter & Scope")
            
            with st.container(border=True):
                st.markdown(f"#### **Project Title:** {project['title']}")
                st.markdown(f"**Site:** {project['site']} | **Product Line:** {project['product_line']}")
                st.markdown(f"**Team:** {', '.join(project['team'])}")
                st.markdown("---")
                st.info(f"**Problem Statement:**\n\n> {project['problem_statement']}")
                st.success(f"**Goal Statement (S.M.A.R.T.):**\n\n> {project['goal_statement']}")
            
            with st.expander("View Define Phase Tools"):
                st.markdown("#### SIPOC Diagram")
                st.info("A high-level map of the process from Supplier to Customer.")
                sipoc_data = {
                    "Suppliers": ["Component Vendors", "Sub-Assembly Line"],
                    "Inputs": ["Capacitors", "PCBs", "Housing"],
                    "Process": ["Inspect -> Assemble -> Solder -> Test"],
                    "Outputs": ["Functional Charging Module"],
                    "Customers": ["Main Assembly Line", "Final Product"]
                }
                
                for category, items in sipoc_data.items():
                    st.markdown(f"**{category}:**")
                    for item in items:
                        st.markdown(f"- {item}")

        # --- MEASURE PHASE ---
        with phase_tabs[1]:
            st.subheader("Measure Phase: Baseline Performance & Measurement System Analysis (MSA)")
            
            process_df = ssm.get_data("process_data")
            specs = ssm.get_data("process_specs")["seal_strength"]
            metric_to_analyze = 'seal_strength'

            st.markdown("##### 1. Process Capability Baseline")
            with st.expander("Learn More: What is Process Capability?"):
                st.markdown("Process Capability (Cpk) and Performance (Ppk) indices measure how well your process can meet customer specifications. **A value of 1.33 is a common minimum target.**")
            
            cp, cpk, pp, ppk = calculate_process_capability(process_df[metric_to_analyze], specs['lsl'], specs['usl'])
            cap_cols = st.columns(4)
            cap_cols[0].metric("Cpk (Baseline)", f"{cpk:.2f}", "Target: > 1.33")
            cap_cols[1].metric("Ppk (Baseline)", f"{ppk:.2f}", "Target: > 1.33")
            cap_cols[2].metric("DPMO (Estimated)", f"{project['baseline_dpmo']:,}")
            cap_cols[3].metric("Sigma Level (Baseline)", f"{1.5 + norm.ppf(1 - project['baseline_dpmo'] / 1_000_000):.2f} σ")

            hist_fig = create_histogram_with_specs(process_df[metric_to_analyze], specs['lsl'], specs['usl'], metric_to_analyze.replace('_', ' ').title())
            st.plotly_chart(hist_fig, use_container_width=True)
            
            st.markdown("---")
            st.markdown("##### 2. Measurement System Analysis (Gage R&R)")
            with st.expander("Learn More: What is Gage R&R?"):
                st.markdown("Gage R&R quantifies how much of your process variation is due to the measurement system itself. **A Total Gage R&R of <10% is excellent, while >30% is unacceptable.** You cannot trust your process data without a reliable measurement system.")
            
            gage_data = ssm.get_data("gage_rr_data")
            results_df, fig1, fig2 = calculate_gage_rr(gage_data)
            
            if not results_df.empty:
                total_grr = results_df.loc['Total Gage R&R', '% Contribution']
                grr_cols = st.columns([1, 2])
                with grr_cols[0]:
                    st.dataframe(results_df.style.format({'% Contribution': '{:.2f}%'}).background_gradient(cmap='Reds', subset=['% Contribution']))
                    if total_grr < 10: st.success(f"**Conclusion:** Measurement System is **Acceptable** ({total_grr:.2f}%).")
                    elif total_grr < 30: st.warning(f"**Conclusion:** Measurement System is **Marginal** ({total_grr:.2f}%).")
                    else: st.error(f"**Conclusion:** Measurement System is **Unacceptable** ({total_grr:.2f}%).")
                with grr_cols[1]:
                    st.plotly_chart(fig1, use_container_width=True)
                    st.plotly_chart(fig2, use_container_width=True)
            else:
                st.error("Gage R&R analysis failed. Please check data and logs.")
                
        # --- ANALYZE PHASE (RCA) ---
        with phase_tabs[2]:
            st.subheader("Analyze Phase: Root Cause Analysis (RCA)")
            st.markdown("Analyze data to identify and verify root causes. This phase moves from symptoms to underlying problems.")
            
            st.markdown("##### Root Cause Analysis Toolkit")
            with st.expander("Learn More: Common RCA Tools"):
                st.markdown("- **5 Whys:** A simple, iterative technique to drill down past symptoms to find the true root cause.\n- **Fishbone Diagram:** A structured brainstorming tool to organize potential causes into categories (e.g., Man, Method, Machine).")

            rca_cols = st.columns(2)
            with rca_cols[0]:
                st.info("**5 Whys Analysis**")
                st.text_input("1. Why is yield low?", "The alignment fixture is inconsistent.", key=f"why1_{project['id']}")
                st.text_input("2. Why is it inconsistent?", "It wears down quickly.", key=f"why2_{project['id']}")
                st.text_input("3. Why does it wear down?", "The material hardness spec is too low.", key=f"why3_{project['id']}")
                st.text_input("4. Why is the spec low?", "It was based on an older, lower-throughput model.", key=f"why4_{project['id']}")
                st.text_input("5. Why was it not updated?", "Process oversight during design transfer.", key=f"why5_{project['id']}")

            with rca_cols[1]:
                render_fishbone_diagram(effect=project['problem_statement'])

            st.markdown("---")
            st.markdown("##### Hypothesis Testing Suite")
            ht_data = ssm.get_data("hypothesis_testing_data")
            ht_tool = st.selectbox("Select Hypothesis Test:", ["2-Sample t-Test (Before vs. After)", "ANOVA (Supplier A vs. B vs. C)"])
            
            if ht_tool == "2-Sample t-Test (Before vs. After)":
                fig, result = perform_t_test(ht_data['before_change'], ht_data['after_change'], "Before Change", "After Change")
                st.plotly_chart(fig, use_container_width=True)
                if result['p_value'] < 0.05: st.success(f"**Conclusion:** The difference is statistically significant (p = {result['p_value']:.4f}).")
                else: st.warning(f"**Conclusion:** The difference is not statistically significant (p = {result['p_value']:.4f}).")

            elif ht_tool == "ANOVA (Supplier A vs. B vs. C)":
                df_anova = pd.melt(ht_data[['supplier_a', 'supplier_b', 'supplier_c']], var_name='group', value_name='value')
                fig, result = perform_anova(df_anova, 'value', 'group', "Component Strength by Supplier")
                st.plotly_chart(fig, use_container_width=True)
                if result['p_value'] < 0.05: st.success(f"**Conclusion:** There is a statistically significant difference between the suppliers (p = {result['p_value']:.4f}).")
                else: st.warning(f"**Conclusion:** There is no significant difference between the suppliers (p = {result['p_value']:.4f}).")
        
        # --- IMPROVE PHASE (DOE) ---
        with phase_tabs[3]:
            st.subheader("Improve Phase: Design of Experiments (DOE) & Optimization")
            st.markdown("Use DOE to find the optimal settings for critical process parameters (the 'vital few' X's).")
            
            doe_data = ssm.get_data("doe_data")
            factors = ['temp', 'time', 'pressure']
            response = 'strength'

            st.info("The following analysis is based on a **3-Factor, 2-Level Full Factorial** experiment to optimize the 'Display Module Bonding Process'.")
            with st.expander("Learn More: Interpreting DOE Plots"):
                st.markdown("- **Main Effects Plot:** Shows the average impact each factor has on the response. The steeper the line, the larger the effect.\n- **Interaction Plot:** Visualizes how the effect of one factor changes depending on the level of another. **Non-parallel (crossed) lines indicate a strong interaction.**\n- **Response Surface:** A 3D map showing the predicted response across a range of factor settings, helping to visualize the optimal process window.")

            main_effects_fig, interaction_fig, surface_fig = create_doe_plots(doe_data, factors, response)
            
            st.plotly_chart(main_effects_fig, use_container_width=True)
            st.plotly_chart(interaction_fig, use_container_width=True)
            st.plotly_chart(surface_fig, use_container_width=True)
            st.success("**DOE Conclusion:** The analysis reveals that **Time** has the largest positive effect on bond strength. A significant interaction between **Time and Temperature** is also observed. The optimal process window appears to be at high Time and moderate Temperature, as visualized on the response surface.")
            
        # --- CONTROL PHASE ---
        with phase_tabs[4]:
            st.subheader("Control Phase: Sustain the Gains")
            st.markdown("Establish process controls and monitoring plans to ensure the improvements are sustained long-term.")

            with st.expander("Learn More: The Control Plan"):
                st.markdown("The Control Plan is a living document that outlines how to maintain the process improvements. It specifies the critical parameters to monitor, the methods and frequency of monitoring, and the reaction plan to follow if the process goes out of control.")

            st.markdown("##### Control Plan")
            control_plan_data = {'Process Step': ['Display Module Bonding'], 'Critical Parameter (Y)': ['Bond Strength'], 'Key Input (X)': ['Temperature'], 'Specification': ['85 ± 5 MPa'], 'Control Method': ['SPC Chart (I-MR)'], 'Reaction Plan': ['Halt line, notify engineer if point is outside control limits.']}
            st.dataframe(pd.DataFrame(control_plan_data), hide_index=True, use_container_width=True)
            
            st.markdown("##### Live SPC Monitoring of Controlled Process")
            controlled_process = np.random.normal(loc=88, scale=0.8, size=100)
            spc_fig = create_control_chart(pd.Series(controlled_process), "Bond Strength", 80, 90)
            st.plotly_chart(spc_fig, use_container_width=True)
            st.success("The new process is stable and in a state of statistical control, demonstrating that the improvements have been successfully sustained.")

    except Exception as e:
        st.error("An error occurred while rendering the DMAIC Toolkit.")
        logger.error(f"Failed to render DMAIC toolkit: {e}", exc_info=True)
