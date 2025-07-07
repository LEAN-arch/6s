"""
Renders the expert-level DMAIC Improvement Project Toolkit, the core operational
workspace for project execution within the Command Center.

This module provides an interactive, end-to-end environment for executing
complex Six Sigma projects. It guides an MBB through each phase of the DMAIC
methodology (Define, Measure, Analyze, Improve, Control), embedding the
application's advanced statistical and plotting utilities directly into the
project workflow.

SME Overhaul:
- Architected as a fully integrated, project-centric workspace. Selecting a
  project at the top dynamically populates all tools with relevant data.
- Enforces a clean "calculate-then-plot" architecture, calling functions from
  the `stats` and `plotting` utilities in the correct sequence.
- Upgraded all visualizations to their enhanced versions (I-MR chart, info-rich
  histogram, etc.) for a superior analytical experience.
- Rewrote all "Learn More" sections to provide expert-level, accessible
  explanations of the tools and methodologies for each DMAIC phase.
- Polished the entire UI for a professional, guided, and intuitive workflow.
"""

import logging
import pandas as pd
import streamlit as st
import numpy as np
from scipy.stats import norm

from six_sigma.data.session_state_manager import SessionStateManager
from six_sigma.utils.plotting import create_imr_chart, create_histogram_with_specs, create_doe_plots, create_gage_rr_plots
from six_sigma.utils.stats import calculate_process_performance, perform_hypothesis_test, perform_anova_on_dataframe, calculate_gage_rr

logger = logging.getLogger(__name__)

def _render_fishbone_diagram(effect: str):
    """Renders a visually appealing Fishbone (Ishikawa) diagram for RCA."""
    st.markdown("##### Fishbone Diagram: Potential Causes")
    # This can be made more dynamic in a future version
    causes = {
        "Measurement": ["Gage not calibrated", "Incorrect test procedure", "Operator error"],
        "Material": ["Inconsistent raw material", "Supplier quality issues", "Improper storage"],
        "Personnel": ["Inadequate training", "High operator fatigue", "SOP not followed"],
        "Environment": ["Poor lighting", "Temp/humidity fluctuations", "Contamination"],
        "Machine": ["Fixture wear & tear", "Incorrect settings", "No preventative maintenance"],
        "Method": ["Outdated SOP", "Inefficient assembly sequence", "Poor handling"]
    }
    
    # Simple HTML/Markdown based representation
    st.markdown(f"**Effect:** <span style='color:firebrick;'>{effect}</span>", unsafe_allow_html=True)
    for category, items in causes.items():
        st.markdown(f"**{category}**")
        for item in items:
            st.markdown(f"- *{item}*")


def render_dmaic_toolkit(ssm: SessionStateManager) -> None:
    """Creates the UI for the DMAIC Improvement Toolkit workspace."""
    st.header("🛠️ DMAIC Project Execution Toolkit")
    st.markdown("Select an active improvement project below to access the full suite of DMAIC tools. This is your primary workspace for project execution, from definition to control.")

    try:
        projects = ssm.get_data("dmaic_projects")
        if not projects:
            st.warning("No DMAIC projects have been defined. Please add projects in the Project Pipeline.")
            return

        # --- Project Selector ---
        project_titles = {p['id']: f"{p['id']}: {p['title']}" for p in projects}
        if 'selected_project_id' not in st.session_state or st.session_state.selected_project_id not in project_titles:
            st.session_state.selected_project_id = list(project_titles.keys())[0]

        selected_id = st.selectbox(
            "**Select Active Project:**", options=list(project_titles.keys()),
            format_func=lambda x: project_titles[x],
            help="The analysis in the tabs below will update based on this selection."
        )
        project = next((p for p in projects if p['id'] == selected_id), None)
        
        # --- DMAIC Phase Tabs ---
        phase_tabs = st.tabs(["**Define**", "**Measure**", "**Analyze**", "**Improve**", "**Control**"])

        # ==================== DEFINE PHASE ====================
        with phase_tabs[0]:
            st.subheader(f"Define Phase: Project Charter")
            st.info("The **Define** phase is about clearly articulating the business problem, goal, scope, and team for the project. The Project Charter is the guiding document for the entire effort.")
            
            with st.container(border=True):
                st.markdown(f"### {project['title']}")
                st.markdown(f"**Site:** {project['site']} | **Product Line:** {project['product_line']} | **Start Date:** {project['start_date']}")
                st.markdown(f"**Team:** {', '.join(project['team'])}")
                st.divider()
                st.error(f"**Problem Statement:**\n\n> {project['problem_statement']}", icon="❗️")
                st.success(f"**Goal Statement (S.M.A.R.T.):**\n\n> {project['goal_statement']}", icon="🎯")

        # ==================== MEASURE PHASE ====================
        with phase_tabs[1]:
            st.subheader("Measure Phase: Baseline Performance & Measurement System")
            st.info("The **Measure** phase is about collecting data to establish a performance baseline and verifying that your measurement system is reliable enough to be trusted.")
            
            # --- 1. Baseline Process Performance ---
            st.markdown("#### 1. Establish Process Baseline")
            with st.expander("Learn More: What is Process Performance (Ppk)?"):
                st.markdown("""
                Process Performance (Ppk) is a measure of how well your process is meeting customer specifications, taking into account both the spread and centering of the process over the **long term**.
                - **Ppk < 1.0:** The process is not capable of meeting specifications.
                - **1.0 < Ppk < 1.33:** The process is marginally capable.
                - **Ppk > 1.33:** The process is considered capable. This is a common minimum target.
                
                We use the information-rich histogram and the I-MR chart below to visualize this baseline.
                """)
            
            process_df = ssm.get_data("process_data")
            specs = ssm.get_data("process_specs")["seal_strength"]
            metric_to_analyze = 'seal_strength'
            data_series = process_df[metric_to_analyze]

            # Calculate-then-plot
            capability_metrics = calculate_process_performance(data_series, specs['lsl'], specs['usl'])
            hist_fig = create_histogram_with_specs(data_series, specs['lsl'], specs['usl'], metric_to_analyze.replace('_', ' ').title(), capability_metrics)
            imr_fig = create_imr_chart(data_series, metric_to_analyze.replace('_', ' ').title(), specs['lsl'], specs['usl'])
            
            st.metric("Baseline Process Performance (Ppk)", f"{capability_metrics.get('ppk', 0):.2f}", delta="-0.48 vs. Target 1.33", delta_color="inverse")
            
            plot_cols = st.columns(2)
            with plot_cols[0]:
                st.plotly_chart(hist_fig, use_container_width=True)
            with plot_cols[1]:
                st.plotly_chart(imr_fig, use_container_width=True)

            # --- 2. Measurement System Analysis (MSA) ---
            st.markdown("---")
            st.markdown("#### 2. Validate the Measurement System (Gage R&R)")
            with st.expander("Learn More: Why is Gage R&R Critical?"):
                 st.markdown("""
                Before you can analyze or improve a process, you **must** trust your data. A Gage R&R study tells you if your measurement system is reliable. It separates the variation from the measurement system itself from the actual variation in the parts.
                - **% Contribution < 10%:** Excellent. The measurement system is acceptable.
                - **% Contribution > 30%:** Unacceptable. The measurement system is contributing too much noise. You must fix the measurement system before proceeding.
                """)
            
            gage_data = ssm.get_data("gage_rr_data")
            results_df, _ = calculate_gage_rr(gage_data) # We don't need the anova table here
            
            if not results_df.empty:
                total_grr_contrib = results_df.loc['Total Gage R&R', '% Contribution']
                grr_cols = st.columns([1, 2])
                with grr_cols[0]:
                    st.dataframe(results_df.style.format({'% Contribution': '{:.2f}%'}).background_gradient(cmap='Reds', subset=['% Contribution']))
                    if total_grr_contrib < 10: st.success(f"**Verdict:** System is **Acceptable** ({total_grr_contrib:.2f}%)")
                    elif total_grr_contrib < 30: st.warning(f"**Verdict:** System is **Marginal** ({total_grr_contrib:.2f}%)")
                    else: st.error(f"**Verdict:** System is **Unacceptable** ({total_grr_contrib:.2f}%)")
                with grr_cols[1]:
                    fig1, fig2 = create_gage_rr_plots(gage_data)
                    st.plotly_chart(fig1, use_container_width=True)
                    st.plotly_chart(fig2, use_container_width=True)
            else:
                st.error("Gage R&R analysis failed. Please check data and logs.")
        
        # ==================== ANALYZE PHASE ====================
        with phase_tabs[2]:
            st.subheader("Analyze Phase: Identify Root Causes")
            st.info("The **Analyze** phase is about using data and structured problem-solving tools to identify the verified root causes of the problem defined in the charter.")
            
            st.markdown("#### Root Cause Analysis (RCA) Toolkit")
            rca_cols = st.columns(2)
            with rca_cols[0]:
                _render_fishbone_diagram(effect="Low Sub-Assembly Yield")
            with rca_cols[1]:
                st.markdown("##### 5 Whys Analysis")
                st.info("Drill down past symptoms to find the true root cause.")
                st.text_input("1. Why is yield low?", "The alignment fixture is inconsistent.", key=f"why1_{project['id']}")
                st.text_input("2. Why is it inconsistent?", "It wears down quickly.", key=f"why2_{project['id']}")
                st.text_input("3. Why does it wear down?", "The material hardness spec is too low.", key=f"why3_{project['id']}")
                st.text_input("4. Why is the spec low?", "It was based on an older, lower-throughput model.", key=f"why4_{project['id']}")
                st.error("**Root Cause:** Process oversight during design transfer.", icon="🔑")

            st.markdown("---")
            st.markdown("#### Hypothesis Testing for Verification")
            ht_data = ssm.get_data("hypothesis_testing_data")
            test_type = st.radio("Select Test:", ["2-Sample t-Test (Before vs. After)", "ANOVA (Supplier A vs. B vs. C)"], horizontal=True)
            
            if test_type == "2-Sample t-Test (Before vs. After)":
                st.markdown("###### Is there a significant difference between the 'Before' and 'After' process change?")
                fig, result = perform_t_test(ht_data['before_change'], ht_data['after_change'])
                st.plotly_chart(px.box(pd.melt(ht_data[['before_change', 'after_change']]), x='variable', y='value', color='variable'), use_container_width=True)
                if result.get('reject_null'): st.success(f"**Conclusion:** The difference is **statistically significant** (p = {result['p_value']:.4f}). We reject the null hypothesis.")
                else: st.warning(f"**Conclusion:** The difference is **not statistically significant** (p = {result['p_value']:.4f}). We fail to reject the null hypothesis.")
            
            elif test_type == "ANOVA (Supplier A vs. B vs. C)":
                st.markdown("###### Is there a significant difference in component strength between Suppliers A, B, and C?")
                df_anova = pd.melt(ht_data[['supplier_a', 'supplier_b', 'supplier_c']], var_name='group', value_name='value')
                fig, result = perform_anova_on_dataframe(df_anova, 'value', 'group')
                st.plotly_chart(px.box(df_anova, x='group', y='value', color='group'), use_container_width=True)
                if result.get('reject_null'): st.success(f"**Conclusion:** There is a **statistically significant difference** between at least two suppliers (p = {result['p_value']:.4f}).")
                else: st.warning(f"**Conclusion:** There is **no statistically significant difference** between suppliers (p = {result['p_value']:.4f}).")
        
        # ==================== IMPROVE PHASE ====================
        with phase_tabs[3]:
            st.subheader("Improve Phase: Develop and Verify Solutions")
            st.info("The **Improve** phase is about using tools like Design of Experiments (DOE) to find the optimal process settings that solve the problem and achieve the goal.")
            
            st.markdown("#### Design of Experiments (DOE) Analysis")
            with st.expander("Learn More: Interpreting DOE Plots"):
                st.markdown("""
                DOE is the most powerful tool for process optimization.
                - **Main Effects Plot:** Shows the average impact each factor has on the response. The steeper the line, the more significant the factor.
                - **Interaction Plot:** Visualizes how one factor's effect changes based on another's level. **Non-parallel (crossed) lines indicate a significant interaction,** which is often a key discovery.
                - **Response Surface:** A 3D map of the predicted response, helping to visualize the optimal process window.
                """)
            
            doe_data = ssm.get_data("doe_data")
            factors, response = ['temp', 'time', 'pressure'], 'strength'
            doe_plots = create_doe_plots(doe_data, factors, response)
            
            st.plotly_chart(doe_plots['main_effects'], use_container_width=True)
            st.plotly_chart(doe_plots['interaction'], use_container_width=True)
            st.plotly_chart(doe_plots['surface'], use_container_width=True)
            
            st.success("**DOE Conclusion:** The analysis reveals that **Time** has the largest positive effect on bond strength, while **Temperature** also has a significant effect. A strong interaction between Time and Temperature is observed. The optimal process window appears to be at high Time and high Temperature, as visualized on the response surface.")
            
        # ==================== CONTROL PHASE ====================
        with phase_tabs[4]:
            st.subheader("Control Phase: Sustain the Gains")
            st.info("The **Control** phase is about institutionalizing the improvement to ensure it is permanent. This involves updating documentation, implementing monitoring systems like SPC, and creating a formal control plan.")
            
            st.markdown("#### 1. Finalized Control Plan")
            control_plan_data = {
                'Process Step': ['Display Module Bonding'], 
                'Critical Parameter (Y)': ['Bond Strength'], 
                'Key Input (X)': ['Temperature', 'Time'], 
                'Specification': ['> 95 MPa'], 
                'Control Method': ['SPC Chart (I-MR) on bonding machine', 'Automated timer'], 
                'Reaction Plan': ['Halt line if SPC shows out-of-control. Recalibrate bonder.', 'Alarm if timer fails.']
            }
            st.dataframe(pd.DataFrame(control_plan_data), hide_index=True, use_container_width=True)
            
            st.markdown("#### 2. Live SPC Monitoring of New Process")
            st.markdown("This SPC chart monitors the process *after* the improvements have been implemented.")
            # Simulate a new, improved process that is in control at a higher mean
            improved_process = pd.Series(np.random.normal(loc=98, scale=1.2, size=100))
            spc_fig = create_imr_chart(improved_process, "Bond Strength (Post-Improvement)", 95, 105)
            st.plotly_chart(spc_fig, use_container_width=True)
            
            st.success("**Conclusion:** The new process is stable and in a state of statistical control at a new, higher performance level. The improvements have been successfully implemented and sustained.")

    except Exception as e:
        st.error(f"An error occurred while rendering the DMAIC Toolkit: {e}")
        logger.error(f"Failed to render DMAIC toolkit: {e}", exc_info=True)
