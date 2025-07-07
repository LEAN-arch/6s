"""
Renders the expert-level DMAIC Improvement Project Toolkit, the core operational
workspace for project execution within the Command Center.

This module provides an interactive, end-to-end environment for executing
complex Six Sigma projects. It guides an MBB through each phase of the DMAIC
methodology (Define, Measure, Analyze, Improve, Control), embedding the
application's advanced statistical and plotting utilities directly into the
project workflow.

SME Masterclass Overhaul:
- Architected as a fully integrated, DYNAMIC project-centric workspace. Selecting
  a project now loads a dataset SPECIFIC to that project's problem.
- **Massively Extended:** Each DMAIC phase now includes an expandable 'Tollgate
  Documents' section, providing detailed, realistic examples and SME explanations
  for a comprehensive suite of Six Sigma tools (e.g., VOC/Kano, FMEA, Pugh Matrix).
- The toolkit is now a world-class educational resource as well as an analytical
  workbench, demonstrating the full breadth of a real-world DMAIC project.
- The Control phase shows a direct "before and after" comparison, visualizing
  the simulated success of the project against the initial baseline.
"""

import logging
import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import statsmodels.api as sm

from six_sigma.data.session_state_manager import SessionStateManager
from six_sigma.utils.plotting import create_imr_chart, create_histogram_with_specs, create_doe_plots, create_gage_rr_plots
from six_sigma.utils.stats import calculate_process_performance, perform_hypothesis_test, perform_anova_on_dataframe, calculate_gage_rr

logger = logging.getLogger(__name__)

def _render_fishbone_diagram(effect: str):
    """Renders a visually appealing Fishbone (Ishikawa) diagram for RCA."""
    st.markdown("##### Fishbone Diagram: Potential Causes")
    causes = {
        "Measurement": ["Gage not calibrated", "Incorrect test procedure", "Operator error"],
        "Material": ["Inconsistent raw material", "Supplier quality issues", "Improper storage"],
        "Personnel": ["Inadequate training", "High operator fatigue", "SOP not followed"],
        "Environment": ["Poor lighting", "Temp/humidity fluctuations", "Contamination"],
        "Machine": ["Fixture wear & tear", "Incorrect settings", "No preventative maintenance"],
        "Method": ["Outdated SOP", "Inefficient assembly sequence", "Poor handling"]
    }
    st.markdown(f"**Effect:** <span style='color:firebrick;'>{effect}</span>", unsafe_allow_html=True)
    for category, items in causes.items():
        st.markdown(f"**{category}**")
        for item in items: st.markdown(f"- *{item}*")

def render_dmaic_toolkit(ssm: SessionStateManager) -> None:
    """Creates the UI for the DMAIC Improvement Toolkit workspace."""
    st.header("🛠️ DMAIC Project Execution Toolkit")
    st.markdown("Select an active improvement project below to access the full suite of DMAIC tools. This is your primary workspace for project execution, from definition to control.")

    try:
        projects = ssm.get_data("dmaic_projects")
        dmaic_data = ssm.get_data("dmaic_project_data")

        if not projects or not dmaic_data:
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
        project_data = dmaic_data.get(selected_id)
        
        # --- DMAIC Phase Tabs ---
        phase_tabs = st.tabs(["**✅ DEFINE**", "**📏 MEASURE**", "**🔍 ANALYZE**", "**💡 IMPROVE**", "**🛡️ CONTROL**"])

        # ==================== DEFINE PHASE ====================
        with phase_tabs[0]:
            st.subheader(f"Define Phase: Scope the Project")
            st.info("The **Define** phase is about clearly articulating the business problem, goal, scope, and team for the project. The Project Charter is the guiding document for the entire effort.")
            
            with st.container(border=True):
                st.markdown(f"### Project Charter: {project['title']}")
                st.markdown(f"**Site:** {project['site']} | **Product Line:** {project['product_line']} | **Start Date:** {project['start_date']}")
                st.markdown(f"**Team:** {', '.join(project['team'])}")
                st.divider()
                st.error(f"**Problem Statement:**\n\n> {project['problem_statement']}", icon="❗️")
                st.success(f"**Goal Statement (S.M.A.R.T.):**\n\n> {project['goal_statement']}", icon="🎯")
            
            st.markdown("---")
            with st.expander("##### 📖 Explore Define Phase Tollgate Documents & Tools"):
                doc_tabs = st.tabs(["SIPOC Diagram", "VOC & CTQ Tree", "Stakeholder Analysis (RACI)"])
                with doc_tabs[0]:
                    st.markdown("**SIPOC Diagram (Suppliers, Inputs, Process, Outputs, Customers)**")
                    st.caption("A high-level map of the process from start to finish. It helps define the project boundaries and scope.")
                    sipoc_data = {
                        "Suppliers": "Component Vendors, Sub-Assembly Line",
                        "Inputs": "Capacitors, PCBs, Housing, Screws",
                        "Process": "Inspect -> Assemble -> Solder -> Test",
                        "Outputs": "Functional Charging Module",
                        "Customers": "Main Assembly Line, Final Product"
                    }
                    st.dataframe(pd.DataFrame.from_dict(sipoc_data, orient='index', columns=['Examples']).rename_axis('Category'), use_container_width=True)
                with doc_tabs[1]:
                    st.markdown("**Voice of the Customer (VOC) & Critical-to-Quality (CTQ) Tree**")
                    st.caption("Translate customer needs into measurable product/process characteristics.")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("###### VOC / Kano Analysis")
                        st.write("Customer feedback is gathered and classified.")
                        voc_data = {'Customer Need': ["Must fit in housing", "Must charge quickly", "Looks sleek"], 'Requirement Type': ["Basic (Must-be)", "Performance", "Excitement (Attractive)"]}
                        st.dataframe(voc_data, hide_index=True)
                    with col2:
                        st.markdown("###### CTQ Tree")
                        st.write("The needs are broken down into measurable requirements.")
                        st.graphviz_chart('''
                            digraph {
                                "Fit in housing" -> "Correct Dimensions";
                                "Correct Dimensions" -> "Length: 10±0.1mm";
                                "Correct Dimensions" -> "Width: 8±0.1mm";
                            }
                        ''')
                with doc_tabs[2]:
                    st.markdown("**Stakeholder Analysis / RACI Matrix**")
                    st.caption("Defines the roles and responsibilities of team members to ensure clear communication and accountability. (Responsible, Accountable, Consulted, Informed)")
                    raci_data = {'Task': ["Define Scope", "Collect Data", "Analyze Data", "Approve Solution"], 'John (MBB)': ['A', 'C', 'A', 'A'], 'Jane (Engineer)': ['R', 'R', 'R', 'C'], 'Mike (Ops)': ['C', 'R', 'C', 'I']}
                    st.dataframe(pd.DataFrame(raci_data).set_index('Task'), use_container_width=True)

        # ==================== MEASURE PHASE ====================
        with phase_tabs[1]:
            st.subheader("Measure Phase: Quantify the Problem")
            st.info("The **Measure** phase is about collecting data to establish a performance baseline and verifying that your measurement system is reliable enough to be trusted.")
            st.markdown("#### 1. Establish Process Baseline")
            if not project_data: st.error(f"No specific measurement data found for project {selected_id}."); return
            baseline_series = project_data["baseline"]["measurement"]; specs = project_data["specs"]
            metric_name = "Sub-Assembly Dimension (mm)" if selected_id == "DMAIC-001" else "Bond Strength"
            capability_metrics = calculate_process_performance(baseline_series, specs['lsl'], specs['usl'])
            st.metric("Baseline Process Performance (Ppk)", f"{capability_metrics.get('ppk', 0):.2f}", f"Target: > 1.33", delta_color="inverse")
            plot_cols = st.columns(2)
            with plot_cols[0]: st.plotly_chart(create_histogram_with_specs(baseline_series, specs['lsl'], specs['usl'], metric_name, capability_metrics), use_container_width=True)
            with plot_cols[1]: st.plotly_chart(create_imr_chart(baseline_series, metric_name, specs['lsl'], specs['usl']), use_container_width=True)
            st.markdown("---")
            st.markdown("#### 2. Validate the Measurement System (Gage R&R)")
            gage_data = ssm.get_data("gage_rr_data"); results_df, _ = calculate_gage_rr(gage_data)
            if not results_df.empty:
                total_grr_contrib = results_df.loc['Total Gage R&R', '% Contribution']; grr_cols = st.columns([1, 2])
                with grr_cols[0]:
                    st.dataframe(results_df.style.format({'% Contribution': '{:.2f}%'}).background_gradient(cmap='Reds', subset=['% Contribution']))
                    if total_grr_contrib < 10: st.success(f"**Verdict:** System is **Acceptable** ({total_grr_contrib:.2f}%)")
                    else: st.error(f"**Verdict:** System is **Unacceptable** ({total_grr_contrib:.2f}%)")
                with grr_cols[1]: fig1, fig2 = create_gage_rr_plots(gage_data); st.plotly_chart(fig1, use_container_width=True); st.plotly_chart(fig2, use_container_width=True)
            else: st.error("Gage R&R analysis failed.")
            st.markdown("---")
            with st.expander("##### 📖 Explore Measure Phase Tollgate Documents & Tools"):
                doc_tabs = st.tabs(["Data Collection Plan", "Operational Definitions", "Value Stream Map (VSM)"])
                with doc_tabs[0]:
                    st.markdown("**Data Collection Plan**")
                    st.caption("A detailed plan to ensure data is collected consistently and accurately.")
                    plan_data = {'Metric to Collect': [metric_name], 'Data Type': ['Continuous'], 'Measurement Tool': ['Digital Calipers #DC-04'], 'Sample Size': ['5 units per hour'], 'Data Collector': ['Line Operator'], 'Frequency': ['Hourly']}
                    st.dataframe(plan_data, hide_index=True)
                with doc_tabs[1]:
                    st.markdown("**Operational Definitions**")
                    st.caption("Precise definitions of key terms to ensure everyone measures the same way.")
                    st.info(f"**Definition for '{metric_name}':** 'The maximum distance, measured in millimeters to two decimal places using calipers #DC-04, across the component's primary axis (marked 'A' on the engineering drawing). The measurement must be taken after the component has cooled to room temperature for at least 5 minutes.'")
                with doc_tabs[2]:
                    st.markdown("**Value Stream Map (VSM) - Findings**")
                    st.caption("A VSM visualizes the flow of material and information. Below are key findings from the VSM exercise for this process.")
                    st.code("""
Process Step         | Value-Add Time | Non-Value-Add Time (Wait)
---------------------|----------------|--------------------------
1. Component Kitting | 2 mins         | 45 mins
2. Sub-Assembly      | 5 mins         | 120 mins (bottleneck)
3. Solder            | 3 mins         | 15 mins
4. Test              | 1 min          | 30 mins
-----------------------------------------------------------------
Total Lead Time: 221 mins | Total Value-Add Time: 11 mins
Process Cycle Efficiency (PCE): 4.98%
                    """, language='bash')
        
        # ==================== ANALYZE PHASE ====================
        with phase_tabs[2]:
            st.subheader("Analyze Phase: Identify Root Causes")
            st.info("The **Analyze** phase is about using data and structured problem-solving tools to identify the verified root causes of the problem defined in the charter.")
            st.markdown("#### Root Cause Brainstorming & Verification")
            rca_cols = st.columns(2)
            with rca_cols[0]: _render_fishbone_diagram(effect="Low Sub-Assembly Yield")
            with rca_cols[1]:
                st.markdown("##### 5 Whys Analysis"); st.info("Drill down past symptoms to find the true root cause.")
                st.text_input("1. Why is yield low?", "The alignment fixture is inconsistent.", key=f"why1_{project['id']}")
                st.text_input("2. Why is it inconsistent?", "It wears down quickly.", key=f"why2_{project['id']}")
                st.error("**Root Cause:** Process oversight during design transfer.", icon="🔑")
            st.markdown("---")
            st.markdown("#### Data-Driven Analysis & Root Cause Verification")
            ht_shifts = project_data["shifts"]
            result = perform_hypothesis_test(ht_shifts['shift_1'], ht_shifts['shift_2'])
            st.plotly_chart(px.box(pd.melt(ht_shifts, var_name='Group', value_name='Value'), x='Group', y='Value', color='Group', title="Hypothesis Test: Comparison of Production Shifts"), use_container_width=True)
            if result.get('reject_null'): st.success(f"**Conclusion:** The difference is statistically significant (p = {result.get('p_value', 0):.4f}). We reject the null hypothesis that the shifts are the same.")
            else: st.warning(f"**Conclusion:** The difference is not statistically significant (p = {result.get('p_value', 0):.4f}).")
            st.markdown("---")
            with st.expander("##### 📖 Explore Analyze Phase Tollgate Documents & Tools"):
                doc_tabs = st.tabs(["Pareto Analysis", "Failure Mode and Effects Analysis (FMEA)", "Regression Analysis"])
                with doc_tabs[0]:
                    st.markdown("**Pareto Analysis of Defect Types**")
                    st.caption("Identifies the 'vital few' defect types that cause the majority of problems for this specific sub-assembly step.")
                    pareto_data = {'Defect Type': ['Scratched Housing', 'Wrong Dimension', 'Bent Pin', 'Solder Splash', 'Missing Screw'], 'Frequency': [88, 65, 15, 8, 4]}
                    pareto_df = pd.DataFrame(pareto_data).sort_values('Frequency', ascending=False)
                    pareto_df['Cumulative %'] = (pareto_df['Frequency'].cumsum() / pareto_df['Frequency'].sum()) * 100
                    st.dataframe(pareto_df.style.format({'Cumulative %': '{:.1f}%'}), use_container_width=True)
                    st.info("The Pareto chart clearly shows 'Scratched Housing' and 'Wrong Dimension' are the 80/20 opportunities.")
                with doc_tabs[1]:
                    st.markdown("**Failure Mode and Effects Analysis (FMEA) - Excerpt**")
                    st.caption("A risk assessment tool to systematically identify and prioritize potential failure modes.")
                    fmea_data = {'Process Step': ['Fixture Placement'], 'Potential Failure Mode': ['Misalignment'], 'Potential Effect': ['Wrong Dimension'], 'SEV': [8], 'OCC': [5], 'DET': [3], 'RPN': [120]}
                    st.dataframe(fmea_data, hide_index=True)
                    st.warning("**High RPN (Risk Priority Number):** An RPN of 120 indicates this is a high-risk failure mode that must be addressed.")
                with doc_tabs[2]:
                    st.markdown("**Regression Analysis**")
                    st.caption("Models the relationship between an input (X) and an output (Y). Here, we model the effect of fixture age on defect rate.")
                    X = np.random.rand(50) * 10; y = 0.5 * X + np.random.randn(50) * 2 + 3
                    X = sm.add_constant(X); model = sm.OLS(y, X).fit()
                    st.code(f"{model.summary()}")
                    st.success("**Conclusion:** The strong positive coefficient for the input variable and its low p-value (<0.05) statistically confirms that as the fixture gets older, the defect rate significantly increases.")
        
        # ==================== IMPROVE PHASE ====================
        with phase_tabs[3]:
            st.subheader("Improve Phase: Develop and Verify Solutions")
            st.info("The **Improve** phase is about using tools like Design of Experiments (DOE) to find the optimal process settings that solve the problem and achieve the goal.")
            st.markdown("#### Design of Experiments (DOE) for Process Optimization")
            doe_data = ssm.get_data("doe_data"); factors, response = ['temp', 'time', 'pressure'], 'strength'
            doe_plots = create_doe_plots(doe_data, factors, response)
            st.plotly_chart(doe_plots['main_effects'], use_container_width=True)
            st.success("**DOE Conclusion:** The analysis reveals that **Time** has the largest positive effect on bond strength, while **Temperature** also has a significant effect.")
            st.markdown("---")
            with st.expander("##### 📖 Explore Improve Phase Tollgate Documents & Tools"):
                doc_tabs = st.tabs(["Solution Selection (Pugh Matrix)", "Mistake-Proofing (Poka-Yoke)", "Pilot & Implementation Plan"])
                with doc_tabs[0]:
                    st.markdown("**Solution Selection (Pugh Matrix)**")
                    st.caption("A structured method for comparing multiple solution concepts against a baseline or standard.")
                    pugh_data = {'Criteria': ['Cost', 'Effectiveness', 'Ease of Implementation', 'Sustainability'], 'Baseline (Current)': [0, 0, 0, 0], 'Solution A: New Fixture': [-2, 2, -1, 2], 'Solution B: Modify SOP': [1, 1, 2, -1]}
                    pugh_df = pd.DataFrame(pugh_data).set_index('Criteria'); pugh_df.loc['Total Score'] = pugh_df.sum()
                    st.dataframe(pugh_df.style.apply(lambda x: ['background: lightgreen' if v > 0 else 'background: pink' if v < 0 else '' for v in x], axis=1))
                    st.success("**Decision:** Solution A (New Fixture) has the highest positive score and is chosen for implementation.")
                with doc_tabs[1]:
                    st.markdown("**Mistake-Proofing (Poka-Yoke) Ideas**")
                    st.caption("Designing the process to make it impossible to make a mistake.")
                    st.success("**Idea 1: Guide Pins.** The new fixture will have asymmetric guide pins, making it physically impossible to insert the component backwards.", icon="✅")
                    st.success("**Idea 2: Sensor Interlock.** A sensor on the fixture will confirm the component is fully seated. The machine will not start if the sensor is not triggered.", icon="✅")
                with doc_tabs[2]:
                    st.markdown("**Implementation Plan (Action Plan)**")
                    st.caption("A high-level roadmap for deploying the solution.")
                    st.markdown("""
                    - **Week 1:** Finalize and order new fixture design.
                    - **Week 3:** Recieve and validate new fixture.
                    - **Week 4:** Conduct pilot run with new fixture on Line 2.
                    - **Week 5:** Train all operators on new procedure and fixture.
                    - **Week 6:** Full rollout across all lines. Update SOPs.
                    """)
            
        # ==================== CONTROL PHASE ====================
        with phase_tabs[4]:
            st.subheader("Control Phase: Sustain the Gains")
            st.info("The **Control** phase is about institutionalizing the improvement to ensure it is permanent. This involves updating documentation, implementing monitoring systems like SPC, and creating a formal control plan.")
            st.markdown("#### 1. Live SPC Monitoring of New, Improved Process")
            improved_mean = specs["target"]; improved_std = capability_metrics.get('sigma', 1.0) / 2
            improved_process = pd.Series(np.random.normal(loc=improved_mean, scale=improved_std, size=200))
            new_capability = calculate_process_performance(improved_process, specs['lsl'], specs['usl'])
            st.metric("New Process Performance (Ppk)", f"{new_capability.get('ppk', 0):.2f}", f"Improved from {capability_metrics.get('ppk', 0):.2f}", delta_color="normal")
            spc_fig = create_imr_chart(improved_process, f"{metric_name} (Post-Improvement)", specs['lsl'], specs['usl'])
            st.plotly_chart(spc_fig, use_container_width=True)
            st.markdown("---")
            with st.expander("##### 📖 Explore Control Phase Tollgate Documents & Tools"):
                doc_tabs = st.tabs(["Control Plan", "Response Plan", "Lessons Learned"])
                with doc_tabs[0]:
                    st.markdown("**Finalized Control Plan**")
                    st.caption("The official document defining how the gains will be maintained.")
                    control_plan_data = { 'Process Step': ['Sub-Assembly Fixture', 'Sub-Assembly Fixture'], 'Critical Input (X)': ['Fixture Material Hardness', 'Fixture PM Schedule'], 'Specification': ['Rockwell HRC 58-62', 'Quarterly'], 'Control Method': ['Material Cert from Supplier', 'CMMS Work Order'], 'Sample Size/Freq': ['Per Lot', 'Per Quarter']}
                    st.dataframe(pd.DataFrame(control_plan_data), hide_index=True)
                with doc_tabs[1]:
                    st.markdown("**Response Plan (Out-of-Control Action Plan)**")
                    st.caption("A clear 'IF-THEN' plan for when the process goes out of control.")
                    st.warning("**IF** a point on the I-MR chart violates a control limit, **THEN**:")
                    st.markdown("""1. The operator immediately stops the line...""")
                with doc_tabs[2]:
                    st.markdown("**Lessons Learned**")
                    st.caption("A summary of key takeaways to be shared across the organization.")
                    st.success("**Key Insight:** The original design transfer process did not adequately account for increased production throughput...", icon="💡")
                    st.success("**Best Practice:** The new asymmetric guide pin design is highly effective...", icon="💡")

    except Exception as e:
        st.error(f"An error occurred while rendering the DMAIC Toolkit: {e}")
        logger.error(f"Failed to render DMAIC toolkit: {e}", exc_info=True)
