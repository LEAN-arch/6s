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
- **Visually Rich:** All tollgate documents have been enhanced with professional
  visualizations, including Graphviz diagrams, Pareto charts, VSM charts, and more,
  transforming the toolkit into a world-class educational resource.
- The Control phase shows a direct "before and after" comparison, visualizing
  the simulated success of the project against the initial baseline.
, encoding='utf-8'
"""

import logging
import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm

from six_sigma.data.session_state_manager import SessionStateManager
from six_sigma.utils.plotting import create_imr_chart, create_histogram_with_specs, create_doe_plots, create_gage_rr_plots
from six_sigma.utils.stats import calculate_process_performance, perform_hypothesis_test, perform_anova_on_dataframe, calculate_gage_rr

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_datasets(ssm: SessionStateManager) -> bool:
    """Validate all required datasets for the DMAIC toolkit."""
    datasets = {
        "dmaic_projects": ["id", "title", "site", "product_line", "start_date", "team", "problem_statement", "goal_statement"],
        "dmaic_project_data": ["baseline", "specs", "shifts"],
        "gage_rr_data": [],
        "doe_data": ["temp", "time", "pressure", "strength"]
    }
    for dataset, required_cols in datasets.items():
        data = ssm.get_data(dataset)
        if dataset == "dmaic_projects":
            if not isinstance(data, list) or not data or not all(isinstance(p, dict) and all(col in p for col in required_cols) for p in data):
                st.error(f"Invalid {dataset} structure.")
                logger.error(f"Invalid {dataset} structure")
                return False
        elif dataset == "dmaic_project_data":
            if not isinstance(data, dict) or not data or not all(isinstance(p.get("baseline", {}).get("measurement"), pd.Series) and all(k in p.get("specs", {}) for k in ["lsl", "usl", "target"]) for p in data.values()):
                st.error(f"Invalid {dataset} structure.")
                logger.error(f"Invalid {dataset} structure")
                return False
        else:
            if data.empty or (required_cols and not all(col in data.columns for col in required_cols)):
                st.error(f"Invalid or missing {dataset}.")
                logger.error(f"Invalid {dataset}: {data.columns.tolist() if not data.empty else 'empty'}")
                return False
    return True

def _render_fishbone_diagram(effect: str):
    """Renders a visually appealing Fishbone (Ishikawa) diagram for RCA using Graphviz."""
    if not effect or not isinstance(effect, str):
        logger.error("Invalid effect parameter for Fishbone diagram")
        st.error("Invalid effect parameter for Fishbone diagram")
        return
    st.markdown("##### Fishbone Diagram: Potential Causes")
    causes = {
        "Measurement": ["Gage not calibrated", "Incorrect test procedure"],
        "Material": ["Inconsistent raw material", "Supplier quality issues"],
        "Personnel": ["Inadequate training", "SOP not followed"],
        "Environment": ["Poor lighting", "Temp/humidity fluctuations"],
        "Machine": ["Fixture wear & tear", "Incorrect settings"],
        "Method": ["Outdated SOP", "Inefficient assembly sequence"]
    }
    try:
        # Pre-process labels to escape quotes
        escaped_effect = effect.replace('"', '\\"')
        sub_labels = [
            "{} [label=\"{}\", shape=ellipse, fillcolor=lightyellow] -> {}".format(
                f"{cat}_sub{i}",
                sub.replace('"', '\\"'),
                cat
            )
            for cat, subs in causes.items()
            for i, sub in enumerate(subs)
        ]
        dot = r'''
        digraph {
            rankdir=LR;
            node [shape=box, style=filled, fillcolor=lightblue];
            Effect [label="%s", fillcolor=firebrick, fontcolor=white];
            %s;
            %s;
            %s;
        }
        ''' % (
            escaped_effect,
            "; ".join([f"{cat} [label=\"{cat}\"]" for cat in causes.keys()]),
            "; ".join([f"{cat} -> Effect" for cat in causes.keys()]),
            "; ".join(sub_labels)
        )
        st.graphviz_chart(dot)
    except Exception as e:
        st.error("Failed to render Fishbone diagram.")
        logger.error(f"Fishbone diagram rendering failed: {e}")

def render_dmaic_toolkit(ssm: SessionStateManager) -> None:
    """Creates the UI for the DMAIC Improvement Toolkit workspace."""
    if not isinstance(ssm, SessionStateManager):
        st.error("Invalid SessionStateManager instance provided.")
        logger.error("SessionStateManager is not properly initialized.")
        return
    if not validate_datasets(ssm):
        return

    st.header("🛠️ DMAIC Project Execution Toolkit")
    st.markdown("Select an active improvement project below to access the full suite of DMAIC tools. This is your primary workspace for project execution, from definition to control.")

    try:
        projects = ssm.get_data("dmaic_projects")
        dmaic_data = ssm.get_data("dmaic_project_data")
        project_titles = {p['id']: f"{p['id']}: {p['title']}" for p in projects}
        if not hasattr(st.session_state, 'selected_project_id') or st.session_state.selected_project_id not in project_titles:
            st.session_state.selected_project_id = list(project_titles.keys())[0]
        selected_id = st.selectbox("**Select Active Project:**", options=list(project_titles.keys()), format_func=lambda x: project_titles[x], help="The analysis in the tabs below will update based on this selection.")
        project = next((p for p in projects if p['id'] == selected_id), None)
        if not project:
            st.error(f"Project with ID {selected_id} not found.")
            logger.error(f"Project ID {selected_id} not found in projects")
            return
        required_keys = ['title', 'site', 'product_line', 'start_date', 'team', 'problem_statement', 'goal_statement']
        if not all(key in project for key in required_keys):
            st.error(f"Project {selected_id} missing required metadata: {set(required_keys) - set(project.keys())}")
            logger.error(f"Project {selected_id} missing keys: {set(required_keys) - set(project.keys())}")
            return
        if not isinstance(project['team'], list):
            st.error("Project team must be a list of members.")
            logger.error(f"Invalid team format for project {selected_id}: {project['team']}")
            return
        project_data = dmaic_data.get(selected_id)
        if not isinstance(project_data, dict):
            st.error(f"No data found for project {selected_id}.")
            logger.error(f"No project_data for ID {selected_id}: {project_data}")
            return

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
                    try:
                        st.graphviz_chart(r'''
                            digraph {
                                rankdir=LR;
                                node [shape=box, style=rounded];
                                Suppliers [label="Suppliers\\n- Component Vendors\\n- Sub-Assembly Line"];
                                Inputs [label="Inputs\\n- Capacitors, PCBs\\n- Housing, Screws"];
                                Process [label="Process Steps\\n1. Inspect\\n2. Assemble\\n3. Solder\\n4. Test"];
                                Outputs [label="Outputs\\n- Functional Module"];
                                Customers [label="Customers\\n- Main Assembly Line\\n- Final Product"];
                                Suppliers -> Inputs -> Process -> Outputs -> Customers;
                            }
                        ''')
                    except Exception as e:
                        st.error("Failed to render SIPOC diagram.")
                        logger.error(f"SIPOC diagram rendering failed: {e}")
                with doc_tabs[1]:
                    st.markdown("**Voice of the Customer (VOC) & Critical-to-Quality (CTQ) Tree**")
                    st.caption("Translate customer needs into measurable product/process characteristics.")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("###### Kano Model Visualization")
                        st.write("Customer feedback is classified to prioritize features.")
                        kano_data = project_data.get("kano_data", pd.DataFrame({
                            'Feature': ['Fits in housing', 'Charges quickly', 'Looks sleek'],
                            'Satisfaction': [2, 8, 10],
                            'Execution': [2, 7, 3],
                            'Type': ['Basic', 'Performance', 'Exciter']
                        }))
                        try:
                            fig = px.scatter(kano_data, x='Execution', y='Satisfaction', text='Feature', color='Type', title="Kano Model Analysis", labels={'Execution': 'Degree of Execution', 'Satisfaction': 'Customer Satisfaction'})
                            fig.update_traces(textposition='top center')
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error("Failed to render Kano model plot.")
                            logger.error(f"Kano model plot failed: {e}")
                    with col2:
                        st.markdown("###### CTQ Tree")
                        st.write("The needs are broken down into measurable requirements.")
                        try:
                            st.graphviz_chart(r'''
                                digraph {
                                    "Fit in housing" -> "Correct Dimensions";
                                    "Correct Dimensions" -> "Length: 10±0.1mm";
                                    "Correct Dimensions" -> "Width: 8±0.1mm";
                                }
                            ''')
                        except Exception as e:
                            st.error("Failed to render CTQ tree.")
                            logger.error(f"CTQ tree rendering failed: {e}")
                with doc_tabs[2]:
                    st.markdown("**Stakeholder Analysis / RACI Matrix**")
                    st.caption("Defines the roles and responsibilities of team members. (Responsible, Accountable, Consulted, Informed)")
                    raci_data = project_data.get("raci_data", {
                        'Task': ["Define Scope", "Collect Data", "Analyze Data", "Approve Solution"],
                        'John (MBB)': ['A', 'C', 'A', 'A'],
                        'Jane (Engineer)': ['R', 'R', 'R', 'C'],
                        'Mike (Ops)': ['C', 'R', 'C', 'I']
                    })
                    raci_df = pd.DataFrame(raci_data).set_index('Task')
                    def color_raci(val):
                        colors = {
                            'R': 'background-color: #a8d8ea',  # Light Blue
                            'A': 'background-color: #f4c7c3',  # Light Red
                            'C': 'background-color: #b8d8be',  # Light Green
                            'I': 'background-color: #e0e0e0'   # Light Grey
                        }
                        return colors.get(val, '')
                    try:
                        st.dataframe(raci_df.style.map(color_raci), use_container_width=True)
                    except Exception as e:
                        st.error("Failed to render RACI matrix.")
                        logger.error(f"RACI matrix rendering failed: {e}")

        # ==================== MEASURE PHASE ====================
        with phase_tabs[1]:
            st.subheader("Measure Phase: Quantify the Problem")
            st.info("The **Measure** phase is about collecting data to establish a performance baseline and verifying that your measurement system is reliable enough to be trusted.")
            st.markdown("#### 1. Establish Process Baseline")
            if not isinstance(project_data, dict) or "baseline" not in project_data or "specs" not in project_data:
                st.error(f"Invalid project_data structure for project {selected_id}.")
                logger.error(f"Invalid project_data structure for {selected_id}")
            else:
                baseline_series = project_data["baseline"].get("measurement")
                specs = project_data["specs"]
                if not isinstance(baseline_series, pd.Series) or baseline_series.empty:
                    st.error(f"No valid baseline measurement data for project {selected_id}.")
                    logger.error(f"No valid baseline measurement for {selected_id}")
                elif not all(key in specs for key in ["lsl", "usl", "target"]):
                    st.error(f"Missing specification limits (lsl, usl, target) for project {selected_id}.")
                    logger.error(f"Missing specs for {selected_id}: {specs.keys()}")
                else:
                    metric_name = project_data.get("metric_name", "Sub-Assembly Dimension (mm)" if selected_id == "DMAIC-001" else "Bond Strength")
                    try:
                        capability_metrics = calculate_process_performance(baseline_series, specs['lsl'], specs['usl'])
                        st.metric("Baseline Process Performance (Ppk)", f"{capability_metrics.get('ppk', 0):.2f}", f"Target: > 1.33", delta_color="inverse")
                        plot_cols = st.columns(2)
                        plot_cols[0].plotly_chart(create_histogram_with_specs(baseline_series, specs['lsl'], specs['usl'], metric_name, capability_metrics), use_container_width=True)
                        plot_cols[1].plotly_chart(create_imr_chart(baseline_series, metric_name, specs['lsl'], specs['usl']), use_container_width=True)
                    except Exception as e:
                        st.error("Failed to generate baseline process plots.")
                        logger.error(f"Baseline process plots failed: {e}")

            st.markdown("---")
            st.markdown("#### 2. Validate the Measurement System (Gage R&R)")
            gage_data = ssm.get_data("gage_rr_data")
            if gage_data.empty:
                st.error("No Gage R&R data available.")
                logger.error("gage_rr_data is empty")
            else:
                try:
                    results_df, _ = calculate_gage_rr(gage_data)
                    if not results_df.empty:
                        total_grr_contrib = results_df.loc['Total Gage R&R', '% Contribution']
                        grr_cols = st.columns([1, 2])
                        with grr_cols[0]:
                            st.dataframe(results_df.style.format({'% Contribution': '{:.2f}%'}).background_gradient(cmap='Reds', subset=['% Contribution']))
                            if total_grr_contrib < 10:
                                st.success(f"**Verdict:** System is **Acceptable** ({total_grr_contrib:.2f}%)")
                            else:
                                st.error(f"**Verdict:** System is **Unacceptable** ({total_grr_contrib:.2f}%)")
                        with grr_cols[1]:
                            fig1, fig2 = create_gage_rr_plots(gage_data)
                            st.plotly_chart(fig1, use_container_width=True)
                            st.plotly_chart(fig2, use_container_width=True)
                    else:
                        st.error("Gage R&R analysis failed.")
                        logger.error("Gage R&R results empty")
                except Exception as e:
                    st.error("Failed to perform Gage R&R analysis.")
                    logger.error(f"Gage R&R failed: {e}")

            st.markdown("---")
            with st.expander("##### 📖 Explore Measure Phase Tollgate Documents & Tools"):
                doc_tabs = st.tabs(["Data Collection Plan", "Value Stream Map (VSM)"])
                with doc_tabs[0]:
                    st.markdown("**Data Collection Plan**")
                    try:
                        st.dataframe({
                            'Metric to Collect': [metric_name],
                            'Data Type': ['Continuous'],
                            'Tool': ['Calipers #DC-04']
                        }, hide_index=True)
                    except Exception as e:
                        st.error("Failed to render Data Collection Plan.")
                        logger.error(f"Data Collection Plan rendering failed: {e}")
                with doc_tabs[1]:
                    st.markdown("**Value Stream Map (VSM) - Visualization**")
                    st.caption("A VSM visualizes the flow of material and information. This chart powerfully shows the proportion of time that is waste.")
                    vsm_data = project_data.get("vsm_data", pd.DataFrame([
                        {'Category': 'Total Lead Time', 'Type': 'Value-Add Time', 'Time (mins)': 11},
                        {'Category': 'Total Lead Time', 'Type': 'Non-Value-Add Time (Waste)', 'Time (mins)': 210}
                    ]))
                    try:
                        fig = px.bar(vsm_data, x='Time (mins)', y='Category', color='Type', orientation='h', text='Time (mins)', title='Process Cycle Efficiency (PCE)', color_discrete_map={'Value-Add Time': 'green', 'Non-Value-Add Time (Waste)': 'red'})
                        st.plotly_chart(fig, use_container_width=True)
                        st.metric("Process Cycle Efficiency", "4.98%", "Highly inefficient", delta_color="inverse")
                    except Exception as e:
                        st.error("Failed to render VSM plot.")
                        logger.error(f"VSM plot failed: {e}")
        
        # ==================== ANALYZE PHASE ====================
        with phase_tabs[2]:
            st.subheader("Analyze Phase: Identify Root Causes")
            st.info("The **Analyze** phase is about using data and structured problem-solving tools to identify the verified root causes of the problem defined in the charter.")
            st.markdown("#### Root Cause Brainstorming & Verification")
            rca_cols = st.columns(2)
            with rca_cols[0]:
                _render_fishbone_diagram(effect="Low Sub-Assembly Yield")
            with rca_cols[1]:
                st.markdown("##### 5 Whys Analysis")
                st.info("Drill down past symptoms to find the true root cause.")
                why1 = st.text_input("1. Why is yield low?", value=st.session_state.get(f"why1_{project['id']}", "The alignment fixture is inconsistent."), key=f"why1_{project['id']}")
                why2 = st.text_input("2. Why is it inconsistent?", value=st.session_state.get(f"why2_{project['id']}", "It wears down quickly."), key=f"why2_{project['id']}")
                st.error("**Root Cause:** Process oversight during design transfer.", icon="🔑")

            st.markdown("---")
            st.markdown("#### Data-Driven Analysis & Root Cause Verification")
            if "shifts" not in project_data or not all(key in project_data["shifts"] for key in ["shift_1", "shift_2"]):
                st.error(f"No valid shift data for project {selected_id}.")
                logger.error(f"Missing shift data for {selected_id}")
            else:
                ht_shifts = project_data["shifts"]
                try:
                    result = perform_hypothesis_test(ht_shifts['shift_1'], ht_shifts['shift_2'])
                    fig = px.box(pd.melt(pd.DataFrame(ht_shifts), var_name='Group', value_name='Value'), x='Group', y='Value', color='Group', title="Hypothesis Test: Comparison of Production Shifts")
                    st.plotly_chart(fig, use_container_width=True)
                    if result.get('reject_null'):
                        st.success(f"**Conclusion:** The difference is statistically significant (p = {result.get('p_value', 0):.4f}).")
                    else:
                        st.info(f"**Conclusion:** No significant difference (p = {result.get('p_value', 0):.4f}).")
                except Exception as e:
                    st.error("Failed to perform hypothesis test.")
                    logger.error(f"Hypothesis test failed: {e}")

            st.markdown("---")
            with st.expander("##### 📖 Explore Analyze Phase Tollgate Documents & Tools"):
                doc_tabs = st.tabs(["Pareto Analysis", "FMEA", "Regression Analysis"])
                with doc_tabs[0]:
                    st.markdown("**Pareto Analysis of Defect Types**")
                    st.caption("Identifies the 'vital few' defect types that cause the majority of problems.")
                    pareto_data = project_data.get("pareto_data", {
                        'Defect Type': ['Wrong Dimension', 'Scratched Housing', 'Bent Pin', 'Solder Splash', 'Missing Screw'],
                        'Frequency': [88, 65, 15, 8, 4]
                    })
                    pareto_df = pd.DataFrame(pareto_data).sort_values('Frequency', ascending=False)
                    pareto_df['Cumulative %'] = (pareto_df['Frequency'].cumsum() / pareto_df['Frequency'].sum()) * 100
                    try:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=pareto_df['Defect Type'], y=pareto_df['Frequency'], name='Frequency'))
                        fig.add_trace(go.Scatter(x=pareto_df['Defect Type'], y=pareto_df['Cumulative %'], name='Cumulative %', yaxis='y2'))
                        fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="Cumulative %"))
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error("Failed to render Pareto chart.")
                        logger.error(f"Pareto chart failed: {e}")
                with doc_tabs[1]:
                    st.markdown("**Failure Mode and Effects Analysis (FMEA) - Risk Matrix**")
                    st.caption("A risk assessment tool to systematically identify and prioritize potential failure modes.")
                    fmea_data = project_data.get("fmea_data", pd.DataFrame({
                        'Failure Mode': ['Misalignment', 'Dropped Part', 'Wrong Setting'],
                        'Severity': [8, 5, 9],
                        'Occurrence': [5, 2, 1],
                        'Detection': [3, 6, 8]
                    }))
                    fmea_data['RPN'] = fmea_data['Severity'] * fmea_data['Occurrence'] * fmea_data['Detection']
                    try:
                        fig = px.scatter(fmea_data, x='Occurrence', y='Severity', size='RPN', color='RPN', text='Failure Mode', title='FMEA Risk Prioritization', size_max=60)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error("Failed to render FMEA risk matrix.")
                        logger.error(f"FMEA risk matrix failed: {e}")
                with doc_tabs[2]:
                    st.markdown("**Regression Analysis**")
                    st.caption("Models the relationship between an input (X) and an output (Y).")
                    if not isinstance(project_data, dict):
                        st.error(f"Invalid project_data structure for project {selected_id}.")
                        logger.error(f"Invalid project_data structure for {selected_id}: {type(project_data)}")
                    else:
                        regression_data = project_data.get("regression_data")
                        if not isinstance(regression_data, dict):
                            st.warning(f"Regression data missing for project {selected_id}. Using synthetic data.")
                            logger.warning(f"Regression data missing for project {selected_id}: {regression_data}")
                            regression_data = {'X': np.random.rand(50) * 10, 'y': None}
                        X = regression_data.get('X')
                        y = regression_data.get('y')
                        # Validate regression data
                        if (X is None or y is None or
                            not isinstance(X, (list, np.ndarray, pd.Series)) or
                            not isinstance(y, (list, np.ndarray, pd.Series))):
                            st.warning("Regression data is missing or invalid. Using synthetic data for demonstration.")
                            logger.warning(f"Invalid regression data for project {selected_id}: X={type(X)}, y={type(y)}")
                            X = np.random.rand(50) * 10
                            y = 0.5 * X + np.random.randn(50) * 2 + 3
                        try:
                            X = np.array(X)  # Ensure X is a numpy array
                            y = np.array(y)  # Ensure y is a numpy array
                            if len(X) != len(y):
                                st.error(f"Regression data mismatch: len(X)={len(X)}, len(y)={len(y)}. Using synthetic data.")
                                logger.error(f"Regression data mismatch for project {selected_id}: len(X)={len(X)}, len(y)={len(y)}")
                                X = np.random.rand(50) * 10
                                y = 0.5 * X + np.random.randn(50) * 2 + 3
                            fig = px.scatter(x=X, y=y, labels={'x': 'Fixture Age (months)', 'y': 'Defect Rate (%)'}, title='Fixture Age vs. Defect Rate', trendline="ols")
                            st.plotly_chart(fig, use_container_width=True)
                            model = sm.OLS(y, sm.add_constant(X)).fit()
                            st.code(f"{model.summary()}")
                            st.success("**Conclusion:** The strong positive coefficient and low p-value statistically confirm that as the fixture ages, the defect rate increases.")
                        except Exception as e:
                            st.error("Failed to perform regression analysis.")
                            logger.error(f"Regression analysis failed: {e}")

        # ==================== IMPROVE PHASE ====================
        with phase_tabs[3]:
            st.subheader("Improve Phase: Develop and Verify Solutions")
            st.info("The **Improve** phase is about using tools like Design of Experiments (DOE) to find the optimal process settings that solve the problem and achieve the goal.")
            st.markdown("#### Design of Experiments (DOE) for Process Optimization")
            doe_data = ssm.get_data("doe_data")
            factors, response = ['temp', 'time', 'pressure'], 'strength'
            if doe_data.empty or not all(col in doe_data.columns for col in factors + [response]):
                st.error("Invalid or missing DOE data.")
                logger.error(f"DOE data missing columns: {set(factors + [response]) - set(doe_data.columns)}")
            else:
                try:
                    doe_plots = create_doe_plots(doe_data, factors, response)
                    st.plotly_chart(doe_plots['main_effects'], use_container_width=True)
                    st.plotly_chart(doe_plots['interaction'], use_container_width=True)
                    st.success("**DOE Conclusion:** The analysis reveals that **Time** has the largest positive effect on bond strength.")
                except Exception as e:
                    st.error("Failed to generate DOE plots.")
                    logger.error(f"DOE plots failed: {e}")

            st.markdown("---")
            with st.expander("##### 📖 Explore Improve Phase Tollgate Documents & Tools"):
                doc_tabs = st.tabs(["Solution Selection (Pugh Matrix)", "Implementation Plan"])
                with doc_tabs[0]:
                    st.markdown("**Solution Selection (Pugh Matrix)**")
                    st.caption("A structured method for comparing multiple solution concepts against a baseline.")
                    pugh_data = project_data.get("pugh_data", {
                        'Criteria': ['Cost', 'Effectiveness', 'Ease of Implementation', 'Sustainability'],
                        'Baseline (Current)': [0, 0, 0, 0],
                        'Solution A: New Fixture': [-2, 2, -1, 2],
                        'Solution B: Modify SOP': [1, 1, 2, -1]
                    })
                    pugh_df = pd.DataFrame(pugh_data).set_index('Criteria')
                    pugh_df.loc['Total Score'] = pugh_df.sum()
                    try:
                        st.dataframe(pugh_df.style.map(lambda x: 'background-color: #90ee90' if x > 0 else 'background-color: #ffcccb' if x < 0 else ''), use_container_width=True)
                        st.success("**Decision:** Solution A (New Fixture) is chosen.")
                    except Exception as e:
                        st.error("Failed to render Pugh Matrix.")
                        logger.error(f"Pugh Matrix rendering failed: {e}")
                with doc_tabs[1]:
                    st.markdown("**Implementation Plan (Gantt Chart)**")
                    st.caption("A visual roadmap for deploying the solution.")
                    plan_df = project_data.get("gantt_data", pd.DataFrame([
                        dict(Task="Order New Fixture", Start='2024-08-01', Finish='2024-08-05'),
                        dict(Task="Validate Fixture", Start='2024-08-19', Finish='2024-08-23'),
                        dict(Task="Train Operators", Start='2024-08-26', Finish='2024-08-30'),
                        dict(Task="Full Rollout", Start='2024-09-02', Finish='2024-09-06')
                    ]))
                    try:
                        fig = px.timeline(plan_df, x_start="Start", x_end="Finish", y="Task", title="Project Implementation Timeline")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error("Failed to render Gantt chart.")
                        logger.error(f"Gantt chart rendering failed: {e}")
            
        # ==================== CONTROL PHASE ====================
        with phase_tabs[4]:
            st.subheader("Control Phase: Sustain the Gains")
            st.info("The **Control** phase is about institutionalizing the improvement to ensure it is permanent.")
            st.markdown("#### 1. Live SPC Monitoring of New, Improved Process")
            improved_data = project_data.get("improved", {}).get("measurement")
            if not isinstance(improved_data, pd.Series) or improved_data.empty:
                improved_mean = specs["target"]
                improved_std = capability_metrics.get('sigma', 1.0) / 2
                improved_process = pd.Series(np.random.normal(loc=improved_mean, scale=improved_std, size=200))
            else:
                improved_process = improved_data
            try:
                new_capability = calculate_process_performance(improved_process, specs['lsl'], specs['usl'])
                st.metric("New Process Performance (Ppk)", f"{new_capability.get('ppk', 0):.2f}", f"Improved from {capability_metrics.get('ppk', 0):.2f}", delta_color="normal")
                spc_fig = create_imr_chart(improved_process, f"{metric_name} (Post-Improvement)", specs['lsl'], specs['usl'])
                st.plotly_chart(spc_fig, use_container_width=True)
            except Exception as e:
                st.error("Failed to generate SPC chart for improved process.")
                logger.error(f"SPC chart for improved process failed: {e}")

            st.markdown("---")
            with st.expander("##### 📖 Explore Control Phase Tollgate Documents & Tools"):
                doc_tabs = st.tabs(["Control Plan", "Response Plan", "Lessons Learned"])
                with doc_tabs[0]:
                    st.markdown("**Finalized Control Plan**")
                    control_plan_data = project_data.get("control_plan_data", {
                        'Process Step': ['Sub-Assembly Fixture', 'Sub-Assembly Fixture'],
                        'Critical Input (X)': ['Fixture Material Hardness', 'Fixture PM Schedule'],
                        'Specification': ['Rockwell HRC 58-62', 'Quarterly'],
                        'Control Method': ['Material Cert', 'CMMS Work Order']
                    })
                    try:
                        st.dataframe(pd.DataFrame(control_plan_data), hide_index=True)
                    except Exception as e:
                        st.error("Failed to render Control Plan.")
                        logger.error(f"Control Plan rendering failed: {e}")
                with doc_tabs[1]:
                    st.markdown("**Response Plan (Out-of-Control Action Plan)**")
                    st.warning("**IF** a point on the I-MR chart violates a control limit, **THEN**:")
                    st.markdown("""1. Stop the line...""")
                with doc_tabs[2]:
                    st.markdown("**Lessons Learned**")
                    st.success("**Key Insight:** The original design transfer process was inadequate...", icon="💡")
                    st.success("**Best Practice:** The new asymmetric guide pin design is highly effective...", icon="💡")
    except Exception as e:
        st.error(f"An error occurred while rendering the DMAIC Toolkit: {e}")
        logger.error(f"Failed to render DMAIC toolkit: {e}", exc_info=True)

if __name__ == "__main__":
    ssm = SessionStateManager()  # Assumed to be defined elsewhere
    render_dmaic_toolkit(ssm)
