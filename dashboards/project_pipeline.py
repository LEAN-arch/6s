# six_sigma/dashboards/project_pipeline.py
"""
Renders the Improvement Project Pipeline dashboard.

This module provides a high-level, portfolio view of all active DMAIC
improvement projects. It allows an MBB to track the overall health of the
improvement program, visualize project timelines, and monitor progress
through the DMAIC phases.
"""

import logging
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from six_sigma.data.session_state_manager import SessionStateManager

logger = logging.getLogger(__name__)

def render_project_pipeline(ssm: SessionStateManager) -> None:
    """
    Creates the UI for the Improvement Project Pipeline dashboard.

    Args:
        ssm (SessionStateManager): The session state manager to access project data.
    """
    st.header("🗂️ Improvement Project Pipeline")
    st.markdown("Track the portfolio of active DMAIC projects across all sites. This dashboard provides a strategic overview of program health, progress, and timelines.")

    try:
        # --- 1. Load and Prepare Data ---
        projects_list = ssm.get_data("dmaic_projects")
        if not projects_list:
            st.warning("No DMAIC projects have been defined.")
            return

        df = pd.DataFrame(projects_list)
        df['start_date'] = pd.to_datetime(df['start_date'])
        # Estimate end dates for Gantt chart based on phase
        phase_duration = {"Define": 30, "Measure": 45, "Analyze": 60, "Improve": 90, "Control": 30}
        df['end_date'] = df.apply(lambda row: row['start_date'] + pd.Timedelta(days=phase_duration.get(row['phase'], 30)), axis=1)

        # --- 2. High-Level Program KPIs ---
        total_projects = len(df)
        projects_in_control = len(df[df['phase'] == 'Control'])
        # Simplified for demo: COPQ saved is a function of projects in control
        total_copq_saved = projects_in_control * 125000 

        st.subheader("Program Health KPIs")
        kpi_cols = st.columns(3)
        kpi_cols[0].metric("Active Projects", total_projects, help="Total number of DMAIC projects in the pipeline.")
        kpi_cols[1].metric("Projects in Control Phase", projects_in_control, help="Projects that have completed the core improvement cycle and are being sustained.")
        kpi_cols[2].metric("Estimated Annualized Savings", f"${total_copq_saved:,.0f}", help="Estimated savings from projects that have reached the Control phase.")

        st.divider()
        
        # --- 3. Visualizations ---
        viz_cols = st.columns(2)
        with viz_cols[0]:
            # --- DMAIC Phase Funnel ---
            st.markdown("**Project Distribution by DMAIC Phase**")
            phase_order = ["Define", "Measure", "Analyze", "Improve", "Control"]
            phase_counts = df['phase'].value_counts().reindex(phase_order, fill_value=0)
            
            fig_funnel = go.Figure(go.Funnel(
                y=phase_counts.index,
                x=phase_counts.values,
                textinfo="value+percent initial",
                marker={"color": ["#1f77b4", "#ff7f0e", "#d62728", "#9467bd", "#2ca02c"]}
            ))
            fig_funnel.update_layout(
                title_text="Project Funnel",
                height=450,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_funnel, use_container_width=True)

        with viz_cols[1]:
            # --- Project Timeline Gantt Chart ---
            st.markdown("**Project Timelines**")
            fig_gantt = px.timeline(
                df,
                x_start="start_date",
                x_end="end_date",
                y="id",
                color="site",
                title="Gantt Chart of Active Projects",
                labels={"id": "Project ID", "site": "Site"},
                hover_name="title"
            )
            fig_gantt.update_yaxes(categoryorder="total ascending")
            fig_gantt.update_layout(
                height=450,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_gantt, use_container_width=True)

        st.divider()

        # --- 4. Detailed Project Table ---
        st.subheader("Project Portfolio Details")
        st.dataframe(
            df[['id', 'title', 'site', 'phase', 'team']],
            hide_index=True,
            use_container_width=True,
            column_config={
                "id": "Project ID",
                "title": st.column_config.TextColumn("Project Title", width="large"),
                "site": "Site",
                "phase": "Current Phase",
                "team": st.column_config.ListColumn("Team (Lead first)")
            }
        )
        
    except Exception as e:
        st.error("An error occurred while rendering the Project Pipeline dashboard.")
        logger.error(f"Failed to render project pipeline: {e}", exc_info=True)
