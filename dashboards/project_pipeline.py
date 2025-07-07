"""
Renders the Improvement Project Pipeline dashboard, the central Project Management
Office (PMO) for the Six Sigma program.

This module provides a high-level, portfolio view of all active and completed
DMAIC improvement projects. It allows an MBB to track the overall health,
velocity, and strategic alignment of the entire improvement program, visualizing
project timelines and progress through the DMAIC phases.

SME Overhaul:
- Enhanced program KPIs to include 'Average Project Cycle Time', a critical metric
  for program velocity.
- Replaced the funnel chart with a clearer horizontal bar chart for visualizing
  project distribution by phase.
- Upgraded the detailed project table to be interactive and information-rich,
  adding a visual progress bar for each project.
- The Gantt chart now visualizes the full projected timeline of each project.
- All elements have been polished for a professional, PMO-grade appearance.
"""

import logging
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import date

from six_sigma.data.session_state_manager import SessionStateManager

logger = logging.getLogger(__name__)

def render_project_pipeline(ssm: SessionStateManager) -> None:
    """Creates the UI for the Improvement Project Pipeline dashboard."""
    st.header("🗂️ Improvement Project Pipeline")
    st.markdown("A Project Management Office (PMO) view to track the portfolio of active DMAIC projects. Use this dashboard for a strategic overview of program health, progress, and timelines.")

    try:
        # --- 1. Load and Prepare Data ---
        projects_list = ssm.get_data("dmaic_projects")
        if not projects_list:
            st.warning("No DMAIC projects have been defined in the data model.")
            return

        df = pd.DataFrame(projects_list)
        df['start_date'] = pd.to_datetime(df['start_date'])
        
        # --- Data Enrichment for Visualization ---
        phase_map = {"Define": 1, "Measure": 2, "Analyze": 3, "Improve": 4, "Control": 5}
        df['phase_numeric'] = df['phase'].map(phase_map)
        df['progress'] = df['phase_numeric'] / len(phase_map)
        
        # Calculate projected end dates based on average duration per phase
        phase_duration_days = {"Define": 30, "Measure": 45, "Analyze": 60, "Improve": 90, "Control": 30}
        total_project_duration = sum(phase_duration_days.values())
        df['end_date'] = df['start_date'] + pd.to_timedelta(total_project_duration, unit='d')
        
        df['cycle_time'] = (pd.to_datetime(date.today()) - df['start_date']).dt.days

        # --- 2. High-Level Program KPIs ---
        st.subheader("Program Health & Performance")
        total_projects = len(df)
        projects_completed = len(df[df['phase'] == 'Control'])
        avg_cycle_time = df[df['phase'] != 'Control']['cycle_time'].mean()
        # Simulate savings for a more dynamic feel
        avg_savings_per_project = 175000
        total_annual_savings = projects_completed * avg_savings_per_project

        kpi_cols = st.columns(4)
        kpi_cols[0].metric("Total Projects", total_projects, help="Total number of DMAIC projects in the pipeline.")
        kpi_cols[1].metric("Completed Projects", projects_completed, help="Projects that have reached the Control phase.")
        kpi_cols[2].metric("Avg. Project Cycle Time (Days)", f"{avg_cycle_time:.0f}", help="Average number of days active projects have been in-flight.")
        kpi_cols[3].metric("Annualized Savings", f"${total_annual_savings:,.0f}", help=f"Estimated savings from completed projects (based on avg. of ${avg_savings_per_project:,.0f}/project).")

        st.divider()

        # --- 3. Visualizations: Distribution and Timelines ---
        st.subheader("Project Distribution & Schedule")
        viz_cols = st.columns(2)
        
        with viz_cols[0]:
            # --- DMAIC Phase Distribution Bar Chart ---
            st.markdown("**Project Count by DMAIC Phase**")
            phase_order = ["Define", "Measure", "Analyze", "Improve", "Control"]
            phase_counts = df['phase'].value_counts().reindex(phase_order, fill_value=0)
            
            fig_bar = go.Figure(go.Bar(
                x=phase_counts.values,
                y=phase_counts.index,
                orientation='h',
                text=phase_counts.values,
                textposition='auto',
                marker=dict(color=px.colors.sequential.Tealgrn, coloraxis=None) # Corrected syntax
            ))
            fig_bar.update_layout(
                yaxis=dict(categoryorder='array', categoryarray=phase_order[::-1]), # Reverse order for top-to-bottom
                xaxis_title="Number of Projects",
                yaxis_title="DMAIC Phase",
                height=450,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with viz_cols[1]:
            # --- Project Timeline Gantt Chart ---
            st.markdown("**Project Timelines (Gantt Chart)**")
            df_sorted = df.sort_values(by='start_date', ascending=False)
            fig_gantt = px.timeline(
                df_sorted,
                x_start="start_date",
                x_end="end_date",
                y="id",
                color="site",
                title=None,
                labels={"id": "Project ID", "site": "Site"},
                hover_name="title",
                custom_data=['phase']
            )
            fig_gantt.update_traces(hovertemplate='<b>%{hovertext}</b><br>Site: %{color}<br>Phase: %{customdata[0]}<extra></extra>')
            fig_gantt.update_yaxes(categoryorder="total ascending")
            fig_gantt.update_layout(
                height=450,
                margin=dict(l=20, r=20, t=40, b=20),
                legend_title_text='Site'
            )
            st.plotly_chart(fig_gantt, use_container_width=True)

        st.divider()

        # --- 4. Detailed Project Portfolio Table ---
        st.subheader("Detailed Project Portfolio")
        st.dataframe(
            df[['id', 'title', 'site', 'phase', 'progress', 'team']],
            hide_index=True,
            use_container_width=True,
            column_config={
                "id": st.column_config.TextColumn("Project ID", help="Unique project identifier."),
                "title": st.column_config.TextColumn("Project Title", width="large"),
                "site": st.column_config.TextColumn("Site", help="Manufacturing site where the project is based."),
                "phase": st.column_config.TextColumn("Current Phase"),
                "progress": st.column_config.ProgressColumn(
                    "Progress",
                    help="Visual indicator of project completion based on DMAIC phase.",
                    format="%.0f%%",
                    min_value=0,
                    max_value=1,
                ),
                "team": st.column_config.ListColumn("Team Members (Lead first)")
            }
        )
        
    except Exception as e:
        st.error(f"An error occurred while rendering the Project Pipeline dashboard: {e}")
        logger.error(f"Failed to render project pipeline: {e}", exc_info=True)
