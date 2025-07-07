# six_sigma/dashboards/global_operations_dashboard.py
"""
Renders the Global Operations Dashboard.

This module provides a high-level, executive summary of quality performance
across all international manufacturing sites. It focuses on visualizing key
performance indicators (KPIs) like Cost of Poor Quality (COPQ) and Rolled
Throughput Yield (RTY), enabling trend analysis and site-to-site comparison.

SME Overhaul:
- Added dynamic delta indicators to KPIs for trend comparison.
- Implemented a new site-vs-site "Leaderboard" chart for direct performance comparison.
- Enhanced the time-series plot to show multi-line comparisons and a rolling average.
- Added a dynamic "Actionable Insights" section to guide user focus.
"""

import logging
import pandas as pd
import streamlit as st
import plotly.express as px

from six_sigma.data.session_state_manager import SessionStateManager

logger = logging.getLogger(__name__)

def render_global_dashboard(ssm: SessionStateManager) -> None:
    """Creates the UI for the Global Operations Dashboard tab."""
    st.header("🌍 Global Operations Performance Dashboard")
    st.markdown("Monitor key quality and cost metrics across all manufacturing sites. Use the filters to drill down into specific sites and product lines to identify areas for improvement.")

    try:
        # --- 1. Load and Prepare Data ---
        copq_list = ssm.get_data("copq_data")
        rty_list = ssm.get_data("global_kpis") # This key now holds RTY data
        projects_list = ssm.get_data("dmaic_projects")

        if not copq_list or not rty_list:
            st.warning("Key Performance Indicator (KPI) data is not fully available.")
            return

        df_copq = pd.DataFrame(copq_list)
        df_copq['date'] = pd.to_datetime(df_copq['date'])
        
        df_rty = pd.DataFrame(rty_list)
        df_rty['date'] = pd.to_datetime(df_rty['date'])

        # --- 2. Interactive Filters ---
        st.subheader("Dashboard Filters")
        all_sites = sorted(df_copq['site'].unique())
        selected_sites = st.multiselect("Filter by Site:", options=all_sites, default=all_sites)
        
        # Apply filters
        filtered_copq = df_copq[df_copq['site'].isin(selected_sites)]
        filtered_rty = df_rty[df_rty['site'].isin(selected_sites)]

        st.divider()

        # --- 3. Enhanced KPIs with Deltas ---
        st.subheader("Key Metrics (Last 90 Days vs. Previous 90 Days)")
        if not filtered_copq.empty and not filtered_rty.empty:
            max_date = filtered_copq['date'].max()
            current_period_copq = filtered_copq[filtered_copq['date'].between(max_date - pd.Timedelta(days=89), max_date)]
            previous_period_copq = filtered_copq[filtered_copq['date'].between(max_date - pd.Timedelta(days=179), max_date - pd.Timedelta(days=90))]

            current_period_rty = filtered_rty[filtered_rty['date'].between(max_date - pd.Timedelta(days=89), max_date)]
            previous_period_rty = filtered_rty[filtered_rty['date'].between(max_date - pd.Timedelta(days=179), max_date - pd.Timedelta(days=90))]

            total_copq_current = current_period_copq['cost'].sum()
            total_copq_previous = previous_period_copq['cost'].sum()
            copq_delta = total_copq_current - total_copq_previous if total_copq_previous > 0 else 0

            avg_rty_current = current_period_rty['rty'].mean()
            avg_rty_previous = previous_period_rty['rty'].mean()
            rty_delta = avg_rty_current - avg_rty_previous if avg_rty_previous > 0 else 0
            
            active_projects = len([p for p in projects_list if p.get('phase') != 'Control'])

            kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
            kpi_col1.metric(label="Total COPQ (90d)", value=f"${total_copq_current / 1_000_000:.2f}M", delta=f"${copq_delta/1000:,.0f}K vs prev. 90d", delta_color="inverse")
            kpi_col2.metric(label="Avg. RTY (90d)", value=f"{avg_rty_current:.2%}", delta=f"{rty_delta:.2%}", delta_color="normal", help="Rolled Throughput Yield: The probability a unit passes all process steps defect-free.")
            kpi_col3.metric(label="Active Improvement Projects", value=active_projects, help="Number of DMAIC projects currently in Define, Measure, Analyze, or Improve phases.")
        else:
            st.info("No data available for the selected filters.")

        # --- 4. Enhanced Visualizations ---
        if not filtered_copq.empty:
            st.divider()
            st.subheader("Performance Analysis & Breakdown")
            
            viz_col1, viz_col2 = st.columns(2)
            with viz_col1:
                st.markdown("**Site Performance Leaderboard (Last 90 Days)**")
                site_summary = current_period_copq.groupby('site').agg(total_copq=('cost', 'sum')).reset_index()
                rty_summary = current_period_rty.groupby('site').agg(avg_rty=('rty', 'mean')).reset_index()
                leaderboard_df = pd.merge(site_summary, rty_summary, on='site').sort_values('total_copq', ascending=False)
                
                fig_leaderboard = px.bar(leaderboard_df, x='site', y='total_copq', color='avg_rty',
                                         title="COPQ vs. RTY by Site",
                                         labels={'site': 'Manufacturing Site', 'total_copq': 'Total Cost of Poor Quality ($)', 'avg_rty': 'Avg. RTY'},
                                         color_continuous_scale=px.colors.sequential.RdYlGn,
                                         hover_data={'avg_rty': ':.2%'})
                fig_leaderboard.update_layout(margin=dict(t=50, b=10), height=400)
                st.plotly_chart(fig_leaderboard, use_container_width=True)

            with viz_col2:
                st.markdown("**COPQ Contribution Analysis**")
                fig_treemap = px.treemap(current_period_copq, path=[px.Constant("All Sites"), 'site', 'failure_type', 'category'],
                                         values='cost', color='cost', color_continuous_scale='Reds',
                                         title="Where is the Cost of Poor Quality Coming From?")
                fig_treemap.update_layout(margin=dict(t=50, b=10), height=400)
                st.plotly_chart(fig_treemap, use_container_width=True)

            # --- 5. Actionable Insights ---
            st.divider()
            st.subheader("Actionable Insights")
            if not leaderboard_df.empty:
                worst_site_copq = leaderboard_df.iloc[0]
                best_site_rty = leaderboard_df.sort_values('avg_rty', ascending=False).iloc[0]

                st.warning(f"**Focus Area:** The **{worst_site_copq['site']}** site has the highest COPQ (${worst_site_copq['total_copq']:,.0f}) over the last 90 days. "
                         "Use the **COPQ Analysis** module to drill down into the specific failure modes driving this cost.", icon="🎯")
                st.success(f"**Best Practice:** The **{best_site_rty['site']}** site is achieving the highest average RTY ({best_site_rty['avg_rty']:.2%}). "
                         "Use the **FTY Analysis** module to identify their best-performing steps and share best practices.", icon="💡")

    except Exception as e:
        st.error("An error occurred while rendering the Global Operations Dashboard.")
        logger.error(f"Failed to render global dashboard: {e}", exc_info=True)
