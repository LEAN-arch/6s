# six_sigma/dashboards/global_operations_dashboard.py
"""
Renders the Global Operations Dashboard.

This module provides a high-level, executive summary of quality performance
across all international manufacturing sites. It focuses on visualizing key
performance indicators (KPIs) like Cost of Poor Quality (COPQ) and First Time
Right (FTR), enabling trend analysis and site-to-site comparison.

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
    """
    Creates the UI for the Global Operations Dashboard tab.

    Args:
        ssm (SessionStateManager): The session state manager to access manufacturing data.
    """
    st.header("🌍 Global Operations Performance Dashboard")
    st.markdown("Monitor key quality and cost metrics across all manufacturing sites. Use the filters to drill down into specific sites and product lines.")

    try:
        # --- 1. Load and Prepare Data ---
        kpi_list = ssm.get_data("global_kpis")
        projects_list = ssm.get_data("dmaic_projects")

        if not kpi_list:
            st.warning("No Key Performance Indicator (KPI) data is available.")
            return

        df_kpi = pd.DataFrame(kpi_list)
        df_kpi['date'] = pd.to_datetime(df_kpi['date'])

        # --- 2. Interactive Filters ---
        st.subheader("Dashboard Filters")
        filt_col1, filt_col2 = st.columns(2)
        all_sites = sorted(df_kpi['site'].unique())
        selected_sites = filt_col1.multiselect("Filter by Site:", options=all_sites, default=all_sites)

        available_products = sorted(df_kpi[df_kpi['site'].isin(selected_sites)]['product_line'].unique()) if selected_sites else sorted(df_kpi['product_line'].unique())
        selected_products = filt_col2.multiselect("Filter by Product Line:", options=available_products, default=available_products)

        filtered_df = df_kpi[df_kpi['site'].isin(selected_sites) & df_kpi['product_line'].isin(selected_products)]

        st.divider()

        # --- 3. Enhanced KPIs with Deltas ---
        st.subheader("Key Metrics (Last 90 Days vs. Previous 90 Days)")
        if not filtered_df.empty:
            max_date = filtered_df['date'].max()
            current_period = filtered_df[filtered_df['date'].between(max_date - pd.Timedelta(days=89), max_date)]
            previous_period = filtered_df[filtered_df['date'].between(max_date - pd.Timedelta(days=179), max_date - pd.Timedelta(days=90))]

            total_copq_current = current_period['copq'].sum()
            total_copq_previous = previous_period['copq'].sum()
            copq_delta = total_copq_current - total_copq_previous if total_copq_previous > 0 else 0

            avg_ftr_current = current_period['ftr_rate'].mean()
            avg_ftr_previous = previous_period['ftr_rate'].mean()
            ftr_delta = avg_ftr_current - avg_ftr_previous if avg_ftr_previous > 0 else 0

            active_projects = len([p for p in projects_list if p.get('phase') != 'Control'])

            kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
            kpi_col1.metric(label="Total COPQ (90d)", value=f"${total_copq_current / 1_000_000:.2f}M", delta=f"${copq_delta/1000:,.0f}K vs prev. 90d", delta_color="inverse")
            kpi_col2.metric(label="Avg. FTR (90d)", value=f"{avg_ftr_current:.2%}", delta=f"{ftr_delta:.2%}", delta_color="normal")
            kpi_col3.metric(label="Active Improvement Projects", value=active_projects, help="Number of DMAIC projects currently in Define, Measure, Analyze, or Improve phases.")
        else:
            st.info("No data available for the selected filters.")

        # --- 4. Enhanced Visualizations ---
        if not filtered_df.empty:
            st.divider()
            st.subheader("Performance Analysis & Breakdown")
            
            # --- New: Site-vs-Site Leaderboard ---
            viz_col1, viz_col2 = st.columns([1, 1])
            with viz_col1:
                st.markdown("**Site Performance Leaderboard**")
                site_summary = current_period.groupby('site').agg(
                    total_copq=('copq', 'sum'),
                    avg_ftr=('ftr_rate', 'mean')
                ).reset_index().sort_values('total_copq', ascending=False)
                
                fig_leaderboard = px.bar(site_summary, x='site', y='total_copq', color='avg_ftr',
                                         title="COPQ vs. FTR by Site (Last 90 Days)",
                                         labels={'site': 'Manufacturing Site', 'total_copq': 'Total Cost of Poor Quality ($)', 'avg_ftr': 'Avg. FTR'},
                                         color_continuous_scale=px.colors.sequential.RdYlGn,
                                         hover_data={'avg_ftr': ':.2%'})
                fig_leaderboard.update_layout(margin=dict(t=50, b=10), height=400)
                st.plotly_chart(fig_leaderboard, use_container_width=True)

            with viz_col2:
                # --- COPQ Treemap for a holistic view ---
                st.markdown("**COPQ Contribution Analysis**")
                copq_breakdown_df = current_period.groupby(['site', 'product_line'])['copq'].sum().reset_index()
                fig_treemap = px.treemap(copq_breakdown_df, path=[px.Constant("All Sites"), 'site', 'product_line'],
                                         values='copq', color='copq', color_continuous_scale='Reds',
                                         title="Where is the Cost of Poor Quality Coming From?")
                fig_treemap.update_layout(margin=dict(t=50, b=10), height=400)
                st.plotly_chart(fig_treemap, use_container_width=True)

            # --- Enhanced Time Series Plot ---
            st.subheader("KPI Trends by Site")
            trend_df = filtered_df.groupby(['date', 'site']).agg(
                copq=('copq', 'mean'),
                ftr_rate=('ftr_rate', 'mean')
            ).reset_index()

            kpi_to_plot = st.radio("Select a KPI to trend:", ['COPQ', 'FTR'], horizontal=True)
            y_col = 'copq' if kpi_to_plot == 'COPQ' else 'ftr_rate'
            
            fig_trend = px.line(trend_df, x='date', y=y_col, color='site',
                                title=f"Daily Average {kpi_to_plot} by Site",
                                labels={'date': 'Date', y_col: kpi_to_plot})
            
            show_rolling_avg = st.checkbox("Show 30-day Rolling Average")
            if show_rolling_avg:
                for site in trend_df['site'].unique():
                    site_df = trend_df[trend_df['site'] == site].copy()
                    site_df['rolling_avg'] = site_df[y_col].rolling(window=30, min_periods=1).mean()
                    fig_trend.add_scatter(x=site_df['date'], y=site_df['rolling_avg'], mode='lines',
                                          name=f'{site} (30d Avg)', line=dict(dash='dash'),
                                          legendgroup=site, showlegend=False)

            st.plotly_chart(fig_trend, use_container_width=True)

            # --- 5. Actionable Insights ---
            st.divider()
            st.subheader("Actionable Insights")
            if not site_summary.empty:
                worst_site_copq = site_summary.iloc[0]
                best_site_ftr = site_summary.sort_values('avg_ftr', ascending=False).iloc[0]

                st.warning(f"**Focus Area:** The **{worst_site_copq['site']}** site has the highest COPQ (${worst_site_copq['total_copq']:,.0f}) over the last 90 days. "
                         "Use the **COPQ Analysis Center** to drill down into the specific failure modes driving this cost.", icon="🎯")
                st.success(f"**Best Practice:** The **{best_site_ftr['site']}** site is achieving the highest average FTR ({best_site_ftr['avg_ftr']:.2%}). "
                         "Consider initiating a Kaizen event to share best practices from this site.", icon="💡")

    except Exception as e:
        st.error("An error occurred while rendering the Global Operations Dashboard.")
        logger.error(f"Failed to render global dashboard: {e}", exc_info=True)
