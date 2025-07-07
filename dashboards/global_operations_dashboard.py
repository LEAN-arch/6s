# six_sigma/dashboards/global_operations_dashboard.py
"""
Renders the Global Operations Dashboard.

This module provides a high-level, executive summary of quality performance
across all international manufacturing sites. It focuses on visualizing key
performance indicators (KPIs) like Cost of Poor Quality (COPQ) and First Time
Right (FTR), enabling trend analysis and site-to-site comparison.
"""

import logging
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure the app's root directory is in the path
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
        df_kpi['copq'] = pd.to_numeric(df_kpi['copq'])
        df_kpi['ftr_rate'] = pd.to_numeric(df_kpi['ftr_rate'])
        df_kpi['scrap_rate'] = pd.to_numeric(df_kpi['scrap_rate'])

        # --- 2. Interactive Filters ---
        st.subheader("Dashboard Filters")
        filt_col1, filt_col2 = st.columns(2)
        
        all_sites = sorted(df_kpi['site'].unique())
        selected_sites = filt_col1.multiselect(
            "Filter by Site:",
            options=all_sites,
            default=all_sites
        )

        # Dynamically populate product options based on selected sites
        if selected_sites:
            available_products = sorted(df_kpi[df_kpi['site'].isin(selected_sites)]['product_line'].unique())
        else:
            available_products = sorted(df_kpi['product_line'].unique())
        
        selected_products = filt_col2.multiselect(
            "Filter by Product Line:",
            options=available_products,
            default=available_products
        )

        # Apply filters
        filtered_df = df_kpi[
            df_kpi['site'].isin(selected_sites) &
            df_kpi['product_line'].isin(selected_products)
        ]

        st.divider()

        # --- 3. Display Key Performance Indicators (KPIs) ---
        st.subheader("Key Metrics (Last 90 Days)")
        if not filtered_df.empty:
            recent_df = filtered_df[filtered_df['date'] >= (filtered_df['date'].max() - pd.Timedelta(days=90))]
            
            total_copq = recent_df['copq'].sum()
            avg_ftr = recent_df['ftr_rate'].mean()
            avg_scrap = recent_df['scrap_rate'].mean()
            active_projects = len(projects_list)

            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
            kpi_col1.metric(
                label="Total Cost of Poor Quality (COPQ)",
                value=f"${total_copq / 1_000_000:.2f}M",
                help="Sum of all costs associated with scrap, rework, and warranty claims in the selected period."
            )
            kpi_col2.metric(
                label="Avg. First Time Right (FTR)",
                value=f"{avg_ftr:.2%}",
                help="Average percentage of units that pass all tests on the first attempt without rework."
            )
            kpi_col3.metric(
                label="Avg. Scrap Rate",
                value=f"{avg_scrap:.2%}",
                help="Average percentage of units that are scrapped during production."
            )
            kpi_col4.metric(
                label="Active DMAIC Projects",
                value=active_projects,
                help="Number of formal, ongoing improvement projects across all sites."
            )
        else:
            st.info("No data available for the selected filters.")

        # --- 4. Visualizations ---
        if not filtered_df.empty:
            # --- Time Series Plot ---
            st.subheader("Quality KPI Trends Over Time")
            kpi_to_plot = st.selectbox(
                "Select a KPI to visualize:",
                options=['Cost of Poor Quality (COPQ)', 'First Time Right (FTR)', 'Scrap Rate']
            )
            
            column_map = {
                'Cost of Poor Quality (COPQ)': 'copq',
                'First Time Right (FTR)': 'ftr_rate',
                'Scrap Rate': 'scrap_rate'
            }
            y_col = column_map[kpi_to_plot]

            trend_df = filtered_df.groupby('date')[y_col].mean().reset_index()
            fig_trend = px.area(
                trend_df, x='date', y=y_col,
                title=f"Daily Average: {kpi_to_plot}",
                labels={'date': 'Date', y_col: kpi_to_plot}
            )
            fig_trend.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_trend, use_container_width=True)

            st.divider()

            # --- Breakdown Charts & Project Summary ---
            viz_col1, viz_col2 = st.columns(2)
            with viz_col1:
                st.subheader("COPQ Breakdown by Site & Product")
                copq_breakdown_df = filtered_df.groupby(['site', 'product_line'])['copq'].sum().reset_index()
                fig_treemap = px.treemap(
                    copq_breakdown_df,
                    path=[px.Constant("All Sites"), 'site', 'product_line'],
                    values='copq',
                    color='copq',
                    color_continuous_scale='Reds',
                    title="Contribution to Total Cost of Poor Quality"
                )
                fig_treemap.update_layout(margin = dict(t=50, l=25, r=25, b=25))
                st.plotly_chart(fig_treemap, use_container_width=True)
            
            with viz_col2:
                st.subheader("Active Improvement Initiatives")
                projects_df = pd.DataFrame(projects_list)
                st.dataframe(
                    projects_df[['id', 'site', 'title', 'phase']],
                    hide_index=True,
                    use_container_width=True
                )

    except Exception as e:
        st.error("An error occurred while rendering the Global Operations Dashboard.")
        logger.error(f"Failed to render global dashboard: {e}", exc_info=True)
