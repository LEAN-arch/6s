# six_sigma/dashboards/copq_dashboard.py
"""
Renders the Cost of Poor Quality (COPQ) Analysis Center.

This module provides a dedicated dashboard for deep-dive analysis into the
drivers of COPQ. It allows an MBB to dissect costs by failure type (internal vs.
external), category, and site, using tools like Pareto analysis to identify the
most impactful areas for cost-saving initiatives.

SME Overhaul:
- Differentiates between Internal and External Failure Costs.
- Upgraded Pareto chart for better readability.
- Added a time-series trend plot to show COPQ composition over time.
- Implemented a more powerful, multi-level treemap for hierarchical drill-down.
"""

import logging
import pandas as pd
import streamlit as st
import plotly.express as px

from six_sigma.data.session_state_manager import SessionStateManager

logger = logging.getLogger(__name__)

def render_copq_dashboard(ssm: SessionStateManager) -> None:
    """
    Creates the UI for the COPQ Analysis Center tab.

    Args:
        ssm (SessionStateManager): The session state manager to access failure data.
    """
    st.header("💰 Cost of Poor Quality (COPQ) Analysis Center")
    st.markdown("Drill down into the specific drivers of internal and external failure costs. Use these tools to identify and prioritize the most significant opportunities for cost reduction.")

    try:
        # --- 1. Load and Prepare Data ---
        copq_data = ssm.get_data("copq_data")
        if not copq_data:
            st.warning("No Cost of Poor Quality data is available.")
            return

        df = pd.DataFrame(copq_data)
        df['date'] = pd.to_datetime(df['date'])
        
        # --- 2. High-Level COPQ Summary ---
        total_copq = df['cost'].sum()
        internal_copq = df[df['failure_type'] == 'Internal']['cost'].sum()
        external_copq = df[df['failure_type'] == 'External']['cost'].sum()

        st.subheader("COPQ Overview (Last 365 Days)")
        kpi_cols = st.columns(3)
        kpi_cols[0].metric("Total COPQ", f"${total_copq/1_000_000:.2f}M")
        kpi_cols[1].metric("Internal Failure Costs", f"${internal_copq/1_000:.1f}K",
                           help="Costs from defects caught before reaching the customer (e.g., scrap, rework).")
        kpi_cols[2].metric("External Failure Costs", f"${external_copq/1_000:.1f}K",
                           help="Costs from defects that reached the customer (e.g., warranty, complaints).")
        st.divider()

        # --- 3. Pareto Analysis of Failure Costs ---
        st.subheader("Pareto Analysis: The 'Vital Few' Failure Modes")
        st.markdown("This chart identifies the few critical failure categories that account for the majority of the cost. **Focus improvement efforts on the categories on the left** for the greatest financial impact.")

        pareto_df = df.groupby('category')['cost'].sum().sort_values(ascending=False).reset_index()
        pareto_df.rename(columns={'cost': 'Total Cost'}, inplace=True)
        pareto_df['Cumulative Percentage'] = (pareto_df['Total Cost'].cumsum() / pareto_df['Total Cost'].sum()) * 100

        fig_pareto = go.Figure()
        fig_pareto.add_trace(go.Bar(
            x=pareto_df['category'], y=pareto_df['Total Cost'],
            name='Cost', text=pareto_df['Total Cost'].apply(lambda x: f'${x/1000:.0f}k'),
            marker_color='#1f77b4'
        ))
        fig_pareto.add_trace(go.Scatter(
            x=pareto_df['category'], y=pareto_df['Cumulative Percentage'],
            name='Cumulative %', mode='lines+markers', yaxis='y2', line=dict(color='#d62728')
        ))

        fig_pareto.update_layout(
            title_text="<b>COPQ by Failure Category</b>",
            xaxis_title="Failure Category", yaxis_title="Total Cost ($)",
            yaxis2=dict(title="Cumulative Percentage", overlaying="y", side="right", range=[0, 105], showgrid=False, ticksuffix="%"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500, uniformtext_minsize=8, uniformtext_mode='hide'
        )
        st.plotly_chart(fig_pareto, use_container_width=True)

        # --- 4. Detailed Breakdown & Time-Series Analysis ---
        st.divider()
        st.subheader("Detailed COPQ Drill-Down")
        
        viz_cols = st.columns(2)
        with viz_cols[0]:
            st.markdown("**Hierarchical Cost Breakdown**")
            fig_treemap = px.treemap(
                df, path=[px.Constant("Total COPQ"), 'failure_type', 'site', 'category'],
                values='cost', color='cost', color_continuous_scale='YlOrRd',
                title="Drill-Down from Failure Type to Site and Category"
            )
            fig_treemap.update_layout(margin=dict(t=50, l=10, r=10, b=10), height=450)
            st.plotly_chart(fig_treemap, use_container_width=True)
            
        with viz_cols[1]:
            st.markdown("**COPQ Composition Over Time**")
            trend_df = df.set_index('date').groupby([pd.Grouper(freq='M'), 'failure_type'])['cost'].sum().reset_index()
            fig_area = px.area(
                trend_df, x='date', y='cost', color='failure_type',
                title="Monthly Internal vs. External Failure Costs",
                labels={'cost': 'Total Monthly Cost ($)'},
                color_discrete_map={"Internal": "#ff7f0e", "External": "#d62728"}
            )
            fig_area.update_layout(height=450, margin=dict(t=50, l=10, r=10, b=10))
            st.plotly_chart(fig_area, use_container_width=True)
            
        # --- 5. Actionable Insights ---
        st.divider()
        st.subheader("Actionable Insights & Next Steps")
        
        if not pareto_df.empty:
            top_failure = pareto_df.iloc[0]
            st.warning(
                f"**Highest Impact Opportunity:** The failure category **'{top_failure['category']}'** accounts for **${top_failure['Total Cost']:,.0f}**, "
                f"which is **{top_failure['Cumulative Percentage']:.1f}%** of the total recorded COPQ."
                f"\n\n**Recommendation:** This is a clear, high-impact target. Launch a new **DMAIC project** in the workspace to investigate the root causes of this failure mode and drive significant cost savings.",
                icon="🎯"
            )
        
    except Exception as e:
        st.error("An error occurred while rendering the COPQ Analysis Center.")
        logger.error(f"Failed to render COPQ dashboard: {e}", exc_info=True)
