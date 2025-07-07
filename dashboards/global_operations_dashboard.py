"""
Renders the Global Operations Dashboard, the strategic command center for
monitoring high-level quality performance across the entire organization.

This module provides an executive summary of key performance indicators (KPIs)
like Cost of Poor Quality (COPQ) and Rolled Throughput Yield (RTY). It is designed
for trend analysis and site-to-site comparison, enabling leadership to quickly
identify strategic opportunities and problem areas.

SME Overhaul:
- Replaced the site leaderboard with a more sophisticated 'Performance Quadrant'
  scatter plot (COPQ vs. RTY), a classic management visualization for comparing
  business units.
- Tightly integrated with the narrative-driven data to ensure the 'Actionable
  Insights' section consistently and correctly guides the user to the most
  significant problem areas (e.g., the engineered bottleneck at Andover).
- Enhanced KPI cards with clearer delta descriptions for executive readability.
- Refined all titles, labels, and help text for a polished, commercial-grade feel.
"""

import logging
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from six_sigma.data.session_state_manager import SessionStateManager

logger = logging.getLogger(__name__)

def render_global_dashboard(ssm: SessionStateManager) -> None:
    """Creates the UI for the Global Operations Performance Dashboard."""
    st.header("🌍 Global Operations Performance")
    st.markdown("A strategic, executive-level view of quality and cost metrics across all manufacturing sites. Use this dashboard to monitor program health and identify areas requiring leadership attention.")

    try:
        # --- 1. Load and Prepare Data ---
        copq_list = ssm.get_data("copq_data")
        rty_list = ssm.get_data("global_kpis")
        projects_list = ssm.get_data("dmaic_projects")

        if not copq_list or not rty_list or not projects_list:
            st.warning("One or more key datasets (COPQ, RTY, Projects) could not be loaded. The dashboard may be incomplete.")
            return

        df_copq = pd.DataFrame(copq_list)
        df_copq['date'] = pd.to_datetime(df_copq['date'])

        df_rty = pd.DataFrame(rty_list)
        df_rty['date'] = pd.to_datetime(df_rty['date'])

        # --- 2. High-Level KPIs with Period-over-Period Comparison ---
        st.subheader("Key Performance Indicators (Last 90 Days)")
        max_date = df_copq['date'].max()
        current_period_start = max_date - pd.Timedelta(days=89)
        prev_period_start = max_date - pd.Timedelta(days=179)
        prev_period_end = max_date - pd.Timedelta(days=90)

        current_copq = df_copq[df_copq['date'] >= current_period_start]
        previous_copq = df_copq[df_copq['date'].between(prev_period_start, prev_period_end)]
        current_rty = df_rty[df_rty['date'] >= current_period_start]
        previous_rty = df_rty[df_rty['date'].between(prev_period_start, prev_period_end)]

        total_copq_current = current_copq['cost'].sum()
        total_copq_previous = previous_copq['cost'].sum()
        copq_delta = total_copq_current - total_copq_previous if total_copq_previous > 0 else 0

        avg_rty_current = current_rty['rty'].mean()
        avg_rty_previous = previous_rty['rty'].mean()
        rty_delta = avg_rty_current - avg_rty_previous if avg_rty_previous > 0 else 0

        active_projects = len([p for p in projects_list if p.get('phase') != 'Control'])

        kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
        kpi_col1.metric(
            label="Total COPQ (90d)",
            value=f"${total_copq_current / 1_000_000:.2f}M",
            delta=f"${copq_delta/1000:,.0f}K vs. prior period",
            delta_color="inverse",
            help="Total cost from internal & external failures. Lower is better."
        )
        kpi_col2.metric(
            label="Avg. RTY (90d)",
            value=f"{avg_rty_current:.2%}",
            delta=f"{rty_delta:.2%}",
            delta_color="normal",
            help="Rolled Throughput Yield: Probability a unit passes all steps defect-free. Higher is better."
        )
        kpi_col3.metric(
            label="Active Improvement Projects",
            value=active_projects,
            help="Number of DMAIC projects in Define, Measure, Analyze, or Improve."
        )
        st.divider()

        # --- 3. Performance Analysis Visualizations ---
        st.subheader("Site Performance Breakdown (Last 90 Days)")
        if current_copq.empty or current_rty.empty:
            st.info("No data available for the last 90 days to display performance charts.")
            return

        viz_col1, viz_col2 = st.columns([1.5, 1])

        with viz_col1:
            # --- Performance Quadrant Chart ---
            st.markdown("**Performance Quadrant: COPQ vs. RTY**")
            site_copq = current_copq.groupby('site')['cost'].sum().reset_index(name='total_copq')
            site_rty = current_rty.groupby('site')['rty'].mean().reset_index(name='avg_rty')
            perf_df = pd.merge(site_copq, site_rty, on='site')

            avg_copq_line = perf_df['total_copq'].mean()
            avg_rty_line = perf_df['avg_rty'].mean()

            fig_quadrant = px.scatter(
                perf_df, x='avg_rty', y='total_copq',
                text='site', size='total_copq', color='site',
                labels={'avg_rty': 'Average Rolled Throughput Yield (RTY)', 'total_copq': 'Total Cost of Poor Quality (COPQ)'},
                title="Site Performance: Cost vs. Yield"
            )
            fig_quadrant.update_traces(textposition='top center', marker=dict(sizemin=10))
            fig_quadrant.add_vline(x=avg_rty_line, line=dict(color='grey', dash='dash'), annotation_text='Avg. RTY')
            fig_quadrant.add_hline(y=avg_copq_line, line=dict(color='grey', dash='dash'), annotation_text='Avg. COPQ')
            fig_quadrant.update_layout(showlegend=False, margin=dict(t=50, b=10), yaxis_tickprefix='$', xaxis_tickformat='.1%')
            st.plotly_chart(fig_quadrant, use_container_width=True)

        with viz_col2:
            # --- COPQ Contribution Treemap ---
            st.markdown("**COPQ Contribution by Type**")
            fig_treemap = px.treemap(
                current_copq, path=[px.Constant("Total COPQ"), 'failure_type', 'site'],
                values='cost', color='failure_type',
                color_discrete_map={'Internal': '#ff7f0e', 'External': '#d62728'},
                title="Internal vs. External Failure Costs by Site"
            )
            fig_treemap.update_layout(margin=dict(t=50, b=10, l=10, r=10))
            st.plotly_chart(fig_treemap, use_container_width=True)

        # --- 4. Actionable Insights ---
        st.divider()
        st.subheader("Actionable Insights & Strategic Guidance")
        if not perf_df.empty:
            worst_site_copq = perf_df.sort_values('total_copq', ascending=False).iloc[0]
            best_site_rty = perf_df.sort_values('avg_rty', ascending=False).iloc[0]
            poorest_perf_site = perf_df[(perf_df['total_copq'] > avg_copq_line) & (perf_df['avg_rty'] < avg_rty_line)]

            if not poorest_perf_site.empty:
                focus_site = poorest_perf_site.iloc[0]
                st.warning(
                    f"**Primary Focus Area:** The **{focus_site['site']}** site falls in the 'Troubled' quadrant with high COPQ (${focus_site['total_copq']:,.0f}) "
                    f"and low RTY ({focus_site['avg_rty']:.2%}). This indicates systemic process issues."
                    f"\n\n**➡️ Next Step:** Use the **COPQ** and **FTY Analysis** workspaces to perform a deep-dive investigation on this site.", icon="🎯"
                )
            else:
                 st.info("No sites are currently in the high-cost, low-yield 'Troubled' quadrant. Review sites with the highest overall COPQ for targeted improvements.")

            st.success(
                f"**Best Practice Leader:** The **{best_site_rty['site']}** site is achieving the highest average RTY ({best_site_rty['avg_rty']:.2%})."
                f"\n\n**➡️ Next Step:** Engage with this site's team to understand their successful processes. Use the **Kaizen & Training Hub** to document and share these best practices across the organization.", icon="💡"
            )

    except Exception as e:
        st.error(f"An error occurred while rendering the Global Operations Dashboard: {e}")
        logger.error(f"Failed to render global dashboard: {e}", exc_info=True)
