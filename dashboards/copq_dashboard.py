"""
Renders the Cost of Poor Quality (COPQ) Analysis Center, a dashboard dedicated
to translating process failures into financial terms.

This module provides a dedicated workbench for deep-dive analysis into the
drivers of COPQ. It allows an MBB to dissect costs by failure type, category,
and site, using tools like Pareto analysis to identify the most impactful areas
for cost-saving initiatives. It serves as the financial justification for the
problems identified in the FTY dashboard.

SME Overhaul:
- Tightly integrated with the data narrative to ensure the Pareto analysis
  correctly identifies the financial impact of the engineered process bottleneck.
- The Pareto chart has been upgraded to a publication-quality standard, including
  an "80/20" line to visually emphasize the vital few.
- The 'Learn More' section provides an expert-level explanation of COPQ strategy.
- The 'Actionable Insights' section is now a dynamic, data-driven call to action.
- All visualizations have been polished for clarity and professional impact.
"""

import logging
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from six_sigma.data.session_state_manager import SessionStateManager

logger = logging.getLogger(__name__)

def render_copq_dashboard(ssm: SessionStateManager) -> None:
    """Creates the UI for the COPQ Analysis Center."""
    st.header("💰 Cost of Poor Quality (COPQ) Analysis")
    st.markdown("Translate quality problems into financial terms. Use this dashboard to identify and prioritize the most significant opportunities for cost reduction through focused improvement projects.")

    with st.expander("Learn More: The Strategic Importance of COPQ"):
        st.markdown("""
        **Cost of Poor Quality (COPQ)** represents all the money a company loses because its processes are not perfect. It's a powerful way to get management attention and justify resources for improvement projects.

        COPQ is typically broken into two main categories:

        - **Internal Failure Costs:** Costs from defects caught *before* reaching the customer.
          - _Examples: Scrapping a bad part, time spent on rework, re-testing a product._
          - _Impact:_ These are bad. They waste time, material, and capacity.

        - **External Failure Costs:** Costs from defects that *reach the customer*.
          - _Examples: Warranty claims, product recalls, field service trips, complaint handling, brand reputation damage._
          - _Impact:_ These are **disastrous**. They are exponentially more expensive than internal failures and can lead to lost customers and market share.

        **A key quality strategy is to convert external failures into internal ones (by improving inspection) and then eliminate the internal failures (by improving the process).**
        """)

    try:
        # --- 1. Load and Prepare Data ---
        copq_data = ssm.get_data("copq_data")
        if not copq_data:
            st.warning("No Cost of Poor Quality data is available in the data model.")
            return

        df = pd.DataFrame(copq_data)
        df['date'] = pd.to_datetime(df['date'])
        
        # --- 2. High-Level COPQ Summary ---
        st.subheader("COPQ Overview (Last 365 Days)")
        total_copq = df['cost'].sum()
        internal_copq = df[df['failure_type'] == 'Internal']['cost'].sum()
        external_copq = df[df['failure_type'] == 'External']['cost'].sum()

        kpi_cols = st.columns(3)
        kpi_cols[0].metric("Total COPQ", f"${total_copq/1_000_000:.2f}M")
        kpi_cols[1].metric(
            "Internal Failure Costs", f"${internal_copq/1_000:.1f}K",
            help="Costs from defects caught before reaching the customer (e.g., scrap, rework)."
        )
        kpi_cols[2].metric(
            "External Failure Costs", f"${external_copq/1_000:.1f}K",
            help="Costs from defects that reached the customer (e.g., warranty, complaints).",
            delta_color="inverse"
        )
        st.divider()

        # --- 3. Pareto Analysis: Identifying the 'Vital Few' ---
        st.subheader("Pareto Analysis: The 'Vital Few' Drivers of Cost")
        st.markdown("The Pareto Principle (or 80/20 rule) states that for many events, roughly 80% of the effects come from 20% of the causes. This chart identifies the few critical failure categories that account for the majority of the cost. **Focus improvement efforts here for the greatest financial impact.**")

        pareto_df = df.groupby('category')['cost'].sum().sort_values(ascending=False).reset_index()
        pareto_df['Cumulative Percentage'] = (pareto_df['cost'].cumsum() / pareto_df['cost'].sum()) * 100

        fig_pareto = go.Figure()
        # Bar chart for costs
        fig_pareto.add_trace(go.Bar(
            x=pareto_df['category'], y=pareto_df['cost'], name='Cost',
            text=pareto_df['cost'].apply(lambda x: f'${x/1000:.0f}k'),
            marker_color='#1f77b4'
        ))
        # Line chart for cumulative percentage
        fig_pareto.add_trace(go.Scatter(
            x=pareto_df['category'], y=pareto_df['Cumulative Percentage'],
            name='Cumulative %', mode='lines+markers', yaxis='y2',
            line=dict(color='firebrick', width=3)
        ))
        # 80% line
        fig_pareto.add_hline(y=80, line=dict(color="grey", dash="dash"), yref="y2", annotation_text="80% Mark")

        fig_pareto.update_layout(
            title_text="<b>COPQ by Failure Category (Pareto Chart)</b>",
            xaxis_title="Failure Category", yaxis_title="Total Cost ($)",
            yaxis=dict(tickprefix="$"),
            yaxis2=dict(title="Cumulative Percentage", overlaying="y", side="right", range=[0, 105], showgrid=False, ticksuffix="%"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500, uniformtext_minsize=8, uniformtext_mode='hide'
        )
        st.plotly_chart(fig_pareto, use_container_width=True)

        # --- 4. Detailed Drill-Down Visualizations ---
        st.divider()
        st.subheader("Detailed COPQ Drill-Down")
        viz_cols = st.columns(2)

        with viz_cols[0]:
            st.markdown("**Hierarchical Cost Breakdown**")
            fig_treemap = px.treemap(
                df, path=[px.Constant("Total COPQ"), 'failure_type', 'site', 'category'],
                values='cost', color='cost', color_continuous_scale='YlOrRd',
                title="Drill-Down: Failure Type → Site → Category"
            )
            fig_treemap.update_layout(margin=dict(t=50, l=10, r=10, b=10), height=450)
            st.plotly_chart(fig_treemap, use_container_width=True)
            
        with viz_cols[1]:
            st.markdown("**COPQ Composition Over Time**")
            trend_df = df.set_index('date').groupby([pd.Grouper(freq='M'), 'failure_type'])['cost'].sum().reset_index()
            fig_area = px.area(
                trend_df, x='date', y='cost', color='failure_type',
                title="Monthly Internal vs. External Failure Costs",
                labels={'date': 'Month', 'cost': 'Total Monthly Cost ($)'},
                color_discrete_map={"Internal": "#ff7f0e", "External": "#d62728"}
            )
            fig_area.update_layout(height=450, margin=dict(t=50, l=10, r=10, b=10), yaxis_tickprefix="$")
            st.plotly_chart(fig_area, use_container_width=True)
            
        # --- 5. Data-Driven Actionable Insights ---
        st.divider()
        st.subheader("Actionable Insights & Next Steps")
        
        if not pareto_df.empty:
            top_failure = pareto_df.iloc[0]
            st.warning(
                f"**Highest Impact Opportunity:** The failure category **'{top_failure['category']}'** is the single largest driver of cost, accounting for **${top_failure['cost']:,.0f}**. "
                f"This represents **{top_failure['Cumulative Percentage']:.1f}%** of the total recorded COPQ."
                f"\n\n**➡️ Next Step:** This is a clear, high-impact target. Launch a new **DMAIC project** to investigate the root causes of this failure mode and drive significant, measurable cost savings.",
                icon="🎯"
            )
        
    except Exception as e:
        st.error(f"An error occurred while rendering the COPQ Analysis Center: {e}")
        logger.error(f"Failed to render COPQ dashboard: {e}", exc_info=True)
