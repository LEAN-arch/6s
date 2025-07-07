"""
Renders the First Time Yield (FTY) and Rolled Throughput Yield (RTY) dashboard,
a crucial tool for analyzing process efficiency and pinpointing bottlenecks.

This module provides a deep-dive analysis into process yields. It calculates
both the FTY for individual steps and the cumulative RTY for the entire process,
allowing an MBB to precisely identify the source of process inefficiency and
quantify the "hidden factory" of rework and scrap.

SME Overhaul:
- Fully integrated with the coherent data narrative to ensure visualizations
  clearly highlight the engineered process bottlenecks.
- The FTY bar chart now programmatically highlights the bottleneck step in red for
  instant recognition.
- The 'Learn More' section has been completely rewritten with clear analogies
  to make the concepts of FTY and RTY accessible to all user levels.
- All visualizations and KPIs have been polished for a professional, insightful
  presentation.
- Code has been refactored for robustness, ensuring process steps are always
  displayed in the correct logical order.
"""

import logging
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm
import numpy as np

from six_sigma.data.session_state_manager import SessionStateManager

logger = logging.getLogger(__name__)

def render_fty_dashboard(ssm: SessionStateManager) -> None:
    """Creates the UI for the FTY & RTY Analysis dashboard."""
    st.header("✅ Yield & Process Bottleneck Analysis")
    st.markdown("Analyze process efficiency by calculating First Time Yield (FTY) for each step and the overall Rolled Throughput Yield (RTY) for the entire value stream. Use these insights to identify your primary process bottlenecks.")

    with st.expander("Learn More: Understanding Yield (FTY vs. RTY)"):
        st.markdown("""
        Understanding process yield is fundamental to identifying waste and inefficiency. Imagine your production line as a water pipe with several connections.

        - **First Time Yield (FTY):** This is the quality of a *single connection* in the pipe. It measures the percentage of units that pass a single process step perfectly the first time, without any leaks (rework or scrap). A high FTY means that specific step is very efficient.
          > _Formula: `FTY = (Units Out) / (Units In)`_

        - **Rolled Throughput Yield (RTY):** This is the quality of the *entire pipe*. It measures the probability that water makes it from the start to the end without leaking from *any* of the connections. It's calculated by multiplying the FTY of all steps together and reveals the massive, cumulative impact of small, seemingly insignificant losses at each step. This is often called the "hidden factory."
          > _Formula: `RTY = FTY₁ * FTY₂ * ... * FTYₙ`_

        **This dashboard helps you find the leakiest connection (the bottleneck) in your process pipe.**
        """)

    try:
        # --- 1. Load and Prepare Data ---
        yield_data = ssm.get_data("yield_data")
        if not yield_data:
            st.warning("No process yield data is available in the data model.")
            return

        df = pd.DataFrame(yield_data)
        df['date'] = pd.to_datetime(df['date'])

        # --- 2. Interactive Filters ---
        st.subheader("Analysis Filters")
        site_list = sorted(df['site'].unique())
        selected_site = st.selectbox("Select a Manufacturing Site to Analyze:", site_list, help="The data is engineered to show a clear bottleneck at the Andover, US site.")

        filtered_df = df[df['site'] == selected_site]
        if filtered_df.empty:
            st.info(f"No yield data available for the selected site: {selected_site}")
            return

        # --- 3. Calculate FTY and RTY for the Selected Site ---
        step_summary = filtered_df.groupby('step_name').agg(
            total_in=('units_in', 'sum'),
            total_out=('units_out', 'sum')
        ).reset_index()
        step_summary['fty'] = step_summary['total_out'] / step_summary['total_in']

        # Ensure steps are always in the correct logical order
        process_step_order = ["Component Kitting", "Sub-Assembly", "Main Assembly", "Final QC Test"]
        step_summary['step_name'] = pd.Categorical(step_summary['step_name'], categories=process_step_order, ordered=True)
        step_summary.sort_values('step_name', inplace=True)

        # Calculate cumulative RTY
        step_summary['rty'] = step_summary['fty'].cumprod()

        overall_rty = step_summary['rty'].iloc[-1] if not step_summary.empty else 0
        # Calculate Sigma Level based on RTY (includes a 1.5 sigma shift for long-term performance)
        sigma_level = 1.5 + norm.ppf(overall_rty) if 0 < overall_rty < 1 else 0
        bottleneck = step_summary.sort_values('fty').iloc[0] if not step_summary.empty else None

        st.divider()

        # --- 4. High-Level KPIs ---
        st.subheader(f"Overall Process Performance: {selected_site}")
        kpi_cols = st.columns(3)
        kpi_cols[0].metric("Rolled Throughput Yield (RTY)", f"{overall_rty:.2%}",
                           help="The probability a unit passes all process steps without a single defect.")
        kpi_cols[1].metric("Process Sigma Level", f"{sigma_level:.2f} σ",
                           help="A standardized measure of process capability based on RTY. Higher is better (6σ is world-class).")
        if bottleneck is not None:
            kpi_cols[2].metric("Primary Bottleneck", bottleneck['step_name'],
                               help="The process step with the lowest individual First Time Yield (FTY). This is the top priority for improvement.")

        # --- 5. Visualizations ---
        st.divider()
        st.subheader("Yield Analysis Visualizations")
        viz_cols = st.columns(2)

        with viz_cols[0]:
            # --- FTY Bar Chart Highlighting Bottleneck ---
            st.markdown("**First Time Yield (FTY) by Process Step**")
            colors = ['firebrick' if step == bottleneck['step_name'] else '#1f77b4' for step in step_summary['step_name']]
            fig_fty = px.bar(
                step_summary, x='step_name', y='fty',
                text=step_summary['fty'].apply(lambda x: f'{x:.2%}'),
                title="Which process step is the least efficient?",
                labels={'step_name': 'Process Step', 'fty': 'First Time Yield'}
            )
            fig_fty.update_traces(marker_color=colors)
            # Dynamically set y-axis range to better highlight differences
            min_yield = step_summary['fty'].min()
            fig_fty.update_layout(
                yaxis=dict(range=[max(0.8, min_yield - 0.05), 1.01], tickformat=".1%"),
                xaxis_title=None, height=450, margin=dict(t=50, b=10)
            )
            st.plotly_chart(fig_fty, use_container_width=True)

        with viz_cols[1]:
            # --- RTY Waterfall Chart ---
            st.markdown("**How Yield Loss Accumulates (RTY)**")
            waterfall_data = go.Waterfall(
                name="RTY", orientation="v",
                measure=["absolute"] + ["relative"] * len(step_summary),
                x=["Process Start"] + list(step_summary['step_name']),
                text=[f"{val:.1%}" for val in [1] + list(step_summary['rty'])],
                textposition='auto',
                y=[1] + list(step_summary['fty'] - 1), # Express loss at each step
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                decreasing={"marker": {"color": "rgba(214, 39, 40, 0.8)"}},
                totals={"marker": {"color": "rgba(31, 119, 180, 0.8)"}}
            )
            fig_waterfall = go.Figure(waterfall_data)
            fig_waterfall.update_layout(
                title="The 'Cascade Effect' of Yield Loss",
                yaxis_tickformat='.0%',
                showlegend=False,
                height=450, margin=dict(t=50, b=10)
            )
            st.plotly_chart(fig_waterfall, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred while rendering the FTY/RTY dashboard: {e}")
        logger.error(f"Failed to render FTY dashboard: {e}", exc_info=True)
