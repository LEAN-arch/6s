# six_sigma/dashboards/fty_dashboard.py
"""
Renders the First Time Yield (FTY) and Rolled Throughput Yield (RTY) dashboard.

This module provides a deep-dive analysis into process yields. It calculates
both the FTY for individual steps and the cumulative RTY for the entire process,
allowing an MBB to precisely identify process bottlenecks and quantify the
"hidden factory" of rework and scrap.

SME Overhaul:
- Calculates both FTY and RTY.
- Adds Sigma Level calculation based on RTY.
- Implements a sophisticated Waterfall chart to show cumulative yield loss.
- Includes clear explanations of the methodologies for all user levels.
"""

import logging
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm

from six_sigma.data.session_state_manager import SessionStateManager

logger = logging.getLogger(__name__)

def render_fty_dashboard(ssm: SessionStateManager) -> None:
    """Creates the UI for the FTY/RTY Analysis dashboard."""
    st.header("✅ First Time Yield (FTY) & RTY Analysis")
    st.markdown("Analyze process efficiency by calculating First Time Yield for each step and the overall Rolled Throughput Yield (RTY) for the entire value stream. Use these insights to identify your primary process bottlenecks.")

    with st.expander("Learn More: Understanding FTY, RTY, and Sigma Level"):
        st.markdown("""
        - **First Time Yield (FTY):** The percentage of units that pass a *single process step* correctly the first time, without needing any rework or scrap. It measures the efficiency of an individual step.
          > _Formula: `FTY = Units Out / Units In`_
        - **Rolled Throughput Yield (RTY):** The probability that a unit will pass through the *entire process* (all steps) without a single defect or rework event. It exposes the "hidden factory" of cumulative losses.
          > _Formula: `RTY = FTY₁ * FTY₂ * ... * FTYₙ`_
        - **Sigma Level (σ):** A standardized measure of process capability, converted from the overall RTY. A higher Sigma Level indicates a more capable, less defective process. A 6σ process has a near-perfect RTY.
          > _Formula: `Sigma Level = 1.5 + Z-score(RTY)` (The 1.5 shift accounts for long-term process drift)_
        """)

    try:
        # --- 1. Load and Prepare Data ---
        yield_data = ssm.get_data("yield_data")
        if not yield_data:
            st.warning("No process yield data is available.")
            return

        df = pd.DataFrame(yield_data)
        df['date'] = pd.to_datetime(df['date'])

        # --- 2. Interactive Filters ---
        st.subheader("Analysis Filters")
        site_list = sorted(df['site'].unique())
        selected_site = st.selectbox("Select a Manufacturing Site to Analyze:", site_list)

        filtered_df = df[df['site'] == selected_site]
        
        # --- 3. Calculate FTY and RTY ---
        step_summary = filtered_df.groupby('step_name').agg(
            total_in=('units_in', 'sum'),
            total_out=('units_out', 'sum')
        ).reset_index()
        step_summary['fty'] = step_summary['total_out'] / step_summary['total_in']
        
        process_step_order = ["Component Kitting", "Sub-Assembly", "Main Assembly", "Final QC Test"]
        step_summary['step_name'] = pd.Categorical(step_summary['step_name'], categories=process_step_order, ordered=True)
        step_summary.sort_values('step_name', inplace=True)
        
        step_summary['rty'] = step_summary['fty'].cumprod()
        
        overall_rty = step_summary['rty'].iloc[-1] if not step_summary.empty else 0
        sigma_level = 1.5 + norm.ppf(overall_rty) if 0 < overall_rty < 1 else 0

        st.divider()

        # --- 4. High-Level KPIs ---
        st.subheader(f"Overall Process Performance for {selected_site}")
        kpi_cols = st.columns(3)
        kpi_cols[0].metric("Rolled Throughput Yield (RTY)", f"{overall_rty:.2%}",
                           help="The probability that a unit passes through all process steps without a single defect.")
        kpi_cols[1].metric("Process Sigma Level", f"{sigma_level:.2f} σ",
                           help="A measure of process capability based on the RTY. Higher is better (6σ is world-class).")
        kpi_cols[2].metric("Primary Bottleneck", step_summary.sort_values('fty').iloc[0]['step_name'],
                           help="The process step with the lowest individual First Time Yield.")

        # --- 5. Visualizations ---
        st.divider()
        st.subheader("Yield Analysis Visualizations")
        
        viz_cols = st.columns(2)
        with viz_cols[0]:
            st.markdown("**First Time Yield (FTY) by Process Step**")
            fig_fty = px.bar(step_summary, x='step_name', y='fty',
                             text=step_summary['fty'].apply(lambda x: f'{x:.2%}'),
                             title="Which step is the least efficient?")
            fig_fty.update_layout(yaxis=dict(range=[max(0.8, step_summary['fty'].min() - 0.02), 1.0], tickformat=".1%"),
                                  xaxis_title=None, yaxis_title="First Time Yield",
                                  height=450, margin=dict(t=50, b=10))
            fig_fty.update_traces(marker_color='#1f77b4')
            st.plotly_chart(fig_fty, use_container_width=True)

        with viz_cols[1]:
            st.markdown("**Rolled Throughput Yield (RTY) Waterfall**")
            
            y_values = [1] + list(step_summary['rty'].values)
            text_values = [f"{val:.1%}" for val in y_values]
            measure_values = ["absolute"] + ["relative"] * len(step_summary)
            x_values = ["Process Start"] + list(step_summary['step_name'].values)

            fig_waterfall = go.Figure(go.Waterfall(
                name="RTY",
                orientation="v",
                measure=measure_values,
                x=x_values,
                text=text_values,
                y=np.diff(np.insert(y_values, 0, 0)),
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                decreasing={"marker": {"color": "#d62728"}},
                increasing={"marker": {"color": "#2ca02c"}},
                totals={"marker": {"color": "#1f77b4"}}
            ))
            
            fig_waterfall.update_layout(
                title="How Yield Loss Accumulates",
                yaxis_tickformat='.0%',
                height=450, margin=dict(t=50, b=10)
            )
            st.plotly_chart(fig_waterfall, use_container_width=True)

    except Exception as e:
        st.error("An error occurred while rendering the FTY/RTY dashboard.")
        logger.error(f"Failed to render FTY dashboard: {e}", exc_info=True)
