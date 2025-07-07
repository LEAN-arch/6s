# six_sigma/dashboards/fty_dashboard.py
"""
Renders the First Time Yield (FTY) and Rolled Throughput Yield (RTY) dashboard.

This module provides a deep-dive analysis into process yields. It calculates
both the FTY for individual steps and the cumulative RTY for the entire process,
allowing an MBB to precisely identify process bottlenecks and quantify the
"hidden factory" of rework and scrap.
"""

import logging
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from six_sigma.data.session_state_manager import SessionStateManager

logger = logging.getLogger(__name__)

def render_fty_dashboard(ssm: SessionStateManager) -> None:
    """
    Creates the UI for the FTY/RTY Analysis dashboard.

    Args:
        ssm (SessionStateManager): The session state manager to access yield data.
    """
    st.header("✅ First Time Yield (FTY) & RTY Analysis")
    st.markdown("Analyze process efficiency by calculating First Time Yield for each step and the overall Rolled Throughput Yield (RTY) for the entire value stream. Use these insights to identify your primary process bottlenecks.")

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
        # Aggregate data over the entire period for selected site
        step_summary = filtered_df.groupby('step_name').agg(
            total_in=('units_in', 'sum'),
            total_out=('units_out', 'sum')
        ).reset_index()
        step_summary['fty'] = step_summary['total_out'] / step_summary['total_in']
        
        # Ensure correct order for RTY calculation
        process_step_order = ["Component Kitting", "Sub-Assembly", "Main Assembly", "Final QC Test"]
        step_summary['step_name'] = pd.Categorical(step_summary['step_name'], categories=process_step_order, ordered=True)
        step_summary.sort_values('step_name', inplace=True)
        
        # Calculate Rolled Throughput Yield
        step_summary['rty'] = step_summary['fty'].cumprod()
        
        overall_rty = step_summary['rty'].iloc[-1] if not step_summary.empty else 0
        sigma_level = 4.5 + stats.norm.ppf(overall_rty) if overall_rty > 0 else 0

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
            # --- FTY Bar Chart ---
            st.markdown("**First Time Yield (FTY) by Process Step**")
            fig_fty = px.bar(step_summary, x='step_name', y='fty',
                             text=step_summary['fty'].apply(lambda x: f'{x:.2%}'),
                             title="Which step is the least efficient?")
            fig_fty.update_layout(yaxis=dict(range=[0.8, 1.0], tickformat=".1%"),
                                  xaxis_title="Process Step", yaxis_title="First Time Yield",
                                  height=450, margin=dict(t=50, b=10))
            fig_fty.update_traces(marker_color='#1f77b4')
            st.plotly_chart(fig_fty, use_container_width=True)

        with viz_cols[1]:
            # --- RTY Waterfall Chart ---
            st.markdown("**Rolled Throughput Yield (RTY) Waterfall**")
            initial_yield = 1.0
            waterfall_data = []
            for i, row in step_summary.iterrows():
                yield_loss = initial_yield * (1 - row['fty'])
                waterfall_data.append(go.Waterfall(
                    name=row['step_name'], orientation="v",
                    measure=["relative"] * len(step_summary),
                    x=[step_summary['step_name']], y=[step_summary.apply(lambda r: r['fty']-1, axis=1)],
                    text=[f"{(row['fty']-1):.2%}" for i, row in step_summary.iterrows()],
                    connector={"line":{"color":"rgb(63, 63, 63)"}}
                ))

            fig_waterfall = go.Figure(go.Waterfall(
                name="Yield", orientation="v",
                x=step_summary['step_name'],
                text=[f"{(row['fty']):.2%}" for i, row in step_summary.iterrows()],
                y=step_summary['fty']-1, # Represent loss
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                decreasing={"marker":{"color":"#d62728"}},
            ))
            fig_waterfall.add_trace(go.Waterfall(
                orientation="v",
                measure=["total"] * (len(step_summary) + 1),
                x=["Start"] + list(step_summary['step_name']),
                y=[1] + list(step_summary['rty'] - step_summary['rty'].shift(1).fillna(0)-1),
                text=[f"{y:.2%}" for y in [1] + list(step_summary['rty'])]
            ))


            waterfall_steps = [
                go.layout.updatemenu.Button(
                    args=["visible", [True, False, False, False]],
                    label="Kitting", method="restyle"
                ),
                go.layout.updatemenu.Button(
                    args=["visible", [False, True, False, False]],
                    label="Sub-Assembly", method="restyle"
                ),
            ]
            
            fig_waterfall = go.Figure()
            initial_val = 1.0
            fig_waterfall.add_trace(go.Waterfall(
                x=["Start"], y=[initial_val], measure=["absolute"],
                text=[f"{initial_val:.1%}"]
            ))
            
            last_val = initial_val
            for _, row in step_summary.iterrows():
                fig_waterfall.add_trace(go.Waterfall(
                    x=[row['step_name']], y=[row['fty']*last_val - last_val],
                    measure=['relative'], text=[f"{(row['fty']*last_val - last_val):.2%}"]
                ))
                last_val *= row['fty']
            
            fig_waterfall.add_trace(go.Waterfall(
                x=["Final RTY"], y=[last_val], measure=['total'],
                text=[f"{last_val:.2%}"]
            ))
            
            fig_waterfall.update_layout(
                title="How Yield Loss Accumulates (Waterfall)",
                waterfallgap=0.3,
                height=450, margin=dict(t=50, b=10)
            )

            st.plotly_chart(fig_waterfall, use_container_width=True)

    except Exception as e:
        st.error("An error occurred while rendering the FTY/RTY dashboard.")
        logger.error(f"Failed to render FTY dashboard: {e}", exc_info=True)
