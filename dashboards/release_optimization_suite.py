# six_sigma/dashboards/release_optimization_suite.py
"""
Renders the Product Release Optimization Suite.

This dashboard provides tools and analysis to move from traditional inspection
to more efficient, data-driven lot acceptance strategies. It allows engineers
to evaluate the effectiveness of release tests, compare sampling plans, and
justify reductions in destructive testing, directly impacting cycle time and
cost.
"""

import logging
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc

logger = logging.getLogger(__name__)

from six_sigma.data.session_state_manager import SessionStateManager

def render_release_optimization_suite(ssm: SessionStateManager) -> None:
    """
    Creates the UI for the Product Release Optimization Suite tab.

    Args:
        ssm (SessionStateManager): The session state manager to access release data.
    """
    st.header("✅ Product Release Optimization Suite")
    st.markdown("Analyze and improve the efficiency and effectiveness of your product release processes. Use these tools to justify data-driven changes to sampling plans and testing strategies.")

    try:
        # --- 1. Load Data ---
        release_data = ssm.get_data("release_data")
        if not release_data:
            st.warning("No product release data is available.")
            return

        df_release = pd.DataFrame(release_data)
        df_release['true_status_numeric'] = df_release['true_status'].apply(lambda x: 1 if x == 'Fail' else 0)

        # --- 2. Define Tabs for Different Optimization Tools ---
        st.info("Select a tool below to analyze your current release testing or simulate alternative strategies.", icon="🔬")
        tool_tabs = st.tabs(["**Test Effectiveness Analysis (ROC)**", "**Sampling Plan Simulator**"])

        # --- TEST EFFECTIVENESS ANALYSIS ---
        with tool_tabs[0]:
            st.subheader("Analyze Release Test Effectiveness")
            st.markdown("This tool assesses how well your current release test measurement predicts the true quality of a batch. A high **Area Under the Curve (AUC)** indicates a powerful, discriminating test that can reliably separate good batches from bad ones.")

            # Calculate ROC Curve
            fpr, tpr, thresholds = roc_curve(df_release['true_status_numeric'], df_release['test_measurement'])
            roc_auc = auc(fpr, tpr)
            
            # Create ROC Plot
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC = {roc_auc:.3f}', line=dict(color='darkblue', width=3)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Chance', line=dict(color='grey', width=2, dash='dash')))
            fig_roc.update_layout(
                title=f"<b>Release Test Performance (AUC = {roc_auc:.3f})</b>",
                xaxis_title='False Positive Rate (Good Batches Failed)',
                yaxis_title='True Positive Rate (Bad Batches Caught)',
                legend=dict(x=0.4, y=0.2, bgcolor='rgba(255,255,255,0.6)'),
                height=500
            )

            kpi_col, plot_col = st.columns([1, 2.5])
            with kpi_col:
                st.metric("Test Discriminatory Power (AUC)", f"{roc_auc:.3f}")
                st.markdown("""
                - **> 0.9:** Excellent Test
                - **0.8 - 0.9:** Good Test
                - **0.7 - 0.8:** Fair Test
                - **< 0.7:** Poor Test
                """)
                if roc_auc < 0.8:
                    st.warning("The current test has limited power to distinguish good vs. bad lots. Consider developing a more effective test method.")
                else:
                    st.success("The current test is effective. You can confidently set data-driven release limits.")

            with plot_col:
                st.plotly_chart(fig_roc, use_container_width=True)

        # --- SAMPLING PLAN SIMULATOR ---
        with tool_tabs[1]:
            st.subheader("Sampling Plan Simulator")
            st.markdown("Compare the effectiveness and cost of different sampling strategies against your historical batch data. This helps justify moving away from 100% inspection or arbitrary sample sizes.")

            sim_cols = st.columns(3)
            lot_size = sim_cols[0].number_input("Average Lot Size", 100, 10000, 1000)
            true_defect_rate = sim_cols[1].slider("Assumed True Defect Rate (%)", 0.1, 10.0, 2.0, 0.1) / 100
            cost_per_sample = sim_cols[2].number_input("Cost per Sample (Destructive Test)", 1, 500, 50)
            
            st.divider()

            # --- Define Sampling Plans ---
            plans = {
                "100% Inspection": {"n": lot_size, "c": 0},
                "ANSI Level II (Normal)": {"n": 80, "c": 3}, # Example values for a lot of 1000
                "Reduced Sampling": {"n": 20, "c": 1}
            }

            results = []
            for name, params in plans.items():
                n, c = params['n'], params['c']
                # Probability of acceptance using binomial distribution
                p_accept = sum([np.math.comb(n, k) * (true_defect_rate**k) * ((1-true_defect_rate)**(n-k)) for k in range(c + 1)])
                
                # Simulate outcomes over 1000 hypothetical lots
                bad_lots_passed = p_accept * 100
                
                total_cost = n * cost_per_sample
                results.append({
                    "Sampling Plan": name,
                    "Sample Size (n)": n,
                    "Acceptance # (c)": c,
                    "Prob. of Acceptance": p_accept,
                    "Consumer's Risk (%)": bad_lots_passed, # % of bad lots that will be accepted
                    "Cost per Lot ($)": total_cost
                })
            
            results_df = pd.DataFrame(results)
            
            st.markdown("##### Simulation Results")
            st.caption(f"Based on a true defect rate of **{true_defect_rate:.1%}**.")
            st.dataframe(
                results_df.style.format({
                    "Prob. of Acceptance": "{:.2%}",
                    "Consumer's Risk (%)": "{:.1f}",
                    "Cost per Lot ($)": "${:,.0f}"
                }).background_gradient(cmap='Reds', subset=["Consumer's Risk (%)"])
                  .background_gradient(cmap='Greens_r', subset=["Cost per Lot ($)"]),
                hide_index=True,
                use_container_width=True
            )
            st.info("""
            **How to Interpret This Table:**
            - **Probability of Acceptance:** The chance a lot with the assumed defect rate will pass inspection.
            - **Consumer's Risk:** The percentage of **bad lots** that would be incorrectly accepted by this plan. Lower is better.
            - **Cost per Lot:** The direct cost of performing the (destructive) test for each batch.
            
            **Goal:** Find a plan that balances an acceptably low Consumer's Risk with a low cost.
            """)

    except Exception as e:
        st.error("An error occurred while rendering the Release Optimization Suite.")
        logger.error(f"Failed to render release optimization suite: {e}", exc_info=True)
