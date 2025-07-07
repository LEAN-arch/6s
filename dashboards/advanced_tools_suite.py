"""
Renders the Advanced Statistical Tools Suite, a standalone workbench for
on-demand, sophisticated statistical analysis.

This module provides direct access to common Six Sigma analytical tools outside
the context of a specific DMAIC project. It's designed as a go-to resource for
an MBB to quickly perform Measurement System Analysis (Gage R&R), analyze
experimental data (DOE), or conduct formal hypothesis tests.

SME Overhaul:
- Architecturally refactored to enforce a clean 'calculate-then-plot' pattern,
  separating statistical computation from visualization.
- Completely rewrote the 'Learn More' sections for each tool to serve as
  expert-level, yet accessible, mini-tutorials on the purpose, execution,
  and interpretation of the analysis.
- The UI for each tool has been polished to provide clear, immediate, and
  color-coded conclusions (e.g., Gage R&R verdict, hypothesis test result).
- Improved robustness with checks for data availability.
"""

import logging
import pandas as pd
import streamlit as st
import plotly.express as px

from six_sigma.data.session_state_manager import SessionStateManager
from six_sigma.utils.plotting import create_doe_plots, create_gage_rr_plots
from six_sigma.utils.stats import calculate_gage_rr, perform_hypothesis_test, perform_anova_on_dataframe

logger = logging.getLogger(__name__)

def render_advanced_tools_suite(ssm: SessionStateManager) -> None:
    """Creates the UI for the Advanced Statistical Tools Suite."""
    st.header("🔬 Advanced Statistical Suite")
    st.markdown("A workbench for on-demand, sophisticated statistical analysis. Use these standalone tools to conduct Measurement System Analysis, analyze experimental data, and perform hypothesis tests.")

    tool_tabs = st.tabs(["**Gage R&R**", "**Design of Experiments (DOE)**", "**Hypothesis Testing**"])

    # ==================== GAGE R&R ====================
    with tool_tabs[0]:
        st.subheader("Measurement System Analysis (Gage R&R)")
        
        with st.expander("Learn More: Understanding Gage R&R"):
            st.markdown("""
            **Purpose:** To determine if your measurement system is reliable and capable. Before you analyze or try to improve a process, you **must** trust your data. This analysis quantifies how much of your process variation is due to the measurement system itself versus the actual variation between parts.
            
            **Key Metric: % Contribution**
            This is the percentage of total variation that is consumed by the measurement system (the sum of Repeatability and Reproducibility).
            - **< 10%:** The measurement system is **Excellent / Acceptable**. You can trust your data.
            - **10% to 30%:** The system is **Marginal**. It may be acceptable for some applications, but improvement is recommended.
            - **> 30%:** The system is **Unacceptable**. The measurement system is creating too much noise, masking the true process variation. You *must* fix the measurement system before analyzing the process.

            **Components of Variation:**
            - **Repeatability (Equipment Variation - EV):** Variation seen when the *same operator* measures the *same part* multiple times. It reflects the inherent variation of the gage itself.
            - **Reproducibility (Appraiser Variation - AV):** Variation seen when *different operators* measure the *same part*. It reflects variation caused by the operators.
            """)
        
        try:
            gage_data = ssm.get_data("gage_rr_data")
            if gage_data is None or gage_data.empty:
                st.warning("No Gage R&R data is available in the data model.")
            else:
                # 1. Calculate
                results_df, _ = calculate_gage_rr(gage_data)
                
                # 2. Plot
                fig1, fig2 = create_gage_rr_plots(gage_data)
                
                # 3. Render
                if not results_df.empty:
                    total_grr_contrib = results_df.loc['Total Gage R&R', '% Contribution']
                    
                    st.info("This analysis uses the ANOVA method to partition variance, as recommended by the Automotive Industry Action Group (AIAG).")
                    grr_cols = st.columns([1, 1.5])
                    with grr_cols[0]:
                        st.markdown("**Gage R&R Results (% Contribution)**")
                        st.dataframe(results_df.style.format({'% Contribution': '{:.2f}%'}).background_gradient(cmap='Reds', subset=['% Contribution']))
                        
                        if total_grr_contrib < 10:
                            st.success(f"**Conclusion:** Measurement System is **Acceptable** ({total_grr_contrib:.2f}%).")
                        elif total_grr_contrib < 30:
                            st.warning(f"**Conclusion:** Measurement System is **Marginal** ({total_grr_contrib:.2f}%).")
                        else:
                            st.error(f"**Conclusion:** Measurement System is **Unacceptable** ({total_grr_contrib:.2f}%).")
                    
                    with grr_cols[1]:
                        st.plotly_chart(fig1, use_container_width=True)
                        st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.error("Gage R&R calculation failed. Check logs for details.")

        except Exception as e:
            st.error(f"An error occurred during Gage R&R analysis: {e}")
            logger.error(f"Failed Gage R&R analysis: {e}", exc_info=True)

    # ==================== DOE ====================
    with tool_tabs[1]:
        st.subheader("Design of Experiments (DOE) Analyzer")
        with st.expander("Learn More: Understanding DOE"):
            st.markdown("""
            **Purpose:** To efficiently and systematically determine the relationship between factors (inputs) affecting a process and the output (response). It is the most powerful tool for process optimization, allowing you to find the "recipe" for the best results.
            
            **Key Plots & Interpretation:**
            - **Main Effects Plot:** Shows the average impact of each factor on the response. The *steeper the line*, the more significant the factor's effect.
            - **Interaction Plot:** Visualizes how the effect of one factor changes depending on the level of another. **Non-parallel (crossed) lines indicate a significant interaction,** which is often a key discovery. For example, "baking time only matters at high temperatures."
            - **3D Response Surface:** A 3D map showing the predicted response across a range of factor settings. It helps visualize the optimal process window (the "peak of the mountain").
            """)
        
        try:
            doe_data = ssm.get_data("doe_data")
            if doe_data is None or doe_data.empty:
                st.warning("No DOE data is available in the data model.")
            else:
                factors = ['temp', 'time', 'pressure']
                response = 'strength'
                
                st.info(f"Analyzing a **3-Factor, 2-Level Full Factorial Design with center points** for the response variable: **'{response}'**.")
                st.dataframe(doe_data.head(), use_container_width=True)

                doe_plots = create_doe_plots(doe_data, factors, response)
                
                st.plotly_chart(doe_plots['main_effects'], use_container_width=True)
                st.plotly_chart(doe_plots['interaction'], use_container_width=True)
                st.plotly_chart(doe_plots['surface'], use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred during DOE analysis: {e}")
            logger.error(f"Failed DOE analysis: {e}", exc_info=True)

    # ==================== HYPOTHESIS TESTING ====================
    with tool_tabs[2]:
        st.subheader("Hypothesis Testing Suite")
        with st.expander("Learn More: Understanding Hypothesis Testing"):
            st.markdown("""
            **Purpose:** To use sample data to make a formal statistical decision about a claim regarding a population. It provides a structured way to answer questions like "Is the new process better than the old one?" or "Are these two suppliers different?"
            
            **Key Concepts:**
            - **Null Hypothesis (H₀):** The default assumption, typically stating there is *no effect* or *no difference* (e.g., the means are equal).
            - **Alternative Hypothesis (Hₐ):** The claim you want to prove (e.g., the means are different).
            - **P-value:** The probability of observing your data (or something more extreme) if the null hypothesis were true.
            
            **Decision Rule:** If the **p-value is less than your significance level (α, typically 0.05)**, you have found a statistically significant result. This means you have strong evidence against the null hypothesis, so you **reject the null hypothesis.**
            """)
        
        try:
            ht_data = ssm.get_data("hypothesis_testing_data")
            if ht_data is None or ht_data.empty:
                st.warning("No Hypothesis Testing data is available in the data model.")
            else:
                test_type = st.selectbox("Select a Hypothesis Test:", ["2-Sample t-Test", "One-Way ANOVA"])
                
                if test_type == "2-Sample t-Test":
                    st.markdown("##### Question: Is there a significant difference between the 'Before' and 'After' process change?")
                    result = perform_hypothesis_test(ht_data['before_change'], ht_data['after_change'])
                    
                    df_plot = pd.melt(ht_data[['before_change', 'after_change']], var_name='Group', value_name='Value')
                    fig = px.box(df_plot, x='Group', y='Value', points='all', title="Comparison: Before vs. After", color='Group')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if result.get('reject_null'):
                        st.success(f"**Conclusion: The difference is statistically significant.** (p-value = {result['p_value']:.4f}). We reject the null hypothesis and conclude that the means are different.")
                    else:
                        st.warning(f"**Conclusion: The difference is not statistically significant.** (p-value = {result['p_value']:.4f}). We fail to reject the null hypothesis.")
                
                elif test_type == "One-Way ANOVA":
                    st.markdown("##### Question: Is there a significant difference in component strength among Suppliers A, B, and C?")
                    df_anova = pd.melt(ht_data[['supplier_a', 'supplier_b', 'supplier_c']], var_name='group', value_name='value')
                    result = perform_anova_on_dataframe(df_anova, 'value', 'group')
                    
                    fig = px.box(df_anova, x='group', y='value', points='all', title="Component Strength by Supplier", color='group')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if result.get('reject_null'):
                        st.success(f"**Conclusion: There is a statistically significant difference** between at least two of the suppliers. (p-value = {result['p_value']:.4f}). Further post-hoc tests would be needed to identify which specific pairs are different.")
                    else:
                        st.warning(f"**Conclusion: There is no statistically significant difference** among the suppliers. (p-value = {result['p_value']:.4f}).")

        except Exception as e:
            st.error(f"An error occurred during Hypothesis Testing: {e}")
            logger.error(f"Failed Hypothesis Testing: {e}", exc_info=True)
