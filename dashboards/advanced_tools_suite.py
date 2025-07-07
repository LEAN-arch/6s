# six_sigma/dashboards/advanced_tools_suite.py
"""
Renders the Advanced Statistical Tools Suite.

This module provides a standalone workbench for performing sophisticated
statistical analyses common in Six Sigma projects, such as Gage R&R,
Design of Experiments (DOE), and Hypothesis Testing. It's designed as a
go-to resource for an MBB outside the context of a specific DMAIC project.
"""

import logging
import pandas as pd
import streamlit as st
from scipy import stats

from six_sigma.data.session_state_manager import SessionStateManager
from six_sigma.utils.plotting import create_gage_rr_plots, create_doe_plots
from six_sigma.utils.stats import calculate_gage_rr, perform_t_test, perform_anova

logger = logging.getLogger(__name__)

def render_advanced_tools_suite(ssm: SessionStateManager) -> None:
    """Creates the UI for the Advanced Tools Suite."""
    st.header("🔮 Advanced Statistical Tools Suite")
    st.markdown("A workbench for on-demand, sophisticated statistical analysis. Use these tools to conduct Measurement System Analysis, analyze experimental data, and perform hypothesis tests.")

    tool_tabs = st.tabs(["**Measurement System Analysis (Gage R&R)**", "**Design of Experiments (DOE)**", "**Hypothesis Testing**"])

    # --- GAGE R&R ---
    with tool_tabs[0]:
        st.subheader("Measurement System Analysis (Gage R&R)")
        with st.expander("Learn More: Understanding Gage R&R"):
            st.markdown("""
            **Purpose:** To determine if your measurement system is reliable and capable. It quantifies how much of your process variation is due to the measurement system itself versus the actual variation between parts.
            
            **Key Metrics:**
            - **% Contribution:** The percentage of total variation attributable to the Gage R&R. A value **<10% is considered acceptable**; >30% is unacceptable.
            - **Repeatability:** Variation from the measurement instrument (gauge) itself.
            - **Reproducibility:** Variation from the appraisers (operators) using the instrument.

            **Interpretation:** You cannot trust your process data without a reliable measurement system. If the Gage R&R % is high, you must improve your measurement system *before* trying to improve your process.
            """)
        
        try:
            gage_data = ssm.get_data("gage_rr_data")
            if gage_data is None or gage_data.empty:
                st.warning("No Gage R&R data available.")
                return

            results_df, main_plot, components_plot = calculate_gage_rr(gage_data)
            total_grr = results_df.loc['Total Gage R&R', '% Contribution']
            
            st.info("This analysis uses the ANOVA method to partition variance, as recommended by the AIAG.")
            grr_cols = st.columns([1, 2])
            with grr_cols[0]:
                st.markdown("**Gage R&R Results**")
                st.dataframe(results_df.style.format({'% Contribution': '{:.2f}%'}).background_gradient(cmap='Reds', subset=['% Contribution']))
                if total_grr < 10: st.success(f"**Conclusion:** Measurement System is **Acceptable** ({total_grr:.2f}%).")
                elif total_grr < 30: st.warning(f"**Conclusion:** Measurement System is **Marginal** ({total_grr:.2f}%).")
                else: st.error(f"**Conclusion:** Measurement System is **Unacceptable** ({total_grr:.2f}%).")
            with grr_cols[1]:
                st.plotly_chart(main_plot, use_container_width=True)
                st.plotly_chart(components_plot, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred during Gage R&R analysis: {e}")
            logger.error(f"Failed Gage R&R analysis: {e}", exc_info=True)


    # --- DOE ---
    with tool_tabs[1]:
        st.subheader("Design of Experiments (DOE) Analyzer")
        with st.expander("Learn More: Understanding DOE"):
            st.markdown("""
            **Purpose:** To efficiently and systematically determine the relationship between factors affecting a process and the output of that process. It is the gold standard for process optimization.
            
            **Key Plots:**
            - **Main Effects Plot:** Shows the average impact of each factor on the response. The steeper the line, the more significant the factor.
            - **Interaction Plot:** Visualizes how the effect of one factor changes depending on the level of another. **Non-parallel (crossed) lines indicate a strong interaction.**
            - **3D Response Surface:** A 3D map showing the predicted response across a range of factor settings, helping to visualize the optimal process window.
            """)
        
        try:
            doe_data = ssm.get_data("doe_data")
            if doe_data is None or doe_data.empty:
                st.warning("No DOE data available.")
                return

            factors = ['temp', 'time', 'pressure']
            response = 'strength'
            
            st.info(f"Analyzing a **3-Factor, 2-Level Full Factorial Design** for the response variable: **'{response}'**.")
            st.dataframe(doe_data, use_container_width=True)

            main_effects_fig, interaction_fig, surface_fig = create_doe_plots(doe_data, factors, response)
            
            st.plotly_chart(main_effects_fig, use_container_width=True)
            st.plotly_chart(interaction_fig, use_container_width=True)
            st.plotly_chart(surface_fig, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred during DOE analysis: {e}")
            logger.error(f"Failed DOE analysis: {e}", exc_info=True)


    # --- HYPOTHESIS TESTING ---
    with tool_tabs[2]:
        st.subheader("Hypothesis Testing Suite")
        with st.expander("Learn More: Understanding Hypothesis Testing"):
            st.markdown("""
            **Purpose:** To use sample data to make a statistical decision about a claim or assumption about a population.
            
            **Key Concepts:**
            - **Null Hypothesis (H₀):** The default assumption, typically stating there is *no effect* or *no difference* (e.g., the means are equal).
            - **Alternative Hypothesis (Hₐ):** The claim we want to test (e.g., the means are different).
            - **P-value:** The probability of observing our data (or more extreme data) if the null hypothesis were true.
            
            **Interpretation:** A small p-value (typically **p < 0.05**) provides strong evidence against the null hypothesis, so you reject the null hypothesis. A large p-value means your data is consistent with the null hypothesis, so you fail to reject it.
            """)
        
        try:
            ht_data = ssm.get_data("hypothesis_testing_data")
            if ht_data is None or ht_data.empty:
                st.warning("No Hypothesis Testing data available.")
                return
            
            test_type = st.selectbox("Select a Hypothesis Test:", ["2-Sample t-Test", "One-Way ANOVA"])
            
            if test_type == "2-Sample t-Test":
                st.markdown("##### Is there a significant difference between the 'Before' and 'After' process change?")
                fig, result = perform_t_test(ht_data['before_change'], ht_data['after_change'], "Before Change", "After Change")
                st.plotly_chart(fig, use_container_width=True)
                if result['p_value'] < 0.05:
                    st.success(f"**Conclusion:** The difference is **statistically significant** (p = {result['p_value']:.4f}). We reject the null hypothesis that the means are equal.")
                else:
                    st.warning(f"**Conclusion:** The difference is **not statistically significant** (p = {result['p_value']:.4f}). We fail to reject the null hypothesis.")
            
            elif test_type == "One-Way ANOVA":
                st.markdown("##### Is there a significant difference in component strength between Suppliers A, B, and C?")
                df_anova = pd.melt(ht_data[['supplier_a', 'supplier_b', 'supplier_c']], var_name='group', value_name='value')
                fig, result = perform_anova(df_anova, 'value', 'group', "Component Strength by Supplier")
                st.plotly_chart(fig, use_container_width=True)
                if result['p_value'] < 0.05:
                    st.success(f"**Conclusion:** There is a **statistically significant difference** between at least two of the suppliers (p = {result['p_value']:.4f}).")
                else:
                    st.warning(f"**Conclusion:** There is **no statistically significant difference** between the suppliers (p = {result['p_value']:.4f}).")

        except Exception as e:
            st.error(f"An error occurred during Hypothesis Testing: {e}")
            logger.error(f"Failed Hypothesis Testing: {e}", exc_info=True)
