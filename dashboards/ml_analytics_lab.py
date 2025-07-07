# six_sigma/dashboards/ml_analytics_lab.py
"""
Renders the Advanced Analytics & Machine Learning Lab.

This module provides a workspace for exploring modern analytical techniques and
comparing them to classical Six Sigma methods. It includes tools for predictive
quality modeling, advanced process optimization, and release test analysis,
empowering an MBB to leverage cutting-edge, data-driven strategies.
"""

import logging
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from statsmodels.formula.api import ols

from six_sigma.data.session_state_manager import SessionStateManager

logger = logging.getLogger(__name__)

def render_ml_analytics_lab(ssm: SessionStateManager) -> None:
    """Creates the UI for the Advanced Analytics & ML Lab."""
    st.header("🔬 Advanced Analytics & ML Lab")
    st.markdown("Explore modern data science techniques to augment classical Six Sigma methodologies. Use this lab to build predictive models, compare optimization techniques, and enhance quality control strategies.")

    tool_tabs = st.tabs(["**1. Predictive Quality Modeling**", "**2. Process Optimization: Classical vs. Modern**", "**3. Release Test Effectiveness**"])

    # --- PREDICTIVE QUALITY MODELING ---
    with tool_tabs[0]:
        st.subheader("Predictive Quality Modeling")
        st.markdown("Develop a model to predict final QC outcomes from in-process sensor data, enabling a shift to 'Quality at the Source'.")
        with st.expander("Learn More: Predictive Quality"):
            st.markdown("Instead of waiting for a final test to find a defect (a lagging indicator), this approach uses machine learning to predict failure based on real-time process data (leading indicators). This allows for early intervention, saving significant costs associated with completing a defective unit.")

        df_pred = ssm.get_data("predictive_quality_data")
        if df_pred.empty:
            st.warning("Predictive quality data is not available.")
            return

        features, target = ['in_process_temp', 'in_process_pressure', 'in_process_vibration'], 'final_qc_outcome'
        X, y = df_pred[features], df_pred[target].apply(lambda x: 1 if x == 'Fail' else 0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced').fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        eval_cols = st.columns([1.5, 2])
        with eval_cols[0]:
            st.markdown("**Model Performance**")
            decision_threshold = st.slider("Set Classification Threshold for 'Fail'", 0.0, 1.0, 0.5, 0.05, key="pred_q_thresh")
            y_pred = (y_pred_proba >= decision_threshold).astype(int); cm = confusion_matrix(y_test, y_pred, labels=[1, 0]); tn, fp, fn, tp = cm.ravel()
            cm_df = pd.DataFrame([[f"True Positive (Caught): {tp}", f"False Negative (Missed): {fn}"], [f"False Positive (Alarm): {fp}", f"True Negative: {tn}"]], columns=["Predicted: Fail", "Predicted: Pass"], index=["Actual: Fail", "Actual: Pass"])
            st.dataframe(cm_df, use_container_width=True)
        with eval_cols[1]:
            st.markdown("**Key Predictive Features**")
            feature_imp = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
            fig_imp = px.bar(feature_imp, x='Importance', y='Feature', orientation='h', title="Which sensor readings are most predictive?"); fig_imp.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=300); st.plotly_chart(fig_imp, use_container_width=True)
        
        st.markdown("##### Business Impact Simulation")
        cost_of_failure = st.number_input("Cost of a Completed Failed Unit ($)", 100, 10000, 5000)
        net_savings = (tp * cost_of_failure) - (fp * 100) # Assume $100 review cost
        st.metric("Estimated Net Savings on Test Set", f"${net_savings:,.0f}", help="Savings from catching failures early minus cost of false alarms.")

    # --- CLASSICAL VS MODERN OPTIMIZATION ---
    with tool_tabs[1]:
        st.subheader("Process Optimization: Classical (RSM) vs. Modern (ML)")
        st.markdown("Compare a traditional Response Surface Methodology (RSM) model with a modern Machine Learning approach (Gaussian Process) for process optimization.")

        df_ml_classical = ssm.get_data("ml_vs_classical_data")
        X, y = df_ml_classical[['x', 'y']], df_ml_classical['z']

        viz_cols = st.columns(2)
        with viz_cols[0]:
            st.markdown("**Classical: Response Surface (OLS)**")
            with st.expander("Learn More"):
                st.markdown("RSM uses Ordinary Least Squares (OLS) to fit a quadratic equation to the data. It's excellent for finding a single, smooth optimum but can struggle with complex, multi-modal surfaces.")
            model_ols = ols('z ~ x + y + I(x**2) + I(y**2) + x*y', data=df_ml_classical).fit()
            grid_pred = model_ols.predict(df_ml_classical[['x','y']]).values.reshape(50, 50)
            fig = go.Figure(data=[go.Surface(z=grid_pred, x=df_ml_classical['x'].unique(), y=df_ml_classical['y'].unique(), colorscale='viridis', opacity=0.9)])
            fig.update_layout(title="RSM Predicted Surface", height=500, margin=dict(l=0,r=0,b=0,t=40)); st.plotly_chart(fig, use_container_width=True)
        with viz_cols[1]:
            st.markdown("**Modern: Gaussian Process (ML)**")
            with st.expander("Learn More"):
                st.markdown("A Gaussian Process model is a flexible, non-parametric ML approach. It can capture highly complex, non-linear relationships that a simple quadratic model might miss, making it more powerful for discovering unexpected optima in a design space.")
            kernel = C(1.0) * RBF(1.0)
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, random_state=42).fit(X, y)
            z_gp = gp.predict(X).reshape(50, 50)
            fig_gp = go.Figure(data=[go.Surface(z=z_gp, x=df_ml_classical['x'].unique(), y=df_ml_classical['y'].unique(), colorscale='plasma', opacity=0.9)])
            fig_gp.update_layout(title="Gaussian Process Predicted Surface", height=500, margin=dict(l=0,r=0,b=0,t=40)); st.plotly_chart(fig_gp, use_container_width=True)
        st.success("**Conclusion:** While both methods identify a similar optimal region, the Gaussian Process model captures more complex local features and nuances in the surface compared to the smoother, quadratic RSM fit. For complex processes, the ML approach may reveal more detailed insights.")

    # --- RELEASE TEST EFFECTIVENESS ---
    with tool_tabs[2]:
        st.subheader("Release Test Effectiveness (ROC Analysis)")
        st.markdown("Assess how well a release test measurement predicts the true quality of a batch. A high **Area Under the Curve (AUC)** indicates a powerful, discriminating test.")
        df_release = ssm.get_data("release_data")
        df_release['true_status_numeric'] = df_release['true_status'].apply(lambda x: 1 if x == 'Fail' else 0)
        fpr, tpr, _ = roc_curve(df_release['true_status_numeric'], df_release['test_measurement'])
        roc_auc = auc(fpr, tpr)
        
        fig_roc = go.Figure(data=[go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC = {roc_auc:.3f}', line=dict(color='darkblue', width=3))])
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Chance', line=dict(color='grey', width=2, dash='dash')))
        fig_roc.update_layout(title=f"<b>Release Test Performance (AUC = {roc_auc:.3f})</b>", xaxis_title='False Positive Rate (Good Batches Failed)', yaxis_title='True Positive Rate (Bad Batches Caught)', legend=dict(x=0.4, y=0.2)); st.plotly_chart(fig_roc, use_container_width=True)
