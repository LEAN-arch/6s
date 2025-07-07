"""
Renders the Machine Learning & Analytics Lab, a sophisticated workspace for
applying modern data science techniques to quality engineering challenges.

This module serves as the R&D hub for the modern MBB. It provides tools for
building predictive quality models and evaluating the effectiveness of release
tests using advanced analytics. This dashboard demonstrates how to augment
classical Six Sigma with cutting-edge, data-driven strategies.

SME Overhaul & Consolidation:
- This module now serves as the single, unified lab for all ML-driven analyses,
  incorporating the essential functionality from the previously separate (and
  now deprecated) `predictive_quality_lab.py` and `release_optimization_suite.py`.
- The 'Predictive Quality' tool has been enhanced with an interactive business
  impact simulator, translating model performance directly into financial terms.
- The 'Release Test Analytics' tool provides a clear, industry-standard ROC
  analysis to evaluate test effectiveness.
- All "Learn More" sections have been rewritten to make complex data science
  concepts accessible and relevant to a quality professional.
"""

import logging
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc

from six_sigma.data.session_state_manager import SessionStateManager

logger = logging.getLogger(__name__)

def render_ml_analytics_lab(ssm: SessionStateManager) -> None:
    """Creates the UI for the Advanced Analytics & ML Lab."""
    st.header("🧪 ML & Analytics Lab")
    st.markdown("Explore modern data science techniques to augment classical Six Sigma methodologies. Use this lab to build predictive models and enhance quality control strategies.")

    tool_tabs = st.tabs(["**1. Predictive Quality Modeling**", "**2. Release Test Effectiveness (ROC Analysis)**"])

    # ==================== PREDICTIVE QUALITY MODELING ====================
    with tool_tabs[0]:
        st.subheader("Predictive Quality Modeling")
        st.markdown("Develop a model to predict final QC outcomes from in-process sensor data, enabling a strategic shift from lagging indicators to leading indicators.")
        
        with st.expander("Learn More: What is Predictive Quality?"):
            st.markdown("""
            **The Old Way (Lagging Indicators):** We produce a complete unit, then perform a final quality test. If it fails, we have already wasted all the time, material, and energy that went into making it.
            
            **The New Way (Leading Indicators):** This approach uses machine learning to find patterns in real-time, in-process sensor data (like temperature, pressure, vibration) that are predictive of a future failure.
            
            By flagging a unit that is *likely* to fail while it's still being made, we can intervene early. This allows us to scrap it with minimal cost, adjust the process in real-time, and prevent the full cost of a completed defect. This is the essence of **"Quality at the Source."**
            """)

        try:
            df_pred = ssm.get_data("predictive_quality_data")
            if df_pred.empty:
                st.warning("Predictive quality data is not available in the data model.")
            else:
                # --- Model Training ---
                features, target = ['in_process_temp', 'in_process_pressure', 'in_process_vibration'], 'final_qc_outcome'
                X, y = df_pred[features], df_pred[target].apply(lambda x: 1 if x == 'Fail' else 0)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
                
                with st.spinner("Training Random Forest model..."):
                    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced').fit(X_train, y_train)
                y_pred_proba = model.predict_proba(X_test)[:, 1] # Probability of 'Fail'

                # --- Interactive Evaluation ---
                st.markdown("#### Model Performance Evaluation")
                eval_cols = st.columns([1.5, 2])
                with eval_cols[0]:
                    st.markdown("**Interactive Confusion Matrix**")
                    decision_threshold = st.slider(
                        "Set Classification Threshold", 0.0, 1.0, 0.5, 0.05,
                        help="Adjust this threshold to see the trade-off. A higher value means the model needs to be more 'certain' before predicting a failure, reducing false alarms but potentially missing more true failures."
                    )
                    y_pred = (y_pred_proba >= decision_threshold).astype(int)
                    cm = confusion_matrix(y_test, y_pred)
                    tn, fp, fn, tp = cm.ravel()
                    
                    cm_df = pd.DataFrame(
                        [[f"Caught Failure (TP): {tp}", f"Missed Failure (FN): {fn}"],
                         [f"False Alarm (FP): {fp}", f"Correct Pass (TN): {tn}"]],
                        columns=["Predicted: Fail", "Predicted: Pass"],
                        index=["Actual: Fail", "Actual: Pass"]
                    )
                    st.dataframe(cm_df, use_container_width=True)

                with eval_cols[1]:
                    st.markdown("**Key Predictive Features**")
                    feature_imp = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
                    fig_imp = px.bar(feature_imp, x='Importance', y='Feature', orientation='h', title="Which sensor readings are most predictive?")
                    fig_imp.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=300)
                    st.plotly_chart(fig_imp, use_container_width=True)
                
                st.divider()
                st.markdown("#### Business Impact Simulation")
                impact_cols = st.columns(2)
                cost_of_failure = impact_cols[0].number_input("Cost of a Completed Failed Unit ($)", 100, 10000, 5000)
                cost_of_review = impact_cols[1].number_input("Cost of a False Alarm (e.g., review time) ($)", 10, 1000, 100)
                
                savings = (tp * cost_of_failure)
                wasted_cost = (fp * cost_of_review)
                net_savings = savings - wasted_cost

                st.info(f"""
                With the current threshold of **{decision_threshold:.2f}**:
                - **Gross Savings:** The model caught **{tp}** failures early, preventing **${savings:,.0f}** in scrap/rework costs.
                - **Investigation Costs:** It triggered **{fp}** false alarms, costing **${wasted_cost:,.0f}** in unnecessary reviews.
                """)
                st.metric("Estimated Net Financial Impact (on Test Set)", f"${net_savings:,.0f}")
        except Exception as e:
            st.error(f"An error occurred during Predictive Quality analysis: {e}")
            logger.error(f"Failed Predictive Quality analysis: {e}", exc_info=True)


    # ==================== RELEASE TEST EFFECTIVENESS (ROC) ====================
    with tool_tabs[1]:
        st.subheader("Release Test Effectiveness (ROC Analysis)")
        st.markdown("Assess how well a go/no-go release test can distinguish between good and bad product lots. This is critical for data-driven lot acceptance decisions.")
        
        with st.expander("Learn More: Understanding ROC and AUC"):
            st.markdown("""
            A **Receiver Operating Characteristic (ROC) curve** is an industry-standard tool for evaluating the performance of a binary classification test.
            
            - **The Curve:** It plots the **True Positive Rate** (how many bad lots you correctly fail) against the **False Positive Rate** (how many good lots you incorrectly fail) at every possible test threshold. An ideal curve "hugs" the top-left corner.
            - **The Diagonal Line:** Represents a test with no discriminatory power (random chance, like a coin flip).
            - **Area Under the Curve (AUC):** This is the key metric. It summarizes the entire curve into a single number. An AUC of 1.0 is a perfect test, while an AUC of 0.5 is a useless test.
            
            **AUC Score Interpretation:**
            - **> 0.9:** Excellent test
            - **0.8 - 0.9:** Good test
            - **0.7 - 0.8:** Fair test
            - **< 0.7:** Poor test
            
            A high AUC score gives you the confidence to set statistically-driven release limits and justify sampling plans.
            """)
        
        try:
            df_release = ssm.get_data("release_data")
            if df_release.empty:
                st.warning("Release test data is not available in the data model.")
            else:
                df_release['true_status_numeric'] = df_release['true_status'].apply(lambda x: 1 if x == 'Fail' else 0)
                fpr, tpr, _ = roc_curve(df_release['true_status_numeric'], df_release['test_measurement'])
                roc_auc = auc(fpr, tpr)
                
                roc_cols = st.columns([1, 2])
                with roc_cols[0]:
                    st.metric("Test Discriminatory Power (AUC)", f"{roc_auc:.3f}")
                    if roc_auc > 0.9: st.success("This is an **excellent** test.")
                    elif roc_auc > 0.8: st.success("This is a **good** test.")
                    elif roc_auc > 0.7: st.warning("This is a **fair** test.")
                    else: st.error("This is a **poor** test.")

                with roc_cols[1]:
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC = {roc_auc:.3f}', line=dict(color='royalblue', width=3)))
                    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Chance', line=dict(color='grey', width=2, dash='dash')))
                    fig_roc.update_layout(
                        title=f"<b>Release Test Performance (ROC Curve)</b>",
                        xaxis_title='False Positive Rate (1 - Specificity)',
                        yaxis_title='True Positive Rate (Sensitivity)',
                        legend=dict(x=0.4, y=0.15),
                        height=400,
                        margin=dict(l=20,r=20,t=40,b=20)
                    )
                    st.plotly_chart(fig_roc, use_container_width=True)
        except Exception as e:
            st.error(f"An error occurred during ROC analysis: {e}")
            logger.error(f"Failed ROC analysis: {e}", exc_info=True)
