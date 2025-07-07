# philips_quality_optimizer/dashboards/predictive_quality_lab.py
"""
Renders the Predictive Quality Lab.

This dashboard provides a workspace for developing and evaluating machine
learning models that predict final product quality from in-process sensor
data. It enables the shift from lagging indicators (final QC) to leading
indicators (in-process metrics), supporting a "Quality at the Source" strategy.
"""

import logging
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc

from philips_quality_optimizer.data.session_state_manager import SessionStateManager

logger = logging.getLogger(__name__)

def render_predictive_quality_lab(ssm: SessionStateManager) -> None:
    """
    Creates the UI for the Predictive Quality Lab tab.

    Args:
        ssm (SessionStateManager): The session state manager to access predictive data.
    """
    st.header("🔮 Predictive Quality Lab")
    st.markdown("Develop machine learning models to predict final QC outcomes from in-process manufacturing data. Use this lab to identify key leading indicators of quality and build a case for real-time, automated process control.")

    try:
        # --- 1. Load Data and Check for Dependencies ---
        df_full = ssm.get_data("predictive_quality_data")

        if df_full.empty:
            st.warning("No predictive quality data is available.")
            return

        # --- 2. Train and Evaluate the Model ---
        st.subheader("Model Training & Evaluation")
        st.markdown("We'll train a Random Forest classifier to predict the `final_qc_outcome` based on the in-process sensor readings.")
        
        # --- Model Setup ---
        features = ['in_process_temp', 'in_process_pressure', 'in_process_vibration']
        target = 'final_qc_outcome'
        
        X = df_full[features]
        y = df_full[target].apply(lambda x: 1 if x == 'Fail' else 0) # Encode target

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        
        # Get predictions on the test set
        y_pred_proba = model.predict_proba(X_test)[:, 1] # Probability of 'Fail'

        # --- Evaluation UI ---
        eval_cols = st.columns([1.5, 2])
        with eval_cols[0]:
            st.markdown("**Model Performance**")
            # --- Confusion Matrix ---
            # Set a threshold for classification
            decision_threshold = st.slider(
                "Set Classification Threshold for 'Fail'",
                min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                help="A higher threshold makes the model more 'cautious' about predicting a failure, reducing false alarms but potentially missing some true failures."
            )
            y_pred = (y_pred_proba >= decision_threshold).astype(int)
            cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
            tn, fp, fn, tp = cm.ravel()
            
            # Display Confusion Matrix as a clear table
            cm_df = pd.DataFrame(
                [[f"True Positive: {tp}", f"False Negative: {fn}"],
                 [f"False Positive: {fp}", f"True Negative: {tn}"]],
                columns=["Predicted: Fail", "Predicted: Pass"],
                index=["Actual: Fail", "Actual: Pass"]
            )
            st.dataframe(cm_df, use_container_width=True)

        with eval_cols[1]:
            # --- Feature Importance ---
            st.markdown("**Key Predictive Features**")
            feature_imp = pd.DataFrame({
                'Feature': features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)

            fig_imp = px.bar(feature_imp, x='Importance', y='Feature', orientation='h', title="Which sensor readings are most predictive?")
            fig_imp.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=300)
            st.plotly_chart(fig_imp, use_container_width=True)
            
        st.divider()
        
        # --- 3. Business Impact Simulation ---
        st.subheader("Business Impact Simulation")
        st.markdown("Simulate the financial impact of deploying this predictive model to stop failing runs early, preventing the cost of completion.")

        impact_cols = st.columns(3)
        cost_of_failure = impact_cols[0].number_input("Cost of a Completed Failed Unit ($)", 100, 10000, 5000)
        cost_of_review = impact_cols[1].number_input("Cost of Reviewing a Flagged Unit ($)", 10, 1000, 100)
        
        # Calculations based on the confusion matrix
        runs_saved_from_failure = tp
        runs_wrongly_flagged = fp
        
        money_saved = runs_saved_from_failure * cost_of_failure
        money_lost = runs_wrongly_flagged * cost_of_review
        net_savings = money_saved - money_lost
        
        impact_cols[2].metric(
            label="Estimated Net Savings (on Test Set)",
            value=f"${net_savings:,.0f}",
            help=f"Savings from catching failures early minus the cost of reviewing false alarms."
        )
        
        st.info(f"""
        **Simulation Summary at {decision_threshold:.0%} Threshold:**
        - The model correctly predicts **{tp}** out of **{tp+fn}** true failures, saving an estimated **${money_saved:,.0f}**.
        - It incorrectly flags **{fp}** good units for review, costing an estimated **${money_lost:,.0f}**.
        - This results in a potential **net saving of ${net_savings:,.0f}** for the number of units in the test set.
        """)
        
    except ImportError:
        st.error("This page requires scikit-learn. Please ensure it is installed (`pip install scikit-learn`).")
    except Exception as e:
        st.error("An error occurred while rendering the Predictive Quality Lab.")
        logger.error(f"Failed to render predictive quality lab: {e}", exc_info=True)
