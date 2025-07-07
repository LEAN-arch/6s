"""
Renders the Machine Learning & Analytics Lab, a sophisticated workspace for
applying and comparing modern data science techniques against classical Six Sigma
statistical methods.

This module is the core R&D and educational hub for the modern MBB. It is
structured as a series of comparative studies, directly pitting a classical
approach against a modern one for common industrial challenges. This design is
intended to build intuition about when to use each type of tool.

SME Overhaul & Consolidation:
- This module is now the definitive, unified lab for all advanced analytics.
- It preserves the original predictive quality and ROC analysis sections.
- It has been massively extended with new comparative studies:
  - Driver Analysis: ANOVA vs. SHAP for explainability.
  - Process Control: SPC vs. Isolation Forest for anomaly detection.
  - Process Optimization: DOE/RSM vs. Bayesian Optimization.
  - Failure Analysis: Manual Binning vs. K-Means Clustering.
- Each tab includes rich SME explanations, real-world analogies, and over 10
  distinct examples to provide a comprehensive learning experience.
"""
import logging
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt

# Scikit-learn and Scikit-optimize imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from skopt import gp_minimize
from skopt.space import Real

# Local application imports
from six_sigma.data.session_state_manager import SessionStateManager
from six_sigma.utils.stats import perform_anova_on_dataframe

logger = logging.getLogger(__name__)

# Helper function to avoid re-running optimization
@st.cache_data
def run_bayesian_optimization(df_opt, n_calls=15):
    bounds = [Real(-5, 5, name='x'), Real(-5, 5, name='y')]
    def objective_func(params):
        x, y = params
        # Find the closest point in our grid to the sampled point
        return -df_opt.loc[((df_opt['x'] - x)**2 + (df_opt['y'] - y)**2).idxmin()]['z']
    
    result = gp_minimize(objective_func, bounds, n_calls=n_calls, random_state=42)
    return result

def render_ml_analytics_lab(ssm: SessionStateManager) -> None:
    """Creates the UI for the ML & Analytics Lab comparative workspace."""
    st.header("🔬 Classical Statistics vs. Modern Machine Learning")
    st.markdown("A comparative lab to understand the strengths and weaknesses of traditional statistical methods versus modern ML approaches for common Six Sigma tasks. This workspace is designed to build intuition and expand your analytical toolkit.")

    # --- Main UI Tabs ---
    tab_list = [
        "**1. Predictive Quality**", "**2. Test Effectiveness**", "**3. Driver Analysis**",
        "**4. Process Control**", "**5. Process Optimization**", "**6. Failure Mode Analysis**"
    ]
    tabs = st.tabs(tab_list)

    # ==================== TAB 1: PREDICTIVE QUALITY (Classification) ====================
    with tabs[0]:
        st.subheader("Challenge 1: Predict Product Failure from In-Process Data")
        with st.expander("SME Deep Dive: Logistic Regression vs. Random Forest"):
            st.markdown("""
            **The Goal:** Build an early-warning system. Can we predict if a product will fail its final test based on sensor readings during production?
            
            #### The Methods
            - **Classical: Logistic Regression** is a statistical workhorse. It finds the best linear boundary to separate the two classes (Pass/Fail).
              - **Analogy (Example 1):** Think of a diligent but junior apprentice. They follow a simple, linear checklist. "If Temperature > 220°C, add 2 points to failure risk. If Pressure > 60 psi, add 3 points." It's easy to understand their logic.
              - **Pros:** Highly interpretable coefficients, statistically rigorous, fast.
              - **Cons:** Struggles with complex, non-linear relationships. It can't easily understand "Temperature only matters if Pressure is also high."

            - **Modern: Random Forest** is an ensemble of many decision trees. It's like asking hundreds of experts for their opinion and taking the majority vote.
              - **Analogy (Example 2):** Think of a seasoned master mechanic. They don't use a simple checklist. They have immense intuition, recognizing thousands of subtle, interacting patterns. "I've seen this strange vibration combined with a slight drop in pressure before... that usually means trouble, but only on Tuesdays."
              - **Pros:** Excellent predictive accuracy, automatically captures non-linearities and interactions.
              - **Cons:** A "black box" – it's hard to understand the exact reasoning of 500 experts voting at once.

            #### SME Verdict
            For **maximum predictive power** to catch failures, **Random Forest** is superior. For **simple, explainable models** to present to stakeholders, **Logistic Regression** is often better.
            """)
        
        df_pred = ssm.get_data("predictive_quality_data")
        if not df_pred.empty:
            features, target = ['in_process_temp', 'in_process_pressure', 'in_process_vibration'], 'final_qc_outcome'
            X, y = df_pred[features], df_pred[target].apply(lambda x: 1 if x == 'Fail' else 0)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            model_rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
            model_lr = LogisticRegression(random_state=42).fit(X_train, y_train)
            
            st.markdown("##### Model Performance Comparison (ROC Curve)")
            # ... (ROC Plot code from previous version) ...
            pred_proba_rf = model_rf.predict_proba(X_test)[:, 1]; pred_proba_lr = model_lr.predict_proba(X_test)[:, 1]
            auc_rf = roc_auc_score(y_test, pred_proba_rf); auc_lr = roc_auc_score(y_test, pred_proba_lr)
            fpr_rf, tpr_rf, _ = roc_curve(y_test, pred_proba_rf); fpr_lr, tpr_lr, _ = roc_curve(y_test, pred_proba_lr)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr_rf, y=tpr_rf, mode='lines', name=f'Random Forest (AUC = {auc_rf:.3f})', line=dict(width=3)))
            fig.add_trace(go.Scatter(x=fpr_lr, y=tpr_lr, mode='lines', name=f'Logistic Regression (AUC = {auc_lr:.3f})', line=dict(width=3)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Chance', line=dict(color='grey', width=2, dash='dash')))
            fig.update_layout(title="<b>Model Performance (ROC Curve)</b>", xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', legend=dict(x=0.4, y=0.15))
            st.plotly_chart(fig, use_container_width=True)

    # ==================== TAB 2: TEST EFFECTIVENESS (Evaluation) ====================
    with tabs[1]:
        st.subheader("Challenge 2: Evaluate the Power of a Go/No-Go Release Test")
        with st.expander("SME Deep Dive: The ROC Curve"):
             st.markdown("""
             **The Goal:** Quantify how good our final product test is. Does a high measurement value truly indicate a bad batch?
             
             #### The Method: Receiver Operating Characteristic (ROC) Analysis
             This is the gold standard for evaluating any binary diagnostic test.
             - **Analogy (Example 3):** Imagine a medical test for a disease. The ROC curve helps us understand the fundamental trade-off: If we make the test *very sensitive* (catching every sick person), we will inevitably get more *false positives* (telling healthy people they are sick). If we make it *very specific* (only identifying the most obvious cases), we will *miss* borderline cases. The ROC curve visualizes this trade-off across all possible cut-off points.
             
             - **AUC (Area Under the Curve)** is the single best metric. An AUC of 1.0 is a perfect test. An AUC of 0.5 is a useless test (a coin flip).
             
             #### Interactive Exploration (Example 4)
             Use the slider below to pick a "cut-off" value on the test measurement. The plot will show where this point lies on the ROC curve, and the table will show the resulting confusion matrix. This lets you find a practical "sweet spot" that balances catching bad lots with not failing too many good ones.
             """)
        df_release = ssm.get_data("release_data")
        if not df_release.empty:
            df_release['true_status_numeric'] = df_release['true_status'].apply(lambda x: 1 if x == 'Fail' else 0)
            fpr, tpr, thresholds = roc_curve(df_release['true_status_numeric'], df_release['test_measurement'])
            roc_auc = auc(fpr, tpr)
            
            slider_val = st.slider("Select Test Cut-off Threshold", float(df_release['test_measurement'].min()), float(df_release['test_measurement'].max()), float(df_release['test_measurement'].mean()))
            
            # Find the closest point on the ROC curve for the slider value
            idx = (np.abs(thresholds - slider_val)).argmin()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {roc_auc:.3f})'))
            fig.add_trace(go.Scatter(x=[fpr[idx]], y=[tpr[idx]], mode='markers', marker=dict(size=15, color='red'), name='Current Threshold'))
            fig.update_layout(title="<b>Interactive ROC Analysis</b>", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            
            # Calculate confusion matrix for the selected threshold
            y_pred = (df_release['test_measurement'] >= slider_val).astype(int)
            cm = confusion_matrix(df_release['true_status_numeric'], y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.metric("Test Power (AUC)", f"{roc_auc:.3f}")
                st.write("Confusion Matrix at this Threshold:")
                cm_df = pd.DataFrame([[f"Caught (TP): {tp}", f"Missed (FN): {fn}"], [f"False Alarm (FP): {fp}", f"Correct (TN): {tn}"]], columns=["Predicted: Fail", "Predicted: Pass"], index=["Actual: Fail", "Actual: Pass"])
                st.dataframe(cm_df)

    # ==================== TAB 3: DRIVER ANALYSIS (Explainability) ====================
    with tabs[2]:
        st.subheader("Challenge 3: Understand the 'Why' Behind Failures")
        with st.expander("SME Deep Dive: ANOVA vs. SHAP"):
            st.markdown("""
            **The Goal:** Move beyond *what* happened to *why* it happened. Which process variables are the most influential drivers of failure?
            
            #### The Methods
            - **Classical: ANOVA (Analysis of Variance)** tests if the average value of an input is significantly different for "Pass" vs. "Fail" groups.
                - **Analogy (Example 5):** It's like a pollster reporting "Voters earning over $100k, on average, preferred Candidate A." It tells you about a group average, a powerful but high-level insight.
            - **Modern: SHAP (SHapley Additive exPlanations)** explains individual predictions from an ML model.
                - **Analogy (Example 6):** It's like an exit poll interview with a single voter. "Why did you vote for Candidate A?" "Well, their tax policy was a big factor (+10 points), but their stance on trade was a negative (-3 points). Overall, I leaned positive." SHAP does this for every feature and every prediction.
            
            #### SME Verdict
            **ANOVA** is excellent for designed experiments to confirm a factor's overall significance. **SHAP** is revolutionary for debugging and understanding complex, real-world (observational) data. It provides both a global view (the summary plot) and a local, per-unit explanation. The SHAP plot below shows that *high* pressure and *high* temperature are strong drivers of failure, an insight ANOVA's single p-value cannot provide.
            """)
        if not df_pred.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Classical: Average Effect (ANOVA)")
                fig_box = px.box(df_pred, x='final_qc_outcome', y='in_process_pressure', title='Pressure by Outcome')
                st.plotly_chart(fig_box, use_container_width=True)
            with col2:
                st.markdown("##### Modern: Individual Explanation (SHAP)")
                fig, ax = plt.subplots(figsize=(6, 4)); shap.summary_plot(shap_values[1], X_test, plot_type="dot", show=False, plot_size=(5, 3)); st.pyplot(fig, bbox_inches='tight'); plt.clf()

    # ==================== TAB 4: PROCESS CONTROL (Anomaly Detection) ====================
    with tabs[3]:
        st.subheader("Challenge 4: Detect Unusual Behavior in a Live Process")
        with st.expander("SME Deep Dive: SPC vs. Isolation Forest"):
            st.markdown("""
            **The Goal:** Monitor a running process and get alerted to any strange or unexpected behavior.
            
            #### The Methods
            - **Classical: SPC (Statistical Process Control) Chart** uses the historical, stable variation of a process to set +/- 3 sigma control limits. It follows a strict set of rules.
                - **Analogy (Example 7):** A security guard with a checklist. "Is anyone running? No. Is anyone shouting? No. Is anyone outside the velvet rope? Yes! Alert!" It's great at catching known rule violations.
            - **Modern: Isolation Forest** is an unsupervised ML algorithm that learns the "shape" of normal data. It flags any point that doesn't conform to this learned shape.
                - **Analogy (Example 8):** A seasoned detective at a party. They don't have a checklist. They just have a "feel" for the room. They might notice someone who is standing too still, or whispering in a corner, or wearing a winter coat in summer. They spot things that aren't against the "rules" but are just... weird.
            
            #### SME Verdict
            **SPC** is the non-negotiable standard for **sustaining control** on a known, stable process. **Isolation Forest** is a powerful **investigative tool** to find "unknown unknowns" or monitor complex systems where simple rules don't apply. Notice below how the ML method catches the big shift like SPC does, but also flags other subtle outliers.
            """)
        df_process = ssm.get_data("process_data")
        if not df_process.empty:
            process_series = df_process['seal_strength']
            iso_forest = IsolationForest(contamination='auto', random_state=42).fit(process_series.values.reshape(-1, 1))
            df_process['anomaly'] = iso_forest.predict(process_series.values.reshape(-1, 1))
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Classical: SPC Chart")
                from six_sigma.utils.plotting import create_imr_chart
                st.plotly_chart(create_imr_chart(process_series, "Seal Strength", 78, 92), use_container_width=True)
            with col2:
                st.markdown("##### Modern: ML Anomaly Detection")
                fig_iso = go.Figure()
                fig_iso.add_trace(go.Scatter(y=df_process['seal_strength'], mode='lines', name='Process Data'))
                anomalies = df_process[df_process['anomaly'] == -1]
                fig_iso.add_trace(go.Scatter(x=anomalies.index, y=anomalies['seal_strength'], mode='markers', name='Detected Anomaly', marker=dict(color='red', size=10, symbol='x')))
                fig_iso.update_layout(title='<b>Isolation Forest Anomaly Detection</b>')
                st.plotly_chart(fig_iso, use_container_width=True)

    # ==================== TAB 5: PROCESS OPTIMIZATION ====================
    with tabs[4]:
        st.subheader("Challenge 5: Efficiently Find the Best Process 'Recipe'")
        with st.expander("SME Deep Dive: DOE/RSM vs. Bayesian Optimization"):
            st.markdown("""
            **The Goal:** Find the combination of inputs (e.g., time, temperature) that gives the best possible output (e.g., maximum strength), using the fewest number of experiments.
            
            #### The Methods
            - **Classical: Design of Experiments (DOE) / Response Surface Methodology (RSM)** involves pre-planning a grid of experiments to run (e.g., a factorial design). After running them, you fit a statistical model (like a quadratic equation) to the results to find the optimum.
                - **Analogy (Example 9):** A baker testing a new cake recipe. They meticulously plan to bake cakes at (low temp, low time), (low temp, high time), (high temp, low time), (high temp, high time), plus a few in the middle. They bake all 9 cakes, taste them, then model the results to declare the best recipe. It's systematic and robust, but front-loaded with work.
            - **Modern: Bayesian Optimization** is a sequential, "smart search" strategy. It uses a flexible ML model (a Gaussian Process) to build a probabilistic map of the response surface. It then uses this map to intelligently decide the *single most informative experiment to run next*.
                - **Analogy (Example 10):** A master chef trying to perfect a sauce. They make one batch. They taste it and think, "Hmm, a bit too salty, but the acidity is promising." Based on that, they don't test a random recipe next; they intelligently decide the next best guess might be "a little less salt, a little more sugar." They learn and adapt after every single experiment.
            
            #### SME Verdict
            **DOE/RSM** is the gold standard for formal, rigorous experimentation where factors can be controlled. **Bayesian Optimization** is incredibly powerful when experiments are very expensive or time-consuming, as it often finds a near-optimal solution with far fewer experimental runs than a full DOE.
            """)
        df_opt = ssm.get_data("optimization_data")
        if not df_opt.empty:
            result = run_bayesian_optimization(df_opt)
            sampled_points = np.array(result.x_iters)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Classical: RSM (Full Grid View)")
                fig = go.Figure(data=go.Contour(z=df_opt['z'], x=df_opt['x'], y=df_opt['y'], colorscale='Viridis'))
                fig.update_layout(title="Full 'True' Response Surface")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown("##### Modern: Bayesian Optimization Path")
                fig = go.Figure(data=go.Contour(z=df_opt['z'], x=df_opt['x'], y=df_opt['y'], showscale=False, colorscale='Viridis', opacity=0.5))
                fig.add_trace(go.Scatter(x=sampled_points[:, 0], y=sampled_points[:, 1], mode='markers+text', text=[str(i+1) for i in range(len(sampled_points))], textposition="top right", marker=dict(color='red', size=10, symbol='x'), name='Sampled Points'))
                fig.update_layout(title="Path of Smart Search (15 Experiments)")
                st.plotly_chart(fig, use_container_width=True)

    # ==================== TAB 6: FAILURE MODE ANALYSIS (Clustering) ====================
    with tabs[5]:
        st.subheader("Challenge 6: Discover Hidden Groups or 'Types' of Failures")
        with st.expander("SME Deep Dive: Manual Binning vs. K-Means Clustering"):
            st.markdown("""
            **The Goal:** We have a cloud of failure data. Are there distinct, natural "families" of failures that we are currently missing?
            
            #### The Methods
            - **Classical: Manual Binning / Histograms.** We look at one variable at a time (e.g., temperature) and draw lines based on our expert knowledge. "Failures above 240°C we'll call 'overheating'. Failures below 190°C we'll call 'incomplete bonding'."
                - **Analogy (Example 11):** Sorting laundry by color. We decide on the categories beforehand: whites, darks, colors. It's simple and based on one dimension.
            - **Modern: K-Means Clustering.** An unsupervised ML algorithm that looks at all variables simultaneously and mathematically finds the best "centers" (centroids) to partition the data into a specified number (K) of distinct groups.
                - **Analogy (Example 12):** A smart sorting machine. It looks at color, fabric type, and item size all at once. It might discover three groups you never thought of: "delicate whites," "heavy-duty darks," and "colorful cottons." It finds the natural, multi-dimensional groupings in the data.

            #### SME Verdict
            **Manual Binning** is useful for simple, one-dimensional problems. **K-Means Clustering** is exceptionally powerful for uncovering hidden, multi-dimensional patterns in failure data, potentially revealing root causes (e.g., "Failure Mode A" is always high-temp/low-pressure, while "Failure Mode B" is low-temp/high-pressure) that would be impossible to see by looking at one variable at a time.
            """)
        df_clust = ssm.get_data("failure_clustering_data")
        if not df_clust.empty:
            X_clust = StandardScaler().fit_transform(df_clust[['temperature', 'pressure']])
            kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto').fit(X_clust)
            df_clust['ml_cluster'] = kmeans.labels_
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Classical: Manual Binning")
                df_clust['manual_bin'] = pd.cut(df_clust['temperature'], bins=[0, 200, 230, 300], labels=['Low Temp', 'Mid Temp', 'High Temp'])
                fig1 = px.scatter(df_clust, x='temperature', y='pressure', color='manual_bin', title='Failures Grouped by Temperature Bins')
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                st.markdown("##### Modern: K-Means Clustering")
                fig2 = px.scatter(df_clust, x='temperature', y='pressure', color='ml_cluster', title='Failures Grouped by ML Clusters', color_continuous_scale=px.colors.qualitative.Plotly)
                st.plotly_chart(fig2, use_container_width=True)
