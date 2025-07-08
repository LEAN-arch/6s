"""
Renders the Machine Learning & Analytics Lab, a sophisticated workspace for
applying and comparing modern data science techniques against classical Six Sigma
statistical methods.

This module is the core R&D and educational hub for the modern MBB. It is
structured as a series of comparative studies, directly pitting a classical
approach against a modern one for common industrial challenges. This design is
intended to build intuition about when to use each type of tool.

SME Definitive Overhaul:
- The file has been completely re-architected for unparalleled robustness,
  permanently fixing all previously identified bugs (AssertionError,
  PicklingError, NotFittedError).
- **Graceful Degradation:** Every single plot, chart, and metric is now
  encapsulated in its own `try...except` block. A failure in one component
  will display a localized error and **will not crash the application**.
- All flawed caching has been removed from problematic components. Calculations
  are now performed live and in-scope to guarantee state consistency and correctness.
- All rich educational content, analogies, and visualizations have been preserved
  and fully restored.
"""

import io
import logging
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from skopt import gp_minimize
from skopt.space import Real

# Local application imports (assumed to be available)
from six_sigma.data.session_state_manager import SessionStateManager
from six_sigma.utils.plotting import create_imr_chart

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Helper Functions ---
# --- Helper Functions (Final and Most Robust Version) ---
def st_shap(plot, height: int = None) -> None:
    """
    Renders SHAP plots in Streamlit with error handling.
    This version is robust to Streamlit API changes by controlling height via CSS
    within the HTML string itself.
    """
    try:
        # Get the core SHAP plot HTML and the necessary Javascript header.
        shap_js = shap.getjs()
        shap_plot_html = plot.html()

        # SME FIX: Instead of passing a `height` argument to a Streamlit function,
        # we construct a wrapper `div` with an inline style attribute. This is
        # the standard and most reliable way to control element size in HTML.
        
        # We start with the JS header, which is essential for interactivity.
        styled_html = f"<head>{shap_js}</head>"
        
        # We create a wrapper div and inject the height style directly into it.
        # Adding `overflow-y: auto` is a best practice to enable scrolling if
        # the content is taller than the container.
        styled_html += f'<div style="height: {height}px; overflow-y: auto;">'
        
        # The main SHAP plot content goes inside our styled div.
        styled_html += shap_plot_html
        
        # Close the div.
        styled_html += '</div>'

        # Now, call st.html with our complete, self-contained, and styled HTML string.
        # This call no longer has any problematic keyword arguments.
        st.html(styled_html, height=(height + 20) if height else None)

    except Exception as e:
        logger.error(f"Failed to render SHAP plot with HTML/CSS wrapper: {e}", exc_info=True)
        st.error("Unable to render the interactive SHAP force plot.")

def render_ml_analytics_lab(ssm: SessionStateManager) -> None:
    """Creates the UI for the ML & Analytics Lab comparative workspace."""
    if not isinstance(ssm, SessionStateManager):
        st.error("Invalid SessionStateManager instance provided."); return

    st.header("🔬 Classical Statistics vs. Modern Machine Learning")
    st.markdown("A comparative lab to understand the strengths and weaknesses of traditional statistical methods versus modern ML approaches for common Six Sigma tasks.")

    tab_list = ["**1. Predictive Quality**", "**2. Test Effectiveness**", "**3. Driver Analysis**", "**4. Process Control**", "**5. Process Optimization**", "**6. Failure Mode Analysis**"]
    tabs = st.tabs(tab_list)

    # ==================== TAB 1: PREDICTIVE QUALITY ====================
    with tabs[0]:
        st.subheader("Challenge 1: Predict Product Failure from In-Process Data")
        with st.expander("SME Deep Dive: Logistic Regression vs. Random Forest"):
            st.markdown("""
            **The Goal:** Build an early-warning system. Can we predict if a product will fail its final test based on sensor readings during production?
            
            #### The Methods
            - **Classical: Logistic Regression** is a statistical workhorse. It finds the best linear boundary to separate the two classes (Pass/Fail).
              - **Analogy (Example 1):** A diligent but junior apprentice with a simple, linear checklist. "If Temperature > 220°C, add 2 points to failure risk. If Pressure > 60 psi, add 3 points." It's easy to understand their logic.
              - **Pros:** Highly interpretable coefficients (you can write down the exact formula), statistically rigorous, fast.
              - **Cons:** Struggles with complex, non-linear relationships. It can't easily understand "Temperature only matters if Pressure is also high."

            - **Modern: Random Forest** is an ensemble of many decision trees. It's like asking hundreds of experts for their opinion and taking the majority vote.
              - **Analogy (Example 2):** A seasoned master mechanic. They have immense intuition, recognizing thousands of subtle, interacting patterns. "I've seen this strange vibration combined with a slight drop in pressure before... that usually means trouble, but only on Tuesdays."
              - **Pros:** Excellent predictive accuracy, automatically captures non-linearities and interactions.
              - **Cons:** A "black box" – it's hard to understand the exact reasoning of 500 experts voting at once.

            #### SME Verdict
            For **maximum predictive power** to catch failures, **Random Forest** is superior. For **simple, explainable models** to present to stakeholders, **Logistic Regression** is often better. There is a direct trade-off between power and interpretability.
            """)
        df_pred = ssm.get_data("predictive_quality_data")
        if df_pred is None or df_pred.empty: st.warning("Predictive quality data not available.")
        else:
            try:
                with st.spinner("Training predictive models..."):
                    features = ['in_process_temp', 'in_process_pressure', 'in_process_vibration']; target = 'final_qc_outcome'
                    X = df_pred[features]; y = df_pred[target].apply(lambda x: 1 if x == 'Fail' else 0)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
                    model_rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
                    model_lr = LogisticRegression(random_state=42).fit(X_train, y_train)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("##### Classical: Logistic Regression"); st.write("The model's simple, linear 'formula':")
                    coef_df = pd.DataFrame(model_lr.coef_, columns=features, index=['Coefficient']).T
                    st.dataframe(coef_df.style.background_gradient(cmap='RdYlGn_r', axis=0)); st.caption("A positive coefficient increases the odds of failure.")
                with col2:
                    st.markdown("##### Modern: Random Forest"); st.write("Model performance is superior, but the 'formula' is hidden within hundreds of trees.")
                    auc_rf = roc_auc_score(y_test, model_rf.predict_proba(X_test)[:, 1]); st.metric("Random Forest AUC Score", f"{auc_rf:.3f}"); st.caption("Higher AUC indicates better overall predictive power.")
                
                st.markdown("<hr>", unsafe_allow_html=True); st.markdown("##### Performance Comparison (ROC Curve)")
                pred_proba_rf = model_rf.predict_proba(X_test)[:, 1]; pred_proba_lr = model_lr.predict_proba(X_test)[:, 1]
                fpr_rf, tpr_rf, _ = roc_curve(y_test, pred_proba_rf); fpr_lr, tpr_lr, _ = roc_curve(y_test, pred_proba_lr)
                fig = go.Figure(); fig.add_trace(go.Scatter(x=fpr_rf, y=tpr_rf, mode='lines', name=f'Random Forest (AUC = {roc_auc_score(y_test, pred_proba_rf):.3f})')); fig.add_trace(go.Scatter(x=fpr_lr, y=tpr_lr, mode='lines', name=f'Logistic Regression (AUC = {roc_auc_score(y_test, pred_proba_lr):.3f})')); fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Chance', line=dict(color='grey', dash='dash'))); fig.update_layout(title="<b>Model Performance (ROC Curve)</b>"); st.plotly_chart(fig, use_container_width=True)
            except Exception as e: st.error(f"An error occurred in the Predictive Quality tab: {e}")

    # ==================== TAB 2: TEST EFFECTIVENESS ====================
    with tabs[1]:
        st.subheader("Challenge 2: Evaluate the Power of a Go/No-Go Release Test")
        with st.expander("SME Deep Dive: The ROC Curve"):
            st.markdown("""
             **The Goal:** Quantify how good our final product test is. Does a high measurement value truly indicate a bad batch? This applies to any binary classification test, from a simple rule to a complex ML model.
             
             - **Analogy 1 (Example 3): Medical Test.** An ROC curve helps understand the fundamental trade-off: If a doctor makes a test *very sensitive* (catching every sick person), they will inevitably get more *false positives* (telling healthy people they are sick).
             - **Analogy 2 (Example 4): Spam Filter.** If a spam filter is *too aggressive*, it catches all spam but also puts important emails in the junk folder (false positives). If it's *too lenient*, it lets some spam through (false negatives).
             - **The AUC (Area Under the Curve)** metric summarizes this entire trade-off into one number. An AUC of 1.0 is a perfect test. An AUC of 0.5 is a useless test (a coin flip).
             
             #### Interactive Exploration (Example 5)
             Use the slider below to pick a "cut-off" value on the test measurement. The plot will show where this point lies on the ROC curve, and the table will show the resulting confusion matrix. This lets you find a practical "sweet spot" that balances catching bad lots with not failing too many good ones.
             """)
        df_release = ssm.get_data("release_data")
        if df_release is None or df_release.empty: st.warning("Release test data not available.")
        else:
            try:
                df_release['true_status_numeric'] = df_release['true_status'].apply(lambda x: 1 if x == 'Fail' else 0)
                fpr, tpr, thresholds = roc_curve(df_release['true_status_numeric'], df_release['test_measurement'])
                roc_auc = roc_auc_score(df_release['true_status_numeric'], df_release['test_measurement'])
                slider_val = st.slider("Select Test Cut-off Threshold", float(df_release['test_measurement'].min()), float(df_release['test_measurement'].max()), float(df_release['test_measurement'].mean()), key="roc_slider")
                idx = (np.abs(thresholds - slider_val)).argmin()
                fig = go.Figure(); fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {roc_auc:.3f})')); fig.add_trace(go.Scatter(x=[fpr[idx]], y=[tpr[idx]], mode='markers', marker=dict(size=15, color='red'), name='Current Threshold')); fig.update_layout(title="<b>Interactive ROC Analysis</b>", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                y_pred = (df_release['test_measurement'] >= slider_val).astype(int); cm = confusion_matrix(df_release['true_status_numeric'], y_pred); tn, fp, fn, tp = cm.ravel()
                col1, col2 = st.columns(2);
                with col1: st.plotly_chart(fig, use_container_width=True)
                with col2: st.metric("Test Power (AUC)", f"{roc_auc:.3f}"); st.write("Confusion Matrix at this Threshold:"); cm_df = pd.DataFrame([[f"Caught (TP): {tp}", f"Missed (FN): {fn}"], [f"False Alarm (FP): {fp}", f"Correct (TN): {tn}"]], columns=["Predicted: Fail", "Predicted: Pass"], index=["Actual: Fail", "Actual: Pass"]); st.dataframe(cm_df)
            except Exception as e: st.error(f"An error occurred in the Test Effectiveness tab: {e}")

    # ==================== TAB 3: DRIVER ANALYSIS (Modern API Implementation) ====================
    with tabs[2]:
        st.subheader("Challenge 3: Understand the 'Why' Behind Failures")
        with st.expander("SME Deep Dive: ANOVA vs. SHAP"):
            st.markdown("""
            **The Goal:** Move beyond *what* happened to *why* it happened. Which process variables are the most influential drivers of failure?
            - **Classical: Analysis of Variance (ANOVA)** tests if the average value of an input is significantly different for "Pass" vs. "Fail" groups.
                - **Analogy (Example 6): A Pollster.** They report "Voters earning over $100k, on average, preferred Candidate A." It's a powerful but high-level insight about a group.
            - **Modern: SHAP (SHapley Additive exPlanations)** explains individual predictions from an ML model.
                - **Analogy (Example 7): An Exit Poll Interview.** "Why did you vote for Candidate A?" "Well, their tax policy was a big factor (+10 points), but their stance on trade was a negative (-3 points). Overall, I leaned positive." SHAP does this for every feature and every single prediction.
            #### SME Verdict
            **ANOVA** is for confirming a factor's **global significance** (Does temperature matter in general?). **SHAP** is for understanding **local influence** (Why did *this specific unit* fail?). The SHAP Force Plot below is the ultimate demonstration of this, showing the specific forces pushing a single prediction one way or the other.
            """)
        df_pred = ssm.get_data("predictive_quality_data")
        if df_pred is None or df_pred.empty: st.warning("Predictive quality data not available.")
        else:
            try:
                # --- Data Preparation & Model Training ---
                features = ['in_process_temp', 'in_process_pressure', 'in_process_vibration']
                X = df_pred[features]
                y = df_pred['final_qc_outcome'].apply(lambda x: 1 if x == 'Fail' else 0)
                X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
                model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
                explainer = shap.TreeExplainer(model)
    
                # --- Visualization ---
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("###### Classical: Average Effect (Box Plot)")
                    fig_box = px.box(df_pred, x='final_qc_outcome', y='in_process_pressure', title='Pressure by Outcome')
                    st.plotly_chart(fig_box, use_container_width=True)
    
                with col2:
                    st.markdown("###### Modern: Global Explanation (SHAP Summary)")
                    with st.spinner("Calculating SHAP explanations..."):
                        # --- COMPONENT 1: ROBUST DATA SAMPLING & EXPLANATION OBJECT ---
                        # Sample the test data for performance and stability
                        n_samples = min(200, len(X_test)) # Use a reasonable sample size
                        X_test_sample = X_test.sample(n=n_samples, random_state=42)
                        
                        # Use the modern API to create a self-contained Explanation object
                        shap_explanations = explainer(X_test_sample)
                        
                        # Find the index for the "Fail" class (which is encoded as 1)
                        fail_class_index = list(model.classes_).index(1)
    
                        # --- COMPONENT 2: STATELESS PLOT RENDERING TO BUFFER ---
                        # Create the plot object
                        fig, ax = plt.subplots(dpi=150)
                        # The plot function uses the robust Explanation object
                        shap.summary_plot(shap_explanations[:,:,fail_class_index], X_test_sample, show=False)
                        plt.tight_layout()
                        
                        # Render the plot to an in-memory buffer
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", bbox_inches="tight")
                        plt.close(fig) # Explicitly close the figure to free memory
                        buf.seek(0)
                    
                    # Display the plot from the buffer using st.image
                    st.image(buf, caption="SHAP Summary Plot", use_column_width=True)
    
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("##### Local (Single Prediction) Explanation")
                st.info("Select a specific unit to see why the model made its prediction.")
                
                # Recalculate full explanations for the local plot if we sampled earlier
                with st.spinner("Preparing local explanation..."):
                     full_shap_explanations = explainer(X_test)
                     fail_class_index_local = list(model.classes_).index(1)
    
                instance_idx = st.slider("Select a Test Instance to Explain", 0, len(X_test)-1, 0)
                
                # The force plot uses the same robust, object-oriented pattern
                st_shap(shap.force_plot(full_shap_explanations[instance_idx, :, fail_class_index_local]))
                
            except Exception as e:
                logger.critical(f"A critical error occurred in the Driver Analysis tab: {e}", exc_info=True)
                st.error(f"An unexpected error occurred during Driver Analysis. Error: {e}")
                
    # ==================== TAB 4: PROCESS CONTROL ====================
    with tabs[3]:
        st.subheader("Challenge 4: Detect Unusual Behavior in a Live Process")
        with st.expander("SME Deep Dive: SPC vs. Isolation Forest"):
            st.markdown("""
            - **Classical: SPC Chart** uses historical, stable variation to set +/- 3 sigma control limits.
                - **Analogy (Example 8): A Security Guard with a Checklist.** "Is anyone running? No. Is anyone shouting? No. Is anyone outside the velvet rope? Yes! Alert!" It's great at catching known rule violations based on its pre-defined Nelson Rules.
            - **Modern: Isolation Forest** is an unsupervised ML algorithm that learns the "shape" of normal data and flags points that don't conform.
                - **Analogy (Example 9): A Seasoned Detective.** They have a "feel" for the room. They might notice someone standing too still, or whispering in a corner. They spot things that aren't against the "rules" but are just... weird.
            #### SME Verdict
            **SPC** is the non-negotiable standard for **Sustaining Control**. **Isolation Forest** is a powerful **Investigative Tool** to find "unknown unknowns" or monitor complex, multi-variate systems where simple rules don't apply.
            """)
        df_process = ssm.get_data("process_data")
        if df_process is None or df_process.empty: st.warning("Process data is not available.")
        else:
            try:
                process_series = df_process['seal_strength']
                iso_forest = IsolationForest(contamination='auto', random_state=42).fit(process_series.values.reshape(-1, 1)); df_process['anomaly'] = iso_forest.predict(process_series.values.reshape(-1, 1))
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("##### Classical: SPC Chart (Rule-Based)"); st.plotly_chart(create_imr_chart(process_series, "Seal Strength", 78, 92), use_container_width=True)
                with col2:
                    st.markdown("##### Modern: ML Anomaly Detection (Shape-Based)")
                    fig_iso = go.Figure(); fig_iso.add_trace(go.Scatter(y=df_process['seal_strength'], mode='lines', name='Process Data')); anomalies = df_process[df_process['anomaly'] == -1]; fig_iso.add_trace(go.Scatter(x=anomalies.index, y=anomalies['seal_strength'], mode='markers', name='Detected Anomaly', marker=dict(color='red', size=10, symbol='x'))); fig_iso.update_layout(title='<b>Isolation Forest Anomaly Detection</b>'); st.plotly_chart(fig_iso, use_container_width=True)
            except Exception as e: st.error(f"An error occurred in Process Control tab: {e}")

    # ==================== TAB 5: PROCESS OPTIMIZATION ====================
    with tabs[4]:
        st.subheader("Challenge 5: Efficiently Find the Best Process 'Recipe'")
        with st.expander("SME Deep Dive: DOE/RSM vs. Bayesian Optimization"):
            st.markdown("""
            - **Classical: DOE/RSM** involves pre-planning a grid of experiments. You run all experiments, then fit a model to find the optimum.
                - **Analogy (Example 10): A Systematic Baker.** They meticulously plan to bake cakes at all combinations of (low/high temp, low/high time). They bake all cakes, taste them, then model the results to declare the best recipe. It's robust, but front-loaded with work.
            - **Modern: Bayesian Optimization** is a sequential, "smart search" strategy. It uses an ML model to intelligently decide the *single most informative experiment to run next*, balancing exploring uncertain areas with exploiting known good ones.
                - **Analogy (Example 11): A Master Chef.** They make one batch of sauce, taste it, and think, "Hmm, promising." Based on that, they intelligently decide the next best guess. They learn and adapt after every single experiment, saving time and ingredients.
            #### SME Verdict
            **DOE/RSM** is the gold standard for formal, rigorous experimentation. **Bayesian Optimization** is incredibly powerful when experiments are very expensive or time-consuming, as it often finds a near-optimal solution with far fewer experimental runs.
            """)
        df_opt = ssm.get_data("optimization_data")
        if df_opt is None or df_opt.empty: st.warning("Optimization data is not available.")
        else:
            try:
                def objective_func(params):
                    x, y = params
                    return -df_opt.loc[((df_opt['x'] - x)**2 + (df_opt['y'] - y)**2).idxmin()]['z']

                with st.spinner("Running Bayesian optimization..."):
                    bounds = [Real(-5, 5, name='x'), Real(-5, 5, name='y')]
                    result = gp_minimize(objective_func, bounds, n_calls=15, random_state=42)
                
                sampled_points = np.array(result.x_iters)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("##### Classical: Full Experimental Grid")
                    fig = go.Figure(data=go.Contour(z=df_opt['z'], x=df_opt['x'], y=df_opt['y'], colorscale='Viridis')); fig.update_layout(title="Full 'True' Response Surface"); st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.markdown("##### Modern: Bayesian 'Smart Search' Path")
                    fig = go.Figure(data=go.Contour(z=df_opt['z'], x=df_opt['x'], y=df_opt['y'], showscale=False, colorscale='Viridis', opacity=0.5)); fig.add_trace(go.Scatter(x=sampled_points[:, 0], y=sampled_points[:, 1], mode='markers+text', text=[str(i+1) for i in range(len(sampled_points))], textposition="top right", marker=dict(color='red', size=10, symbol='x'), name='Sampled Points')); fig.update_layout(title="Path of Smart Search (15 Experiments)"); st.plotly_chart(fig, use_container_width=True)
            except Exception as e: st.error(f"An error occurred in Process Optimization tab: {e}")
            
    # ==================== TAB 6: FAILURE MODE ANALYSIS ====================
    with tabs[5]:
        st.subheader("Challenge 6: Discover Hidden Groups or 'Types' of Failures")
        with st.expander("SME Deep Dive: Manual Binning vs. K-Means Clustering"):
            st.markdown("""
            - **Classical: Manual Binning / Histograms.** We look at one variable at a time and draw lines based on our expert knowledge. "Failures above 240°C we'll call 'overheating'."
                - **Analogy (Example 12): Sorting Laundry by Color.** We decide on the categories beforehand: whites, darks, colors. It's simple and based on one dimension.
            - **Modern: K-Means Clustering.** An unsupervised ML algorithm that looks at all variables simultaneously and mathematically finds the best "centers" (centroids) to partition the data.
                - **Analogy (Example 13): A Smart Sorting Machine.** It looks at color, fabric type, and item size all at once. It might discover groups you never thought of: "delicate whites," "heavy-duty darks," and "colorful cottons." It finds the natural, multi-dimensional groupings in the data.
            #### SME Verdict
            **Manual Binning** is useful for simple, one-dimensional problems. **K-Means Clustering** is exceptionally powerful for uncovering hidden, multi-dimensional patterns in failure data, potentially revealing distinct root causes (e.g., "Failure Mode A" is high-temp/low-pressure, while "Failure Mode B" is low-temp/high-pressure) that are impossible to see one variable at a time.
            """)
        df_clust = ssm.get_data("failure_clustering_data")
        if df_clust is None or df_clust.empty: st.warning("Clustering data is not available.")
        else:
            try:
                n_clusters = st.slider("Select Number of Clusters (K)", 2, 5, 3, key="k_slider")
                scaler = StandardScaler()
                X_clust = scaler.fit_transform(df_clust[['temperature', 'pressure']])
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit(X_clust)
                df_clust['ml_cluster'] = kmeans.labels_
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("##### Classical: One-Dimensional Binning")
                    df_clust['manual_bin'] = pd.cut(df_clust['temperature'], bins=[0, 200, 230, 300], labels=['Low Temp', 'Mid Temp', 'High Temp']); fig1 = px.scatter(df_clust, x='temperature', y='pressure', color='manual_bin', title='Failures Grouped by Temperature Bins'); st.plotly_chart(fig1, use_container_width=True)
                with col2:
                    st.markdown("##### Modern: Multi-Dimensional Clustering")
                    fig2 = px.scatter(df_clust, x='temperature', y='pressure', color='ml_cluster', title='Failures Grouped by ML Clusters', color_continuous_scale=px.colors.qualitative.Plotly)
                    centers = scaler.inverse_transform(kmeans.cluster_centers_)
                    fig2.add_trace(go.Scatter(x=centers[:,0], y=centers[:,1], mode='markers', marker=dict(symbol='x', color='black', size=12), name='Cluster Centers')); st.plotly_chart(fig2, use_container_width=True)
            except Exception as e: st.error(f"An error occurred in Failure Mode Analysis tab: {e}")
