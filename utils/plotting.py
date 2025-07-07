"""
Provides a suite of sophisticated, standardized plotting utilities for creating
expert-level statistical visualizations.

This module is the dedicated visualization engine for the Six Sigma Command Center.
It contains helper functions that generate various Plotly figures, ensuring a
consistent, professional, and publication-quality look and feel across the entire
application.

SME Overhaul:
- Architecturally decoupled from all calculation logic. Functions now take data
  and pre-calculated metrics as inputs, adhering to the Single Responsibility
  Principle.
- Upgraded the control chart to a standard I-MR (Individuals and Moving Range) chart.
- Massively enhanced the histogram to include a statistical summary box (with Ppk),
  and a normal distribution curve overlay.
- Refined all plots for superior aesthetics, clarity, and information density.
- Improved robustness to handle empty or invalid data gracefully.
"""

import logging
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.formula.api import ols
from scipy.stats import norm

logger = logging.getLogger(__name__)


def create_imr_chart(
    data_series: pd.Series, title: str, lsl: float, usl: float
) -> go.Figure:
    """
    Creates a high-quality Individual and Moving Range (I-MR) control chart.

    An I-MR chart is used for continuous data where observations are not in
    subgroups. It consists of two plots:
    1. The Individuals (I) chart: Plots individual data points.
    2. The Moving Range (MR) chart: Plots the range between consecutive points.

    Args:
        data_series (pd.Series): The series of individual measurements.
        title (str): The main title for the chart.
        lsl (float): The Lower Specification Limit for the I-chart.
        usl (float): The Upper Specification Limit for the I-chart.

    Returns:
        go.Figure: A Plotly figure object containing the I-MR chart.
    """
    if data_series.empty or len(data_series) < 2:
        fig = go.Figure()
        fig.update_layout(title_text=f"<b>I-MR Chart for {title}</b><br><sup>No data available</sup>", height=500)
        return fig

    # --- Calculations ---
    moving_range = abs(data_series.diff()).dropna()
    avg_moving_range = moving_range.mean()
    mean = data_series.mean()

    # Control Limits for I-Chart (based on Moving Range)
    ucl_i = mean + 3 * (avg_moving_range / 1.128)
    lcl_i = mean - 3 * (avg_moving_range / 1.128)

    # Control Limits for MR-Chart
    ucl_mr = avg_moving_range * 3.267
    lcl_mr = 0 # LCL for MR chart is always 0

    # --- Figure Creation ---
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3], vertical_spacing=0.05
    )

    # --- I-Chart (Top Plot) ---
    fig.add_trace(go.Scatter(y=data_series, mode='lines+markers', name='Measurements', line=dict(color='royalblue')), row=1, col=1)
    fig.add_hline(y=usl, line=dict(color="firebrick", width=2, dash="solid"), name="USL", annotation_text="USL", annotation_position="top right", row=1, col=1)
    fig.add_hline(y=lsl, line=dict(color="firebrick", width=2, dash="solid"), name="LSL", annotation_text="LSL", annotation_position="bottom right", row=1, col=1)
    fig.add_hline(y=ucl_i, line=dict(color="darkorange", width=2, dash="dash"), name="UCL", row=1, col=1)
    fig.add_hline(y=lcl_i, line=dict(color="darkorange", width=2, dash="dash"), name="LCL", row=1, col=1)
    fig.add_hline(y=mean, line=dict(color="darkgreen", width=1.5, dash="solid"), name="Mean", row=1, col=1)

    out_of_control = data_series[(data_series > ucl_i) | (data_series < lcl_i)]
    if not out_of_control.empty:
        fig.add_trace(go.Scatter(x=out_of_control.index, y=out_of_control, mode='markers', name='Out of Control', marker=dict(color='red', size=10, symbol='x')), row=1, col=1)

    # --- MR-Chart (Bottom Plot) ---
    fig.add_trace(go.Scatter(y=moving_range, mode='lines+markers', name='Moving Range', line=dict(color='grey')), row=2, col=1)
    fig.add_hline(y=ucl_mr, line=dict(color="darkorange", width=2, dash="dash"), name="UCL (MR)", row=2, col=1)
    fig.add_hline(y=avg_moving_range, line=dict(color="darkgreen", width=1.5, dash="solid"), name="Avg. MR", row=2, col=1)

    out_of_control_mr = moving_range[moving_range > ucl_mr]
    if not out_of_control_mr.empty:
        fig.add_trace(go.Scatter(x=out_of_control_mr.index, y=out_of_control_mr, mode='markers', name='Out of Control (MR)', marker=dict(color='red', size=8, symbol='x')), row=2, col=1)

    # --- Layout and Formatting ---
    fig.update_layout(
        title=f"<b>I-MR Control Chart for {title}</b>",
        yaxis_title="Measurement", yaxis2_title="Moving Range",
        xaxis2_title="Observation Number",
        showlegend=False, height=500, margin=dict(t=80, b=50, l=50, r=50)
    )
    return fig


def create_histogram_with_specs(
    data_series: pd.Series,
    lsl: float,
    usl: float,
    title: str,
    capability_metrics: Dict[str, float]
) -> go.Figure:
    """
    Creates an information-rich histogram with specification limits, a normal
    curve overlay, and a statistical summary box.

    Args:
        data_series (pd.Series): The measurement data.
        lsl (float): The Lower Specification Limit.
        usl (float): The Upper Specification Limit.
        title (str): The main title for the chart.
        capability_metrics (Dict[str, float]): A dictionary containing pre-calculated
                                              metrics like 'ppk' and 'sigma'.

    Returns:
        go.Figure: A Plotly figure object containing the histogram.
    """
    if data_series.empty:
        fig = go.Figure()
        fig.update_layout(title_text=f"<b>Process Distribution for {title}</b><br><sup>No data available</sup>")
        return fig

    mean = data_series.mean()
    std_dev = capability_metrics.get('sigma', data_series.std())

    fig = go.Figure()

    # Histogram trace
    fig.add_trace(go.Histogram(x=data_series, name='Frequency', histnorm='probability density', marker_color='#1f77b4', opacity=0.7))

    # Normal curve overlay
    x_norm = np.linspace(data_series.min(), data_series.max(), 200)
    y_norm = norm.pdf(x_norm, mean, std_dev)
    fig.add_trace(go.Scatter(x=x_norm, y=y_norm, mode='lines', name='Normal Fit', line=dict(color='darkgreen', width=2)))

    # Specification and Mean lines
    fig.add_vline(x=lsl, line=dict(color="firebrick", width=2, dash="dash"), annotation_text=f"LSL: {lsl}", annotation_position="top left")
    fig.add_vline(x=usl, line=dict(color="firebrick", width=2, dash="dash"), annotation_text=f"USL: {usl}", annotation_position="top right")
    fig.add_vline(x=mean, line=dict(color="black", width=1.5, dash="dot"), annotation_text=f"Mean: {mean:.3f}")

    # Statistical summary box annotation
    stats_text = (
        f"<b>Statistics</b><br>"
        f"Mean: {mean:.3f}<br>"
        f"Std Dev: {std_dev:.3f}<br>"
        f"N: {len(data_series)}<br>"
        f"<b>Ppk: {capability_metrics.get('ppk', 0):.2f}</b>"
    )
    fig.add_annotation(
        text=stats_text, align='left', showarrow=False,
        xref='paper', yref='paper', x=0.98, y=0.98,
        bgcolor='rgba(255, 255, 255, 0.7)', bordercolor='black', borderwidth=1
    )

    fig.update_layout(
        title=f"<b>Process Distribution for {title}</b>",
        xaxis_title="Measurement Value", yaxis_title="Density",
        showlegend=False, margin=dict(t=80)
    )
    return fig


def create_gage_rr_plots(df: pd.DataFrame) -> Tuple[go.Figure, go.Figure]:
    """
    Creates standard plots for a Gage R&R study: Variation by Operator and
    Part-Operator Interaction.

    Args:
        df (pd.DataFrame): The raw data from the Gage R&R study.

    Returns:
        Tuple[go.Figure, go.Figure]: A tuple containing the two Plotly figures.
    """
    if df.empty:
        fig1 = go.Figure().update_layout(title_text="<b>Variation by Operator</b><br><sup>No data available</sup>")
        fig2 = go.Figure().update_layout(title_text="<b>Part-Operator Interaction</b><br><sup>No data available</sup>")
        return fig1, fig2

    # Variation by Operator Box Plot
    fig1 = px.box(df, x='operator', y='measurement', color='operator',
                  title="<b>Measurement Variation by Operator</b>",
                  points="all", labels={'measurement': 'Measurement', 'operator': 'Operator'})
    fig1.update_traces(quartilemethod="exclusive")
    fig1.update_layout(showlegend=False)

    # Part-Operator Interaction Plot
    interaction_df = df.groupby(['operator', 'part_id'])['measurement'].mean().reset_index()
    fig2 = px.line(interaction_df, x='part_id', y='measurement', color='operator',
                   title="<b>Part-Operator Interaction Plot</b>",
                   markers=True, labels={'part_id': 'Part ID', 'measurement': 'Average Measurement'})
    fig2.update_traces(marker=dict(size=8))
    fig2.update_layout(legend_title_text='Operator')

    return fig1, fig2


def create_doe_plots(df: pd.DataFrame, factors: List[str], response: str) -> Dict[str, go.Figure]:
    """
    Creates a dictionary of standard DOE plots: Main Effects, Interaction, and
    a 3D Response Surface.

    Args:
        df (pd.DataFrame): The experimental design data.
        factors (List[str]): The names of the factor columns.
        response (str): The name of the response column.

    Returns:
        Dict[str, go.Figure]: A dictionary of Plotly figures keyed by plot type.
    """
    if df.empty or not all(f in df.columns for f in factors) or response not in df.columns:
        return {
            "main_effects": go.Figure().update_layout(title="Main Effects (No Data)"),
            "interaction": go.Figure().update_layout(title="Interaction (No Data)"),
            "surface": go.Figure().update_layout(title="Response Surface (No Data)")
        }

    # Main Effects Plot
    main_effects_fig = make_subplots(rows=1, cols=len(factors), subplot_titles=[f.title() for f in factors])
    for i, factor in enumerate(factors):
        effect_data = df.groupby(factor)[response].mean().reset_index()
        main_effects_fig.add_trace(go.Scatter(x=effect_data[factor], y=effect_data[response], mode='lines+markers', line=dict(color='#1f77b4')), row=1, col=i+1)
    main_effects_fig.update_layout(title_text="<b>Main Effects Plots</b>", showlegend=False, height=350)

    # Interaction Plots
    interaction_pairs = [('temp', 'time'), ('temp', 'pressure'), ('time', 'pressure')]
    interaction_fig = make_subplots(rows=1, cols=3, subplot_titles=[f"{p[0].title()}:{p[1].title()}" for p in interaction_pairs])
    for i, (f1, f2) in enumerate(interaction_pairs):
        interaction_df = df.groupby([f1, f2])[response].mean().reset_index()
        for level in interaction_df[f2].unique():
            subset = interaction_df[interaction_df[f2] == level]
            interaction_fig.add_trace(go.Scatter(x=subset[f1], y=subset[response], name=f'{f2}={level}', mode='lines+markers', legendgroup=f'group{i}'), row=1, col=i+1)
    interaction_fig.update_layout(title_text="<b>Interaction Plots</b>", height=350)

    # 3D Response Surface Plot
    surface_fig = go.Figure()
    try:
        # Fit a quadratic model for the surface
        formula = f"Q('{response}') ~ Q('{factors[0]}')*Q('{factors[1]}') + I(Q('{factors[0]}')**2) + I(Q('{factors[1]}')**2)"
        model = ols(formula, data=df).fit()
        f1_range = np.linspace(df[factors[0]].min(), df[factors[0]].max(), 30)
        f2_range = np.linspace(df[factors[1]].min(), df[factors[1]].max(), 30)
        grid_x, grid_y = np.meshgrid(f1_range, f2_range)
        grid_df = pd.DataFrame({factors[0]: grid_x.flatten(), factors[1]: grid_y.flatten()})
        grid_df['predicted'] = model.predict(grid_df)

        surface_fig.add_trace(go.Surface(z=grid_df['predicted'].values.reshape(grid_x.shape), x=grid_x, y=grid_y, colorscale='Viridis', opacity=0.9, name='Predicted Surface'))
        surface_fig.add_trace(go.Scatter3d(x=df[factors[0]], y=df[factors[1]], z=df[response], mode='markers', marker=dict(size=5, color='red', symbol='circle'), name='Actual Data Points'))
        surface_fig.update_layout(
            title=f'<b>Response Surface: {factors[0].title()} vs {factors[1].title()}</b>',
            scene=dict(xaxis_title=factors[0].title(), yaxis_title=factors[1].title(), zaxis_title=f'Predicted {response.title()}'),
            height=500, margin=dict(l=0, r=0, b=0, t=50)
        )
    except Exception as e:
        logger.error(f"Could not fit RSM model for 3D plot: {e}")
        surface_fig.update_layout(title="3D Response Surface (Model Fit Error)")

    return {"main_effects": main_effects_fig, "interaction": interaction_fig, "surface": surface_fig}
