# 6s/utils/plotting.py
"""
Utility functions for creating standardized statistical plots.

This module contains helper functions that generate various Plotly figures
used throughout the 6σ Quality Command Center, particularly within the
DMAIC and Product Release toolkits. This centralization ensures a consistent
and professional visual style for all data analysis.
"""

import logging
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

logger = logging.getLogger(__name__)

def create_control_chart(
    data_series: pd.Series,
    title: str,
    lsl: float = None,
    usl: float = None
) -> go.Figure:
    """
    Creates an Individual and Moving Range (I-MR) control chart.

    This is a standard SPC tool for monitoring the stability of a process over time.

    Args:
        data_series (pd.Series): A pandas Series of measurement data.
        title (str): The title for the chart.
        lsl (float, optional): Lower Specification Limit. Defaults to None.
        usl (float, optional): Upper Specification Limit. Defaults to None.

    Returns:
        go.Figure: A Plotly figure object containing the I-MR chart.
    """
    if data_series.empty:
        return go.Figure().update_layout(title="No data available for control chart.")

    # --- Individuals (I) Chart ---
    mean = data_series.mean()
    mr = abs(data_series.diff()).dropna()
    mr_mean = mr.mean()
    
    # Control Limits for I-Chart
    ucl_i = mean + 3 * (mr_mean / 1.128)
    lcl_i = mean - 3 * (mr_mean / 1.128)

    fig = go.Figure()
    # Add Spec Limits if they exist
    if usl:
        fig.add_hline(y=usl, line_dash="solid", line_color="red", line_width=2, name="USL",
                      annotation_text="USL", annotation_position="top right")
    if lsl:
        fig.add_hline(y=lsl, line_dash="solid", line_color="red", line_width=2, name="LSL",
                      annotation_text="LSL", annotation_position="bottom right")

    # Add Control Limits
    fig.add_hline(y=ucl_i, line_dash="dash", line_color="orange", name="UCL")
    fig.add_hline(y=lcl_i, line_dash="dash", line_color="orange", name="LCL")
    
    # Add Center Line
    fig.add_hline(y=mean, line_dash="solid", line_color="green", name="Mean")
    
    # Plot Data
    fig.add_trace(go.Scatter(
        y=data_series,
        mode='lines+markers',
        name='Measurements',
        line=dict(color='royalblue')
    ))

    # Identify out-of-control points
    out_of_control = data_series[(data_series > ucl_i) | (data_series < lcl_i)]
    if not out_of_control.empty:
        fig.add_trace(go.Scatter(
            x=out_of_control.index,
            y=out_of_control,
            mode='markers',
            name='Out of Control',
            marker=dict(color='red', size=10, symbol='x')
        ))

    fig.update_layout(
        title=f"<b>Control Chart (I-Chart) for {title}</b>",
        xaxis_title="Observation Number",
        yaxis_title="Measurement Value",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def create_histogram_with_specs(
    data_series: pd.Series,
    lsl: float,
    usl: float,
    title: str
) -> go.Figure:
    """
    Creates a histogram of the data distribution overlaid with specification limits.

    Args:
        data_series (pd.Series): A pandas Series of measurement data.
        lsl (float): Lower Specification Limit.
        usl (float): Upper Specification Limit.
        title (str): The title for the chart.

    Returns:
        go.Figure: A Plotly figure object.
    """
    if data_series.empty:
        return go.Figure().update_layout(title="No data available for histogram.")

    mean = data_series.mean()
    std = data_series.std()
    
    fig = px.histogram(
        data_series,
        nbins=40,
        title=f"<b>Process Distribution for {title}</b>"
    )

    fig.add_vline(x=lsl, line_dash="dash", line_color="red", annotation_text=f"LSL: {lsl}")
    fig.add_vline(x=usl, line_dash="dash", line_color="red", annotation_text=f"USL: {usl}")
    fig.add_vline(x=mean, line_dash="dot", line_color="black", annotation_text=f"Mean: {mean:.2f}")

    fig.update_layout(
        xaxis_title="Measurement Value",
        yaxis_title="Frequency",
        showlegend=False
    )
    return fig
