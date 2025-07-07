# six_sigma/utils/plotting.py
"""
Utility functions for creating standardized, expert-level statistical plots.

This module contains helper functions that generate various Plotly figures
used throughout the Six Sigma Command Center. It has been overhauled to
produce sophisticated, publication-quality visualizations for Gage R&R,
DOE, and other advanced statistical methods.
"""

import logging
from typing import Tuple, List # <--- FIX: Added missing import
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.formula.api import ols

logger = logging.getLogger(__name__)

def create_control_chart(
    data_series: pd.Series, title: str, lsl: float, usl: float
) -> go.Figure:
    """Creates an Individual (I) control chart."""
    if data_series.empty or len(data_series) < 2:
        return go.Figure().update_layout(title="No data available for control chart.")

    mean = data_series.mean()
    mr = abs(data_series.diff()).dropna()
    if mr.empty: mr_mean = np.nan
    else: mr_mean = mr.mean()
    
    if pd.isna(mr_mean) or mr_mean == 0:
        ucl_i, lcl_i = np.nan, np.nan
    else:
        ucl_i, lcl_i = mean + 3 * (mr_mean / 1.128), mean - 3 * (mr_mean / 1.128)

    fig = go.Figure()
    fig.add_hline(y=usl, line=dict(color="red", width=2, dash="solid"), name="USL", annotation_text="USL", annotation_position="top right")
    fig.add_hline(y=lsl, line=dict(color="red", width=2, dash="solid"), name="LSL", annotation_text="LSL", annotation_position="bottom right")
    if not pd.isna(ucl_i):
        fig.add_hline(y=ucl_i, line=dict(color="orange", width=2, dash="dash"), name="UCL")
        fig.add_hline(y=lcl_i, line=dict(color="orange", width=2, dash="dash"), name="LCL")
    fig.add_hline(y=mean, line=dict(color="green", width=2, dash="solid"), name="Mean")
    
    fig.add_trace(go.Scatter(y=data_series, mode='lines+markers', name='Measurements', line=dict(color='royalblue')))
    
    if not pd.isna(ucl_i):
        out_of_control = data_series[(data_series > ucl_i) | (data_series < lcl_i)]
        if not out_of_control.empty:
            fig.add_trace(go.Scatter(x=out_of_control.index, y=out_of_control, mode='markers', name='Out of Control', marker=dict(color='red', size=10, symbol='x')))

    fig.update_layout(title=f"<b>Control Chart (I-Chart) for {title}</b>", xaxis_title="Observation Number", yaxis_title="Measurement Value", showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

def create_histogram_with_specs(
    data_series: pd.Series, lsl: float, usl: float, title: str
) -> go.Figure:
    """Creates a histogram overlaid with specification limits."""
    if data_series.empty: return go.Figure().update_layout(title="No data available for histogram.")
    
    mean = data_series.mean()
    fig = px.histogram(data_series, nbins=40, title=f"<b>Process Distribution for {title}</b>", marginal="box")
    fig.add_vline(x=lsl, line=dict(color="red", width=2, dash="dash"), annotation_text=f"LSL: {lsl}")
    fig.add_vline(x=usl, line=dict(color="red", width=2, dash="dash"), annotation_text=f"USL: {usl}")
    fig.add_vline(x=mean, line=dict(color="black", width=2, dash="dot"), annotation_text=f"Mean: {mean:.3f}")
    fig.update_layout(xaxis_title="Measurement Value", yaxis_title="Frequency", showlegend=False)
    return fig

def create_gage_rr_plots(df: pd.DataFrame) -> Tuple[go.Figure, go.Figure]:
    """Creates standard plots for a Gage R&R study: variation breakdown and interaction."""
    fig1 = px.box(df, x='operator', y='measurement', title="<b>Measurement Variation by Operator</b>", points="all")
    fig1.update_traces(quartilemethod="exclusive")
    
    interaction_df = df.groupby(['operator', 'part_id'])['measurement'].mean().reset_index()
    fig2 = px.line(interaction_df, x='part_id', y='measurement', color='operator',
                   title="<b>Part-Operator Interaction Plot</b>", markers=True)
    fig2.update_traces(marker=dict(size=8))
    fig2.update_layout(xaxis_title="Part ID", yaxis_title="Average Measurement")
    return fig1, fig2

def create_doe_plots(df: pd.DataFrame, factors: List[str], response: str) -> Tuple[go.Figure, go.Figure, go.Figure]:
    """Creates Main Effects, Interaction, and 3D Surface plots for a DOE."""
    main_effects_fig = make_subplots(rows=1, cols=len(factors), subplot_titles=[f.title() for f in factors])
    for i, factor in enumerate(factors):
        effect_data = df.groupby(factor)[response].mean().reset_index()
        main_effects_fig.add_trace(go.Scatter(x=effect_data[factor], y=effect_data[response], mode='lines+markers'), row=1, col=i+1)
    main_effects_fig.update_layout(title_text="<b>Main Effects Plots</b>", showlegend=False, height=350)

    interaction_fig = make_subplots(rows=1, cols=3, subplot_titles=("Temp:Time", "Temp:Pressure", "Time:Pressure"))
    interaction_pairs = [('temp', 'time'), ('temp', 'pressure'), ('time', 'pressure')]
    for i, (f1, f2) in enumerate(interaction_pairs):
        interaction_df = df.groupby([f1, f2])[response].mean().reset_index()
        for level in interaction_df[f2].unique():
            subset = interaction_df[interaction_df[f2] == level]
            interaction_fig.add_trace(go.Scatter(x=subset[f1], y=subset[response], name=f'{f2}={level}', mode='lines+markers', legendgroup=f'group{i}'), row=1, col=i+1)
    interaction_fig.update_layout(title_text="<b>Interaction Plots</b>", height=350)

    formula = f"Q('{response}') ~ Q('{factors[0]}')*Q('{factors[1]}') + I(Q('{factors[0]}')**2) + I(Q('{factors[1]}')**2)"
    try:
        model = ols(formula, data=df).fit()
        f1_range = np.linspace(df[factors[0]].min(), df[factors[0]].max(), 20)
        f2_range = np.linspace(df[factors[1]].min(), df[factors[1]].max(), 20)
        grid_x, grid_y = np.meshgrid(f1_range, f2_range)
        grid_df = pd.DataFrame({factors[0]: grid_x.flatten(), factors[1]: grid_y.flatten()})
        grid_df['predicted'] = model.predict(grid_df)
        
        surface_fig = go.Figure(data=[go.Surface(z=grid_df['predicted'].values.reshape(grid_x.shape), x=grid_x, y=grid_y, colorscale='Viridis')])
        surface_fig.add_trace(go.Scatter3d(x=df[factors[0]], y=df[factors[1]], z=df[response], mode='markers', marker=dict(size=5, color='red')))
        surface_fig.update_layout(title=f'<b>Response Surface: {factors[0].title()} vs {factors[1].title()}</b>', scene=dict(xaxis_title=factors[0], yaxis_title=factors[1], zaxis_title=f'Predicted {response}'), height=500)
    except Exception as e:
        logger.error(f"Could not fit RSM model for 3D plot: {e}")
        surface_fig = go.Figure().update_layout(title="3D Response Surface (Model Fit Error)")
    
    return main_effects_fig, interaction_fig, surface_fig
