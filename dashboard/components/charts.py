"""
Reusable chart components for the dashboard.

Provides functions to create common visualizations using Plotly.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional


def create_win_probability_gauge(probability: float, team_name: str) -> go.Figure:
    """
    Create a gauge chart showing win probability.

    Args:
        probability: Win probability (0-1)
        team_name: Team name for title

    Returns:
        Plotly figure
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{team_name} Win Probability"},
        delta={'reference': 50, 'suffix': '%'},
        gauge={
            'axis': {'range': [None, 100], 'ticksuffix': '%'},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "lightgray"},
                {'range': [40, 60], 'color': "gray"},
                {'range': [60, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))

    fig.update_layout(height=300)
    return fig


def create_accuracy_trend(df: pd.DataFrame) -> go.Figure:
    """
    Create a line chart showing accuracy over time.

    Args:
        df: DataFrame with 'date' and 'accuracy' columns

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['accuracy'],
        mode='lines+markers',
        name='Accuracy',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))

    # Add 50% reference line
    fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                  annotation_text="Random (50%)")

    fig.update_layout(
        title="Model Accuracy Over Time",
        xaxis_title="Date",
        yaxis_title="Accuracy",
        yaxis=dict(tickformat='.1%', range=[0, 1]),
        hovermode='x unified',
        height=400
    )

    return fig


def create_calibration_plot(df: pd.DataFrame) -> go.Figure:
    """
    Create a calibration plot (predicted vs actual probabilities).

    Args:
        df: DataFrame with 'predicted_prob' and 'won' columns

    Returns:
        Plotly figure
    """
    # Bin predictions
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    df['bin'] = pd.cut(df['predicted_prob'], bins=bins, labels=bin_centers)

    # Calculate actual win rate per bin
    calibration = df.groupby('bin', observed=True).agg({
        'won': ['mean', 'count']
    }).reset_index()

    calibration.columns = ['predicted', 'actual', 'count']

    fig = go.Figure()

    # Calibration curve
    fig.add_trace(go.Scatter(
        x=calibration['predicted'],
        y=calibration['actual'],
        mode='markers+lines',
        name='Model Calibration',
        marker=dict(size=calibration['count'] / 2, color='blue'),
        line=dict(color='blue', width=2)
    ))

    # Perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Perfect Calibration',
        line=dict(color='red', dash='dash', width=2)
    ))

    fig.update_layout(
        title="Calibration Plot (Predicted vs Actual Win Rate)",
        xaxis_title="Predicted Probability",
        yaxis_title="Actual Win Rate",
        xaxis=dict(tickformat='.0%', range=[0, 1]),
        yaxis=dict(tickformat='.0%', range=[0, 1]),
        height=500
    )

    return fig


def create_roi_by_confidence(df: pd.DataFrame) -> go.Figure:
    """
    Create a bar chart showing ROI by confidence bucket.

    Args:
        df: DataFrame with betting results

    Returns:
        Plotly figure
    """
    # Create confidence buckets
    df['confidence_bucket'] = pd.cut(
        df['predicted_prob'],
        bins=[0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        labels=['50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
    )

    # Group by confidence bucket
    roi_by_conf = df.groupby('confidence_bucket', observed=True).agg({
        'won': ['mean', 'count']
    }).reset_index()

    roi_by_conf.columns = ['confidence', 'win_rate', 'count']

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=roi_by_conf['confidence'],
        y=roi_by_conf['win_rate'],
        text=roi_by_conf['count'].apply(lambda x: f'n={x}'),
        textposition='outside',
        marker_color='lightblue'
    ))

    fig.update_layout(
        title="Win Rate by Confidence Level",
        xaxis_title="Confidence Bucket",
        yaxis_title="Win Rate",
        yaxis=dict(tickformat='.1%'),
        height=400
    )

    return fig


def create_team_comparison(team_stats: dict) -> go.Figure:
    """
    Create a radar chart comparing team statistics.

    Args:
        team_stats: Dictionary with team statistics

    Returns:
        Plotly figure
    """
    categories = list(team_stats.keys())
    values = list(team_stats.values())

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Team Stats'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values) * 1.1]
            )
        ),
        showlegend=False,
        height=400
    )

    return fig


def create_betting_history_table(df: pd.DataFrame) -> go.Figure:
    """
    Create an interactive table of betting history.

    Args:
        df: DataFrame with betting history

    Returns:
        Plotly figure
    """
    # Select relevant columns
    columns = ['date', 'team', 'opponent', 'predicted_prob', 'won', 'recommended_stake']
    display_df = df[columns].copy()

    # Format columns
    display_df['predicted_prob'] = display_df['predicted_prob'].apply(lambda x: f'{x:.1%}')
    display_df['won'] = display_df['won'].apply(lambda x: '✅' if x == 1 else '❌')
    display_df['recommended_stake'] = display_df['recommended_stake'].apply(lambda x: f'${x:.2f}')

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Date', 'Team', 'Opponent', 'Win Prob', 'Result', 'Stake'],
            fill_color='paleturquoise',
            align='left',
            font=dict(size=12, color='black')
        ),
        cells=dict(
            values=[display_df[col] for col in display_df.columns],
            fill_color='lavender',
            align='left',
            font=dict(size=11)
        )
    )])

    fig.update_layout(height=500)
    return fig


def create_odds_comparison(model_prob: float, market_prob: float, team_name: str) -> go.Figure:
    """
    Create a bar chart comparing model odds vs market odds.

    Args:
        model_prob: Model's predicted probability
        market_prob: Market implied probability
        team_name: Team name

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=['Model Probability', 'Market Probability'],
        y=[model_prob * 100, market_prob * 100],
        marker_color=['blue', 'orange'],
        text=[f'{model_prob:.1%}', f'{market_prob:.1%}'],
        textposition='outside'
    ))

    edge = model_prob - market_prob
    edge_color = 'green' if edge > 0 else 'red'

    fig.update_layout(
        title=f"{team_name} - Model vs Market",
        yaxis_title="Probability (%)",
        yaxis=dict(range=[0, 100]),
        annotations=[
            dict(
                x=0.5,
                y=max(model_prob, market_prob) * 100 + 10,
                text=f"Edge: {edge:+.1%}",
                showarrow=False,
                font=dict(size=14, color=edge_color, weight='bold')
            )
        ],
        height=350
    )

    return fig


def create_profit_curve(df: pd.DataFrame) -> go.Figure:
    """
    Create a cumulative profit curve over time.

    Args:
        df: DataFrame with betting results and stakes

    Returns:
        Plotly figure
    """
    # Calculate profit per bet (simplified)
    df = df.copy()
    df['profit'] = df.apply(
        lambda row: row['recommended_stake'] * 0.9 if row['won'] == 1 else -row['recommended_stake'],
        axis=1
    )
    df['cumulative_profit'] = df['profit'].cumsum()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['cumulative_profit'],
        mode='lines',
        name='Cumulative Profit',
        line=dict(color='green', width=2),
        fill='tozeroy'
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="red")

    fig.update_layout(
        title="Cumulative Profit/Loss Over Time",
        xaxis_title="Bet Number",
        yaxis_title="Profit ($)",
        hovermode='x unified',
        height=400
    )

    return fig
