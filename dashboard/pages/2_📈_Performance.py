"""
Performance Analytics Page

Shows model performance metrics, accuracy trends, and calibration analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import (
    load_all_predictions,
    load_betting_statistics,
    calculate_model_metrics
)
from components.charts import (
    create_calibration_plot,
    create_roi_by_confidence,
    create_profit_curve
)

st.set_page_config(page_title="Performance Analytics", page_icon="📈", layout="wide")

st.title("📈 Model Performance Analytics")

# Load data
all_predictions = load_all_predictions(limit=50)
betting_stats = load_betting_statistics()

if all_predictions is None:
    st.warning("⚠️ No historical prediction data available.")
    st.info("""
    **Note:** Performance metrics require historical predictions with actual game results.

    Keep running the prediction scripts daily to build up historical data!
    """)
    st.stop()

st.success(f"✅ Loaded {len(all_predictions)} historical predictions")

# Calculate overall metrics
if 'won' in all_predictions.columns and 'predicted_prob' in all_predictions.columns:
    metrics = calculate_model_metrics(all_predictions)

    st.markdown("## 🎯 Overall Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        accuracy = metrics.get('accuracy', 0)
        st.metric(
            "Accuracy",
            f"{accuracy:.1%}",
            delta=f"{(accuracy - 0.5):.1%} vs random"
        )

    with col2:
        brier = metrics.get('brier_score', 0)
        st.metric(
            "Brier Score",
            f"{brier:.4f}",
            delta="Lower is better",
            delta_color="inverse"
        )

    with col3:
        logloss = metrics.get('log_loss', 0)
        st.metric(
            "Log Loss",
            f"{logloss:.4f}",
            delta="Lower is better",
            delta_color="inverse"
        )

    with col4:
        total_bets = metrics.get('total_bets', 0)
        st.metric("Total Predictions", total_bets)

    st.markdown("---")

    # Calibration Plot
    st.markdown("## 📊 Calibration Analysis")
    st.markdown("""
    The calibration plot shows how well the model's predicted probabilities match actual outcomes.
    A well-calibrated model's predictions should align with the diagonal line.
    """)

    try:
        calib_fig = create_calibration_plot(all_predictions)
        st.plotly_chart(calib_fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating calibration plot: {e}")

    st.markdown("---")

    # Performance by confidence level
    st.markdown("## 🎲 Performance by Confidence Level")
    st.markdown("Win rate broken down by how confident the model was in its predictions.")

    try:
        roi_fig = create_roi_by_confidence(all_predictions)
        st.plotly_chart(roi_fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating ROI chart: {e}")

    # Detailed breakdown table
    st.markdown("### Detailed Breakdown")

    all_predictions['confidence_bucket'] = pd.cut(
        all_predictions['predicted_prob'],
        bins=[0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        labels=['50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
    )

    breakdown = all_predictions.groupby('confidence_bucket', observed=True).agg({
        'won': ['count', 'sum', 'mean'],
        'predicted_prob': 'mean'
    }).round(3)

    breakdown.columns = ['Games', 'Wins', 'Win Rate', 'Avg Predicted']
    breakdown['Win Rate'] = breakdown['Win Rate'].apply(lambda x: f'{x:.1%}')
    breakdown['Avg Predicted'] = breakdown['Avg Predicted'].apply(lambda x: f'{x:.1%}')

    st.dataframe(breakdown, use_container_width=True)

else:
    st.warning("⚠️ Prediction data doesn't contain both 'won' and 'predicted_prob' columns needed for analysis.")

st.markdown("---")

# Accuracy over time
st.markdown("## 📅 Performance Over Time")

if 'prediction_date' in all_predictions.columns and 'won' in all_predictions.columns:
    # Calculate daily accuracy
    daily_acc = all_predictions.groupby('prediction_date').agg({
        'won': 'mean',
        'predicted_prob': 'count'
    }).reset_index()
    daily_acc.columns = ['date', 'accuracy', 'game_count']
    daily_acc = daily_acc.sort_values('date')

    # Simple line chart
    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=daily_acc['date'],
        y=daily_acc['accuracy'],
        mode='lines+markers',
        name='Daily Accuracy',
        line=dict(color='blue', width=2),
        marker=dict(size=8),
        hovertemplate='<b>Date:</b> %{x}<br><b>Accuracy:</b> %{y:.1%}<extra></extra>'
    ))

    fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                  annotation_text="Random (50%)")

    fig.update_layout(
        title="Model Accuracy Over Time",
        xaxis_title="Date",
        yaxis_title="Accuracy",
        yaxis=dict(tickformat='.0%', range=[0, 1]),
        hovermode='x unified',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show rolling average
    st.markdown("### Rolling 7-Day Average")

    if len(daily_acc) >= 7:
        daily_acc['rolling_acc'] = daily_acc['accuracy'].rolling(window=7, min_periods=1).mean()

        fig2 = go.Figure()

        fig2.add_trace(go.Scatter(
            x=daily_acc['date'],
            y=daily_acc['rolling_acc'],
            mode='lines',
            name='7-Day Rolling Average',
            line=dict(color='green', width=3),
            fill='tozeroy'
        ))

        fig2.add_hline(y=0.5, line_dash="dash", line_color="red")

        fig2.update_layout(
            title="7-Day Rolling Average Accuracy",
            xaxis_title="Date",
            yaxis_title="Accuracy",
            yaxis=dict(tickformat='.0%', range=[0, 1]),
            height=350
        )

        st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("📊 Time series analysis requires prediction_date field in the data.")

st.markdown("---")

# Model interpretation
st.markdown("## 🧠 Model Insights")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### ✅ Strengths")
    st.markdown("""
    - **LightGBM Algorithm**: Gradient boosting for high accuracy
    - **Rolling Features**: 9-game moving averages capture recent form
    - **Probability Calibration**: Platt + Isotonic scaling for reliable probabilities
    - **Home Court Advantage**: Home/away status as a feature
    """)

with col2:
    st.markdown("### ⚠️ Limitations")
    st.markdown("""
    - **No Injury Data**: Player availability not factored in
    - **No Roster Changes**: Trades/signings not reflected immediately
    - **Schedule Difficulty**: Strength of schedule not explicitly modeled
    - **Rest Days**: Back-to-back games impact not fully captured
    """)

# Data quality checks
st.markdown("---")
st.markdown("## 🔍 Data Quality")

quality_col1, quality_col2, quality_col3 = st.columns(3)

with quality_col1:
    missing_won = all_predictions['won'].isna().sum() if 'won' in all_predictions.columns else len(all_predictions)
    pct_complete = (1 - missing_won / len(all_predictions)) * 100
    st.metric("Data Completeness", f"{pct_complete:.1f}%")

with quality_col2:
    unique_teams = all_predictions['team'].nunique() if 'team' in all_predictions.columns else 0
    st.metric("Teams Covered", unique_teams)

with quality_col3:
    date_range = "N/A"
    if 'prediction_date' in all_predictions.columns:
        dates = pd.to_datetime(all_predictions['prediction_date'], errors='coerce')
        date_range = f"{dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}"
    st.metric("Date Range", date_range)

st.caption("Performance metrics are calculated from historical predictions where actual game outcomes are available.")
