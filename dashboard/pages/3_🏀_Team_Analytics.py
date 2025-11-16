"""
Team Analytics Page

Analyzes performance by team, showing team-specific statistics and trends.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import load_all_predictions, load_historical_games, get_team_stats

st.set_page_config(page_title="Team Analytics", page_icon="🏀", layout="wide")

st.title("🏀 Team Analytics")

# Load data
all_predictions = load_all_predictions(limit=50)
historical_games = load_historical_games(days_back=60)

if all_predictions is None:
    st.warning("⚠️ No prediction data available.")
    st.stop()

# Get list of teams
teams = sorted(all_predictions['team'].unique()) if 'team' in all_predictions.columns else []

if not teams:
    st.error("No team data found in predictions.")
    st.stop()

st.sidebar.header("🔍 Select Team")
selected_team = st.sidebar.selectbox("Choose a team", teams)

st.header(f"Analysis for {selected_team}")

# Filter data for selected team
team_preds = all_predictions[all_predictions['team'] == selected_team].copy()

# Overview metrics
st.markdown("## 📊 Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_games = len(team_preds)
    st.metric("Total Predictions", total_games)

with col2:
    if 'won' in team_preds.columns:
        wins = team_preds['won'].sum()
        win_rate = wins / total_games if total_games > 0 else 0
        st.metric("Wins", f"{int(wins)}", delta=f"{win_rate:.1%} win rate")
    else:
        st.metric("Wins", "N/A")

with col3:
    if 'home' in team_preds.columns:
        home_games = (team_preds['home'] == 1).sum()
        st.metric("Home Games", home_games)
    else:
        st.metric("Home Games", "N/A")

with col4:
    if 'predicted_prob' in team_preds.columns:
        avg_prob = team_preds['predicted_prob'].mean()
        st.metric("Avg Win Probability", f"{avg_prob:.1%}")
    else:
        st.metric("Avg Win Probability", "N/A")

st.markdown("---")

# Home vs Away Performance
st.markdown("## 🏠 Home vs Away Performance")

if 'home' in team_preds.columns and 'won' in team_preds.columns:
    home_away_stats = team_preds.groupby('home').agg({
        'won': ['count', 'sum', 'mean']
    }).round(3)

    home_away_stats.columns = ['Games', 'Wins', 'Win Rate']

    # Create a nice display
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏠 Home Games")
        if 1 in home_away_stats.index:
            st.metric("Games", int(home_away_stats.loc[1, 'Games']))
            st.metric("Wins", int(home_away_stats.loc[1, 'Wins']))
            st.metric("Win Rate", f"{home_away_stats.loc[1, 'Win Rate']:.1%}")
        else:
            st.info("No home games data")

    with col2:
        st.subheader("✈️ Away Games")
        if 0 in home_away_stats.index:
            st.metric("Games", int(home_away_stats.loc[0, 'Games']))
            st.metric("Wins", int(home_away_stats.loc[0, 'Wins']))
            st.metric("Win Rate", f"{home_away_stats.loc[0, 'Win Rate']:.1%}")
        else:
            st.info("No away games data")

else:
    st.info("Home/Away data not available")

st.markdown("---")

# Recent performance trend
st.markdown("## 📈 Recent Performance Trend")

if 'prediction_date' in team_preds.columns and 'won' in team_preds.columns:
    # Sort by date
    team_preds_sorted = team_preds.sort_values('prediction_date')

    # Calculate rolling win rate
    if len(team_preds_sorted) >= 5:
        team_preds_sorted['rolling_win_rate'] = team_preds_sorted['won'].rolling(window=5, min_periods=1).mean()

        import plotly.graph_objects as go

        fig = go.Figure()

        # Add scatter for individual games
        fig.add_trace(go.Scatter(
            x=team_preds_sorted['prediction_date'],
            y=team_preds_sorted['won'],
            mode='markers',
            name='Game Result',
            marker=dict(size=10, color=team_preds_sorted['won'], colorscale='RdYlGn'),
            hovertemplate='<b>Date:</b> %{x}<br><b>Result:</b> %{y}<extra></extra>'
        ))

        # Add rolling average line
        fig.add_trace(go.Scatter(
            x=team_preds_sorted['prediction_date'],
            y=team_preds_sorted['rolling_win_rate'],
            mode='lines',
            name='5-Game Rolling Win Rate',
            line=dict(color='blue', width=3)
        ))

        fig.update_layout(
            title=f"{selected_team} Performance Over Time",
            xaxis_title="Date",
            yaxis_title="Win Rate",
            yaxis=dict(tickformat='.0%', range=[-0.1, 1.1]),
            hovermode='x unified',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough games for trend analysis (minimum 5 games required)")

else:
    st.info("Date and outcome data not available for trend analysis")

st.markdown("---")

# Performance vs different opponents
st.markdown("## 🎯 Performance by Opponent")

if 'team_opp' in team_preds.columns and 'won' in team_preds.columns:
    opp_stats = team_preds.groupby('team_opp').agg({
        'won': ['count', 'sum', 'mean']
    }).round(3)

    opp_stats.columns = ['Games', 'Wins', 'Win Rate']
    opp_stats = opp_stats.sort_values('Win Rate', ascending=False)

    # Show top and bottom opponents
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ✅ Best Matchups")
        top_opps = opp_stats.head(5)
        if len(top_opps) > 0:
            for idx, row in top_opps.iterrows():
                st.write(f"**{idx}**: {int(row['Wins'])}-{int(row['Games'] - row['Wins'])} ({row['Win Rate']:.1%})")
        else:
            st.info("Not enough data")

    with col2:
        st.markdown("### ⚠️ Toughest Matchups")
        bottom_opps = opp_stats.tail(5)
        if len(bottom_opps) > 0:
            for idx, row in bottom_opps.iterrows():
                st.write(f"**{idx}**: {int(row['Wins'])}-{int(row['Games'] - row['Wins'])} ({row['Win Rate']:.1%})")
        else:
            st.info("Not enough data")

    # Full opponent breakdown
    with st.expander("📋 Full Opponent Breakdown"):
        opp_stats_display = opp_stats.copy()
        opp_stats_display['Win Rate'] = opp_stats_display['Win Rate'].apply(lambda x: f'{x:.1%}')
        opp_stats_display['Games'] = opp_stats_display['Games'].astype(int)
        opp_stats_display['Wins'] = opp_stats_display['Wins'].astype(int)
        st.dataframe(opp_stats_display, use_container_width=True)

else:
    st.info("Opponent data not available")

st.markdown("---")

# Model confidence analysis
st.markdown("## 🎲 Model Confidence Analysis")

if 'predicted_prob' in team_preds.columns and 'won' in team_preds.columns:
    # Create confidence buckets
    team_preds['confidence'] = pd.cut(
        team_preds['predicted_prob'],
        bins=[0, 0.5, 0.6, 0.7, 0.8, 1.0],
        labels=['50-60%', '60-70%', '70-80%', '80%+']
    )

    conf_stats = team_preds.groupby('confidence', observed=True).agg({
        'won': ['count', 'mean']
    }).round(3)

    conf_stats.columns = ['Games', 'Win Rate']

    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=conf_stats.index.astype(str),
        y=conf_stats['Win Rate'],
        text=conf_stats['Games'].apply(lambda x: f'n={int(x)}'),
        textposition='outside',
        marker_color='lightblue'
    ))

    fig.update_layout(
        title=f"{selected_team} - Win Rate by Model Confidence",
        xaxis_title="Predicted Win Probability",
        yaxis_title="Actual Win Rate",
        yaxis=dict(tickformat='.0%'),
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Interpretation**: When the model is very confident (80%+), how often does the team actually win?
    A well-calibrated model should show higher actual win rates at higher confidence levels.
    """)

else:
    st.info("Confidence analysis requires predicted probabilities and outcomes")

st.markdown("---")

# Recent games table
st.markdown("## 📋 Recent Games")

if len(team_preds) > 0:
    # Sort by date descending
    recent = team_preds.sort_values('prediction_date', ascending=False).head(10) if 'prediction_date' in team_preds.columns else team_preds.head(10)

    # Select and format columns
    display_cols = ['prediction_date', 'team_opp', 'home', 'predicted_prob', 'won']
    display_cols = [col for col in display_cols if col in recent.columns]

    if display_cols:
        recent_display = recent[display_cols].copy()

        # Format columns
        if 'home' in recent_display.columns:
            recent_display['home'] = recent_display['home'].apply(lambda x: '🏠' if x == 1 else '✈️')
        if 'predicted_prob' in recent_display.columns:
            recent_display['predicted_prob'] = recent_display['predicted_prob'].apply(lambda x: f'{x:.1%}')
        if 'won' in recent_display.columns:
            recent_display['won'] = recent_display['won'].apply(lambda x: '✅ W' if x == 1 else '❌ L')

        # Rename columns for display
        column_names = {
            'prediction_date': 'Date',
            'team_opp': 'Opponent',
            'home': 'H/A',
            'predicted_prob': 'Win Prob',
            'won': 'Result'
        }
        recent_display = recent_display.rename(columns=column_names)

        st.dataframe(recent_display, use_container_width=True, hide_index=True)

    else:
        st.info("Limited data available for display")

else:
    st.info("No recent games data available")
