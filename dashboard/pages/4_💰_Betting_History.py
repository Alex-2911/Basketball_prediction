"""
Betting History Page

Shows historical betting recommendations, profitability analysis, and ROI tracking.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import load_all_predictions, load_enriched_predictions

st.set_page_config(page_title="Betting History", page_icon="💰", layout="wide")

st.title("💰 Betting History & ROI Analysis")

# Load data
all_predictions = load_all_predictions(limit=100)
enriched = load_enriched_predictions()

if all_predictions is None:
    st.warning("⚠️ No betting history available.")
    st.info("Run predictions over time to build up betting history!")
    st.stop()

# Filter for games where we have both predictions and outcomes
if 'won' in all_predictions.columns and 'recommended_stake' in all_predictions.columns:
    betting_history = all_predictions[all_predictions['recommended_stake'] > 0].copy()
else:
    betting_history = all_predictions.copy()

st.success(f"✅ Loaded {len(betting_history)} betting opportunities")

# Overall ROI metrics
st.markdown("## 💵 Overall Performance")

if len(betting_history) > 0 and 'won' in betting_history.columns and 'recommended_stake' in betting_history.columns:
    # Calculate profit/loss (simplified: assume -110 odds for all bets)
    # In reality, would use actual odds from the data
    betting_history['profit'] = betting_history.apply(
        lambda row: row['recommended_stake'] * 0.91 if row['won'] == 1 else -row['recommended_stake'],
        axis=1
    )

    betting_history['cumulative_profit'] = betting_history['profit'].cumsum()

    total_staked = betting_history['recommended_stake'].sum()
    total_profit = betting_history['profit'].sum()
    roi = (total_profit / total_staked * 100) if total_staked > 0 else 0
    win_rate = betting_history['won'].mean()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Bets", len(betting_history))

    with col2:
        st.metric("Total Staked", f"${total_staked:.2f}")

    with col3:
        profit_color = "normal" if total_profit > 0 else "inverse"
        st.metric("Total Profit", f"${total_profit:.2f}", delta=f"{roi:.1f}% ROI", delta_color=profit_color)

    with col4:
        st.metric("Win Rate", f"{win_rate:.1%}", delta=f"{(win_rate - 0.5):.1%} vs 50%")

    st.markdown("---")

    # Profit curve
    st.markdown("## 📈 Cumulative Profit/Loss")

    import plotly.graph_objects as go

    # Sort by date if available
    if 'prediction_date' in betting_history.columns:
        betting_history = betting_history.sort_values('prediction_date')
        x_axis = betting_history['prediction_date']
        x_title = "Date"
    else:
        x_axis = range(len(betting_history))
        x_title = "Bet Number"

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_axis,
        y=betting_history['cumulative_profit'],
        mode='lines',
        name='Cumulative Profit',
        line=dict(color='green' if total_profit > 0 else 'red', width=2),
        fill='tozeroy'
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="black")

    fig.update_layout(
        title="Cumulative Profit/Loss Over Time",
        xaxis_title=x_title,
        yaxis_title="Profit ($)",
        hovermode='x unified',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Performance by stake size
    st.markdown("## 💵 Performance by Stake Size")

    betting_history['stake_bucket'] = pd.cut(
        betting_history['recommended_stake'],
        bins=[0, 10, 20, 30, 50, 1000],
        labels=['$0-10', '$10-20', '$20-30', '$30-50', '$50+']
    )

    stake_stats = betting_history.groupby('stake_bucket', observed=True).agg({
        'won': ['count', 'sum', 'mean'],
        'profit': 'sum',
        'recommended_stake': 'sum'
    }).round(2)

    stake_stats.columns = ['Bets', 'Wins', 'Win Rate', 'Profit', 'Total Staked']
    stake_stats['ROI'] = (stake_stats['Profit'] / stake_stats['Total Staked'] * 100).round(1)

    # Display as DataFrame
    stake_stats_display = stake_stats.copy()
    stake_stats_display['Bets'] = stake_stats_display['Bets'].astype(int)
    stake_stats_display['Wins'] = stake_stats_display['Wins'].astype(int)
    stake_stats_display['Win Rate'] = stake_stats_display['Win Rate'].apply(lambda x: f'{x:.1%}')
    stake_stats_display['Profit'] = stake_stats_display['Profit'].apply(lambda x: f'${x:.2f}')
    stake_stats_display['Total Staked'] = stake_stats_display['Total Staked'].apply(lambda x: f'${x:.2f}')
    stake_stats_display['ROI'] = stake_stats_display['ROI'].apply(lambda x: f'{x:.1f}%')

    st.dataframe(stake_stats_display, use_container_width=True)

    st.markdown("---")

    # Best and worst bets
    st.markdown("## 🏆 Best & Worst Bets")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ✅ Top 5 Winners")
        top_wins = betting_history.nlargest(5, 'profit')

        for idx, row in top_wins.iterrows():
            profit = row['profit']
            st.success(
                f"**{row.get('team', 'N/A')}** vs {row.get('team_opp', 'N/A')}: "
                f"+${profit:.2f} (Stake: ${row['recommended_stake']:.2f}, "
                f"Prob: {row.get('predicted_prob', 0):.1%})"
            )

    with col2:
        st.markdown("### ❌ Top 5 Losses")
        top_losses = betting_history.nsmallest(5, 'profit')

        for idx, row in top_losses.iterrows():
            profit = row['profit']
            st.error(
                f"**{row.get('team', 'N/A')}** vs {row.get('team_opp', 'N/A')}: "
                f"${profit:.2f} (Stake: ${row['recommended_stake']:.2f}, "
                f"Prob: {row.get('predicted_prob', 0):.1%})"
            )

    st.markdown("---")

    # Full betting history table
    st.markdown("## 📋 Full Betting History")

    # Date range filter
    if 'prediction_date' in betting_history.columns:
        betting_history['date'] = pd.to_datetime(betting_history['prediction_date'])
        min_date = betting_history['date'].min()
        max_date = betting_history['date'].max()

        date_range = st.date_input(
            "Filter by date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        if len(date_range) == 2:
            filtered_history = betting_history[
                (betting_history['date'] >= pd.to_datetime(date_range[0])) &
                (betting_history['date'] <= pd.to_datetime(date_range[1]))
            ]
        else:
            filtered_history = betting_history
    else:
        filtered_history = betting_history

    # Display columns
    display_cols = ['prediction_date', 'team', 'team_opp', 'home', 'predicted_prob',
                    'recommended_stake', 'won', 'profit']
    display_cols = [col for col in display_cols if col in filtered_history.columns]

    if display_cols:
        history_display = filtered_history[display_cols].copy()

        # Format columns
        if 'home' in history_display.columns:
            history_display['home'] = history_display['home'].apply(lambda x: '🏠' if x == 1 else '✈️')
        if 'predicted_prob' in history_display.columns:
            history_display['predicted_prob'] = history_display['predicted_prob'].apply(lambda x: f'{x:.1%}')
        if 'recommended_stake' in history_display.columns:
            history_display['recommended_stake'] = history_display['recommended_stake'].apply(lambda x: f'${x:.2f}')
        if 'won' in history_display.columns:
            history_display['won'] = history_display['won'].apply(lambda x: '✅' if x == 1 else '❌')
        if 'profit' in history_display.columns:
            history_display['profit'] = history_display['profit'].apply(
                lambda x: f'+${x:.2f}' if x > 0 else f'${x:.2f}'
            )

        # Rename columns
        column_names = {
            'prediction_date': 'Date',
            'team': 'Team',
            'team_opp': 'Opponent',
            'home': 'H/A',
            'predicted_prob': 'Win Prob',
            'recommended_stake': 'Stake',
            'won': 'Result',
            'profit': 'Profit'
        }
        history_display = history_display.rename(columns=column_names)

        st.dataframe(history_display, use_container_width=True, hide_index=True, height=400)

        # Download button
        csv = filtered_history.to_csv(index=False)
        st.download_button(
            label="📥 Download Betting History as CSV",
            data=csv,
            file_name=f"betting_history_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

else:
    st.info("""
    ℹ️ **Betting History Requires:**
    - Historical predictions with actual game outcomes (`won` column)
    - Recommended stake sizes (`recommended_stake` column)

    Run the full prediction pipeline (scripts 3-6) over time to build betting history!
    """)

st.markdown("---")

# Kelly Criterion explanation
with st.expander("ℹ️ About Kelly Criterion Staking"):
    st.markdown("""
    ### Kelly Criterion Formula

    The **Kelly Criterion** is a mathematical formula for optimal bet sizing:

    ```
    Kelly % = (bp - q) / b
    ```

    Where:
    - **b** = decimal odds - 1 (e.g., 2.0 odds → b = 1.0)
    - **p** = probability of winning (model's prediction)
    - **q** = probability of losing (1 - p)

    ### Benefits:
    - **Maximizes long-term growth** of your bankroll
    - **Prevents overbetting** on uncertain outcomes
    - **Scales stakes** based on edge and confidence

    ### Our Implementation:
    - Uses **fractional Kelly** (typically 25-50% of full Kelly) for safety
    - Only recommends bets when model has **positive expected value**
    - Caps maximum stake size to protect bankroll

    ### Example:
    - Model predicts 60% win probability
    - Market odds: 2.0 (50% implied)
    - Edge: 10%
    - Kelly suggests betting ~10% of bankroll
    - Fractional Kelly (25%): bet ~2.5% of bankroll
    """)
