"""
Today's Games Page

Displays upcoming NBA games with predictions and betting recommendations.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import load_latest_predictions, load_enriched_predictions
from components.charts import create_win_probability_gauge, create_odds_comparison

st.set_page_config(page_title="Today's Games", page_icon="📊", layout="wide")

st.title("📊 Today's NBA Games & Predictions")

# Load data
enriched_df = load_enriched_predictions()
predictions_df = load_latest_predictions()

if enriched_df is None and predictions_df is None:
    st.warning("⚠️ No prediction data available. Run script 3 to generate predictions.")
    st.info("""
    **To generate predictions:**
    ```bash
    cd 2026/src
    python 3_predict_games_hybrid_2026.py
    ```
    """)
    st.stop()

# Use enriched if available, otherwise use basic predictions
df = enriched_df if enriched_df is not None else predictions_df

st.success(f"✅ Loaded {len(df)} games from latest predictions")

# Filters
st.sidebar.header("🔍 Filters")

# Confidence filter
min_confidence = st.sidebar.slider(
    "Minimum Win Probability",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    format="%.0f%%"
)

# Filter data
filtered_df = df[df['predicted_prob'] >= min_confidence].copy() if 'predicted_prob' in df.columns else df

# Sort by confidence (if available)
if 'predicted_prob' in filtered_df.columns:
    filtered_df = filtered_df.sort_values('predicted_prob', ascending=False)

st.markdown(f"### Showing {len(filtered_df)} games with win probability ≥ {min_confidence:.0%}")

# Display games in cards
for idx, row in filtered_df.iterrows():
    with st.expander(
        f"🏀 {row.get('team', 'N/A')} vs {row.get('team_opp', 'N/A')} - "
        f"Win Prob: {row.get('predicted_prob', 0):.1%}",
        expanded=False
    ):
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            st.subheader("📈 Prediction")

            # Win probability
            if 'predicted_prob' in row:
                st.metric(
                    "Win Probability",
                    f"{row['predicted_prob']:.1%}",
                    delta=f"{(row['predicted_prob'] - 0.5):.1%} vs 50%"
                )

            # Team info
            st.write(f"**Team:** {row.get('team', 'N/A')}")
            st.write(f"**Opponent:** {row.get('team_opp', 'N/A')}")
            st.write(f"**Home/Away:** {'Home' if row.get('home', 0) == 1 else 'Away'}")

        with col2:
            st.subheader("💰 Betting Info")

            # Odds information
            if 'odds' in row:
                st.metric("Market Odds", f"{row['odds']:.2f}")

                # Calculate implied probability from odds
                if row['odds'] > 0:
                    implied_prob = 1 / row['odds']
                    st.metric(
                        "Implied Probability",
                        f"{implied_prob:.1%}",
                        delta=f"{(row.get('predicted_prob', 0) - implied_prob):.1%} edge"
                    )

            # Recommended stake
            if 'recommended_stake' in row:
                stake = row['recommended_stake']
                if stake > 0:
                    st.success(f"💵 **Recommended Stake:** ${stake:.2f}")

                    if 'kelly_fraction' in row:
                        st.write(f"Kelly Fraction: {row['kelly_fraction']:.1%}")
                else:
                    st.info("🚫 No bet recommended (insufficient edge)")

        with col3:
            st.subheader("🎯 Confidence")

            # Confidence gauge
            if 'predicted_prob' in row:
                # Simple text-based confidence indicator
                prob = row['predicted_prob']
                if prob >= 0.7:
                    st.success(f"🟢 High\n{prob:.0%}")
                elif prob >= 0.6:
                    st.info(f"🟡 Medium\n{prob:.0%}")
                else:
                    st.warning(f"🟠 Low\n{prob:.0%}")

        # Additional details in a second row
        st.markdown("---")
        detail_col1, detail_col2 = st.columns(2)

        with detail_col1:
            if 'date' in row:
                st.write(f"📅 **Date:** {row['date']}")
            if 'season' in row:
                st.write(f"🏆 **Season:** {row['season']}")

        with detail_col2:
            # Model features (if available)
            feature_cols = [col for col in row.index if col.startswith('pts') or col.startswith('reb') or col.startswith('ast')]
            if feature_cols:
                st.write("**Recent Stats (features):**")
                for col in feature_cols[:3]:  # Show top 3
                    st.write(f"- {col}: {row[col]:.1f}" if pd.notna(row[col]) else f"- {col}: N/A")

st.markdown("---")

# Summary statistics
st.subheader("📊 Summary Statistics")

summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

with summary_col1:
    st.metric("Total Games", len(filtered_df))

with summary_col2:
    if 'recommended_stake' in filtered_df.columns:
        bet_count = (filtered_df['recommended_stake'] > 0).sum()
        st.metric("Recommended Bets", bet_count)
    else:
        st.metric("Recommended Bets", "N/A")

with summary_col3:
    if 'predicted_prob' in filtered_df.columns:
        avg_confidence = filtered_df['predicted_prob'].mean()
        st.metric("Avg Win Probability", f"{avg_confidence:.1%}")
    else:
        st.metric("Avg Win Probability", "N/A")

with summary_col4:
    if 'recommended_stake' in filtered_df.columns:
        total_stake = filtered_df['recommended_stake'].sum()
        st.metric("Total Stake", f"${total_stake:.2f}")
    else:
        st.metric("Total Stake", "N/A")

# Download predictions
st.markdown("---")
st.subheader("💾 Download Data")

csv = filtered_df.to_csv(index=False)
st.download_button(
    label="📥 Download Predictions as CSV",
    data=csv,
    file_name=f"nba_predictions_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
    mime="text/csv"
)
