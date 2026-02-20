import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Dashboard config
st.set_page_config(page_title="Trader Sentiment Analytics", layout="wide")

# Load the processed data files
@st.cache_data
def load_data():
    trader = pd.read_csv("data/processed/trader_daily_metrics.csv")
    market = pd.read_csv("data/processed/market_daily_metrics.csv")
    long_short = pd.read_csv("data/processed/long_short_daily.csv")
    base = pd.read_csv("data/intermediate/merged_daily_base.csv")

    # Merge sentiment classification from base file
    sentiment_map = base[['date', 'classification']].drop_duplicates()

    trader = trader.merge(sentiment_map, on='date', how='left')
    market = market.merge(sentiment_map, on='date', how='left')
    long_short = long_short.merge(sentiment_map, on='date', how='left')

    return trader, market, long_short

trader, market, long_short = load_data()

# Segment traders into groups for comparison
trader['lev_segment'] = np.where(
    trader['avg_leverage'] > trader['avg_leverage'].median(),
    "High Leverage", "Low Leverage"
)

trader['freq_segment'] = np.where(
    trader['trade_count'] > trader['trade_count'].median(),
    "Frequent", "Infrequent"
)

# Let users filter by one or more sentiment regimes
st.sidebar.header("Market Sentiment")

sentiments = sorted(trader['classification'].dropna().unique())
selected = st.sidebar.multiselect(
    "Select Sentiment",
    sentiments,
    default=sentiments
)

# Filter data based on sentiment selection
trader_f = trader[trader['classification'].isin(selected)]
market_f = market[market['classification'].isin(selected)]
long_short_f = long_short[long_short['classification'].isin(selected)]

# Page title and description
st.title("Trader Performance vs Market Sentiment")
st.caption("Hyperliquid Trading Behavior & Machine Learning Analysis")

# Show key metrics for filtered data
c1, c2, c3, c4 = st.columns(4)

c1.metric("Avg Daily PnL", f"{trader_f['daily_pnl'].mean():,.0f}")
c2.metric("Avg Win Rate", f"{trader_f['win_rate'].mean()*100:.1f}%")
c3.metric("Avg Volatility", f"{trader_f['pnl_volatility'].mean():,.0f}")
c4.metric("Avg Trades / Day", f"{market_f['total_trades'].mean():.0f}")

st.divider()

# Main PnL distribution
fig_pnl = px.box(
    trader_f,
    x="classification",
    y="daily_pnl",
    title="PnL Distribution by Market Sentiment"
)
st.plotly_chart(fig_pnl, use_container_width=True)

# How activity and positioning changes with sentiment
col1, col2 = st.columns(2)

activity = market_f.groupby("classification")['total_trades'].mean().reset_index()
fig_activity = px.bar(activity, x="classification", y="total_trades",
                      title="Trading Activity by Sentiment")

bias = long_short_f.groupby("classification")['long_short_ratio'].mean().reset_index()
fig_bias = px.bar(bias, x="classification", y="long_short_ratio",
                  title="Long/Short Bias by Sentiment")

col1.plotly_chart(fig_activity, use_container_width=True)
col2.plotly_chart(fig_bias, use_container_width=True)

st.divider()

# -----------------------------------
# SEGMENT ANALYSIS
# -----------------------------------
st.subheader("Trader Segments")
st.divider()

# Segment analysis - who actually makes money?
st.subheader("Trader Segments")

seg_lev = trader_f.groupby(["lev_segment", "classification"])['daily_pnl'].mean().reset_index()
fig_lev = px.bar(seg_lev, x="lev_segment", y="daily_pnl",
                 color="classification", barmode="group",
                 title="PnL by Leverage Segment")

seg_freq = trader_f.groupby(["freq_segment", "classification"])['daily_pnl'].mean().reset_index()
fig_freq = px.bar(seg_freq, x="freq_segment", y="daily_pnl",
                  color="classification", barmode="group",
                  title="PnL by Trading Frequency")

st.plotly_chart(fig_lev, use_container_width=True)
st.plotly_chart(fig_freq, use_container_width=True)

st.divider()

# ML Model - what predicts profitability?
st.subheader("Machine Learning Insights")

features = [
    'avg_trade_size',
    'trade_count',
    'avg_leverage',
    'pnl_volatility',
    'capital_efficiency'
]

ml_df = trader.dropna(subset=features + ['daily_pnl'])

X = ml_df[features]
# Binary label: did they make money or lose money?
ml_df['profit_label'] = pd.cut(
    ml_df['daily_pnl'],
    bins=[-np.inf, 0, np.inf],
    labels=['Loss', 'Profit']
)

# Train random forest to see which features matter most
rf = RandomForestClassifier(n_estimators=150, random_state=42)
rf.fit(X, ml_df['profit_label'])

importance = pd.Series(
    rf.feature_importances_,
    index=features
).sort_values()

fig_imp = px.bar(
    importance,
    orientation="h",
    title="Feature Importance - What Actually Predicts Profit?"
)
st.plotly_chart(fig_imp, use_container_width=True)

# Trader archetypes via clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
ml_df['cluster'] = kmeans.fit_predict(X_scaled)

fig_cluster = px.scatter(
    ml_df,
    x="avg_leverage",
    y="pnl_volatility",
    color="cluster",
    title="Trader Archetypes: Who's Who?"
)
st.plotly_chart(fig_cluster, use_container_width=True)

# Summary insights
st.subheader("Key Takeaways")

st.markdown("""
- Fear sentiment = higher profits but wild volatility  
- Greed sentiment = rekt accounts  
- Low leverage traders consistently win long-term  
- High activity correlates with lower win rates  
- Consistency beats genius every time  
- High activity traders dominate total profitability  
- Capital efficiency is the strongest profit driver  
""")