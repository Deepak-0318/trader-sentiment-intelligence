# Trader Performance vs Market Sentiment Analysis

## Objective
Analyze how Bitcoin market sentiment (Fear/Greed) impacts trader behavior and performance on Hyperliquid.

## Data
- Fear & Greed Index (daily sentiment)
- Hyperliquid historical trade data (211k+ trades)

## Methodology
- Data cleaning and daily alignment
- Feature engineering for performance and risk metrics
- Behavioral segmentation of traders
- Predictive ML model for profitability buckets

## Key Insights
- Fear markets are more profitable but volatile
- Overconfidence during Greed reduces returns
- Low leverage traders outperform consistently
- Traders cluster into four behavioral archetypes

## Modeling
- Random Forest classifier (98% accuracy)
- Most important drivers: capital efficiency, risk-adjusted return, volatility

## Tools
Python, Pandas, Scikit-learn, Matplotlib, Power BI (planned)

