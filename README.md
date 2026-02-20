## Trader Performance vs Market Sentiment

Just started wondering: do traders perform differently when the market is fearful vs greedy? Grabbed 200k+ Hyperliquid trades, aligned them with daily sentiment data from the Fear & Greed Index, and started digging.

What we found: traders absolutely behave differently based on sentiment, and it actually matters for profitability. Some of it makes sense (buy dips during fear = higher win rate), some of it is just emotional bias leading to bad decisions.

## The Data

Got two main sources:

**Fear & Greed Index** - Daily sentiment classifications (Extreme Fear through Extreme Greed). This gives us the market psychology backdrop.

**Hyperliquid Trade History** - 211,000+ actual trades. Account info, size, direction, PnL, fees, timestamps. The real stuff traders were doing.

## What We Built

Started with basic data cleanup - checked for gaps, duplicates, weird timestamps. Then grouped daily trades by sentiment regime.

Built daily metrics for each trader: PnL, win rate, volatility, capital efficiency, leverage used. Segmented traders by behavior (high/low leverage, frequent/infrequent, consistent/inconsistent). 

Then analyzed which behaviors actually made money across different market conditions. Added a Random Forest model to see what actually predicts profitability. K-means clustering grouped traders into archetypes to understand different play styles.

## Key Findings

**Fear sentiment:** Traders buy dips, win rates spike, but volatility is insane. Profit potential is there if you've got the risk management to handle 40-60% daily swings.

**Greed sentiment:** Leverage gets crazy, momentum chasing kicks in, win rates tank. Most profitable trades come right before the crash. Basically everyone loses money running into brick walls.

**Neutral periods:** Boring and consistent. Controlled leverage, steady activity, predictable returns. Good for validating new strategies without sentiment noise.

**Low leverage always wins** - not just in total returns, but especially risk-adjusted. Even taking the same trades, smaller positions = way better long-term curves. High leverage traders look great until they don't.

**Frequency is overrated** - High frequency traders make more total dollars only because they're trading bigger. Per trade and risk-adjusted? Selective traders crushing it. Quality over quantity works.

**Consistency beats everything** - Traders who grind the same process regardless of conditions outperform wildly. The archetypes: disciplined grinders, YOLO margin traders, quiet hodlers, scalpers.

## Quick Start

Just wanna run it locally?

```bash
pip install -r requirements.txt
streamlit run app.py
```

That'll spin up a dashboard where you can filter by sentiment and dig into the data.

## Bottom Line

If you're trading crypto and not thinking about position sizing relative to market conditions, you're leaving money on the table. Fear periods are brutal but profitable if you're disciplined. Greed periods blow people up. Consistency beats genius every time.

Low leverage boring accounts beat high leverage flashy ones over any period longer than a month. That's just math.

The data doesn't lie - sentiment matters, but your process matters more.

## Tools Used

Python (Pandas, NumPy, Matplotlib, Seaborn)
Scikit-learn
Plotly
Streamlit