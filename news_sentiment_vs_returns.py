# %% [markdown]
# ## News Sentiment vs Stock Returns
#
# Mini research: Do daily news headline sentiment and next-day stock returns correlate?
#
# - Tickers: AAPL, TSLA, NVDA (you can change this list)
# - Data sources:
#   - News: NewsAPI.org (headlines)
#   - Prices: Yahoo Finance (via yfinance)

# %% 
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import yfinance as yf
from newsapi import NewsApiClient

from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Aplati un éventuel MultiIndex de colonnes en colonnes simples."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            "_".join(str(c) for c in col).strip("_")
            for col in df.columns.to_list()
        ]
    return df
    
# Ensure VADER is available
nltk.download("vader_lexicon")

sns.set(style="whitegrid")

# %% [markdown]
# ### Configuration

# %%
# IMPORTANT: Put your real NewsAPI key here, but NEVER commit it to Git or share it.
NEWSAPI_KEY = "YOUR API KEY HERE"  # 

# Tickers to analyze
TICKERS = ["AAPL", "TSLA", "NVDA"]

# Map tickers to company names (for better NewsAPI queries)
TICKER_TO_NAME = {
    "AAPL": "Apple",
    "TSLA": "Tesla",
    "NVDA": "Nvidia",
}

# Date range for analysis 
END_DATE = datetime.today().date()
# NewsAPI free tier: only ~30 days back allowed
MAX_NEWSAPI_LOOKBACK_DAYS = 28
START_DATE = END_DATE - timedelta(days=MAX_NEWSAPI_LOOKBACK_DAYS)
print(f"Analyzing from {START_DATE} to {END_DATE}")

print(f"Analyzing from {START_DATE} to {END_DATE}")

# Initialize clients
newsapi = NewsApiClient(api_key=NEWSAPI_KEY)
sia = SentimentIntensityAnalyzer()

# %% [markdown]
# ### Helper: Fetch headlines from NewsAPI

# %%
def fetch_headlines_for_ticker(ticker, start_date, end_date, page_size=50):
    """
    Fetch up to `page_size` headlines for a given ticker using its company name.
    NewsAPI dev/free accounts are limited to 100 results total, so we only request one page.
    """
    company = TICKER_TO_NAME.get(ticker, ticker)
    from_param = start_date.isoformat()
    to_param = end_date.isoformat()

    response = newsapi.get_everything(
        q=company,
        from_param=from_param,
        to=to_param,
        language="en",
        sort_by="relevancy",
        page=1,
        page_size=min(page_size, 100),  # dev plan max 100
    )

    articles = response.get("articles", [])
    rows = []
    for art in articles:
        rows.append(
            {
                "ticker": ticker,
                "company": company,
                "headline": art.get("title"),
                "description": art.get("description"),
                "url": art.get("url"),
                "publisher": (art.get("source") or {}).get("name"),
                "published_at": art.get("publishedAt"),
            }
        )

    return pd.DataFrame(rows)
    while True:
        response = newsapi.get_everything(
            q=company,
            from_param=from_param,
            to=to_param,
            language="en",
            sort_by="relevancy",
            page=page,
            page_size=page_size,
        )

        articles = response.get("articles", [])
        if not articles:
            break

        for art in articles:
            all_articles.append(
                {
                    "ticker": ticker,
                    "company": company,
                    "headline": art.get("title"),
                    "description": art.get("description"),
                    "url": art.get("url"),
                    "publisher": (art.get("source") or {}).get("name"),
                    "published_at": art.get("publishedAt"),
                }
            )

        # Stop if we hit the last page
        total_results = response.get("totalResults", 0)
        if page * page_size >= total_results:
            break

        page += 1
        # To be nice to the API, you could add a short sleep here

    return pd.DataFrame(all_articles)


# %% [markdown]
# ### Step 1: Collect headlines for 1–3 tickers

# %%
news_dfs = []
for t in TICKERS:
    print(f"Fetching headlines for {t}...")
    df_t = fetch_headlines_for_ticker(t, START_DATE, END_DATE)
    print(f"  Got {len(df_t)} articles for {t}")
    news_dfs.append(df_t)

news_df = pd.concat(news_dfs, ignore_index=True) if news_dfs else pd.DataFrame()
print("Combined news shape:", news_df.shape)

# Basic cleaning
news_df = news_df.dropna(subset=["headline"])
news_df["published_at"] = pd.to_datetime(news_df["published_at"])
news_df["date"] = news_df["published_at"].dt.date

news_df.head()

# %% [markdown]
# ### Quick EDA on headlines & publishers

publisher_stats = (
    news_df[["ticker", "publisher"]]
    .groupby("ticker")
    .nunique()
    .rename(columns={"publisher": "num_publishers"})
)

print("\n=== Number of distinct news publishers per ticker ===")
print(publisher_stats.to_string())

top_publishers = (
    news_df["publisher"]
    .value_counts()
    .head(10)
    .rename("num_articles")
    .to_frame()
)

print("\n=== Top 10 news publishers (all tickers) ===")
print(top_publishers.to_string())

plt.figure(figsize=(8, 5))

ax = sns.countplot(
    data=news_df,
    x="ticker",
    order=TICKERS,
    palette="Blues_d",
)

ax.set_title("Nombre de titres de presse par action", fontsize=14, pad=15)
ax.set_xlabel("Ticker", fontsize=12)
ax.set_ylabel("Nombre de titres", fontsize=12)

for p in ax.patches:
    height = p.get_height()
    ax.annotate(
        f"{int(height)}",
        (p.get_x() + p.get_width() / 2., height),
        ha="center",
        va="bottom",
        fontsize=11,
        xytext=(0, 3),
        textcoords="offset points",
    )

ax.tick_params(axis="both", labelsize=11)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Step 2: Sentiment scoring with VADER

# %%
def compute_vader_sentiment(text):
    if not isinstance(text, str):
        return np.nan
    scores = sia.polarity_scores(text)
    return scores["compound"]

news_df["sentiment"] = news_df["headline"].apply(compute_vader_sentiment)

# Daily average sentiment per ticker
daily_sentiment = (
    news_df.groupby(["ticker", "date"])["sentiment"]
    .mean()
    .reset_index()
    .rename(columns={"sentiment": "avg_sentiment"})
)

daily_sentiment.head()

# ### Step 3: Get daily prices and returns from Yahoo Finance
# %%
def fetch_prices_for_ticker(ticker, start_date, end_date):
    data = yf.download(
        ticker,
        start=start_date,
        end=end_date + timedelta(days=1),  # yfinance end is exclusive
        auto_adjust=True,
        progress=False,
    )
    data = data.reset_index()
    data["date"] = data["Date"].dt.date
    return data[["date", "Close"]].rename(columns={"Close": "close"})

# On récupère les prix et on calcule les rendements TICKER PAR TICKER,
# sans aucun groupby ensuite (pour éviter les MultiIndex).
price_dfs = []

for t in TICKERS:
    print(f"Fetching prices for {t}...")
    df_p = fetch_prices_for_ticker(t, START_DATE, END_DATE)

    # tri par date
    df_p = df_p.sort_values("date").copy()

    # on garde le ticker dans une colonne
    df_p["ticker"] = t

    # rendements simple et du lendemain pour CE ticker
    df_p["return_1d"] = df_p["close"].pct_change()
    df_p["return_1d_lead"] = df_p["return_1d"].shift(-1)

    price_dfs.append(df_p)

# on concatène tout
prices_df = pd.concat(price_dfs, ignore_index=True)

print("prices_df.head():")
print(prices_df.head())

# ### Step 4: Merge sentiment and returns

# ### Step 4: Merge sentiment and returns

# Aplatir les colonnes + remettre un index simple
daily_sentiment = flatten_columns(daily_sentiment).reset_index(drop=True)
prices_df = flatten_columns(prices_df).reset_index(drop=True)

print("daily_sentiment columns:", daily_sentiment.columns)
print("prices_df columns:", prices_df.columns)

merged = pd.merge(
    daily_sentiment,
    prices_df,
    on=["ticker", "date"],
    how="inner",
)

print("merged.head():")
print(merged.head())

merged = merged.dropna(subset=["avg_sentiment", "return_1d", "return_1d_lead"])
print("Merged shape:", merged.shape)

merged = pd.merge(
    daily_sentiment,
    prices_df,
    on=["ticker", "date"],
    how="inner",
)

print("merged.head():")
print(merged.head())

# Drop days with missing returns or sentiment
merged = merged.dropna(subset=["avg_sentiment", "return_1d", "return_1d_lead"])
print("Merged shape:", merged.shape)
# Small sample of merged data for sanity check
cols_to_show = ["ticker", "date", "avg_sentiment", "return_1d", "return_1d_lead"]
print("\n=== Sample of merged daily data ===")
print(merged[cols_to_show].head(10).to_string(index=False))

# %% [markdown]
# ### Step 5: Correlation analysis (same-day and next-day)

# %%
results = []

for t in TICKERS:
    sub = merged[merged["ticker"] == t]

    corr_same = sub["avg_sentiment"].corr(sub["return_1d"])
    corr_next = sub["avg_sentiment"].corr(sub["return_1d_lead"])

    results.append(
        {
            "ticker": t,
            "corr_sentiment_same_day_return": corr_same,
            "corr_sentiment_next_day_return": corr_next,
            "n_days": len(sub),
        }
    )

corr_df = pd.DataFrame(results)

# Cleaner column names + arrondi
corr_df = corr_df.rename(
    columns={
        "corr_sentiment_same_day_return": "corr_same_day",
        "corr_sentiment_next_day_return": "corr_next_day",
        "n_days": "num_days",
    }
).round(3)

print("\n=== Correlation between daily sentiment and returns ===")
print(corr_df.to_string(index=False))

# Quick textual interpretation
for _, row in corr_df.iterrows():
    t = row["ticker"]
    cs = row["corr_same_day"]
    cn = row["corr_next_day"]
    print(
        f"\nTicker {t}: "
        f"same-day corr = {cs:+.3f}, next-day corr = {cn:+.3f} "
        f"(based on {int(row['num_days'])} days)."
    )

# %% [markdown]
# ### Step 6: Visualization

# %%
for t in TICKERS:
    sub = merged[merged["ticker"] == t]

    if sub.empty:
        continue

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f"{t}: Daily Avg Sentiment vs Returns")

    # Time series
    axes[0].plot(sub["date"], sub["avg_sentiment"], label="Avg Sentiment", color="tab:blue")
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Sentiment")
    axes[0].legend()

    axes[1].bar(sub["date"], sub["return_1d"], color="tab:orange", label="Daily Return")
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("Return")
    axes[1].set_xlabel("Date")
    axes[1].legend()

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Scatter sentiment vs return
for t in TICKERS:
    sub = merged[merged["ticker"] == t]
    if sub.empty:
        continue

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{t}: Sentiment vs Returns")

    sns.regplot(
        ax=axes[0],
        data=sub,
        x="avg_sentiment",
        y="return_1d",
        scatter_kws={"alpha": 0.6},
    )
    axes[0].set_title("Same-day returns")

    sns.regplot(
        ax=axes[1],
        data=sub,
        x="avg_sentiment",
        y="return_1d_lead",
        scatter_kws={"alpha": 0.6},
    )
    axes[1].set_title("Next-day returns")

    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### (Optional) TA-Lib / PyNance technical indicators
#
# Example: add RSI or MACD and see whether combining technical signals with sentiment gives a stronger relationship.

# %%
try:
    import talib

    def add_technical_indicators(df):
        df = df.sort_values("date")
        close = df["close"].values.astype(float)
        df["rsi_14"] = talib.RSI(close, timeperiod=14)
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        df["macd"] = macd
        df["macd_signal"] = macd_signal
        return df

    prices_with_ta = prices_df.groupby("ticker", group_keys=False).apply(add_technical_indicators)

    merged_ta = pd.merge(
        daily_sentiment,
        prices_with_ta,
        on=["ticker", "date"],
        how="inner",
    ).dropna(subset=["avg_sentiment", "return_1d", "rsi_14"])

    print("With TA shape:", merged_ta.shape)

    # Example: correlation between sentiment and RSI, plus RSI and returns
    for t in TICKERS:
        sub = merged_ta[merged_ta["ticker"] == t]
        if sub.empty:
            continue
        print(f"\nTicker {t}:")
        print("  corr(sentiment, RSI 14):", sub["avg_sentiment"].corr(sub["rsi_14"]))
        print("  corr(RSI 14, same-day return):", sub["rsi_14"].corr(sub["return_1d"]))

except ImportError:
    print("TA-Lib not installed or failed to import; skipping technical indicators.")
# %%
print("\n=== Mini-summary ===")
for _, row in corr_df.iterrows():
    t = row["ticker"]
    cs = row["corr_same_day"]
    cn = row["corr_next_day"]
    if abs(cs) < 0.1 and abs(cn) < 0.1:
        msg = "No clear linear relationship between sentiment and returns."
    else:
        direction = "positive" if (abs(cs) > abs(cn) and cs > 0) or (abs(cn) >= abs(cs) and cn > 0) else "negative"
        msg = f"Evidence of a {direction} relationship between sentiment and returns."

    print(f"- {t}: {msg} (same-day {cs:+.3f}, next-day {cn:+.3f})")