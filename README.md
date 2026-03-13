## News Sentiment vs Stock Returns

### Project Overview

This mini research project investigates whether **daily financial news sentiment** is related to **stock returns** for three large‑cap US equities:

- `AAPL` (Apple)  
- `TSLA` (Tesla)  
- `NVDA` (Nvidia)

The pipeline:

- Collects recent news headlines per ticker (NewsAPI)
- Computes daily sentiment scores (VADER)
- Fetches historical prices and daily returns (Yahoo Finance via `yfinance`)
- Merges sentiment and returns at the `(ticker, date)` level
- Produces tables, correlations, and plots to explore:
  - Same‑day relationship: sentiment vs return on the same date
  - Next‑day relationship: sentiment vs return on the following trading day

---

### Data Sources

#### News

- API: `https://newsapi.org/`
- Fields used:
  - `headline` (title)
  - `description`
  - `url`
  - `publisher` (source name)
  - `publishedAt`
- Query strategy:
  - One query per ticker, using the company name (`Apple`, `Tesla`, `Nvidia`)
  - Restrict to the last ~28 days (NewsAPI free‑tier lookback limit)
  - Maximum one page of results per ticker (≤100 articles, free‑tier max)

#### Prices

- Source: Yahoo Finance via `yfinance`
- Fields used:
  - `Date` (converted to `date`)
  - `Close` (auto‑adjusted)
- Derived metrics:
  - `return_1d`: simple daily percent change of adjusted close
  - `return_1d_lead`: next‑day return (shifted by −1), to test predictive power

---

### Methods

#### Sentiment Analysis

- Library: **NLTK VADER** (`SentimentIntensityAnalyzer`)
- Input: article `headline`
- For each headline:
  - Compute VADER `compound` score in \([-1, 1]\)
- For each `(ticker, date)`:
  - Compute daily average sentiment:
    - `avg_sentiment = mean(compound scores of all headlines that day)`

#### Returns and Alignment

1. For each ticker:
   - Download prices for `[START_DATE, END_DATE]`
   - Sort by `date`
   - Compute:
     - `return_1d = close.pct_change()`
     - `return_1d_lead = return_1d.shift(-1)`

2. Merge sentiment and returns:
   - Inner join on `["ticker", "date"]`
   - Final dataset columns (key ones):
     - `ticker`, `date`
     - `avg_sentiment`
     - `return_1d` (same‑day)
     - `return_1d_lead` (next‑day)

#### Quantitative Analysis

- Descriptive statistics by ticker:
  - mean, std, min, max of `avg_sentiment`, `return_1d`, `return_1d_lead`
- Correlations by ticker:
  - `corr_same_day = corr(avg_sentiment, return_1d)`
  - `corr_next_day = corr(avg_sentiment, return_1d_lead)`
- Textual interpretation:
  - Short English summary per ticker based on correlation magnitudes

#### Visualizations

- EDA on news:
  - Bar chart: number of headlines per ticker (with counts annotated on bars)
  - Publisher tables:
    - Number of distinct publishers per ticker
    - Top 10 publishers across all tickers

- Time‑series (for each ticker):
  - Upper panel: daily `avg_sentiment`
  - Lower panel: bar plot of `return_1d`

- Scatter + regression (for each ticker):
  - `avg_sentiment` vs `return_1d` (same‑day)
  - `avg_sentiment` vs `return_1d_lead` (next‑day)

- Optional:
  - Technical indicators with TA‑Lib (e.g. RSI, MACD) and their correlations with sentiment and returns.

---

### Requirements

#### Python Packages

Install in a virtual environment:


- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `yfinance`
- `newsapi_python`
- `nltk`
- `scikit-learn`
- `ta-lib` (optional)

 
