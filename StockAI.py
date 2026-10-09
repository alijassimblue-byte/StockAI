
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="StockAI", page_icon="📈", layout="wide")

DEFAULT_TICKERS = [
    "NVDA","MSFT","AAPL","AMZN","GOOGL","META","AVGO","TSLA","AMD","NFLX",
    "JPM","V","MA","LLY","COST","WMT","ORCL","CRM","ADBE","QCOM"
]

HORIZONS = [5, 10, 20, 40]

@st.cache_data(ttl=900, show_spinner=False)
def download_prices(tickers, period="5y"):
    data = yf.download(
        list(tickers), period=period, interval="1d",
        auto_adjust=True, progress=False, group_by="column", threads=True
    )
    if data.empty:
        return {}
    out = {}
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            try:
                df = data.xs(t, axis=1, level=1).copy()
            except Exception:
                try:
                    df = data[t].copy()
                except Exception:
                    continue
            if "Close" in df.columns:
                out[t] = df.dropna(how="all")
    else:
        t = tickers[0]
        out[t] = data.copy()
    return out

@st.cache_data(ttl=900, show_spinner=False)
def market_data():
    return download_prices(["SPY","QQQ"], "5y")

def rsi(s, n=14):
    delta = s.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    down = (-delta.clip(upper=0)).rolling(n).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def make_features(df, market_close):
    x = df.copy()
    c, h, l, v = x["Close"], x["High"], x["Low"], x["Volume"]

    f = pd.DataFrame(index=x.index)
    f["ret_1"] = c.pct_change(1)
    f["ret_5"] = c.pct_change(5)
    f["ret_10"] = c.pct_change(10)
    f["ret_20"] = c.pct_change(20)
    f["ret_60"] = c.pct_change(60)
    f["vol_10"] = c.pct_change().rolling(10).std()
    f["vol_20"] = c.pct_change().rolling(20).std()
    f["vol_60"] = c.pct_change().rolling(60).std()
    f["sma_10_ratio"] = c / c.rolling(10).mean() - 1
    f["sma_20_ratio"] = c / c.rolling(20).mean() - 1
    f["sma_50_ratio"] = c / c.rolling(50).mean() - 1
    f["sma_200_ratio"] = c / c.rolling(200).mean() - 1
    f["rsi_14"] = rsi(c)
    f["volume_ratio"] = v / v.rolling(20).mean()
    f["range_14"] = ((h-l)/c).rolling(14).mean()
    f["drawdown_60"] = c / c.rolling(60).max() - 1

    m = market_close.reindex(f.index).ffill()
    f["market_ret_5"] = m.pct_change(5)
    f["market_ret_20"] = m.pct_change(20)
    f["relative_ret_20"] = f["ret_20"] - f["market_ret_20"]

    return f.replace([np.inf, -np.inf], np.nan)

def train_models(df, market_close):
    features = make_features(df, market_close)
    models, diagnostics = {}, {}
    Xcols = list(features.columns)

    for h in HORIZONS:
        y = df["Close"].shift(-h) / df["Close"] - 1
        data = features.join(y.rename("target")).dropna()
        if len(data) < 300:
            continue

        # Time-ordered split: never train on future rows.
        split = int(len(data) * 0.8)
        train, test = data.iloc[:split], data.iloc[split:]
        model = HistGradientBoostingRegressor(
            max_iter=250, learning_rate=0.045, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=42
        )
        model.fit(train[Xcols], train["target"])
        pred = model.predict(test[Xcols])
        diagnostics[h] = mean_absolute_error(test["target"], pred)
        models[h] = (model, Xcols)

    return features, models, diagnostics

def score_stock(df, market_close):
    features, models, diagnostics = train_models(df, market_close)
    latest = features.dropna().iloc[[-1]]
    result = {}

    if latest.empty:
        return None

    for h, (model, cols) in models.items():
        result[h] = float(model.predict(latest[cols])[0])

    # Risk-adjusted opportunity score. This deliberately penalizes volatility.
    vol = float(features["vol_20"].dropna().iloc[-1])
    for h in HORIZONS:
        if h in result:
            result[h] = result[h] / max(vol * np.sqrt(h/5), 0.02)

    best_h = max(result, key=result.get)
    raw_features = {
        "RSI": float(features["rsi_14"].dropna().iloc[-1]),
        "20D momentum": float(features["ret_20"].dropna().iloc[-1]),
        "60D momentum": float(features["ret_60"].dropna().iloc[-1]),
        "20D volatility": vol,
        "vs 50D average": float(features["sma_50_ratio"].dropna().iloc[-1]),
        "vs 200D average": float(features["sma_200_ratio"].dropna().iloc[-1]),
        "volume ratio": float(features["volume_ratio"].dropna().iloc[-1]),
    }

    return {
        "features": raw_features,
        "predictions": result,
        "best_horizon": best_h,
        "errors": diagnostics,
        "latest_price": float(df["Close"].iloc[-1]),
        "feature_frame": features,
    }

def backtest(df, market_close, horizon=20):
    features = make_features(df, market_close)
    y = df["Close"].shift(-horizon) / df["Close"] - 1
    data = features.join(y.rename("target")).dropna()
    if len(data) < 500:
        return None

    Xcols = list(features.columns)
    equity = [1.0]
    buy_hold = [1.0]
    dates = []

    # Walk-forward: train only on information available before each test point.
    step = max(10, horizon)
    start = max(300, int(len(data) * 0.6))
    for i in range(start, len(data)-horizon, step):
        train = data.iloc[:i]
        test = data.iloc[i:min(i+step, len(data)-horizon)]
        model = HistGradientBoostingRegressor(
            max_iter=180, learning_rate=0.05, max_leaf_nodes=12,
            l2_regularization=1.0, random_state=42
        )
        model.fit(train[Xcols], train["target"])
        p = model.predict(test[Xcols])
        # Enter only when predicted return is positive enough to overcome a
        # simple safety threshold. This is a research backtest, not a trading rule.
        for actual, pred in zip(test["target"], p):
            strategy_ret = actual if pred > 0.02 else 0.0
            equity.append(equity[-1] * (1 + strategy_ret))
            buy_hold.append(buy_hold[-1] * (1 + actual))
            dates.append(test.index[len(dates) % len(test.index)])

    if len(equity) < 2:
        return None
    eq = pd.Series(equity)
    peak = eq.cummax()
    max_dd = float((eq/peak - 1).min())
    return {
        "strategy_return": equity[-1]-1,
        "buy_hold_return": buy_hold[-1]-1,
        "max_drawdown": max_dd,
        "observations": len(equity)-1
    }

st.title("📈 StockAI")
st.caption("Historical-pattern research engine • ML forecasts • holding-period analysis • walk-forward backtesting")

st.info(
    "Educational research only. StockAI cannot know the future or guarantee maximum profit. "
    "Its ranking is a model output based on historical data and should not be treated as personalized financial advice."
)

with st.sidebar:
    st.header("Universe")
    tickers_text = st.text_area(
        "Tickers (comma-separated)",
        ",".join(DEFAULT_TICKERS),
        height=130
    )
    period = st.selectbox("Historical training window", ["2y","5y","10y"], index=1)
    min_prediction = st.slider("Minimum predicted return to highlight", -0.10, 0.20, 0.03, 0.01)
    run = st.button("🔄 Analyze market", use_container_width=True)

tickers = [x.strip().upper() for x in tickers_text.split(",") if x.strip()]
if not tickers:
    st.warning("Enter at least one ticker.")
    st.stop()

if run or "prices" not in st.session_state:
    with st.spinner("Downloading market history and training time-series models..."):
        st.session_state.prices = download_prices(tickers, period)
        st.session_state.market = market_data()

prices = st.session_state.prices
market = st.session_state.market

if not prices:
    st.error("No market data was returned. Check the tickers and your internet connection.")
    st.stop()

spy = market.get("SPY")
if spy is None:
    st.error("SPY market benchmark data is unavailable.")
    st.stop()
market_close = spy["Close"]

rows = []
details = {}
progress = st.progress(0)

for n, (ticker, df) in enumerate(prices.items(), 1):
    try:
        if len(df) < 350:
            continue
        result = score_stock(df, market_close)
        if result:
            best_h = result["best_horizon"]
            pred = result["predictions"][best_h]
            score = pred / max(result["features"]["20D volatility"], 0.02)
            rows.append({
                "Ticker": ticker,
                "Price": result["latest_price"],
                "Best hold": f"{best_h} days",
                "Predicted return": pred,
                "Model score": score,
                "RSI": result["features"]["RSI"],
                "20D momentum": result["features"]["20D momentum"],
                "60D momentum": result["features"]["60D momentum"],
            })
            details[ticker] = result
    except Exception:
        pass
    progress.progress(n / max(len(prices),1))
progress.empty()

ranking = pd.DataFrame(rows)
if ranking.empty:
    st.error("Not enough clean historical data to score these stocks.")
    st.stop()

ranking = ranking.sort_values(["Predicted return","Model score"], ascending=False).reset_index(drop=True)

st.subheader("🏆 Current model ranking")
top = ranking.iloc[0]

c1,c2,c3,c4 = st.columns(4)
c1.metric("Top-ranked ticker", top["Ticker"])
c2.metric("Latest price", f"${top['Price']:,.2f}")
c3.metric("Best modeled hold", top["Best hold"])
c4.metric("Predicted return", f"{top['Predicted return']*100:.1f}%")

if top["Predicted return"] >= min_prediction:
    st.success(
        f"Research signal: **{top['Ticker']}** currently has the strongest modeled "
        f"risk-adjusted opportunity in your selected universe, with a modeled holding period of **{top['Best hold']}**."
    )
else:
    st.warning("The model does not see a strong enough positive opportunity under your current threshold.")

display = ranking.copy()
display["Predicted return"] = display["Predicted return"] * 100
display["20D momentum"] = display["20D momentum"] * 100
display["60D momentum"] = display["60D momentum"] * 100
st.dataframe(
    display[["Ticker","Price","Best hold","Predicted return","Model score","RSI","20D momentum","60D momentum"]]
    .style.format({"Price":"${:.2f}","Predicted return":"{:.2f}%","Model score":"{:.2f}","RSI":"{:.1f}","20D momentum":"{:.2f}%","60D momentum":"{:.2f}%"}),
    use_container_width=True,
    hide_index=True
)

st.subheader("🔎 Stock deep dive")
selected = st.selectbox("Choose a stock", ranking["Ticker"].tolist(), index=0)
df = prices[selected]
r = details[selected]

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
    name=selected
))
fig.update_layout(height=500, xaxis_rangeslider_visible=False, title=f"{selected} historical price")
st.plotly_chart(fig, use_container_width=True)

a,b = st.columns(2)
with a:
    st.markdown("### Model holding-period forecast")
    forecast = pd.DataFrame({
        "Holding period": [f"{h} days" for h in HORIZONS if h in r["predictions"]],
        "Model signal": [r["predictions"][h] for h in HORIZONS if h in r["predictions"]]
    })
    forecast["Model signal"] = forecast["Model signal"] * 100
    st.bar_chart(forecast.set_index("Holding period"))
    st.caption("The signal is risk-adjusted model output, not a guaranteed percentage return.")

with b:
    st.markdown("### Current technical state")
    for k,v in r["features"].items():
        if "ratio" in k or "momentum" in k or "average" in k:
            st.write(f"**{k}:** {v*100:.2f}%")
        elif "volatility" in k:
            st.write(f"**{k}:** {v*100:.2f}%")
        else:
            st.write(f"**{k}:** {v:.2f}")

st.subheader("🧪 Walk-forward backtest")
st.caption("This trains sequentially on older data and tests on later data to reduce look-ahead bias.")
bt_h = st.selectbox("Backtest horizon", HORIZONS, index=2)
if st.button("Run backtest"):
    with st.spinner("Running walk-forward backtest..."):
        bt = backtest(df, market_close, bt_h)
    if bt:
        x,y,z = st.columns(3)
        x.metric("Strategy return", f"{bt['strategy_return']*100:.1f}%")
        y.metric("Buy-and-hold comparison", f"{bt['buy_hold_return']*100:.1f}%")
        z.metric("Max drawdown", f"{bt['max_drawdown']*100:.1f}%")
        st.write(f"Test observations: {bt['observations']}")
    else:
        st.warning("Not enough data for this backtest.")

st.subheader("🧠 How StockAI learns")
st.markdown("""
1. **Historical price behavior:** 1/5/10/20/60-day returns, moving-average relationships, volatility and drawdowns.
2. **Momentum + mean-reversion clues:** RSI, distance from moving averages and recent performance.
3. **Volume behavior:** current volume relative to its recent average.
4. **Market context:** SPY returns and the stock's relative performance against the market.
5. **Multiple holding periods:** separate models estimate 5, 10, 20 and 40 trading-day outcomes.
6. **Risk adjustment:** volatile predictions are penalized so the biggest raw forecast does not automatically win.
7. **Walk-forward testing:** the backtest respects time order instead of randomly mixing past and future data.
""")

st.caption("Model limitations: historical patterns can disappear, news can overwhelm technical signals, and predictions have uncertainty. Never assume a model can guarantee maximum profit.")
