import io
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

# ============================================================
# CONSTANTS
# ============================================================
DEFAULT_TICKERS = [
    "NVDA","MSFT","AAPL","AMZN","GOOGL","META","AVGO","TSLA","AMD","NFLX",
    "JPM","V","MA","LLY","COST","WMT","ORCL","CRM","ADBE","QCOM"
]
BENCHMARKS = ["SPY", "QQQ", "IWM", "DIA"]
HORIZONS = [5, 10, 20, 40]
FIELDS = ["Open", "High", "Low", "Close", "Volume"]
MIN_ROWS = 350

SRC_YAHOO = "Yahoo Finance (live download)"
SRC_CSV = "Uploaded prices.csv"

SPEED_FAST = "Fast (recommended)"
SPEED_STANDARD = "Standard (slower, more thorough)"
SPEED_PARAMS = {
    SPEED_FAST: dict(max_iter=80, learning_rate=0.10, max_leaf_nodes=12, bt_iter=60, bt_lr=0.10),
    SPEED_STANDARD: dict(max_iter=250, learning_rate=0.045, max_leaf_nodes=15, bt_iter=180, bt_lr=0.05),
}

PAGES = [
    "🏠 Dashboard", "🔍 Stock Screener", "🤖 AI Predictions", "📊 Stock Analysis",
    "🧪 Backtesting", "⭐ Watchlist", "💼 Portfolio", "🗄️ Data Center",
    "🧠 Model Lab", "⚙️ Settings",
]

THEMES = {
    "Dark": dict(bg="#0E1117", panel="#1A1F2B", border="#2B3242", text="#E8EAED",
                 muted="#9AA0A6", accent="#4F8BF9", good="#2ECC71", bad="#E74C3C",
                 plotly="plotly_dark"),
    "Light": dict(bg="#FFFFFF", panel="#F3F5F9", border="#D9DEE8", text="#1F2430",
                  muted="#5F6775", accent="#2563EB", good="#1E9E57", bad="#D64545",
                  plotly="plotly_white"),
    "Midnight Blue": dict(bg="#0A1128", panel="#12204A", border="#24376F", text="#E6ECFF",
                          muted="#8FA3D6", accent="#5CC8FF", good="#3DDC97", bad="#FF6B6B",
                          plotly="plotly_dark"),
    "Forest": dict(bg="#0F1D16", panel="#183326", border="#26523B", text="#E4F2E8",
                   muted="#8DB59B", accent="#4CD97B", good="#4CD97B", bad="#F07167",
                   plotly="plotly_dark"),
    "Sunset": dict(bg="#1C1116", panel="#2E1A22", border="#4A2B37", text="#FBE9E3",
                   muted="#C9A29A", accent="#FF7A59", good="#7BD88F", bad="#FF5D73",
                   plotly="plotly_dark"),
}

CSS_TEMPLATE = """
<style>
.stApp { background: __BG__; color: __TEXT__; }
[data-testid="stHeader"] { background: __BG__; }
[data-testid="stSidebar"] { background: __PANEL__; border-right: 1px solid __BORDER__; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
.stApp p, .stApp li, .stApp label,
.stApp div[data-testid="stMarkdownContainer"], .stApp div[data-testid="stCaptionContainer"] { color: __TEXT__; }
.stApp div[data-testid="stCaptionContainer"] p { color: __MUTED__; }
[data-testid="stMetric"] { background: __PANEL__; border: 1px solid __BORDER__;
    border-radius: 14px; padding: 14px 16px; }
[data-testid="stMetricLabel"] p { color: __MUTED__ !important; }
[data-testid="stMetricValue"] { color: __TEXT__; }
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
    background: __PANEL__; color: __TEXT__; border: 1px solid __BORDER__; border-radius: 10px; }
.stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {
    border-color: __ACCENT__; color: __ACCENT__; }
input, textarea, [data-baseweb="select"] > div, [data-baseweb="input"] {
    background: __PANEL__ !important; color: __TEXT__ !important; }
[data-baseweb="popover"], [data-baseweb="popover"] ul, [role="listbox"] {
    background: __PANEL__ !important; color: __TEXT__ !important; }
[data-baseweb="popover"] li { color: __TEXT__ !important; }
[data-testid="stSidebar"] [role="radiogroup"] label {
    padding: 6px 10px; border-radius: 10px; width: 100%; }
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: __BG__; }
.stProgress > div > div > div > div { background-color: __ACCENT__; }
[data-baseweb="slider"] div[role="slider"] { background-color: __ACCENT__ !important; }
hr { border-color: __BORDER__; }
</style>
"""


def apply_theme(name):
    t = THEMES[name]
    css = CSS_TEMPLATE
    for key, token in [("bg", "__BG__"), ("panel", "__PANEL__"), ("border", "__BORDER__"),
                       ("text", "__TEXT__"), ("muted", "__MUTED__"), ("accent", "__ACCENT__")]:
        css = css.replace(token, t[key])
    st.markdown(css, unsafe_allow_html=True)


def T():
    return THEMES[st.session_state.get("theme", "Dark")]


def style_fig(fig, height=420, title=None):
    t = T()
    fig.update_layout(
        template=t["plotly"], height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=t["text"]),
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        title=title,
    )
    return fig


# ============================================================
# DATA LOADING
# ============================================================
def tidy(df):
    """Normalize any OHLCV frame: naive datetime index, numeric columns, required columns."""
    if df is None or len(df) == 0:
        return None
    df = df.copy()
    idx = pd.to_datetime(df.index, errors="coerce", utc=True)
    mask = ~idx.isna()
    df = df[mask]
    df.index = idx[mask].tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Close" not in df.columns:
        return None
    df = df.dropna(subset=["Close"])
    for c in ["Open", "High", "Low"]:
        if c not in df.columns:
            df[c] = df["Close"]
        df[c] = df[c].fillna(df["Close"])
    if "Volume" not in df.columns or df["Volume"].fillna(0).sum() == 0:
        df["Volume"] = 1.0
    df["Volume"] = df["Volume"].fillna(0)
    out = df[FIELDS]
    return out if len(out) else None


@st.cache_data(ttl=900, show_spinner=False)
def download_prices(tickers, period="5y"):
    tickers = list(tickers)
    data = yf.download(
        tickers, period=period, interval="1d",
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
                d = tidy(df.dropna(how="all"))
                if d is not None:
                    out[t] = d
    else:
        d = tidy(data.copy())
        if d is not None:
            out[tickers[0]] = d
    return out


@st.cache_data(ttl=900, show_spinner=False)
def market_data(period="5y"):
    return download_prices(["SPY", "QQQ"], period)


def parse_prices_csv(raw):
    """Accepts: long format (Date,Ticker,Open..Volume), yfinance multi-header export,
    or a simple wide file (Date + one Close column per ticker)."""
    prices, fmt = {}, None

    # A) long format
    try:
        flat = pd.read_csv(io.BytesIO(raw))
        low = {str(c).strip().lower(): c for c in flat.columns}
        tcol = low.get("ticker") or low.get("symbol")
        dcol = low.get("date") or low.get("datetime")
        if tcol and dcol and "close" in low:
            flat[dcol] = pd.to_datetime(flat[dcol], errors="coerce", utc=True)
            for t, g in flat.groupby(tcol):
                g = g.set_index(dcol)
                g = g.rename(columns={c: str(c).strip().title() for c in g.columns})
                cols = [c for c in FIELDS if c in g.columns]
                d = tidy(g[cols])
                if d is not None:
                    prices[str(t).strip().upper()] = d
            fmt = "long format (Date, Ticker, prices)"
    except Exception:
        pass

    # B) yfinance multi-header export
    if not prices:
        try:
            mh = pd.read_csv(io.BytesIO(raw), header=[0, 1], index_col=0)
            if isinstance(mh.columns, pd.MultiIndex):
                l0 = set(map(str, mh.columns.get_level_values(0)))
                l1 = set(map(str, mh.columns.get_level_values(1)))
                if "Close" in l0 or "Close" in l1:
                    field_level = 0 if "Close" in l0 else 1
                    tick_level = 1 - field_level
                    for t in pd.unique(mh.columns.get_level_values(tick_level)):
                        sub = mh.xs(t, axis=1, level=tick_level)
                        sub = sub[[c for c in sub.columns if c in FIELDS]]
                        d = tidy(sub)
                        if d is not None:
                            prices[str(t).strip().upper()] = d
                    fmt = "yfinance multi-header export"
        except Exception:
            pass

    # C) simple wide file: Date + one column per ticker (Close only)
    if not prices:
        try:
            wide = pd.read_csv(io.BytesIO(raw), index_col=0)
            for c in wide.columns:
                if str(c).lower().startswith("unnamed"):
                    continue
                d = tidy(pd.DataFrame({"Close": wide[c]}))
                if d is not None:
                    prices[str(c).strip().upper()] = d
            if prices:
                fmt = "wide format (Close only: High/Low/Volume are approximated)"
        except Exception:
            pass

    prices = {t: d for t, d in prices.items() if len(d) >= 30}
    return prices, fmt


def trim(df, years):
    if not years:
        return df
    cutoff = df.index.max() - pd.DateOffset(years=years)
    return df[df.index >= cutoff]


def get_market_close(prices, source, years):
    if source == SRC_CSV:
        csv = st.session_state.get("csv_prices", {})
        if "SPY" in csv:
            return csv["SPY"]["Close"]
        try:
            m = market_data("max" if not years else f"{years}y").get("SPY")
            if m is not None:
                return m["Close"]
        except Exception:
            pass
        closes = pd.concat({t: d["Close"] for t, d in prices.items()}, axis=1)
        return closes.pct_change().mean(axis=1).fillna(0).add(1).cumprod()
    m = market_data("max" if not years else f"{years}y").get("SPY")
    return None if m is None else m["Close"]


# ============================================================
# MODEL
# ============================================================
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
    f["range_14"] = ((h - l) / c).rolling(14).mean()
    f["drawdown_60"] = c / c.rolling(60).max() - 1

    m = market_close.reindex(f.index).ffill()
    f["market_ret_5"] = m.pct_change(5)
    f["market_ret_20"] = m.pct_change(20)
    f["relative_ret_20"] = f["ret_20"] - f["market_ret_20"]

    return f.replace([np.inf, -np.inf], np.nan)


def train_models(df, market_close, speed=SPEED_FAST):
    sp = SPEED_PARAMS[speed]
    features = make_features(df, market_close)
    models, diagnostics = {}, {}
    Xcols = list(features.columns)

    for h in HORIZONS:
        y = df["Close"].shift(-h) / df["Close"] - 1
        data = features.join(y.rename("target")).dropna()
        if len(data) < 300:
            continue

        # Time-ordered split with a gap of h rows so training targets never
        # overlap the test period.
        split = int(len(data) * 0.8)
        train, test = data.iloc[:max(split - h, 1)], data.iloc[split:]
        model = HistGradientBoostingRegressor(
            max_iter=sp["max_iter"], learning_rate=sp["learning_rate"],
            max_leaf_nodes=sp["max_leaf_nodes"], l2_regularization=1.0, random_state=42
        )
        model.fit(train[Xcols], train["target"])
        pred = model.predict(test[Xcols])
        diagnostics[h] = {
            "MAE": float(mean_absolute_error(test["target"], pred)),
            "Baseline MAE": float(mean_absolute_error(
                test["target"], np.full(len(test), train["target"].mean()))),
            "Train rows": len(train),
            "Test rows": len(test),
        }
        models[h] = (model, Xcols)

    return features, models, diagnostics


def score_stock(df, market_close, speed=SPEED_FAST):
    features, models, diagnostics = train_models(df, market_close, speed)
    valid = features.dropna()
    if valid.empty or not models:
        return None
    latest = valid.iloc[[-1]]

    raw = {h: float(model.predict(latest[cols])[0]) for h, (model, cols) in models.items()}

    # Risk-adjusted signal: raw forecast divided by recent volatility.
    vol = float(features["vol_20"].dropna().iloc[-1])
    adjusted = {h: raw[h] / max(vol * np.sqrt(h / 5), 0.02) for h in raw}

    best_h = max(adjusted, key=adjusted.get)
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
        "raw_predictions": raw,
        "predictions": adjusted,
        "best_horizon": best_h,
        "errors": diagnostics,
        "latest_price": float(df["Close"].iloc[-1]),
        "feature_frame": features,
    }


def backtest(df, market_close, horizon=20, cost=0.001, threshold=0.02, speed=SPEED_FAST):
    """Walk-forward, non-overlapping trades, purge gap before each test block,
    and a per-trade cost."""
    sp = SPEED_PARAMS[speed]
    features = make_features(df, market_close)
    y = df["Close"].shift(-horizon) / df["Close"] - 1
    data = features.join(y.rename("target")).dropna()
    if len(data) < 500:
        return None

    Xcols = list(features.columns)
    step = max(10, horizon)
    start = max(300, int(len(data) * 0.6))
    dates, strat, bh, entered = [], [], [], []

    for i in range(start, len(data) - horizon, step):
        train = data.iloc[:max(i - horizon, 0)]
        if len(train) < 250:
            continue
        test = data.iloc[i:min(i + step, len(data) - horizon)]
        model = HistGradientBoostingRegressor(
            max_iter=sp["bt_iter"], learning_rate=sp["bt_lr"], max_leaf_nodes=12,
            l2_regularization=1.0, random_state=42
        )
        model.fit(train[Xcols], train["target"])
        p = model.predict(test[Xcols])
        for j in range(0, len(test), horizon):
            actual = float(test["target"].iloc[j])
            enter = float(p[j]) > threshold
            strat.append(actual - cost if enter else 0.0)
            bh.append(actual)
            entered.append(enter)
            dates.append(test.index[j])

    if not strat:
        return None

    eq = (1 + pd.Series(strat, index=dates)).cumprod()
    bhs = (1 + pd.Series(bh, index=dates)).cumprod()
    wins = [s > 0 for s, e in zip(strat, entered) if e]
    return {
        "strategy_return": float(eq.iloc[-1] - 1),
        "buy_hold_return": float(bhs.iloc[-1] - 1),
        "max_drawdown": float((eq / eq.cummax() - 1).min()),
        "observations": len(strat),
        "trades": int(sum(entered)),
        "win_rate": (sum(wins) / len(wins)) if wins else None,
        "equity": eq,
        "buy_hold": bhs,
    }


# ============================================================
# ANALYSIS RUNNER
# ============================================================
def run_analysis():
    source = st.session_state.get("data_source", SRC_YAHOO)
    window = st.session_state.get("window", "5y")
    speed = st.session_state.get("speed", SPEED_FAST)
    max_stocks = int(st.session_state.get("max_stocks", 40))
    years = {"2y": 2, "5y": 5, "10y": 10, "All": None}[window]

    if source == SRC_CSV:
        csv = st.session_state.get("csv_prices")
        if not csv:
            st.error("No CSV loaded yet. Upload prices.csv in the 🗄️ Data Center first.")
            return False
        ver = st.session_state.get("csv_version", 0)
        chosen = st.session_state.get(f"csv_universe_{ver}") or [t for t in csv if t not in BENCHMARKS]
        chosen = chosen[:max_stocks]
        prices = {t: trim(csv[t], years) for t in chosen if t in csv}
    else:
        tickers = [x.strip().upper() for x in st.session_state.get("tickers_text", ",".join(DEFAULT_TICKERS)).split(",") if x.strip()]
        if not tickers:
            st.warning("Enter at least one ticker in the sidebar.")
            return False
        tickers = tickers[:max_stocks]
        with st.spinner("Downloading market history from Yahoo Finance..."):
            prices = download_prices(tickers, "max" if years is None else window)

    if not prices:
        st.error("No market data was returned. Check the tickers, your data source, or your internet connection.")
        return False

    market_close = get_market_close(prices, source, years)
    if market_close is None:
        st.error("SPY market benchmark data is unavailable.")
        return False

    cache = st.session_state.setdefault("score_cache", {})
    mkey = str(market_close.index[-1]) if len(market_close) else ""

    rows, details = [], {}
    progress = st.progress(0, text="Training models...")
    total = max(len(prices), 1)
    for n, (ticker, df) in enumerate(prices.items(), 1):
        try:
            if len(df) >= MIN_ROWS:
                key = (ticker, len(df), str(df.index[-1]), speed, window, source, mkey)
                result = cache.get(key)
                if result is None:
                    progress.progress((n - 1) / total, text=f"Training {ticker}... ({n}/{total})")
                    result = score_stock(df, market_close, speed)
                    if result:
                        cache[key] = result
                if result:
                    bh = result["best_horizon"]
                    rows.append({
                        "Ticker": ticker,
                        "Price": result["latest_price"],
                        "Best hold": f"{bh} days",
                        "Predicted return": result["raw_predictions"][bh],
                        "Risk-adj score": result["predictions"][bh],
                        "RSI": result["features"]["RSI"],
                        "20D momentum": result["features"]["20D momentum"],
                        "60D momentum": result["features"]["60D momentum"],
                    })
                    details[ticker] = result
        except Exception:
            pass
        progress.progress(n / total, text=f"Done {n}/{total}")
    progress.empty()

    if not rows:
        st.error(f"Not enough clean history to score these stocks (need at least {MIN_ROWS} trading days each).")
        return False

    ranking = (pd.DataFrame(rows)
               .sort_values("Risk-adj score", ascending=False)
               .reset_index(drop=True))
    st.session_state["analysis"] = {
        "ranking": ranking, "details": details, "prices": prices,
        "market_close": market_close, "source": source, "window": window,
        "speed": speed, "ran_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    }
    return True


def require_analysis():
    if "analysis" not in st.session_state:
        if not run_analysis():
            st.stop()
    return st.session_state["analysis"]


def show_table(df):
    d = df.copy()
    for c in ["Predicted return", "20D momentum", "60D momentum"]:
        if c in d:
            d[c] = d[c] * 100
    fmt = {"Price": "${:,.2f}", "Predicted return": "{:.2f}%", "Risk-adj score": "{:.2f}",
           "RSI": "{:.1f}", "20D momentum": "{:.2f}%", "60D momentum": "{:.2f}%"}
    fmt = {k: v for k, v in fmt.items() if k in d}
    st.dataframe(d.style.format(fmt), use_container_width=True, hide_index=True)


DISCLAIMER = ("Educational research only. StockAI cannot know the future or guarantee profit. "
              "Model output is based on historical patterns and is not personalized financial advice.")

# ============================================================
# SIDEBAR
# ============================================================
if "theme" not in st.session_state:
    st.session_state["theme"] = "Dark"

with st.sidebar:
    st.markdown("## 📈 StockAI")
    st.caption("ML market research dashboard")
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.divider()
    st.selectbox("🎨 Theme", list(THEMES), key="theme")
    st.divider()
    st.markdown("**Analysis settings**")
    _source = st.session_state.get("data_source", SRC_YAHOO)
    st.caption(f"Data source: {_source}")
    if _source == SRC_CSV and st.session_state.get("csv_prices"):
        _avail = sorted(t for t in st.session_state["csv_prices"] if t not in BENCHMARKS)
        st.multiselect("Tickers from CSV", _avail, default=_avail,
                       key=f"csv_universe_{st.session_state.get('csv_version', 0)}")
    else:
        st.text_area("Tickers (comma-separated)", ",".join(DEFAULT_TICKERS), height=130, key="tickers_text")
    st.selectbox("Historical training window", ["2y", "5y", "10y", "All"], index=1, key="window")
    st.selectbox("Speed", [SPEED_FAST, SPEED_STANDARD], key="speed",
                 help="Fast uses lighter models and finishes several times quicker.")
    st.number_input("Max stocks to analyze", min_value=5, max_value=500, value=40, step=5, key="max_stocks",
                    help="Only the first N tickers are analyzed. Raise it once you know how fast it runs.")
    run_clicked = st.button("🔄 Analyze market", use_container_width=True)
    st.caption("Finished stocks are remembered, so re-running skips them.")

apply_theme(st.session_state["theme"])

if run_clicked:
    run_analysis()

min_pred = st.session_state.get("min_prediction", 0.03)

# ============================================================
# PAGES
# ============================================================
if page == "🏠 Dashboard":
    A = require_analysis()
    rk = A["ranking"]
    top = rk.iloc[0]
    st.title("🏠 Dashboard")
    st.caption("Historical-pattern research engine • ML forecasts • holding-period analysis • walk-forward backtesting")
    st.info(DISCLAIMER)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Top-ranked ticker", top["Ticker"])
    c2.metric("Latest price", f"${top['Price']:,.2f}")
    c3.metric("Best modeled hold", top["Best hold"])
    c4.metric("Predicted return", f"{top['Predicted return'] * 100:.1f}%")

    if top["Predicted return"] >= min_pred:
        st.success(f"Research signal: **{top['Ticker']}** has the strongest modeled risk-adjusted "
                   f"opportunity in your universe, with a modeled holding period of **{top['Best hold']}**.")
    else:
        st.warning("The model does not see a strong enough positive opportunity under your current threshold.")

    d1, d2, d3 = st.columns(3)
    d1.metric("Stocks scored", len(rk))
    d2.metric("Data source", "CSV" if A["source"] == SRC_CSV else "Yahoo")
    d3.metric("Last analysis", A["ran_at"])

    left, right = st.columns(2)
    with left:
        t10 = rk.head(10).iloc[::-1]
        fig = go.Figure(go.Bar(x=t10["Risk-adj score"], y=t10["Ticker"], orientation="h",
                               marker_color=T()["accent"]))
        st.plotly_chart(style_fig(fig, 380, "Top 10 by risk-adjusted score"), use_container_width=True)
    with right:
        mc = A["market_close"].dropna().tail(252)
        fig = go.Figure(go.Scatter(x=mc.index, y=mc.values, mode="lines", line=dict(color=T()["accent"])))
        st.plotly_chart(style_fig(fig, 380, "Market benchmark (last ~1 year)"), use_container_width=True)

    st.subheader("Top 5 right now")
    show_table(rk.head(5))

elif page == "🔍 Stock Screener":
    A = require_analysis()
    rk = A["ranking"]
    st.title("🔍 Stock Screener")
    f1, f2, f3 = st.columns(3)
    min_ret = f1.slider("Min predicted return (%)", -10.0, 20.0, -10.0, 0.5)
    rsi_rng = f2.slider("RSI range", 0, 100, (0, 100))
    min_mom = f3.slider("Min 60D momentum (%)", -50.0, 100.0, -50.0, 1.0)
    g1, g2 = st.columns(2)
    holds = g1.multiselect("Best hold", sorted(rk["Best hold"].unique(), key=lambda s: int(s.split()[0])),
                           default=sorted(rk["Best hold"].unique(), key=lambda s: int(s.split()[0])))
    search = g2.text_input("Ticker contains", "").strip().upper()

    flt = rk[(rk["Predicted return"] * 100 >= min_ret)
             & (rk["RSI"].between(rsi_rng[0], rsi_rng[1]))
             & (rk["60D momentum"] * 100 >= min_mom)
             & (rk["Best hold"].isin(holds))]
    if search:
        flt = flt[flt["Ticker"].str.contains(search)]

    st.write(f"**{len(flt)}** of {len(rk)} stocks match.")
    show_table(flt)
    st.download_button("⬇️ Download results (CSV)", flt.to_csv(index=False), "screener_results.csv", "text/csv")

    if len(flt):
        fig = go.Figure(go.Scatter(
            x=flt["20D momentum"] * 100, y=flt["Predicted return"] * 100,
            mode="markers+text", text=flt["Ticker"], textposition="top center",
            marker=dict(size=10, color=T()["accent"])))
        fig.update_xaxes(title="20D momentum (%)")
        fig.update_yaxes(title="Predicted return (%)")
        st.plotly_chart(style_fig(fig, 460, "Momentum vs predicted return"), use_container_width=True)

elif page == "🤖 AI Predictions":
    A = require_analysis()
    rk, details = A["ranking"], A["details"]
    st.title("🤖 AI Predictions")
    top = rk.iloc[0]
    if top["Predicted return"] >= min_pred:
        st.success(f"Top modeled opportunity: **{top['Ticker']}** • hold **{top['Best hold']}** • "
                   f"predicted return **{top['Predicted return'] * 100:.1f}%**")
    else:
        st.warning("No stock clears your minimum predicted return threshold right now.")
    st.subheader("Full ranking")
    show_table(rk)
    st.caption("Ranked by risk-adjusted score (forecast divided by recent volatility). "
               "'Predicted return' is the raw model forecast at the best hold length.")

    st.subheader("Risk-adjusted signal by holding period")
    top_n = rk.head(25)["Ticker"].tolist()
    z = [[details[t]["predictions"].get(h, np.nan) for h in HORIZONS] for t in top_n]
    fig = go.Figure(go.Heatmap(z=z, x=[f"{h}d" for h in HORIZONS], y=top_n, colorscale="RdYlGn",
                               zmid=0, colorbar=dict(title="Signal")))
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(style_fig(fig, max(380, 22 * len(top_n) + 80)), use_container_width=True)

elif page == "📊 Stock Analysis":
    A = require_analysis()
    rk, details, prices = A["ranking"], A["details"], A["prices"]
    st.title("📊 Stock Analysis")
    s1, s2 = st.columns([2, 1])
    selected = s1.selectbox("Choose a stock", rk["Ticker"].tolist(), index=0)
    rng = s2.selectbox("Chart range", ["6M", "1Y", "2Y", "All"], index=1)
    df, r = prices[selected], details[selected]

    n = {"6M": 126, "1Y": 252, "2Y": 504, "All": None}[rng]
    overlays = st.multiselect("Moving averages", ["20D", "50D", "200D"], default=["50D", "200D"])
    view = df.tail(n) if n else df
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=view.index, open=view["Open"], high=view["High"], low=view["Low"],
                                 close=view["Close"], name=selected))
    for ma in overlays:
        w = int(ma[:-1])
        s = df["Close"].rolling(w).mean()
        s = s.tail(n) if n else s
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=f"SMA {w}", line=dict(width=1.5)))
    fig.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(style_fig(fig, 520, f"{selected} price"), use_container_width=True)

    a, b = st.columns(2)
    with a:
        hs = [h for h in HORIZONS if h in r["raw_predictions"]]
        fig = go.Figure(go.Bar(x=[f"{h} days" for h in hs],
                               y=[r["raw_predictions"][h] * 100 for h in hs],
                               marker_color=T()["accent"]))
        st.plotly_chart(style_fig(fig, 320, "Raw model forecast (%)"), use_container_width=True)
        fig = go.Figure(go.Bar(x=[f"{h} days" for h in hs], y=[r["predictions"][h] for h in hs],
                               marker_color=T()["good"]))
        st.plotly_chart(style_fig(fig, 320, "Risk-adjusted signal"), use_container_width=True)
        st.caption("Signals are model outputs, not guaranteed returns.")
    with b:
        st.markdown("### Current technical state")
        for k, v in r["features"].items():
            if "momentum" in k or "average" in k or "volatility" in k:
                st.write(f"**{k}:** {v * 100:.2f}%")
            else:
                st.write(f"**{k}:** {v:.2f}")
        rs = r["feature_frame"]["rsi_14"].dropna().tail(n or 504)
        fig = go.Figure(go.Scatter(x=rs.index, y=rs.values, mode="lines", line=dict(color=T()["accent"])))
        fig.add_hline(y=70, line_dash="dot")
        fig.add_hline(y=30, line_dash="dot")
        st.plotly_chart(style_fig(fig, 260, "RSI (14)"), use_container_width=True)

elif page == "🧪 Backtesting":
    A = require_analysis()
    rk, prices, market_close = A["ranking"], A["prices"], A["market_close"]
    st.title("🧪 Walk-forward Backtest")
    st.caption("Trains only on older data, leaves a gap before each test block, uses non-overlapping trades "
               "and subtracts a trading cost. A research tool, not a trading rule.")
    b1, b2, b3 = st.columns(3)
    tk = b1.selectbox("Stock", rk["Ticker"].tolist(), index=0)
    bt_h = b2.selectbox("Backtest horizon (days)", HORIZONS, index=2)
    cost = b3.number_input("Cost per trade (%)", 0.0, 2.0, 0.10, 0.05) / 100
    thr = st.slider("Enter only when predicted return exceeds (%)", 0.0, 10.0, 2.0, 0.5) / 100

    if st.button("▶️ Run backtest"):
        with st.spinner("Running walk-forward backtest..."):
            bt = backtest(prices[tk], market_close, bt_h, cost, thr, st.session_state.get("speed", SPEED_FAST))
        if bt:
            x, y, z, w = st.columns(4)
            x.metric("Strategy return", f"{bt['strategy_return'] * 100:.1f}%")
            y.metric("Buy-and-hold (same windows)", f"{bt['buy_hold_return'] * 100:.1f}%")
            z.metric("Max drawdown", f"{bt['max_drawdown'] * 100:.1f}%")
            w.metric("Win rate", "n/a" if bt["win_rate"] is None else f"{bt['win_rate'] * 100:.0f}%")
            st.write(f"Test windows: {bt['observations']} • Trades taken: {bt['trades']}")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=bt["equity"].index, y=bt["equity"].values, name="Strategy",
                                     line=dict(color=T()["accent"])))
            fig.add_trace(go.Scatter(x=bt["buy_hold"].index, y=bt["buy_hold"].values, name="Buy & hold",
                                     line=dict(color=T()["muted"], dash="dot")))
            st.plotly_chart(style_fig(fig, 420, "Growth of $1"), use_container_width=True)
        else:
            st.warning("Not enough data for this backtest (needs roughly 500+ usable rows).")

elif page == "⭐ Watchlist":
    A = require_analysis()
    rk, prices = A["ranking"], A["prices"]
    st.title("⭐ Watchlist")
    options = rk["Ticker"].tolist()
    if "watchlist_widget" not in st.session_state:
        st.session_state["watchlist_widget"] = list(st.session_state.get("watchlist", []))
    st.session_state["watchlist_widget"] = [t for t in st.session_state["watchlist_widget"] if t in options]
    sel = st.multiselect("Your watchlist", options, key="watchlist_widget")
    st.session_state["watchlist"] = sel

    if sel:
        show_table(rk[rk["Ticker"].isin(sel)])
        fig = go.Figure()
        for t in sel:
            s = prices[t]["Close"].tail(252)
            fig.add_trace(go.Scatter(x=s.index, y=s / s.iloc[0] * 100, mode="lines", name=t))
        st.plotly_chart(style_fig(fig, 420, "Last ~1 year, rebased to 100"), use_container_width=True)
    else:
        st.info("Pick some tickers above to build your watchlist.")

elif page == "💼 Portfolio":
    A = require_analysis()
    prices, details = A["prices"], A["details"]
    st.title("💼 Portfolio (hypothetical)")
    st.caption("Enter hypothetical holdings and click Save. Prices come from the loaded dataset.")
    saved = st.session_state.get("portfolio_df", pd.DataFrame({
        "Ticker": ["AAPL", "MSFT"], "Shares": [10.0, 5.0], "Avg cost": [150.0, 300.0]}))
    with st.form("pf_form"):
        edited = st.data_editor(saved, num_rows="dynamic", use_container_width=True, hide_index=True,
                                column_config={
                                    "Ticker": st.column_config.TextColumn("Ticker"),
                                    "Shares": st.column_config.NumberColumn("Shares", min_value=0.0),
                                    "Avg cost": st.column_config.NumberColumn("Avg cost ($)", min_value=0.0)})
        saved_clicked = st.form_submit_button("💾 Save holdings")
    if saved_clicked:
        pf_new = edited.dropna(subset=["Ticker"]).copy()
        pf_new["Ticker"] = pf_new["Ticker"].astype(str).str.upper().str.strip()
        pf_new["Shares"] = pd.to_numeric(pf_new["Shares"], errors="coerce").fillna(0.0)
        pf_new["Avg cost"] = pd.to_numeric(pf_new["Avg cost"], errors="coerce").fillna(0.0)
        st.session_state["portfolio_df"] = pf_new.reset_index(drop=True)
        saved = st.session_state["portfolio_df"]

    rows = []
    for _, r in saved.dropna(subset=["Ticker"]).iterrows():
        t = str(r["Ticker"]).upper().strip()
        sh, cost_b = float(r["Shares"]), float(r["Avg cost"])
        px = float(prices[t]["Close"].iloc[-1]) if t in prices else np.nan
        value, basis = sh * px, sh * cost_b
        d = details.get(t)
        rows.append({
            "Ticker": t, "Shares": sh, "Avg cost": cost_b, "Last price": px,
            "Value": value, "P/L $": value - basis,
            "P/L %": (value / basis - 1) * 100 if basis > 0 else np.nan,
            "Model best hold": f"{d['best_horizon']} days" if d else "n/a",
            "Model predicted return %": d["raw_predictions"][d["best_horizon"]] * 100 if d else np.nan,
        })
    if rows:
        pf = pd.DataFrame(rows)
        tv, tp = pf["Value"].sum(skipna=True), pf["P/L $"].sum(skipna=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Total value", f"${tv:,.2f}")
        m2.metric("Total P/L", f"${tp:,.2f}")
        m3.metric("Positions", len(pf))
        st.dataframe(pf.style.format({
            "Shares": "{:,.2f}", "Avg cost": "${:,.2f}", "Last price": "${:,.2f}", "Value": "${:,.2f}",
            "P/L $": "${:,.2f}", "P/L %": "{:.1f}%", "Model predicted return %": "{:.2f}%"}),
            use_container_width=True, hide_index=True)
        alloc = pf.dropna(subset=["Value"])
        if len(alloc) and alloc["Value"].sum() > 0:
            fig = go.Figure(go.Pie(labels=alloc["Ticker"], values=alloc["Value"], hole=0.5))
            st.plotly_chart(style_fig(fig, 380, "Allocation"), use_container_width=True)
        missing = pf[pf["Last price"].isna()]["Ticker"].tolist()
        if missing:
            st.warning(f"No price data loaded for: {', '.join(missing)}. Add them to your universe and re-run.")

elif page == "🗄️ Data Center":
    st.title("🗄️ Data Center")
    st.caption("Choose where StockAI gets its price history, and check the data before analyzing.")

    prev = st.session_state.get("data_source", SRC_YAHOO)
    choice = st.radio("Data source", [SRC_YAHOO, SRC_CSV], index=[SRC_YAHOO, SRC_CSV].index(prev), horizontal=True)
    if choice != prev:
        st.session_state["data_source"] = choice
        st.session_state.pop("analysis", None)
        st.rerun()

    st.subheader("Upload prices.csv")
    up = st.file_uploader("Upload the CSV you created in Colab", type=["csv"])
    if up is not None:
        sig = (up.name, up.size)
        if st.session_state.get("csv_sig") != sig:
            with st.spinner("Reading and validating file..."):
                parsed, fmt = parse_prices_csv(up.getvalue())
            if parsed:
                st.session_state["csv_prices"] = parsed
                st.session_state["csv_sig"] = sig
                st.session_state["csv_fmt"] = fmt
                st.session_state["csv_version"] = st.session_state.get("csv_version", 0) + 1
                st.session_state["data_source"] = SRC_CSV
                st.session_state.pop("analysis", None)
                st.rerun()
            else:
                st.error("Couldn't find usable price data in that file. See the accepted formats below.")

    with st.expander("Accepted CSV formats"):
        st.markdown(
            "- **Long:** columns `Date, Ticker, Open, High, Low, Close, Volume`\n"
            "- **yfinance export:** the file saved directly from `yf.download(...).to_csv(...)` (two header rows)\n"
            "- **Wide:** `Date` plus one Close column per ticker (High/Low/Volume are approximated)\n\n"
            "Each ticker needs at least 350 trading days to be scored.")

    csv = st.session_state.get("csv_prices")
    if csv:
        st.success(f"CSV loaded: {len(csv)} tickers • format: {st.session_state.get('csv_fmt')}")
        rep = []
        for t, d in csv.items():
            gaps = int((d.index.to_series().diff().dt.days > 7).sum())
            rep.append({"Ticker": t, "Start": d.index.min().date(), "End": d.index.max().date(),
                        "Rows": len(d), "Gaps > 7 days": gaps,
                        "Usable": "✅" if len(d) >= MIN_ROWS else "⚠️ too short"})
        rep = pd.DataFrame(rep).sort_values("Ticker")
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Tickers", len(rep))
        v2.metric("Usable", int((rep["Usable"] == "✅").sum()))
        v3.metric("Earliest date", str(rep["Start"].min()))
        v4.metric("Latest date", str(rep["End"].max()))
        if "SPY" not in csv:
            st.warning("SPY isn't in your CSV, so StockAI will fetch the SPY benchmark from Yahoo Finance.")
        st.dataframe(rep, use_container_width=True, hide_index=True)
        pick = st.selectbox("Preview a ticker", sorted(csv))
        st.dataframe(csv[pick].tail(10), use_container_width=True)
        if st.button("🗑️ Remove uploaded data"):
            for k in ["csv_prices", "csv_sig", "csv_fmt"]:
                st.session_state.pop(k, None)
            st.session_state["data_source"] = SRC_YAHOO
            st.session_state.pop("analysis", None)
            st.rerun()
    else:
        st.info("No CSV uploaded yet. You can still use live Yahoo Finance data.")

    st.subheader("Current analysis")
    A = st.session_state.get("analysis")
    if A:
        st.write(f"Source: **{A['source']}** • Window: **{A['window']}** • Speed: **{A.get('speed', '')}** • "
                 f"Stocks scored: **{len(A['ranking'])}** • Last run: **{A['ran_at']}**")
    else:
        st.write("Nothing analyzed yet. Click **🔄 Analyze market** in the sidebar.")

elif page == "🧠 Model Lab":
    A = require_analysis()
    rk, details = A["ranking"], A["details"]
    st.title("🧠 Model Lab")
    tk = st.selectbox("Stock", rk["Ticker"].tolist(), index=0)
    diag = details[tk]["errors"]
    rows = []
    for h in HORIZONS:
        if h in diag:
            d = diag[h]
            rows.append({"Horizon": f"{h} days", "Model MAE": d["MAE"] * 100,
                         "Baseline MAE": d["Baseline MAE"] * 100,
                         "Improvement vs baseline": (1 - d["MAE"] / d["Baseline MAE"]) * 100,
                         "Train rows": d["Train rows"], "Test rows": d["Test rows"]})
    if rows:
        md = pd.DataFrame(rows)
        st.dataframe(md.style.format({"Model MAE": "{:.2f}%", "Baseline MAE": "{:.2f}%",
                                      "Improvement vs baseline": "{:.1f}%"}),
                     use_container_width=True, hide_index=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=md["Horizon"], y=md["Model MAE"], name="Model", marker_color=T()["accent"]))
        fig.add_trace(go.Bar(x=md["Horizon"], y=md["Baseline MAE"], name="Baseline (mean)", marker_color=T()["muted"]))
        fig.update_layout(barmode="group")
        st.plotly_chart(style_fig(fig, 340, "Out-of-sample error (lower is better)"), use_container_width=True)
        st.caption("If the model's error isn't clearly below the baseline, its signal for that stock and "
                   "horizon is weak and shouldn't be trusted.")
    st.subheader("How StockAI learns")
    st.markdown("""
1. **Price behavior:** 1/5/10/20/60-day returns, moving-average relationships, volatility and drawdowns.
2. **Momentum + mean-reversion clues:** RSI, distance from moving averages and recent performance.
3. **Volume behavior:** current volume relative to its recent average.
4. **Market context:** SPY returns and the stock's relative performance against the market.
5. **Multiple horizons:** separate models estimate 5, 10, 20 and 40 trading-day outcomes.
6. **Risk adjustment:** forecasts are divided by recent volatility so the largest raw forecast doesn't automatically win.
7. **Time-ordered testing:** training data always comes before test data, with a gap so targets never overlap.
    """)
    sp = SPEED_PARAMS[A.get("speed", SPEED_FAST)]
    st.markdown(f"**Model:** HistGradientBoostingRegressor ({sp['max_iter']} iterations, "
                f"learning rate {sp['learning_rate']}, {sp['max_leaf_nodes']} leaves, L2 = 1.0) • "
                f"Speed mode: {A.get('speed', SPEED_FAST)}")

elif page == "⚙️ Settings":
    st.title("⚙️ Settings")
    st.subheader("Signal threshold")
    val = st.slider("Minimum predicted return to highlight (%)", -10.0, 20.0,
                    float(st.session_state.get("min_prediction", 0.03) * 100), 0.5)
    st.session_state["min_prediction"] = val / 100

    st.subheader("Themes")
    st.caption("Switch themes from the sidebar. Available palettes:")
    cols = st.columns(len(THEMES))
    for col, (name, t) in zip(cols, THEMES.items()):
        col.markdown(
            f"<div style='background:{t['bg']};border:2px solid {t['accent']};border-radius:12px;"
            f"padding:14px;text-align:center'><span style='color:{t['text']};font-weight:600'>{name}</span><br>"
            f"<span style='color:{t['accent']}'>■■■</span></div>", unsafe_allow_html=True)
    st.caption("Tables use Streamlit's own colors. To restyle those too, use the ⋮ menu → Settings → Theme.")

    st.subheader("Maintenance")
    m1, m2, m3 = st.columns(3)
    if m1.button("♻️ Clear cached downloads"):
        st.cache_data.clear()
        st.success("Cache cleared.")
    if m2.button("🧹 Reset analysis results"):
        st.session_state.pop("analysis", None)
        st.success("Analysis cleared. It will re-run when you open a page that needs it.")
    if m3.button("🗑️ Clear trained-model cache"):
        st.session_state.pop("score_cache", None)
        st.success("Trained results cleared. The next analysis will retrain every stock.")

    st.subheader("About")
    st.info(DISCLAIMER)

st.divider()
st.caption("Model limitations: historical patterns can disappear, news can overwhelm technical signals, "
           "and predictions carry uncertainty. Never assume a model can guarantee profit.")
