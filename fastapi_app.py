"""
TrendFire FastAPI Backend
--------------------------
Exposes the same TTM Squeeze / MWD scanning logic used in the Streamlit app
as a clean REST API — so any frontend (React, Lovable, mobile app, or the
existing Streamlit UI) can call it instead of duplicating the logic.

Run locally:
    pip install fastapi uvicorn yfinance pandas numpy --break-system-packages
    uvicorn fastapi_app:app --reload --port 8000

Then visit http://localhost:8000/docs for interactive API documentation
(FastAPI auto-generates this — no extra work needed).

Endpoints:
    GET  /api/health
    GET  /api/stocks/default-fno
    GET  /api/stocks/default-equity
    GET  /api/fno/scan?symbols=RELIANCE,TCS,INFY
    GET  /api/fno/stock/{symbol}
    GET  /api/equity/scan?symbols=RELIANCE,TCS,INFY
    GET  /api/equity/stock/{symbol}
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
import yfinance as yf
import pandas as pd
import numpy as np

app = FastAPI(
    title="TrendFire API",
    description="TTM Squeeze screening for NSE stocks — F&O momentum + Equity positioning",
    version="1.0.0",
)

# Allow any frontend (React dev server, Lovable preview, Streamlit, etc.) to call this API.
# Tighten allow_origins to your actual frontend domain(s) once you have one, for security.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Core squeeze logic (identical to the Streamlit app) ─────────────────────
def linreg_series(series, length):
    values = series.values
    result = np.full(len(values), np.nan)
    x = np.arange(length)
    for i in range(length - 1, len(values)):
        y = values[i - length + 1:i + 1]
        if np.any(np.isnan(y)):
            continue
        slope, intercept = np.polyfit(x, y, 1)
        result[i] = slope * (length - 1) + intercept
    return pd.Series(result, index=series.index)


def compute_squeeze(df, length=20, mult_bb=2.0, mult_kc=1.5):
    close, high, low = df["Close"], df["High"], df["Low"]
    basis = close.rolling(length).mean()
    dev = mult_bb * close.rolling(length).std()
    upperBB, lowerBB = basis + dev, basis - dev
    ma = close.rolling(length).mean()
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    rng = tr.rolling(length).mean()
    upperKC, lowerKC = ma + rng * mult_kc, ma - rng * mult_kc
    sqzOn = (lowerBB > lowerKC) & (upperBB < upperKC)
    sqzOff = (lowerBB < lowerKC) & (upperBB > upperKC)
    highestH = high.rolling(length).max()
    lowestL = low.rolling(length).min()
    avgVal = ((highestH + lowestL) / 2 + close.rolling(length).mean()) / 2
    val = linreg_series(close - avgVal, length)
    return val, sqzOn, sqzOff


def get_status(val, val_prev):
    if pd.isna(val) or pd.isna(val_prev):
        return "— No Data"
    if val > 0 and val > val_prev:
        return "🟢 Bullish Momentum"
    elif val > 0:
        return "🟡 Topping"
    elif val < 0 and val < val_prev:
        return "🔴 Bearish Momentum"
    return "🟠 Bottoming"


def get_sqz_label(sqzOn, sqzOff):
    if sqzOn:
        return "Compression Active"
    if sqzOff:
        return "Fired"
    return "—"


def get_trade_status(wVal, dVal, dValPrev, dSqzOn, hVal, hValPrev):
    if any(pd.isna(x) for x in [wVal, dVal, dValPrev, hVal, hValPrev]):
        return "— Insufficient Data"
    if wVal > 0:
        if dVal < 0 and dVal >= dValPrev and dSqzOn:
            return "🏆 Prime Setup"
        elif dVal > 0 and dVal > dValPrev:
            return "✅ Confirmed Bullish"
        elif dVal > 0 and dVal <= dValPrev:
            if hVal < 0 and hVal < hValPrev:
                return "⚠️ Extended (Weakening)"
            elif hVal < 0 and hVal >= hValPrev:
                return "🎯 Consolidating"
            elif hVal > 0 and hVal > hValPrev:
                return "⚠️ Extended (Gaining)"
            return "⚠️ Extended"
        elif dVal < 0 and dVal >= dValPrev:
            return "⏳ Building Momentum"
        return "❌ Avoid"
    return "❌ Avoid"


def get_equity_status(wVal, wValPrev, dVal, dValPrev, dSqzOn):
    if any(pd.isna(x) for x in [wVal, wValPrev, dVal, dValPrev]):
        return "— Insufficient Data"
    if wVal > 0:
        if dVal > 0 and dVal > dValPrev:
            return "✅ Accumulate"
        elif dSqzOn:
            return "⏳ Watchlist"
        elif dVal < 0:
            return "🔎 Hold / Monitor"
        return "⏳ Watchlist"
    return "❌ Exit / Avoid"


def get_double_confirmation(w_val, d_val, h_val, h_val_prev):
    if any(pd.isna(x) for x in [w_val, d_val, h_val, h_val_prev]):
        return None
    if w_val > 0 and d_val > 0 and h_val > 0 and h_val_prev > 0:
        return "✅ Bullish Confirmed"
    if w_val < 0 and d_val < 0 and h_val < 0 and h_val_prev < 0:
        return "🔻 Bearish Confirmed"
    return None


def fmt_vol(v):
    if pd.isna(v) or v == 0:
        return "—"
    if v >= 1e7:
        return f"{v/1e7:.1f}Cr"
    if v >= 1e5:
        return f"{v/1e5:.1f}L"
    if v >= 1e3:
        return f"{v/1e3:.0f}K"
    return str(int(v))


def pct_change_of(close_series):
    if len(close_series) >= 2:
        prev_close = float(close_series.iloc[-2])
        last_close = float(close_series.iloc[-1])
        if prev_close > 0:
            return ((last_close - prev_close) / prev_close) * 100
    return None


# ── Data fetching ────────────────────────────────────────────────────────────
def fetch_fno_data(symbol: str):
    ticker = f"{symbol.strip().upper()}.NS"
    daily = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
    if daily.empty:
        return None
    if isinstance(daily.columns, pd.MultiIndex):
        daily.columns = daily.columns.get_level_values(0)
    daily = daily.dropna(subset=["Close"])
    weekly = daily.resample("W").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    monthly = daily.resample("ME").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    hourly = yf.download(ticker, period="30d", interval="60m", progress=False, auto_adjust=True)
    if isinstance(hourly.columns, pd.MultiIndex):
        hourly.columns = hourly.columns.get_level_values(0)
    hourly = hourly.dropna(subset=["Close"])
    return {"D": daily, "W": weekly, "M": monthly, "H": hourly}


def fetch_equity_data(symbol: str):
    ticker = f"{symbol.strip().upper()}.NS"
    daily = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
    if daily.empty:
        return None
    if isinstance(daily.columns, pd.MultiIndex):
        daily.columns = daily.columns.get_level_values(0)
    daily = daily.dropna(subset=["Close"])
    weekly = daily.resample("W").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    return {"D": daily, "W": weekly}


# ── Analysis (returns plain dicts — perfect for JSON API responses) ─────────
def analyze_fno_stock(symbol: str) -> Optional[dict]:
    data = fetch_fno_data(symbol)
    if data is None or data["H"].empty or len(data["D"]) < 60:
        return None

    results = {}
    for tf in ["M", "W", "D", "H"]:
        val, sqzOn, sqzOff = compute_squeeze(data[tf])
        results[tf] = {
            "val": val.iloc[-1], "val_prev": val.iloc[-2],
            "sqzOn": bool(sqzOn.iloc[-1]), "sqzOff": bool(sqzOff.iloc[-1]),
        }

    daily_vol = data["D"]["Volume"]
    vol_today = float(daily_vol.iloc[-1])
    avg_vol = float(daily_vol.iloc[-21:-1].mean()) if len(daily_vol) >= 21 else float(daily_vol.mean())
    vol_ratio = (vol_today / avg_vol) if avg_vol > 0 else None

    double_confirm = get_double_confirmation(
        results["W"]["val"], results["D"]["val"], results["H"]["val"], results["H"]["val_prev"]
    )

    return {
        "symbol": symbol.strip().upper(),
        "monthly": {
            "status": get_status(results["M"]["val"], results["M"]["val_prev"]),
            "squeeze": get_sqz_label(results["M"]["sqzOn"], results["M"]["sqzOff"]),
        },
        "weekly": {
            "status": get_status(results["W"]["val"], results["W"]["val_prev"]),
            "squeeze": get_sqz_label(results["W"]["sqzOn"], results["W"]["sqzOff"]),
        },
        "daily": {
            "status": get_status(results["D"]["val"], results["D"]["val_prev"]),
            "squeeze": get_sqz_label(results["D"]["sqzOn"], results["D"]["sqzOff"]),
        },
        "hourly": {
            "status": get_status(results["H"]["val"], results["H"]["val_prev"]),
            "squeeze": get_sqz_label(results["H"]["sqzOn"], results["H"]["sqzOff"]),
        },
        "trade": get_trade_status(
            results["W"]["val"], results["D"]["val"], results["D"]["val_prev"],
            results["D"]["sqzOn"], results["H"]["val"], results["H"]["val_prev"],
        ),
        "double_confirmation": double_confirm,
        "volume": fmt_vol(vol_today),
        "avg_volume": fmt_vol(avg_vol),
        "vol_ratio": round(vol_ratio, 2) if vol_ratio is not None else None,
        "pct_change": round(pct_change_of(data["D"]["Close"]), 2) if pct_change_of(data["D"]["Close"]) is not None else None,
    }


def analyze_equity_stock(symbol: str) -> Optional[dict]:
    data = fetch_equity_data(symbol)
    if data is None or len(data["D"]) < 60:
        return None

    results = {}
    for tf in ["W", "D"]:
        val, sqzOn, sqzOff = compute_squeeze(data[tf])
        results[tf] = {
            "val": val.iloc[-1], "val_prev": val.iloc[-2],
            "sqzOn": bool(sqzOn.iloc[-1]), "sqzOff": bool(sqzOff.iloc[-1]),
        }

    daily_vol = data["D"]["Volume"]
    vol_today = float(daily_vol.iloc[-1])
    avg_vol = float(daily_vol.iloc[-21:-1].mean()) if len(daily_vol) >= 21 else float(daily_vol.mean())
    vol_ratio = (vol_today / avg_vol) if avg_vol > 0 else None

    return {
        "symbol": symbol.strip().upper(),
        "weekly": {
            "status": get_status(results["W"]["val"], results["W"]["val_prev"]),
            "squeeze": get_sqz_label(results["W"]["sqzOn"], results["W"]["sqzOff"]),
        },
        "daily": {
            "status": get_status(results["D"]["val"], results["D"]["val_prev"]),
            "squeeze": get_sqz_label(results["D"]["sqzOn"], results["D"]["sqzOff"]),
        },
        "trade": get_equity_status(
            results["W"]["val"], results["W"]["val_prev"],
            results["D"]["val"], results["D"]["val_prev"], results["D"]["sqzOn"],
        ),
        "volume": fmt_vol(vol_today),
        "avg_volume": fmt_vol(avg_vol),
        "vol_ratio": round(vol_ratio, 2) if vol_ratio is not None else None,
        "pct_change": round(pct_change_of(data["D"]["Close"]), 2) if pct_change_of(data["D"]["Close"]) is not None else None,
    }


def parallel_analyze(symbols: list[str], analyze_fn, max_workers: int = 12):
    results, failed = [], []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sym = {executor.submit(analyze_fn, s): s for s in symbols}
        for future in as_completed(future_to_sym):
            sym = future_to_sym[future]
            try:
                r = future.result()
                if r:
                    results.append(r)
                else:
                    failed.append(sym)
            except Exception:
                failed.append(sym)
    return results, failed


# ── Default stock lists (same as the Streamlit app) ─────────────────────────
DEFAULT_FNO_STOCKS = [
    "360ONE", "ABB", "ABCAPITAL", "ADANIENSOL", "ADANIENT", "ADANIGREEN", "ADANIPORTS", "ADANIPOWER",
    "AFFLE", "ALKEM", "AMBER", "AMBUJACEM", "ANGELONE", "APLAPOLLO", "APOLLOHOSP", "ASHOKLEY", "ASIANPAINT",
    "ASTRAL", "AUBANK", "AUROPHARMA", "AXISBANK", "BAJAJ-AUTO", "BAJAJFINSV", "BAJAJHLDNG", "BAJFINANCE",
    "BANDHANBNK", "BANKBARODA", "BANKINDIA", "BDL", "BEL", "BHARATFORG", "BHARTIARTL", "BHEL", "BIOCON",
    "BLUESTARCO", "BOSCHLTD", "BPCL", "BRITANNIA", "BSE", "CAMS", "CANBK", "CDSL", "CGPOWER", "CHOLAFIN",
    "CIPLA", "COALINDIA", "COCHINSHIP", "COFORGE", "COLPAL", "CONCOR", "CROMPTON", "CUMMINSIND", "DABUR",
    "DALBHARAT", "DELHIVERY", "DIVISLAB", "DIXON", "DLF", "DMART", "DRREDDY", "EICHERMOT", "ETERNAL",
    "EXIDEIND", "FEDERALBNK", "FORCEMOT", "FORTIS", "GAIL", "GLENMARK", "GMRAIRPORT", "GODFRYPHLP",
    "GODREJCP", "GODREJPROP", "GRASIM", "HAL", "HAVELLS", "HCLTECH", "HDFCAMC", "HDFCBANK", "HDFCLIFE",
    "HEROMOTOCO", "HINDALCO", "HINDPETRO", "HINDUNILVR", "HINDZINC", "HYUNDAI", "ICICIBANK", "ICICIGI",
    "ICICIPRULI", "IDEA", "IDFCFIRSTB", "IEX", "INDHOTEL", "INDIANB", "INDIGO", "INDUSINDBK", "INDUSTOWER",
    "INFY", "INOXWIND", "IOC", "IREDA", "IRFC", "ITC", "JINDALSTEL", "JIOFIN", "JSWENERGY", "JSWSTEEL",
    "JUBLFOOD", "KALYANKJIL", "KAYNES", "KEI", "KFINTECH", "KOTAKBANK", "KPITTECH", "LAURUSLABS",
    "LICHSGFIN", "LICI", "LODHA", "LT", "LTF", "LTM", "LUPIN", "M&M", "MANAPPURAM", "MANKIND", "MARICO",
    "MARUTI", "MAXHEALTH", "MAZDOCK", "MCX", "MFSL", "MOTHERSON", "MOTILALOFS", "MPHASIS", "MUTHOOTFIN",
    "NAM-INDIA", "NATIONALUM", "NAUKRI", "NBCC", "NESTLEIND", "NHPC", "NMDC", "NTPC", "NUVAMA", "NYKAA",
    "OBEROIRLTY", "OFSS", "OIL", "ONGC", "PAGEIND", "PATANJALI", "PAYTM", "PERSISTENT", "PETRONET", "PFC",
    "PGEL", "PHOENIXLTD", "PIDILITIND", "PIIND", "PNB", "PNBHOUSING", "POLICYBZR", "POLYCAB", "POWERGRID",
    "POWERINDIA", "PREMIERENE", "PRESTIGE", "RADICO", "RBLBANK", "RECLTD", "RELIANCE", "RVNL", "SAIL",
    "SBICARD", "SBILIFE", "SBIN", "SHREECEM", "SHRIRAMFIN", "SIEMENS", "SOLARINDS", "SONACOMS", "SRF",
    "SUNPHARMA", "SUPREMEIND", "SUZLON", "SWIGGY", "TATACONSUM", "TATAELXSI", "TATAPOWER", "TATASTEEL",
    "TCS", "TECHM", "TIINDIA", "TITAN", "TMPV", "TORNTPHARM", "TRENT", "TVSMOTOR", "ULTRACEMCO",
    "UNIONBANK", "UNITDSPR", "UNOMINDA", "UPL", "VBL", "VEDL", "VMM", "VOLTAS", "WAAREEENER", "WIPRO",
    "YESBANK", "ZYDUSLIFE",
]
DEFAULT_EQUITY_STOCKS = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR",
    "ITC", "LT", "ASIANPAINT", "MARUTI",
]


# ── API Routes ───────────────────────────────────────────────────────────────
def fetch_index_data(ticker: str):
    """For index tickers like ^NSEI — no .NS suffix, already the full yfinance symbol."""
    daily = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
    if daily.empty:
        return None
    if isinstance(daily.columns, pd.MultiIndex):
        daily.columns = daily.columns.get_level_values(0)
    daily = daily.dropna(subset=["Close"])
    weekly = daily.resample("W").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    monthly = daily.resample("ME").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    return {"D": daily, "W": weekly, "M": monthly}


def analyze_index(ticker: str, label: str) -> Optional[dict]:
    data = fetch_index_data(ticker)
    if data is None or len(data["D"]) < 60:
        return None
    results = {}
    for tf in ["M", "W", "D"]:
        val, sqzOn, sqzOff = compute_squeeze(data[tf])
        results[tf] = {
            "val": val.iloc[-1], "val_prev": val.iloc[-2],
            "sqzOn": bool(sqzOn.iloc[-1]), "sqzOff": bool(sqzOff.iloc[-1]),
        }
    return {
        "name": label,
        "last_close": round(float(data["D"]["Close"].iloc[-1]), 2),
        "pct_change": round(pct_change_of(data["D"]["Close"]), 2) if pct_change_of(data["D"]["Close"]) is not None else None,
        "monthly": {"status": get_status(results["M"]["val"], results["M"]["val_prev"]), "squeeze": get_sqz_label(results["M"]["sqzOn"], results["M"]["sqzOff"])},
        "weekly":  {"status": get_status(results["W"]["val"], results["W"]["val_prev"]), "squeeze": get_sqz_label(results["W"]["sqzOn"], results["W"]["sqzOff"])},
        "daily":   {"status": get_status(results["D"]["val"], results["D"]["val_prev"]), "squeeze": get_sqz_label(results["D"]["sqzOn"], results["D"]["sqzOff"])},
    }


@app.get("/api/index/nifty")
def nifty_index():
    result = analyze_index("^NSEI", "NIFTY 50")
    if result is None:
        raise HTTPException(status_code=502, detail="Could not fetch NIFTY 50 index data right now")
    return result


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "TrendFire API"}


@app.get("/api/stocks/default-fno")
def default_fno_stocks():
    return {"stocks": DEFAULT_FNO_STOCKS, "count": len(DEFAULT_FNO_STOCKS)}


@app.get("/api/stocks/default-equity")
def default_equity_stocks():
    return {"stocks": DEFAULT_EQUITY_STOCKS, "count": len(DEFAULT_EQUITY_STOCKS)}


@app.get("/api/fno/stock/{symbol}")
def fno_single_stock(symbol: str):
    result = analyze_fno_stock(symbol)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Could not fetch data for {symbol}")
    return result


@app.get("/api/fno/scan")
def fno_scan(symbols: str = Query(..., description="Comma-separated stock symbols, e.g. RELIANCE,TCS,INFY")):
    symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
    if not symbol_list:
        raise HTTPException(status_code=400, detail="No symbols provided")
    results, failed = parallel_analyze(symbol_list, analyze_fno_stock)
    return {"results": results, "failed": failed, "scanned": len(symbol_list), "succeeded": len(results)}


@app.get("/api/equity/stock/{symbol}")
def equity_single_stock(symbol: str):
    result = analyze_equity_stock(symbol)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Could not fetch data for {symbol}")
    return result


@app.get("/api/equity/scan")
def equity_scan(symbols: str = Query(..., description="Comma-separated stock symbols, e.g. RELIANCE,TCS,INFY")):
    symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
    if not symbol_list:
        raise HTTPException(status_code=400, detail="No symbols provided")
    results, failed = parallel_analyze(symbol_list, analyze_equity_stock)
    return {"results": results, "failed": failed, "scanned": len(symbol_list), "succeeded": len(results)}
