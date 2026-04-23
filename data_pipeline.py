# ============================================================
# data_pipeline.py
# OHLCV + 3 Technical Indicators
#
# Features (8 total):
#   [0] Open        log-return
#   [1] High        log-return
#   [2] Low         log-return
#   [3] Close       log-return  <- target 1
#   [4] Volume      log-return
#   [5] RSI_14      raw value (0-100)
#   [6] MACD        raw value
#   [7] BB_position raw value (0-1)
#
# Targets: Open log-return + Close log-return
# ============================================================
import numpy as np
import pandas as pd
import yfinance as yf
import time
import os
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.preprocessing import StandardScaler

# Column indices
OPEN_IDX   = 0
HIGH_IDX   = 1
LOW_IDX    = 2
CLOSE_IDX  = 3
VOLUME_IDX = 4
RSI_IDX    = 5
MACD_IDX   = 6
BB_IDX     = 7

N_FEATURES = 8
N_OUTPUTS  = 2
SEQ_LEN    = 60


def load_price(symbol, years=10):
    cache_dir = os.path.join(os.path.dirname(__file__), "cache_prices")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{symbol}_{years}y.pkl")

    if os.path.exists(cache_path):
        df = pd.read_pickle(cache_path)
        loaded_from_cache = not df.empty
    else:
        loaded_from_cache = False

    if not loaded_from_cache:
        start_ts = pd.Timestamp.today() - pd.DateOffset(years=years)

        last_err = None
        for _ in range(4):
            try:
                df = yf.download(
                    symbol,
                    start=start_ts,
                    progress=False,
                    threads=False,
                )
                if df is not None and not df.empty:
                    break
                last_err = ValueError(f"Empty dataframe for {symbol}")
            except Exception as e:
                last_err = e
                df = None
            time.sleep(3)
        else:
            raise ValueError(f"yfinance download failed for {symbol}: {last_err}")

    if df.empty:
        raise ValueError(f"No data for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[df.index.dayofweek < 5]
    holidays = USFederalHolidayCalendar().holidays(
        start=df.index.min(), end=df.index.max()
    )
    df = df[~df.index.isin(holidays)]
    df.dropna(inplace=True)

    # Save clean cache
    df.to_pickle(cache_path)
    return df


def compute_rsi(series, period=14):
    """
    RSI - Relative Strength Index (0-100)
    Why: Most used momentum indicator.
    Captures overbought/oversold conditions.
    LSTM learns mean-reversion patterns from it.
    """
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def compute_macd(series, fast=12, slow=26):
    """
    MACD line = EMA(12) - EMA(26)
    Why: Captures trend direction and momentum.
    Positive = uptrend, Negative = downtrend.
    Widely watched so partially self-fulfilling.
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow


def compute_bb_position(series, period=20, std_dev=2):
    """
    Bollinger Band Position (0 to 1)
    0 = lower band (oversold), 1 = upper band (overbought)
    Why: Different from RSI - price-band based not momentum.
    Captures volatility regime and mean reversion.
    """
    ma       = series.rolling(period).mean()
    std      = series.rolling(period).std()
    upper    = ma + (std_dev * std)
    lower    = ma - (std_dev * std)
    position = (series - lower) / (upper - lower + 1e-10)
    return position.clip(0, 1)


def add_indicators(df):
    """Add RSI, MACD, BB_position to dataframe."""
    close            = df["Close"]
    df["RSI_14"]     = compute_rsi(close)
    df["MACD"]       = compute_macd(close)
    df["BB_position"]= compute_bb_position(close)
    df.dropna(inplace=True)
    return df


def build_feature_matrix(df):
    """
    Build 8-feature matrix.
    OHLCV -> log returns (stationary)
    Indicators -> raw values (already normalized)
    """
    raw_ohlcv   = df[["Open","High","Low","Close","Volume"]].values
    price_rets  = np.log(raw_ohlcv[1:] / (raw_ohlcv[:-1] + 1e-10))
    indicators  = df[["RSI_14","MACD","BB_position"]].values[1:]
    features    = np.column_stack([price_rets, indicators])
    return features, raw_ohlcv


def scale_features(features):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    return scaled, scaler


def inverse_transform_col(scaled_values, col_idx, scaler):
    """
    Correct inverse transform for a single column.
    Uses scaler internals directly - no dummy matrix distortion.
    result = scaled_value * std + mean
    """
    mean = scaler.mean_[col_idx]
    std  = scaler.scale_[col_idx]
    return np.array(scaled_values) * std + mean


def create_sequences(scaled_data, seq_len=SEQ_LEN):
    """
    X shape: (N, seq_len, 8)
    y shape: (N, 2) -> [Open return, Close return]
    """
    X, y = [], []
    for i in range(seq_len, len(scaled_data)):
        X.append(scaled_data[i - seq_len:i, :])
        y.append([scaled_data[i, OPEN_IDX],
                  scaled_data[i, CLOSE_IDX]])
    return np.array(X), np.array(y)


def train_test_split_temporal(X, y, train_ratio=0.7):
    """Temporal split - 70% train, 15% validation, 15% test."""
    train_end = int(len(X) * train_ratio)
    val_end = int(len(X) * 0.85)
    return X[:train_end], X[train_end:val_end], X[val_end:], y[:train_end], y[train_end:val_end], y[val_end:]


def prepare_data(symbol, years=10):
    """Full pipeline - call once per symbol."""
    df                           = load_price(symbol, years)
    df                           = add_indicators(df)
    features, raw_ohlcv          = build_feature_matrix(df)
    # features are built from returns df.iloc[1:], so dates must align to df index starting at 2nd row
    dates_features               = df.index[1:]
    scaled, scaler               = scale_features(features)
    X, y                         = create_sequences(scaled)
    X_orig, _                    = create_sequences(raw_ohlcv)  # Original (unscaled) for baseline
    X_train, X_val, X_test, y_train, y_val, y_test = train_test_split_temporal(X, y)
    return {
        "df": df, "features": features, "raw_ohlcv": raw_ohlcv,
        "scaled": scaled, "scaler": scaler,
        "X": X, "y": y,
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "X_orig": X_orig,
        "dates_features": dates_features,
    }
