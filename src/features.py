"""
features.py

Feature engineering for stock return prediction.
All functions take a raw OHLCV DataFrame (output of load_ticker()) and return
a DataFrame with new columns added. No data outside the current row's history
is ever used — enforcing zero lookahead bias.

Usage:
    from src.features import build_features
    from src.data_pipeline import load_ticker

    df = load_ticker("AAPL")
    features = build_features(df)
"""

import pandas as pd
import numpy as np


# ── Individual feature functions ──────────────────────────────────────────────
# Each function:
#   - accepts a DataFrame with at minimum a 'Close' column and a DatetimeIndex
#   - returns the same DataFrame with one or more new columns added
#   - never peeks forward in time (no .shift(-n), no future data)


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds simple daily return and forward 5-day return (the prediction target).

    Columns added:
        ret_1d    : today's close-to-close return (the input feature)
        target_5d : forward 5-day cumulative return (what we're predicting)

    Note: target_5d uses .shift(-5) — this is intentional and only used as
    the label (y), never as an input feature (X). Be careful not to
    accidentally include it in your feature matrix.
    """
    df = df.copy()

    # TODO: compute log return: log(Close_t / Close_{t-1})
    df["ret_1d"] = None

    # TODO: compute forward 5-day cumulative return as the prediction target
    # Hint: use .shift(-5) on Close, then compute return vs today's Close
    df["target_5d"] = None

    return df


def add_momentum(df: pd.DataFrame, windows: list[int] = [5, 10, 21]) -> pd.DataFrame:
    """
    Adds price momentum features: return over the past N days.

    Columns added:
        mom_5d, mom_10d, mom_21d  (or whatever windows are passed)

    Momentum = log(Close_t / Close_{t-N})
    A positive value means the stock has risen over that window.
    """
    df = df.copy()

    for w in windows:
        col = f"mom_{w}d"
        # TODO: compute log return from N days ago to today
        df[col] = None

    return df


def add_rolling_volatility(df: pd.DataFrame, windows: list[int] = [10, 21]) -> pd.DataFrame:
    """
    Adds rolling historical volatility: std of daily returns over N days.

    Columns added:
        vol_10d, vol_21d  (annualised by multiplying by sqrt(252))

    High volatility = noisier, harder-to-predict stock.
    """
    df = df.copy()

    if "ret_1d" not in df.columns:
        raise ValueError("Call add_returns() before add_rolling_volatility().")

    for w in windows:
        col = f"vol_{w}d"
        # TODO: rolling std of ret_1d over w days, annualised
        df[col] = None

    return df


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Adds the Relative Strength Index (RSI) over a given lookback window.

    Columns added:
        rsi_14  (default)

    RSI ranges 0–100.
        > 70 : conventionally considered overbought
        < 30 : conventionally considered oversold

    Formula:
        RS  = avg_gain / avg_loss  (over the window)
        RSI = 100 - (100 / (1 + RS))
    """
    df = df.copy()

    if "ret_1d" not in df.columns:
        raise ValueError("Call add_returns() before add_rsi().")

    col = f"rsi_{window}"

    # TODO: separate ret_1d into gains (positive) and losses (negative)
    # TODO: compute rolling mean of gains and rolling mean of abs(losses)
    # TODO: compute RS and RSI from those
    df[col] = None

    return df


def add_moving_average_crossover(
    df: pd.DataFrame, short: int = 10, long: int = 50
) -> pd.DataFrame:
    """
    Adds a moving average crossover signal.

    Columns added:
        ma_10     : short-window simple moving average of Close
        ma_50     : long-window simple moving average of Close
        ma_cross  : ma_10 / ma_50 - 1  (positive = short MA above long MA)

    A positive ma_cross suggests recent upward momentum relative to the longer
    trend — a classic technical signal.
    """
    df = df.copy()

    short_col = f"ma_{short}"
    long_col  = f"ma_{long}"

    # TODO: compute rolling means of Close for each window
    df[short_col] = None
    df[long_col]  = None

    # TODO: compute the crossover ratio
    df["ma_cross"] = None

    return df


def add_volume_features(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Adds volume-based features to capture unusual trading activity.

    Columns added:
        vol_ratio : today's volume / rolling mean volume over `window` days
                    > 1 means above-average volume (potential breakout signal)
    """
    df = df.copy()

    # TODO: compute rolling mean of Volume, then ratio of today's volume to it
    df["vol_ratio"] = None

    return df


# ── Master builder ────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs all feature functions in the correct order and drops rows with NaNs
    (which appear at the start of the series due to rolling windows).

    This is the only function dataset.py and the notebooks need to import.

    Returns a DataFrame ready to split into X (features) and y (target_5d).
    """
    df = add_returns(df)
    df = add_momentum(df)
    df = add_rolling_volatility(df)
    df = add_rsi(df)
    df = add_moving_average_crossover(df)
    df = add_volume_features(df)

    # Drop the warmup rows where rolling windows don't have enough history
    df = df.dropna()

    return df


# ── Feature/target split helper ───────────────────────────────────────────────

# Columns that are inputs to the model (X)
FEATURE_COLS = [
    "ret_1d",
    "mom_5d", "mom_10d", "mom_21d",
    "vol_10d", "vol_21d",
    "rsi_14",
    "ma_cross",
    "vol_ratio",
]

# Column that is the prediction target (y)
TARGET_COL = "target_5d"


def get_X_y(df: pd.DataFrame):
    """
    Splits a feature DataFrame into X (input matrix) and y (target vector).

    Usage:
        features = build_features(load_ticker("AAPL"))
        X, y = get_X_y(features)
    """
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}. Did you run build_features()?")

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return X, y
