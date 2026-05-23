# data_pipeline.py
# Purpose: scrapes historical S&P 500 data from yfinance.
# This information comes from a well-maintained Wikipedia article.
# Usage: python3 src/data_pipeline.py

# Import dependencies
import os
import time
import logging
import pandas as pd
import yfinance as yf

# ── Config ────────────────────────────────────────────────────────────────────

START_DATE  = "2015-01-01"
END_DATE    = "2025-12-31"
RAW_DIR     = "data/raw"
INTERVAL    = "1d"          # daily bars
BATCH_SIZE  = 50            # tickers per yfinance batch call
SLEEP_SEC   = 1.5           # pause between batches (be a polite API citizen)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_sp500_tickers() -> list[str]:
    """
    Scrapes the current S&P 500 constituent list from Wikipedia.
    Returns a list of ticker strings, with '.' replaced by '-' for yfinance
    compatibility (e.g. BRK.B -> BRK-B).
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
    log.info(f"Found {len(tickers)} S&P 500 tickers.")
    return tickers


def download_batch(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Downloads OHLCV data for a batch of tickers in a single yfinance call.
    Returns a MultiIndex DataFrame: columns = (field, ticker).
    """
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=INTERVAL,
        auto_adjust=True,   # adjusts for splits and dividends automatically
        progress=False,
        threads=True,
    )
    return data


def save_ticker(ticker: str, df: pd.DataFrame) -> None:
    """
    Saves a single ticker's OHLCV data to data/raw/<TICKER>.csv.
    Skips if the DataFrame is empty or has fewer than 252 rows (~1 trading year).
    """
    if df.empty or len(df) < 252:
        log.warning(f"Skipping {ticker}: insufficient data ({len(df)} rows).")
        return

    path = os.path.join(RAW_DIR, f"{ticker}.csv")
    df.to_csv(path)
    log.info(f"Saved {ticker} -> {path} ({len(df)} rows).")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run():
    os.makedirs(RAW_DIR, exist_ok=True)

    tickers = get_sp500_tickers()

    # Process in batches to avoid overwhelming the API
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i : i + BATCH_SIZE]
        log.info(f"Downloading batch {i // BATCH_SIZE + 1} "
                 f"({i+1}–{min(i+BATCH_SIZE, len(tickers))} of {len(tickers)})...")

        try:
            raw = download_batch(batch, START_DATE, END_DATE)
        except Exception as e:
            log.error(f"Batch download failed: {e}. Skipping batch.")
            continue

        # raw.columns is a MultiIndex: (field, ticker)
        # If only one ticker was returned, yfinance may drop the ticker level.
        if isinstance(raw.columns, pd.MultiIndex):
            for ticker in batch:
                if ticker not in raw.columns.get_level_values(1):
                    log.warning(f"{ticker} not in response. Possibly delisted.")
                    continue
                # Select all fields (Open, High, Low, Close, Volume) for this ticker
                ticker_df = raw.xs(ticker, axis=1, level=1).dropna(how="all")
                save_ticker(ticker, ticker_df)
        else:
            # Single-ticker fallback (batch of 1)
            ticker_df = raw.dropna(how="all")
            save_ticker(batch[0], ticker_df)

        time.sleep(SLEEP_SEC)

    log.info("Pipeline complete.")


def load_ticker(ticker: str) -> pd.DataFrame:
    """
    Convenience loader: reads data/raw/<TICKER>.csv and returns a DataFrame
    with a DatetimeIndex. Use this in features.py and dataset.py.

    Example:
        df = load_ticker("AAPL")
        print(df.head())
    """
    path = os.path.join(RAW_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No data for {ticker}. Run data_pipeline.py first.")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "Date"
    return df


if __name__ == "__main__":
    run()
