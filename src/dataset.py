"""
dataset.py

PyTorch Dataset and DataLoader for stock return prediction.
Handles the time-ordered train/val/test split and tensor conversion.

Usage:
    from src.dataset import get_dataloaders

    train_loader, val_loader, test_loader = get_dataloaders(tickers=["AAPL", "MSFT", "GOOG"])
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from src.data_pipeline import load_ticker
from src.features import build_features, get_X_y

# ── Split dates ───────────────────────────────────────────────────────────────
# Strictly time-ordered: train → val → test. No shuffling across boundaries.
# Adjust these if you want a different horizon.

TRAIN_END = "2020-12-31"
VAL_END   = "2022-12-31"
# test set is everything after VAL_END


# ── Dataset ───────────────────────────────────────────────────────────────────

class StockReturnDataset(Dataset):
    """
    A PyTorch Dataset that holds (features, target) pairs for a set of
    stock tickers over a given date range.

    Each sample is one (ticker, date) observation — a flat feature vector
    and a scalar return target.

    Args:
        tickers   : list of ticker strings (e.g. ["AAPL", "MSFT"])
        start     : start date string "YYYY-MM-DD" (inclusive)
        end       : end date string "YYYY-MM-DD" (inclusive)
        scaler    : a fitted StandardScaler, or None (fits one on this data)
    """

    def __init__(
        self,
        tickers: list[str],
        start: str,
        end: str,
        scaler: StandardScaler | None = None,
    ):
        self.tickers = tickers
        self.start   = start
        self.end     = end

        X_all, y_all = self._load_and_combine(tickers, start, end)

        # Fit scaler on training data; reuse fitted scaler for val/test
        if scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_all)
        else:
            self.scaler = scaler
            X_scaled = self.scaler.transform(X_all)

        # TODO: convert X_scaled and y_all to float32 tensors and store as
        # self.X and self.y
        self.X = None
        self.y = None

    def _load_and_combine(
        self, tickers: list[str], start: str, end: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Loads each ticker, builds features, slices to date range, and
        concatenates into a single (X, y) array pair.

        Returns:
            X : np.ndarray of shape (n_samples, n_features)
            y : np.ndarray of shape (n_samples,)
        """
        X_parts, y_parts = [], []

        for ticker in tickers:
            try:
                raw      = load_ticker(ticker)
                features = build_features(raw)
                features = features.loc[start:end]

                if features.empty:
                    continue

                X, y = get_X_y(features)
                X_parts.append(X.values)
                y_parts.append(y.values)

            except FileNotFoundError:
                print(f"Warning: no data for {ticker}, skipping.")

        if not X_parts:
            raise ValueError("No valid tickers found. Did you run data_pipeline.py?")

        # TODO: concatenate X_parts and y_parts into two numpy arrays and return
        return None, None

    def __len__(self) -> int:
        # TODO: return number of samples
        return 0

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: return (self.X[idx], self.y[idx])
        return None, None


# ── DataLoader factory ────────────────────────────────────────────────────────

def get_dataloaders(
    tickers: list[str],
    batch_size: int = 256,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Builds train, val, and test DataLoaders using strict time-ordering.
    The scaler is fitted on training data only and reused for val and test.

    Args:
        tickers    : list of ticker strings
        batch_size : samples per batch

    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = StockReturnDataset(
        tickers, start="2015-01-01", end=TRAIN_END
    )
    # Pass the fitted scaler to val and test — never refit on them
    val_dataset = StockReturnDataset(
        tickers, start=TRAIN_END, end=VAL_END, scaler=train_dataset.scaler
    )
    test_dataset = StockReturnDataset(
        tickers, start=VAL_END, end="2024-12-31", scaler=train_dataset.scaler
    )

    # TODO: wrap each dataset in a DataLoader
    # shuffle=True for train, False for val and test
    train_loader = None
    val_loader   = None
    test_loader  = None

    return train_loader, val_loader, test_loader
