"""
train.py

Training loop, validation loop, early stopping, and checkpoint saving.

Usage:
    python src/train.py --model mlp --epochs 50 --lr 1e-3
"""

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import get_dataloaders
from src.models import get_model
from src.features import FEATURE_COLS

CHECKPOINT_DIR = "checkpoints"
N_FEATURES     = len(FEATURE_COLS)


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Runs one full pass over the training DataLoader.

    Returns:
        mean training loss over all batches
    """
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # TODO:
        # 1. zero the gradients
        # 2. forward pass through model
        # 3. compute loss against y_batch
        # 4. backward pass
        # 5. optimizer step
        # 6. accumulate loss into total_loss

    return total_loss / len(loader)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Runs a forward pass over a DataLoader without computing gradients.
    Used for validation and test evaluation.

    Returns:
        mean loss over all batches
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # TODO: forward pass and accumulate loss (no backward needed)

    return total_loss / len(loader)


# ── Early stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stops training when validation loss stops improving.
    Saves the best model checkpoint automatically.

    Args:
        patience  : how many epochs to wait after last improvement (default: 10)
        min_delta : minimum change to count as improvement (default: 1e-4)
        path      : where to save the best checkpoint
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, path: str = "best.pt"):
        self.patience   = patience
        self.min_delta  = min_delta
        self.path       = path
        self.best_loss  = float("inf")
        self.counter    = 0
        self.should_stop = False

    def step(self, val_loss: float, model: nn.Module) -> None:
        """
        Call this after each epoch with the current validation loss.
        Sets self.should_stop = True when patience is exhausted.
        """
        # TODO:
        # - if val_loss improved by more than min_delta, save checkpoint and reset counter
        # - otherwise increment counter
        # - if counter >= patience, set self.should_stop = True
        pass

    def _save_checkpoint(self, model: nn.Module) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save(model.state_dict(), self.path)


# ── Main training function ────────────────────────────────────────────────────

def train(
    model_name: str  = "mlp",
    tickers: list    = None,
    epochs: int      = 50,
    lr: float        = 1e-3,
    batch_size: int  = 256,
    patience: int    = 10,
):
    """
    Full training run: loads data, initialises model, trains with early stopping.

    Args:
        model_name : "mlp", "cnn", or "lstm"
        tickers    : list of ticker strings (defaults to a small sample)
        epochs     : maximum training epochs
        lr         : learning rate
        batch_size : samples per batch
        patience   : early stopping patience
    """
    if tickers is None:
        # Small default set for quick experimentation
        tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "JPM",
                   "JNJ", "XOM", "WMT", "PG", "MA"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader, _ = get_dataloaders(tickers, batch_size=batch_size)

    # Model
    model = get_model(model_name).to(device)
    print(f"Model: {model_name} | Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimiser
    # TODO: define criterion (MSELoss is a natural start for regression)
    # TODO: define optimizer (Adam is a safe default)
    criterion = None
    optimizer = None

    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pt")
    early_stopping  = EarlyStopping(patience=patience, path=checkpoint_path)

    # Training loop
    print(f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>10} | {'Time':>6}")
    print("-" * 42)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss   = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        print(f"{epoch:>6} | {train_loss:>10.5f} | {val_loss:>10.5f} | {elapsed:>5.1f}s")

        early_stopping.step(val_loss, model)
        if early_stopping.should_stop:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    print(f"\nBest checkpoint saved to: {checkpoint_path}")
    return checkpoint_path


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a stock return prediction model.")
    parser.add_argument("--model",      type=str,   default="mlp",  choices=["mlp", "cnn", "lstm"])
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int,   default=256)
    parser.add_argument("--patience",   type=int,   default=10)
    args = parser.parse_args()

    train(
        model_name=args.model,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
    )
