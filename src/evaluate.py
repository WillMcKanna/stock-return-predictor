"""
evaluate.py

Loads a saved model checkpoint and evaluates it on the held-out test set.
Uses metrics appropriate for financial return prediction — not just MSE.

Usage:
    python src/evaluate.py --model mlp
    python src/evaluate.py --model cnn --checkpoint checkpoints/cnn_best.pt
"""

import argparse

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from src.dataset import get_dataloaders
from src.models import get_model
from src.features import FEATURE_COLS

N_FEATURES = len(FEATURE_COLS)


# ── Metrics ───────────────────────────────────────────────────────────────────

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Standard MSE. Included for completeness but not the primary metric."""
    # TODO: implement
    return 0.0


def rank_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Spearman rank correlation between predicted and actual returns.

    This is the primary metric for financial ML. We care more about correctly
    ranking stocks (which ones will do better than others) than the exact
    predicted return value. A rank correlation of 0.05–0.10 is considered
    meaningful in practice.

    Returns a value in [-1, 1]. Higher is better.
    """
    # TODO: use scipy.stats.spearmanr and return the correlation coefficient
    # Hint: spearmanr returns a result object; access .statistic or .correlation
    return 0.0


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Fraction of predictions where the sign of the predicted return matches
    the sign of the actual return.

    A naive coin-flip baseline is 0.50. Anything above ~0.52 is meaningful.

    Returns a value in [0, 1]. Higher is better.
    """
    # TODO: compare np.sign(y_true) and np.sign(y_pred), compute fraction matching
    return 0.0


def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    The Information Coefficient (IC) is the cross-sectional rank correlation
    between predicted and realised returns at each time step, averaged over time.

    This is the gold-standard metric in quantitative finance.

    Note: computing true IC requires grouping by date, which means your
    test set needs to retain date information. This is left as an extension —
    implement it once your basic evaluation pipeline is working.
    """
    # TODO (extension): group predictions and actuals by date,
    # compute spearmanr within each date cross-section, return the mean.
    raise NotImplementedError("IC requires date-indexed predictions. See docstring.")


# ── Prediction loop ───────────────────────────────────────────────────────────

def get_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Runs inference over a DataLoader and returns all predictions and targets
    as numpy arrays.

    Returns:
        y_true : np.ndarray of shape (n_samples,)
        y_pred : np.ndarray of shape (n_samples,)
    """
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)

            # TODO: run model forward pass, move predictions to CPU,
            # append to all_preds and all_targets

    # TODO: concatenate lists into numpy arrays and return
    return None, None


# ── Main evaluation function ──────────────────────────────────────────────────

def evaluate_model(model_name: str, checkpoint_path: str = None):
    """
    Loads a checkpoint and prints a full evaluation report on the test set.

    Args:
        model_name      : "mlp", "cnn", or "lstm"
        checkpoint_path : path to .pt file (defaults to checkpoints/<model>_best.pt)
    """
    if checkpoint_path is None:
        checkpoint_path = f"checkpoints/{model_name}_best.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and checkpoint
    model = get_model(model_name).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded checkpoint: {checkpoint_path}")

    # Load test data (use a small default ticker set matching what you trained on)
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "JPM",
               "JNJ", "XOM", "WMT", "PG", "MA"]
    _, _, test_loader = get_dataloaders(tickers)

    # Get predictions
    y_true, y_pred = get_predictions(model, test_loader, device)

    # Report
    mse   = mean_squared_error(y_true, y_pred)
    rc    = rank_correlation(y_true, y_pred)
    da    = directional_accuracy(y_true, y_pred)

    print(f"\n── Test Set Evaluation: {model_name.upper()} ──────────────────")
    print(f"  MSE                : {mse:.6f}")
    print(f"  Rank Correlation   : {rc:.4f}   (0.05–0.10 = meaningful in practice)")
    print(f"  Directional Acc.   : {da:.4f}   (0.50 = coin flip baseline)")
    print(f"──────────────────────────────────────────────────────")

    # Honest framing — worth keeping in your notebook write-up too
    print("""
Note: Low rank correlation and directional accuracy near 0.50 are expected.
Equity returns are notoriously hard to predict. The goal of this project is
to build a rigorous ML pipeline and understand the difficulty of the problem —
not to claim alpha. A well-documented honest result is more impressive to
employers than an inflated one.
    """)

    return {"mse": mse, "rank_corr": rc, "dir_acc": da}


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test set.")
    parser.add_argument("--model",      type=str, default="mlp", choices=["mlp", "cnn", "lstm"])
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    evaluate_model(args.model, args.checkpoint)
