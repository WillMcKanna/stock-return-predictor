"""
models.py

PyTorch model definitions for stock return prediction.
All models take a flat feature vector and output a scalar return prediction.

Usage:
    from src.models import MLP, CNN1D, LSTM
    from src.features import FEATURE_COLS

    n_features = len(FEATURE_COLS)
    model = MLP(input_dim=n_features)
"""

import torch
import torch.nn as nn
from src.features import FEATURE_COLS

N_FEATURES = len(FEATURE_COLS)  # convenience constant


# ── MLP ───────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """
    Multi-Layer Perceptron with dropout and batch normalization.
    The simplest architecture — a strong baseline before trying CNN or LSTM.

    Architecture:
        Linear(input_dim → hidden_dim)
        BatchNorm1d → ReLU → Dropout
        Linear(hidden_dim → hidden_dim // 2)
        BatchNorm1d → ReLU → Dropout
        Linear(hidden_dim // 2 → 1)

    Args:
        input_dim  : number of input features (default: N_FEATURES)
        hidden_dim : width of the first hidden layer (default: 128)
        dropout    : dropout probability (default: 0.3)
    """

    def __init__(
        self,
        input_dim: int  = N_FEATURES,
        hidden_dim: int = 128,
        dropout: float  = 0.3,
    ):
        super().__init__()

        # TODO: define self.network as an nn.Sequential block following
        # the architecture described above.
        # Hint: nn.Linear, nn.BatchNorm1d, nn.ReLU, nn.Dropout
        self.network = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : tensor of shape (batch_size, input_dim)
        Returns:
            tensor of shape (batch_size,)  — scalar prediction per sample
        """
        # TODO: pass x through self.network and squeeze the output dimension
        return None


# ── 1D CNN ────────────────────────────────────────────────────────────────────

class CNN1D(nn.Module):
    """
    1D Convolutional Network that treats the feature vector as a sequence.

    Each feature is treated as a channel; the conv layer learns local patterns
    across groups of features. This is a natural extension of the MLP —
    worth trying to see if it picks up on feature interactions.

    Architecture:
        Reshape input to (batch, 1, input_dim)  ← 1 channel, input_dim "timesteps"
        Conv1d(1 → n_filters, kernel_size) → ReLU → AdaptiveAvgPool1d(8)
        Flatten
        Linear(n_filters * 8 → 64) → ReLU → Dropout
        Linear(64 → 1)

    Args:
        input_dim   : number of input features
        n_filters   : number of conv filters (default: 32)
        kernel_size : conv kernel size (default: 3)
        dropout     : dropout probability (default: 0.3)
    """

    def __init__(
        self,
        input_dim: int   = N_FEATURES,
        n_filters: int   = 32,
        kernel_size: int = 3,
        dropout: float   = 0.3,
    ):
        super().__init__()

        # TODO: define self.conv_block as an nn.Sequential:
        #   Conv1d(in_channels=1, out_channels=n_filters, kernel_size=kernel_size, padding='same')
        #   ReLU
        #   AdaptiveAvgPool1d(8)
        self.conv_block = None

        # TODO: define self.fc_block as an nn.Sequential:
        #   Linear(n_filters * 8 → 64)
        #   ReLU
        #   Dropout(dropout)
        #   Linear(64 → 1)
        self.fc_block = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : tensor of shape (batch_size, input_dim)
        Returns:
            tensor of shape (batch_size,)
        """
        # TODO:
        # 1. unsqueeze x to (batch, 1, input_dim) for Conv1d
        # 2. pass through self.conv_block
        # 3. flatten to (batch, n_filters * 8)
        # 4. pass through self.fc_block and squeeze output
        return None


# ── LSTM ──────────────────────────────────────────────────────────────────────

class LSTMModel(nn.Module):
    """
    LSTM that treats each feature as a timestep in a sequence.
    More natural for time-series data than the MLP or CNN.

    Note: this requires reshaping your input into a sequence.
    As implemented here, each feature dimension is one "timestep" with a
    single feature — a simple approach. A richer version would use a sliding
    window of past days as the sequence, which is a good extension to discuss
    in your write-up.

    Architecture:
        Reshape input to (batch, input_dim, 1)
        LSTM(input_size=1, hidden_size, num_layers, batch_first=True)
        Take final hidden state → Linear(hidden_size → 1)

    Args:
        input_dim   : number of input features (= sequence length here)
        hidden_size : LSTM hidden state size (default: 64)
        num_layers  : number of stacked LSTM layers (default: 2)
        dropout     : dropout between LSTM layers (default: 0.3)
    """

    def __init__(
        self,
        input_dim: int   = N_FEATURES,
        hidden_size: int = 64,
        num_layers: int  = 2,
        dropout: float   = 0.3,
    ):
        super().__init__()

        # TODO: define self.lstm as an nn.LSTM
        # Remember: batch_first=True means input shape is (batch, seq_len, input_size)
        # dropout only applies between layers, so set to 0 if num_layers == 1
        self.lstm = None

        # TODO: define self.fc as nn.Linear(hidden_size → 1)
        self.fc = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : tensor of shape (batch_size, input_dim)
        Returns:
            tensor of shape (batch_size,)
        """
        # TODO:
        # 1. reshape x to (batch, input_dim, 1) — treating each feature as a timestep
        # 2. pass through self.lstm → output shape: (batch, seq_len, hidden_size)
        # 3. take the final timestep: output[:, -1, :]
        # 4. pass through self.fc and squeeze
        return None


# ── Model factory ─────────────────────────────────────────────────────────────

MODELS = {
    "mlp":  MLP,
    "cnn":  CNN1D,
    "lstm": LSTMModel,
}

def get_model(name: str, **kwargs) -> nn.Module:
    """
    Convenience factory for use in train.py and notebooks.

    Usage:
        model = get_model("mlp", hidden_dim=256, dropout=0.4)
        model = get_model("cnn", n_filters=64)
        model = get_model("lstm", hidden_size=128, num_layers=2)
    """
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(MODELS.keys())}")
    return MODELS[name](**kwargs)
