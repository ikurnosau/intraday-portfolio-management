import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMClassifier(nn.Module):
    """
    LSTM-based classifier for 3-class classification on sequence data.

    Args:
        input_dim (int): Number of features per time step (default: 37).
        hidden_dim (int): Number of features in LSTM hidden state (default: 128).
        num_layers (int): Number of stacked LSTM layers (default: 2).
        bidirectional (bool): Use bidirectional LSTM (default: False).
        dropout (float): Dropout probability between LSTM layers (default: 0.5).
    """
    def __init__(self,
                 input_dim=37,
                 n_class=3,
                 hidden_dim=128,
                 num_layers=2,
                 bidirectional=False,
                 dropout=0.0):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.n_class = n_class

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim * self.num_directions, self.n_class)

    def forward(self, x):
        """
        Forward pass through LSTM classifier.

        Args:
            x (torch.Tensor): Input tensor of shape (N, T, input_dim),
                               where T is sequence length (e.g., 10).

        Returns:
            logits (torch.Tensor): Output tensor of shape (N, 3).
        """

        # LSTM forward
        out, _ = self.lstm(x)  # out: (N, T, hidden_dim * num_directions)

        # Use the last time-step's output for classification
        last_out = out[:, -1, :]  # (N, hidden_dim * num_directions)
        logits = self.fc(last_out)  # (N, 3)
        return logits