import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

class MLP(nn.Module):
    """Multilayer perceptron (MLP) for both classification and regression tasks.

    Parameters
    ----------
    input_dim : int, default=37
        Number of input features per sample.
    output_dim : int, default=3
        Number of outputs:
        - For output_dim > 1: Classification with output_dim classes (outputs logits)
        - For output_dim = 1: Regression (outputs a single continuous value)
    hidden_dims : list[int], default=[128, 64]
        Width of each hidden layer. The *i*-th entry creates the *i*-th hidden
        ``nn.Linear`` layer. Provide an empty list for a linear model without
        hidden layers.
    dropout : float, default=0.0
        Dropout probability applied *after* each hidden layer (except the
        output layer).
    activation : nn.Module, default=nn.ReLU(inplace=True)
        Activation function inserted after each hidden ``nn.Linear`` layer.
        Must be a *class instance* (e.g. ``nn.ReLU()``) rather than the class
        itself.
    batch_norm : bool, default=False
        If ``True`` a ``nn.BatchNorm1d`` layer is inserted after every hidden
        ``nn.Linear`` layer.
    """

    def __init__(
        self,
        input_dim: int = 37,
        output_dim: int = 3,
        hidden_dims: list[int] | tuple[int, ...] = [128, 64],
        dropout: float = 0.0,
        activation: nn.Module = nn.ReLU(inplace=True),
        batch_norm: bool = False,
    ) -> None:
        super(MLP, self).__init__()

        layers: list[nn.Module] = []
        last_dim = input_dim

        # Hidden layers -------------------------------------------------------
        for hidden_dim in hidden_dims:
            # Linear transformation
            layers.append(nn.Linear(last_dim, hidden_dim))
            # Optional batch norm
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            # Non-linearity
            layers.append(activation)
            # Regularisation
            layers.append(nn.Dropout(p=dropout))
            # Update for next layer
            last_dim = hidden_dim

        # Output layer --------------------------------------------------------
        layers.append(nn.Linear(last_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Compute predictions for input *x*.
        
        Returns class logits for output_dim > 1 (classification)
        or continuous values for output_dim = 1 (regression).
        """
        return self.model(x)
    