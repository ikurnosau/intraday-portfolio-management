from pytorch_tcn import TCN
import torch
import torch.nn as nn
import torch.nn.functional as F


class TCNPredictor(nn.Module):
    def __init__(self,
                 num_inputs: int,
                 num_channels: list[int],
                 output_dim: int=1,
                 kernel_size: int=3,
                 dropout: float = 0.1,
                 use_norm: str = 'weight_norm',
                 use_skip_connections: bool = True):
        super().__init__()

        self.output_dim = output_dim

        hidden_dim=num_channels[-1]

        self.tcn_block = TCN(
            num_inputs=num_inputs,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            use_norm=use_norm,
            use_skip_connections=use_skip_connections,
            input_shape='NLC',
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        B, A, T, F = x.shape

        # (B*A, T, F_concat) → TCN → (B*A, H)
        x_flat = x.reshape(B * A, T, F)
        tcn_out = self.tcn_block(x_flat)[:, -1, :]
        tcn_out = self.norm(tcn_out)
        tcn_out = self.dropout(tcn_out)
        tcn_out = tcn_out.reshape(B, A, -1)

        out = self.fc(tcn_out)
        if self.output_dim == 1:
            out = out.squeeze(-1)  # (B, A)

        return out