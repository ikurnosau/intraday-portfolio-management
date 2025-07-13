import torch
import torch.nn as nn


class ResidualConvBlock(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int, 
                 dilation: int,
                 use_layer_norm: bool = True,
                 dropout: float = 0.2):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            padding=(kernel_size - 1) * dilation, 
            dilation=dilation)

        if use_layer_norm:
            self.ln1 = nn.LayerNorm(out_channels)
        else:
            self.ln1 = None

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)

        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else None

    def forward(self, x: torch.Tensor):
        # (B, C, T) → (B, C, T)

        residual = self.downsample(x) if self.downsample else x
        out = self.conv1(x)
        if self.ln1: 
            out = out.transpose(1, 2)
            out = self.ln1(out)
            out = out.transpose(1, 2)

        out = self.dropout(out)
        # Ensure temporal dimension matches residual for the addition
        if out.shape[-1] != residual.shape[-1]:
            out = out[:, :, :residual.shape[-1]]
        
        return self.activation(out + residual)


class TCN(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 hidden_channels: int, 
                 kernel_size: int, 
                 num_layers: int, 
                 output_dim: int,
                 use_layer_norm: bool = True,
                 dropout: float = 0.2):
        super().__init__()

        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            cur_in_channels = in_channels if i == 0 else hidden_channels
            layers.append(ResidualConvBlock(
                cur_in_channels, 
                hidden_channels, 
                kernel_size, 
                dilation, 
                use_layer_norm,
                dropout))

        self.tcn = nn.Sequential(*layers)
        # Learnable attention weight generator over the temporal dimension
        # Projects hidden channels → scalar score for each time step
        self.attention = nn.Conv1d(hidden_channels, 1, kernel_size=1)
        self.fc = nn.Linear(hidden_channels, output_dim)
        self.dropout = nn.Dropout(dropout)

        if use_layer_norm:
            self.norm = nn.LayerNorm(hidden_channels)
        else:
            self.norm = None

        self.output_dim = output_dim

    def forward(self, x: torch.Tensor):
        # (B, A, T, C) → (B, A, output_size) if output_size > 1, else (B, A)

        B, A, T, C = x.shape
        x = x.reshape(B * A, T, C)
        x = x.transpose(1, 2)  # Pytorch conv1d expects channels first: (B, C, T)
        x = self.tcn(x)  # (B*A, hidden_channels, T_out)

        # Compute attention weights across the temporal dimension
        # ALternatieve to x[:, :, -1]
        attn_scores = self.attention(x)              # (B*A, 1, T_out)
        attn_weights = torch.softmax(attn_scores, 2) # softmax over T_out
        x = (x * attn_weights).sum(dim=2)            # (B*A, hidden_channels)

        if self.norm:
            x = self.norm(x)
        x = self.dropout(x)

        out = self.fc(x)
        out = out.reshape(B, A, -1)

        if self.output_dim == 1:
            out = out.squeeze(-1)  # (B, A)

        return out



