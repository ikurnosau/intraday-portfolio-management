import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeSeriesTransformer(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 num_heads: int,
                 num_layers: int,
                 seq_len: int,
                 feat_dim:int,
                 output_dim: int=1,
                 dropout: float = 0.1,):
        super().__init__()
        self.output_dim = output_dim

        self.input_proj = nn.Linear(feat_dim, hidden_dim)
        self.pos_embed = nn.Embedding(seq_len, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        B, A, T, F = x.shape

        # (B*A, T, F_concat) → TCN → (B*A, H)
        x_flat = x.reshape(B * A, T, F)

        x_features = self.input_proj(x_flat)
        x_position = self.pos_embed(torch.arange(T, device=x.device))
        x_combined = x_features + x_position
        x_encoded = self.encoder(x_combined)[:, -1, :]

        x_encoded = self.norm(x_encoded)
        x_encoded = self.dropout(x_encoded)
        x_encoded = x_encoded.reshape(B, A, -1)

        out = self.fc(x_encoded)
        if self.output_dim == 1:
            out = out.squeeze(-1)  # (B, A)

        return out