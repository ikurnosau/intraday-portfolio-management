import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalSpatialMultiLayer(nn.Module):
    def __init__(self,
                 num_assets: int,
                 input_dim: int,
                 hidden_dim: int = 64,
                 output_dim: int = 1,
                 num_layers: int = 1,
                 bidirectional: bool = False,
                 dropout: float = 0.1,
                 use_spatial_attention: bool = True,
                 num_heads: int = 4,
                 use_asset_embedding: bool = True,
                 asset_embed_dim: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim if not bidirectional else hidden_dim * 2
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.use_spatial_attention = use_spatial_attention
        self.use_asset_embedding = use_asset_embedding

        self.dropout = nn.Dropout(dropout)
        self.fc_output= nn.Linear(self.hidden_dim, output_dim)

        if self.use_asset_embedding:
            self.shared_asset_embed = nn.Embedding(num_assets, asset_embed_dim)

        self.lstms = nn.ModuleList()
        self.temporal_norms = nn.ModuleList()
        self.asset_projs = nn.ModuleList()
        self.spatial_attns = nn.ModuleList()
        self.spatial_norms = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()

        for layer_idx in range(num_layers):
            self.lstms.append(nn.LSTM(
                input_size=input_dim if layer_idx == 0 else self.hidden_dim,
                hidden_size=self.hidden_dim // self.num_directions,
                num_layers=1,
                batch_first=True,
                bidirectional=bidirectional,
            ))
            self.temporal_norms.append(nn.LayerNorm(self.hidden_dim))

            if use_asset_embedding:
                self.asset_projs.append(nn.Linear(asset_embed_dim, self.hidden_dim, bias=False))

            if use_spatial_attention:
                self.spatial_attns.append(nn.MultiheadAttention(
                    embed_dim=self.hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                ))
                self.spatial_norms.append(nn.LayerNorm(self.hidden_dim))

                self.ffns.append(nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                ))
                self.ffn_norms.append(nn.LayerNorm(self.hidden_dim))

    def forward(self, x: torch.Tensor):
        B, A, T, F = x.shape
        for layer_idx in range(self.num_layers):
            x_flat = x.reshape(B * A, T, x.shape[-1]) # (B*A, T, F)
            x, _ = self.lstms[layer_idx](x_flat) # (B*A, T, H)
            x = self.temporal_norms[layer_idx](x)
            x = self.dropout(x)
            x = x.reshape(B, A, T, -1) # (B, A, T, H)

            x = x.permute(0, 2, 1, 3) # (B, T, A, H)
            x = x.reshape(B * T, A, -1) # (B*T, A, H)
            if self.use_asset_embedding:
                asset_ids = torch.arange(A, device=x.device)
                asset_emb = self.shared_asset_embed(asset_ids)  # (A, E)
                asset_emb = self.asset_projs[layer_idx](asset_emb)  # (A, H)
                x = x + asset_emb.unsqueeze(0).expand(B * T, A, -1) # (B*T, A, H)

            if self.use_spatial_attention:
                x_attn, _ = self.spatial_attns[layer_idx](x, x, x)
                x = self.spatial_norms[layer_idx](x_attn + x)
                x = self.dropout(x)
                x = x + self.ffns[layer_idx](self.ffn_norms[layer_idx](x))

            x = x.reshape(B, T, A, -1) # (B, T, A, H)
            x = x.permute(0, 2, 1, 3) # (B, A, T, H)
        
        x = self.fc_output(x[:, :, -1, :]) # (B, A, H)
        if self.output_dim == 1:
            x = x.squeeze(-1)  # (B, A)

        return x
