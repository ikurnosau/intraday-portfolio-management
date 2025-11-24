import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalTransformerLayer(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_heads: int,
                 num_layers: int,
                 seq_len: int,
                 dropout: float = 0.1):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_embed = nn.Embedding(seq_len, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=2 * hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor):
        B, T, F = x.shape

        x_features = self.input_proj(x)
        x_position = self.pos_embed(torch.arange(T, device=x.device))
        x_combined = x_features + x_position
        x_encoded = self.encoder(x_combined)

        return x_encoded, None


class TemporalSpatialMultiLayer(nn.Module):
    """
    One possible improvement I see here is to apply attention and embedding only to the last output of the temporal processor at the last layer, 
    so that the last layer of this system would mimic the TemporalSpatial, but with capacity to increase overall depth 
    """
    def __init__(self,
                 num_assets: int,
                 input_dim: int,
                 seq_len: int,
                 hidden_dim: int = 64,
                 output_dim: int = 1,
                 num_layers: int = 1,
                 bidirectional: bool = False,
                 dropout: float = 0.1,
                 use_spatial_attention: bool = True,
                 num_heads: int = 4,
                 use_asset_embedding: bool = True,
                 asset_embed_dim: int = 16,
                 use_ffn: bool = False,
                 temporal_processor: str = 'lstm'):
        super().__init__()
        self.temporal_processor = temporal_processor
        if temporal_processor == 'lstm':
            self.hidden_dim = hidden_dim if not bidirectional else hidden_dim * 2
            self.num_directions = 2 if bidirectional else 1
        elif temporal_processor == 'transformer':
            self.hidden_dim = hidden_dim
        else:
            raise ValueError(f"Invalid temporal processor: {temporal_processor}")

        self.output_dim = output_dim
        self.num_layers = num_layers
        self.use_spatial_attention = use_spatial_attention
        self.use_asset_embedding = use_asset_embedding
        self.use_ffn = use_ffn

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.fc_output= nn.Linear(self.hidden_dim, output_dim)

        if self.use_asset_embedding:
            self.shared_asset_embed = nn.Embedding(num_assets, asset_embed_dim)

        self.temporals = nn.ModuleList()
        self.asset_projs = nn.ModuleList()
        self.spatial_attns = nn.ModuleList()

        if self.use_ffn:
            self.ffns = nn.ModuleList()

        for layer_idx in range(num_layers):
            if temporal_processor == 'lstm':
                self.temporals.append(nn.LSTM(
                    input_size=input_dim if layer_idx == 0 else self.hidden_dim,
                    hidden_size=self.hidden_dim // self.num_directions,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=bidirectional,
                ))
            elif temporal_processor == 'transformer':
                self.temporals.append(TemporalTransformerLayer(
                    input_dim=input_dim if layer_idx == 0 else self.hidden_dim,
                    hidden_dim=self.hidden_dim,
                    num_heads=num_heads,
                    num_layers=1,
                    seq_len=seq_len,
                    dropout=dropout
                ))

            if use_asset_embedding:
                self.asset_projs.append(nn.Linear(asset_embed_dim, self.hidden_dim, bias=False))

            if use_spatial_attention:
                self.spatial_attns.append(nn.MultiheadAttention(
                    embed_dim=self.hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                ))

                if self.use_ffn and layer_idx < num_layers - 1:
                    self.ffns.append(nn.Sequential(
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(self.hidden_dim , self.hidden_dim),
                    ))

    def forward(self, x: torch.Tensor):
        B, A, T, F = x.shape
        for layer_idx in range(self.num_layers):
            x_flat = x.reshape(B * A, T, x.shape[-1]) # (B*A, T, F)
            x, _ = self.temporals[layer_idx](x_flat) # (B*A, T, H)
            if self.temporal_processor != 'transformer':
                x = self.norm(x)
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
                x = self.norm(x_attn + x)
                x = self.dropout(x)

                if self.use_ffn and layer_idx < self.num_layers - 1:
                    x = x + self.ffns[layer_idx](self.norm(x))

            x = x.reshape(B, T, A, -1) # (B, T, A, H)
            x = x.permute(0, 2, 1, 3) # (B, A, T, H)
        
        x = self.fc_output(x[:, :, -1, :]) # (B, A, H)
        if self.output_dim == 1:
            x = x.squeeze(-1)  # (B, A)

        return x
