import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalSpatial(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 output_dim: int = 1,
                 lstm_layers: int = 1,
                 bidirectional: bool = False,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 use_spatial_attention: bool = True,
                 num_assets: int | None = None,
                 asset_embed_dim: int | None = None,
                 pre_embedding_dim: int | None = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_directions = 2 if bidirectional else 1
        self.use_spatial_attention = use_spatial_attention

        # --------------------------------------------------------------------------------------------------
        # 0. (Optional) asset ID embedding ---------------------------------------------------------------
        # A small learnable vector for each asset is added to the latent representation so the model can
        # distinguish between tickers and learn asset-specific biases.  We match the embedding size to the
        # hidden state so no extra projection is required.
        # --------------------------------------------------------------------------------------------------
        self.use_asset_embedding = num_assets is not None and (asset_embed_dim or 0) > 0
        if self.use_asset_embedding:
            self.asset_embed = nn.Embedding(num_assets, asset_embed_dim or hidden_dim * self.num_directions)
            # If embed dim smaller than hidden, project up; if larger, project down.
            if (asset_embed_dim or hidden_dim * self.num_directions) != hidden_dim * self.num_directions:
                self.asset_proj = nn.Linear(asset_embed_dim or hidden_dim * self.num_directions,
                                            hidden_dim * self.num_directions,
                                            bias=False)
            else:
                self.asset_proj = None

        # --------------------------------------------------------------------------------------------------
        # 0.a (Optional) pre-LSTM asset embedding -----------------------------------------------------------
        # A separate asset ID embedding concatenated to the raw features at each time step, allowing the
        # temporal encoder to condition its dynamics on asset identity.
        # --------------------------------------------------------------------------------------------------
        self.use_pre_asset_embedding = num_assets is not None and (pre_embedding_dim or 0) > 0
        if self.use_pre_asset_embedding:
            self.pre_asset_embed = nn.Embedding(num_assets, pre_embedding_dim)

        # 1. Temporal encoder (shared across assets)
        lstm_input_dim = input_dim + (pre_embedding_dim or 0 if self.use_pre_asset_embedding else 0)
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # 2. Spatial self-attention across assets (optional)
        if use_spatial_attention:
            self.spatial_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim * self.num_directions,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )

        # 3. Output projection
        self.fc = nn.Linear(hidden_dim * self.num_directions, output_dim)

        # Normalisation + regularisation
        self.norm = nn.LayerNorm(hidden_dim * self.num_directions)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Args:
            x: Tensor of shape (batch, asset, seq_len, feat)
        Returns:
            logits: Tensor of shape (batch, asset, output_dim)
        """

        B, A, T, F = x.shape

        # Add pre-LSTM asset embedding if enabled -----------------------------------------------------------
        if self.use_pre_asset_embedding:
            asset_ids = torch.arange(A, device=x.device)
            pre_emb = self.pre_asset_embed(asset_ids)  # (A, E_pre)
            pre_emb = pre_emb.unsqueeze(0).unsqueeze(2)  # (1, A, 1, E_pre)
            pre_emb = pre_emb.expand(B, -1, T, -1)      # (B, A, T, E_pre)
            x = torch.cat([x, pre_emb], dim=-1)          # (B, A, T, F + E_pre)

        # (B*A, T, F_concat) → LSTM → (B*A, H)
        x_flat = x.reshape(B * A, T, x.shape[-1])
        out, _ = self.lstm(x_flat)
        h = out[:, -1, :]  # last time step (B*A, H)
        h = self.norm(h)
        h = self.dropout(h)

        # reshape to (B, A, H)
        h = h.reshape(B, A, -1)

        # Add asset embedding if enabled ------------------------------------------------------------------
        if self.use_asset_embedding:
            asset_ids = torch.arange(A, device=x.device)
            asset_emb = self.asset_embed(asset_ids)  # (A, E)
            if self.asset_proj is not None:
                asset_emb = self.asset_proj(asset_emb)  # (A, H)
            asset_emb = asset_emb.unsqueeze(0).expand(B, -1, -1)  # (B, A, H)
            h = h + asset_emb

        # Apply spatial attention if enabled
        if self.use_spatial_attention:
            h_attn, _ = self.spatial_attn(h, h, h)
            h = self.norm(h_attn + h)  # residual + norm
            h = self.dropout(h)

        out = self.fc(h)  # (B, A, output_dim)

        # Regression convenience: squeeze the trailing dim so that the shape
        # matches common loss functions (B, A) instead of (B, A, 1).
        if self.output_dim == 1:
            out = out.squeeze(-1)  # (B, A)

        return out