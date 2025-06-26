import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalSpatialClassifier(nn.Module):
    """A light-weight model that processes a portfolio tensor
    (batch, asset, seq_len, feat) and outputs class logits per asset.

    Architecture:
      1. Shared LSTM encodes each asset's sequence → latent vector.
      2. (Optional) Single multi-head self-attention layer lets assets attend to
         each other within the same time slice (spatial dependency).
      3. Linear layer produces logits for n_class per asset.

    Args:
        input_dim: Number of input features per time step
        hidden_dim: Size of LSTM hidden state
        n_class: Number of output classes
        lstm_layers: Number of stacked LSTM layers
        bidirectional: Whether to use bidirectional LSTM
        num_heads: Number of attention heads for spatial attention
        dropout: Dropout probability
        use_spatial_attention: Whether to use spatial attention layer (default: True)
        num_assets: Number of assets (optional)
        asset_embed_dim: Dimension of asset embedding (optional)
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 n_class: int = 3,
                 lstm_layers: int = 1,
                 bidirectional: bool = False,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 use_spatial_attention: bool = True,
                 num_assets: int | None = None,
                 asset_embed_dim: int | None = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_class = n_class
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

        # 1. Temporal encoder (shared across assets)
        self.lstm = nn.LSTM(
            input_size=input_dim,
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
        self.fc = nn.Linear(hidden_dim * self.num_directions, n_class)

        # Normalisation + regularisation
        self.norm = nn.LayerNorm(hidden_dim * self.num_directions)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Args:
            x: Tensor of shape (batch, asset, seq_len, feat)
        Returns:
            logits: Tensor of shape (batch, asset, n_class)
        """
        B, A, T, F = x.shape

        # (B*A, T, F) → LSTM → (B*A, H)
        x_flat = x.view(B * A, T, F)
        out, _ = self.lstm(x_flat)
        h = out[:, -1, :]  # last time step (B*A, H)
        h = self.norm(h)
        h = self.dropout(h)

        # reshape to (B, A, H)
        h = h.view(B, A, -1)

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

        logits = self.fc(h)  # (B, A, n_class)
        return logits


if __name__ == "__main__":
    # Test both configurations
    B, A, T, F = 8, 50, 60, 4
    dummy = torch.randn(B, A, T, F)
    
    # With spatial attention
    model_with_attn = TemporalSpatialClassifier(input_dim=F, hidden_dim=64, n_class=3, use_spatial_attention=True)
    out_with_attn = model_with_attn(dummy)
    print("Output shape with attention:", out_with_attn.shape)  # expected (8, 50, 3)
    
    # Without spatial attention
    model_no_attn = TemporalSpatialClassifier(input_dim=F, hidden_dim=64, n_class=3, use_spatial_attention=False)
    out_no_attn = model_no_attn(dummy)
    print("Output shape without attention:", out_no_attn.shape)  # expected (8, 50, 3) 