"""
Transformer-based Seq2Seq for TC track forecasting.

Architecture:
- Input projection: obs features → d_model
- Sinusoidal + learned positional encoding
- Transformer encoder (self-attention over obs window)
- Transformer decoder (cross-attention: pred queries → obs keys)
- Output head: d_model → 2 (delta_lon, delta_lat)

Key design: uses "query embeddings" for the decoder (like DETR),
one learnable embedding per prediction step. This avoids autoregressive
coupling at test time and allows parallel decoding.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.shape[1], :]
        return self.dropout(x)


class TransformerForecaster(nn.Module):
    """
    Non-autoregressive Transformer for TC track forecasting.

    Parameters
    ----------
    obs_feat_dim  : input feature dimension per obs timestep
    d_model       : transformer model dimension
    nhead         : number of attention heads
    num_enc_layers: encoder transformer layers
    num_dec_layers: decoder transformer layers
    pred_len      : number of prediction steps
    dim_feedforward: FFN hidden size
    dropout       : dropout rate
    env_dim       : env feature dim (or None)
    n_basins      : for basin embedding (0 = disabled)
    basin_emb_dim : basin embedding size
    """

    def __init__(
        self,
        obs_feat_dim: int = 6,
        d_model: int = 128,
        nhead: int = 4,
        num_enc_layers: int = 3,
        num_dec_layers: int = 2,
        pred_len: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        env_dim: Optional[int] = None,
        n_basins: int = 6,
        basin_emb_dim: int = 16,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.d_model = d_model
        self.use_env = env_dim is not None
        self.use_basin = n_basins > 0

        in_dim = obs_feat_dim
        if self.use_env and env_dim is not None:
            self.env_proj = nn.Sequential(
                nn.Linear(env_dim, 64), nn.ReLU(), nn.Linear(64, 32)
            )
            in_dim += 32
        if self.use_basin:
            self.basin_emb = nn.Embedding(n_basins, basin_emb_dim)
            in_dim += basin_emb_dim

        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_enc_layers)

        # Learnable prediction query embeddings — one per pred step
        self.pred_queries = nn.Parameter(torch.randn(1, pred_len, d_model))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_dec_layers)

        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pred_queries, std=0.02)
        for p in self.output_head.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        obs: torch.Tensor,               # (B, obs_len, 4)
        obs_rel: torch.Tensor,           # (B, obs_len, 2)
        env: Optional[torch.Tensor] = None,  # (B, obs_len, env_dim)
        basin_idx: Optional[torch.Tensor] = None,  # (B,)
    ) -> torch.Tensor:
        B = obs.shape[0]

        # Build per-timestep feature
        feat = torch.cat([obs, obs_rel], dim=-1)  # (B, obs_len, 6)

        if self.use_env and env is not None:
            env_proj = self.env_proj(env)  # (B, obs_len, 32)
            feat = torch.cat([feat, env_proj], dim=-1)

        if self.use_basin and basin_idx is not None:
            b_emb = self.basin_emb(basin_idx)  # (B, basin_emb_dim)
            b_t   = b_emb.unsqueeze(1).expand(-1, obs.shape[1], -1)
            feat  = torch.cat([feat, b_t], dim=-1)

        # Encoder
        enc_in = self.pos_enc(self.input_proj(feat))   # (B, obs_len, d_model)
        memory = self.encoder(enc_in)                   # (B, obs_len, d_model)

        # Decoder — expand learnable queries over batch
        queries = self.pred_queries.expand(B, -1, -1)  # (B, pred_len, d_model)
        dec_out = self.decoder(queries, memory)         # (B, pred_len, d_model)

        return self.output_head(dec_out)                # (B, pred_len, 2)
