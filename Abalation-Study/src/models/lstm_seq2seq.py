"""
LSTM Encoder-Decoder for TC track forecasting.

Design choices:
- Encoder processes obs window via 2-layer LSTM
- Decoder autoregressively predicts relative displacements
- Optional: env features concatenated at decoder input
- Optional: basin embedding for conditioned generalization
- Input = [lon, lat, pres, wind, delta_lon, delta_lat] (absolute + relative)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor):
        """
        x : (B, obs_len, input_dim)
        Returns: output (B, obs_len, hidden_dim), (h_n, c_n)
        """
        return self.lstm(x)


class LSTMDecoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 2,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward_step(
        self,
        x: torch.Tensor,               # (B, 1, input_dim)
        hidden: tuple,
    ):
        out, hidden = self.lstm(x, hidden)  # out: (B, 1, hidden_dim)
        pred = self.out_proj(out)           # (B, 1, output_dim)
        return pred, hidden


class LSTMSeq2Seq(nn.Module):
    """
    LSTM Seq2Seq with optional env features and basin conditioning.

    Parameters
    ----------
    obs_feat_dim : int
        Dimension of per-timestep observation features (default 6: 4 abs + 2 rel).
    hidden_dim   : int
        LSTM hidden state dimension.
    pred_len     : int
        Number of prediction steps.
    env_dim      : int or None
        Dimension of environmental features (per timestep). If None, env not used.
    n_basins     : int
        Number of basins for basin embedding (set to 0 to disable).
    basin_emb_dim: int
        Basin embedding size.
    num_layers   : int
        Number of LSTM layers.
    dropout      : float
        Dropout rate.
    """

    def __init__(
        self,
        obs_feat_dim: int = 6,
        hidden_dim: int = 128,
        pred_len: int = 4,
        env_dim: Optional[int] = None,
        n_basins: int = 6,
        basin_emb_dim: int = 16,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.use_env = env_dim is not None
        self.use_basin = n_basins > 0

        enc_input_dim = obs_feat_dim
        if self.use_basin:
            self.basin_emb = nn.Embedding(n_basins, basin_emb_dim)
            enc_input_dim += basin_emb_dim

        self.encoder = LSTMEncoder(enc_input_dim, hidden_dim, num_layers, dropout)

        # Decoder input: last pred (2) + optional env (env_dim) + optional basin emb
        dec_input_dim = 2
        if self.use_env and env_dim is not None:
            self.env_proj = nn.Linear(env_dim, 32)
            dec_input_dim += 32
        if self.use_basin:
            dec_input_dim += basin_emb_dim

        self.decoder = LSTMDecoder(dec_input_dim, hidden_dim, output_dim=2,
                                   num_layers=num_layers, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        obs: torch.Tensor,           # (B, obs_len, 4)   absolute
        obs_rel: torch.Tensor,       # (B, obs_len, 2)   relative
        env: Optional[torch.Tensor] = None,  # (B, obs_len, env_dim)
        basin_idx: Optional[torch.Tensor] = None,  # (B,)
    ) -> torch.Tensor:
        """
        Returns pred_rel : (B, pred_len, 2) relative displacements.
        """
        B = obs.shape[0]
        device = obs.device

        # Build encoder input: concat abs features and relative displacements
        enc_input = torch.cat([obs, obs_rel], dim=-1)  # (B, obs_len, 6)

        # Optional basin embedding (broadcast over time)
        basin_emb = None
        if self.use_basin and basin_idx is not None:
            basin_emb = self.basin_emb(basin_idx)  # (B, basin_emb_dim)
            basin_t = basin_emb.unsqueeze(1).expand(-1, obs.shape[1], -1)
            enc_input = torch.cat([enc_input, basin_t], dim=-1)

        _, (h_n, c_n) = self.encoder(enc_input)  # h_n: (num_layers, B, hidden)

        # Autoregressive decoding
        preds = []
        dec_input_vel = obs_rel[:, -1:, :2]  # (B, 1, 2) — seed with last observed velocity (lon/lat rel)

        hidden = (h_n, c_n)
        for step in range(self.pred_len):
            dec_in = dec_input_vel  # (B, 1, 2)

            if self.use_env and env is not None:
                # Use last obs env features for all pred steps (simplification)
                env_last = self.env_proj(env[:, -1, :])  # (B, 32)
                env_t = env_last.unsqueeze(1)             # (B, 1, 32)
                dec_in = torch.cat([dec_in, env_t], dim=-1)

            if self.use_basin and basin_emb is not None:
                bas_t = basin_emb.unsqueeze(1)             # (B, 1, basin_emb_dim)
                dec_in = torch.cat([dec_in, bas_t], dim=-1)

            pred_step, hidden = self.decoder.forward_step(dec_in, hidden)
            # pred_step: (B, 1, 2)
            preds.append(pred_step)
            dec_input_vel = pred_step  # feed prediction back

        return torch.cat(preds, dim=1)  # (B, pred_len, 2)


# ---------------------------------------------------------------------------
# Variant: bilinear attention on encoder outputs
# ---------------------------------------------------------------------------

class LSTMSeq2SeqAttn(LSTMSeq2Seq):
    """
    LSTM Seq2Seq with Bahdanau-style additive attention over encoder states.
    Allows decoder to selectively attend to relevant parts of the obs window.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn_W = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.attn_U = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.attn_v = nn.Linear(self.hidden_dim, 1)

    def _attend(self, dec_h: torch.Tensor, enc_outs: torch.Tensor) -> torch.Tensor:
        """
        dec_h    : (B, hidden_dim) — top decoder hidden state
        enc_outs : (B, obs_len, hidden_dim) — all encoder outputs
        Returns  : context (B, hidden_dim)
        """
        dec_h_exp = dec_h.unsqueeze(1).expand_as(enc_outs)  # (B, obs_len, hidden)
        energy = torch.tanh(self.attn_W(enc_outs) + self.attn_U(dec_h_exp))
        scores = self.attn_v(energy).squeeze(-1)             # (B, obs_len)
        weights = torch.softmax(scores, dim=-1).unsqueeze(1)  # (B, 1, obs_len)
        context = torch.bmm(weights, enc_outs).squeeze(1)    # (B, hidden)
        return context

    def forward(
        self,
        obs: torch.Tensor,
        obs_rel: torch.Tensor,
        env=None,
        basin_idx=None,
    ) -> torch.Tensor:
        B = obs.shape[0]

        enc_input = torch.cat([obs, obs_rel], dim=-1)
        basin_emb = None
        if self.use_basin and basin_idx is not None:
            basin_emb = self.basin_emb(basin_idx)
            basin_t = basin_emb.unsqueeze(1).expand(-1, obs.shape[1], -1)
            enc_input = torch.cat([enc_input, basin_t], dim=-1)

        enc_outs, (h_n, c_n) = self.encoder(enc_input)

        preds = []
        dec_input_vel = obs_rel[:, -1:, :2]
        hidden = (h_n, c_n)

        for step in range(self.pred_len):
            dec_h_top = hidden[0][-1]                       # (B, hidden)
            context = self._attend(dec_h_top, enc_outs)     # (B, hidden)

            dec_in = dec_input_vel                          # (B, 1, 2)
            if self.use_env and env is not None:
                env_last = self.env_proj(env[:, -1, :])
                env_t = env_last.unsqueeze(1)
                dec_in = torch.cat([dec_in, env_t], dim=-1)
            if self.use_basin and basin_emb is not None:
                bas_t = basin_emb.unsqueeze(1)
                dec_in = torch.cat([dec_in, bas_t], dim=-1)

            # Inject attention context into decoder hidden state
            h_list = list(hidden[0].unbind(0))
            h_list[-1] = h_list[-1] + self.dropout(context)
            new_h = torch.stack(h_list, dim=0)
            hidden_mod = (new_h, hidden[1])

            pred_step, hidden = self.decoder.forward_step(dec_in, hidden_mod)
            preds.append(pred_step)
            dec_input_vel = pred_step

        return torch.cat(preds, dim=1)  # (B, pred_len, 2)
