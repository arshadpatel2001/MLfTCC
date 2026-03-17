"""
TropiCycloneNet Backbone (Multimodal Encoder)
=============================================
Implements the three-branch multimodal encoder matching the TCNM architecture
(Huang et al., Nature Communications 2025), adapted for domain generalization.

Architecture:
  Branch 1: 3D-CNN Encoder  ← Data_3d (13, 81, 81)
  Branch 2: MLP Encoder     ← Data_1d (4,)
  Branch 3: Env-T-Net       ← Env-Data (46,)
  Fusion:   Feature concat + projection → representation z ∈ R^d

For PhysIRM, the final representation is split into:
  z_phys ← physics-feature sub-space (invariant across basins)
  z_env  ← synoptic sub-space (basin-specific, allowed to shift)
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Utility layers ────────────────────────────────────────────────────────────

class ConvBnRelu(nn.Sequential):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(in_c, out_c, k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )


class ResBlock2D(nn.Module):
    """Lightweight 2D residual block for ERA5 spatial encoding."""
    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            ConvBnRelu(channels, channels),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))


class SqueezeExcite(nn.Module):
    """Channel attention – upweights physically important feature maps."""
    def __init__(self, channels: int, ratio: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // ratio),
            nn.ReLU(),
            nn.Linear(channels // ratio, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


# ── Branch 1: ERA5 Spatial Encoder ───────────────────────────────────────────

class SpatialEncoder(nn.Module):
    """
    Encodes Data_3d (13, 81, 81) ERA5 patches into a fixed-length vector.

    Design inspired by TCNM's 3D-Data Encoder and Hurricast's spatial branch.
    Uses multi-scale convolutions to capture both local (eyewall) and
    mesoscale (200–500km) atmospheric structure.

    Output: feature vector of size `embed_dim`.
    """

    def __init__(self, in_channels: int = 13, embed_dim: int = 256, dropout: float = 0.1):
        super().__init__()

        self.stem = nn.Sequential(
            ConvBnRelu(in_channels, 32, k=7, s=2, p=3),  # 81 → 41
            ConvBnRelu(32, 64, k=3, s=2, p=1),            # 41 → 21
        )

        self.layer1 = nn.Sequential(
            ConvBnRelu(64, 128, k=3, s=2, p=1),  # 21 → 11
            ResBlock2D(128, dropout),
            SqueezeExcite(128),
        )

        self.layer2 = nn.Sequential(
            ConvBnRelu(128, 256, k=3, s=2, p=1),  # 11 → 6
            ResBlock2D(256, dropout),
            SqueezeExcite(256),
        )

        self.layer3 = nn.Sequential(
            ConvBnRelu(256, 512, k=3, s=2, p=1),  # 6 → 3
            ResBlock2D(512, dropout),
        )

        # Global + local pooling (captures both mean and spatial variance)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        # Pressure-level attention: learn which levels matter most
        self.level_attn = nn.Sequential(
            nn.Linear(512 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 13, 81, 81) ERA5 patch tensor
        Returns:
            z: (B, embed_dim) spatial feature vector
        """
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        # Concatenate global average and max pool for richer representation
        feat = torch.cat([self.gap(h).squeeze(-1).squeeze(-1),
                          self.gmp(h).squeeze(-1).squeeze(-1)], dim=-1)  # (B, 1024)
        return self.level_attn(feat)  # (B, embed_dim)


# ── Branch 2: Tabular Track Encoder ──────────────────────────────────────────

class TrackEncoder(nn.Module):
    """
    Encodes Data_1d tabular features [LONG_norm, LAT_norm, PRES_norm, WND_norm]
    into a feature vector.

    Output: feature vector of size `embed_dim`.
    """

    def __init__(self, in_dim: int = 4, embed_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Branch 3: Environmental Context Encoder (Env-T-Net) ──────────────────────

class EnvEncoder(nn.Module):
    """
    Encodes the 77-dimensional Env-Data vector.

    Actual TCND field layout (77 total):
      month(12) + area(6) + intensity_class(6) + wind(1) + move_velocity(1)
      + location_long(36) + location_lat(12)
      + hist_dir12(1) + hist_dir24(1) + hist_inte24(1)
    """

    def __init__(self, in_dim: int = 77, embed_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.month_emb    = nn.Linear(12, 32)
        self.area_emb     = nn.Linear(6,  16)
        self.icls_emb     = nn.Linear(6,  16)
        self.scalar_emb   = nn.Linear(2,  16)   # wind + move_velocity
        self.loc_long_emb = nn.Linear(36, 32)
        self.loc_lat_emb  = nn.Linear(12, 16)
        self.hist_emb     = nn.Linear(3,  16)   # dir12 + dir24 + inte24 scalars

        fused_dim = 32 + 16 + 16 + 16 + 32 + 16 + 16  # 144

        self.fuse = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 77) → z: (B, embed_dim)"""
        month    = x[:,  0:12]
        area     = x[:, 12:18]
        icls     = x[:, 18:24]
        scalars  = x[:, 24:26]
        loc_long = x[:, 26:62]
        loc_lat  = x[:, 62:74]
        hist     = x[:, 74:77]

        parts = [
            F.relu(self.month_emb(month)),
            F.relu(self.area_emb(area)),
            F.relu(self.icls_emb(icls)),
            F.relu(self.scalar_emb(scalars)),
            F.relu(self.loc_long_emb(loc_long)),
            F.relu(self.loc_lat_emb(loc_lat)),
            F.relu(self.hist_emb(hist)),
        ]
        return self.fuse(torch.cat(parts, dim=-1))


class PhysicsEncoder(nn.Module):
    """
    Encodes the 8-dimensional physics feature vector into the invariant
    sub-space used by PhysIRM.

    Physical features:
      [SST anomaly, wind shear, Coriolis, MPI proxy, BL moisture,
       outflow temp, steering, current intensity]
    """

    def __init__(self, in_dim: int = 8, phys_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, phys_dim),
            nn.LayerNorm(phys_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Fusion + Prediction Heads ─────────────────────────────────────────────────

class MultimodalBackbone(nn.Module):
    """
    Full multimodal backbone: fuses spatial, track, and env branches.

    Produces:
      - z:       full joint representation (phys + env concatenated)
      - z_phys:  physics sub-representation (for PhysIRM invariance penalty)
      - z_env:   synoptic sub-representation (basin-specific features)
    """

    def __init__(
        self,
        spatial_embed: int = 256,
        track_embed:   int = 64,
        env_embed:     int = 128,
        phys_dim:      int = 64,
        final_dim:     int = 256,
        dropout:       float = 0.1,
        use_3d:        bool = True,
        use_env:       bool = True,
    ):
        super().__init__()
        self.use_3d  = use_3d
        self.use_env = use_env
        self.phys_dim = phys_dim

        self.spatial_enc = SpatialEncoder(embed_dim=spatial_embed, dropout=dropout) if use_3d else None
        self.track_enc   = TrackEncoder(embed_dim=track_embed, dropout=dropout)
        self.env_enc     = EnvEncoder(embed_dim=env_embed, dropout=dropout) if use_env else None
        self.phys_enc    = PhysicsEncoder(phys_dim=phys_dim, dropout=dropout)

        # Compute fused dimension
        fused_in = track_embed
        if use_3d:  fused_in += spatial_embed
        if use_env: fused_in += env_embed

        # Projection into final representation space
        self.projector = nn.Sequential(
            nn.Linear(fused_in, final_dim * 2),
            nn.LayerNorm(final_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim * 2, final_dim),
            nn.LayerNorm(final_dim),
        )

        # Split z into [z_env | z_phys] via learned routing
        # z_env_dim + z_phys_dim = final_dim
        self.env_dim  = final_dim - phys_dim
        self.final_dim = final_dim

        # Align physics encoder output dimension with phys sub-space
        self.phys_align = nn.Linear(phys_dim, phys_dim)

    def forward(self, batch: dict) -> dict:
        """
        Args:
            batch: dict with keys data_1d, data_3d, env_data, phys_features
        Returns:
            dict with z, z_phys, z_env
        """
        parts = []

        if self.use_3d:
            parts.append(self.spatial_enc(batch["data_3d"]))

        parts.append(self.track_enc(batch["data_1d"]))

        if self.use_env:
            parts.append(self.env_enc(batch["env_data"]))

        z_full = self.projector(torch.cat(parts, dim=-1))  # (B, final_dim)

        # Split into [z_env | z_phys]
        z_env  = z_full[:, :self.env_dim]                  # (B, env_dim)
        z_phys = z_full[:, self.env_dim:]                  # (B, phys_dim)

        # Blend z_phys with physics encoder output (anchors to physical meaning)
        phys_enc_out = self.phys_enc(batch["phys_features"])  # (B, phys_dim)
        z_phys = z_phys + self.phys_align(phys_enc_out)       # residual physics injection

        return {
            "z":      z_full,    # full representation (ERM, CORAL, DANN use this)
            "z_phys": z_phys,    # physics sub-space (PhysIRM invariance target)
            "z_env":  z_env,     # synoptic sub-space (PhysIRM allows to shift)
        }

    def get_output_dim(self) -> int:
        return self.final_dim


class TaskHeads(nn.Module):
    """
    Two-task prediction heads:
      1. Intensity change classification (5 classes)
      2. Direction classification (8 classes)

    Designed to share the same backbone, with separate final layers.
    """

    def __init__(self, in_dim: int = 256, n_intensity: int = 5, n_direction: int = 8,
                 dropout: float = 0.1):
        super().__init__()

        self.intensity_head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_intensity),
        )

        self.direction_head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_direction),
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits_intensity: (B, 5)
            logits_direction:  (B, 8)
        """
        return self.intensity_head(z), self.direction_head(z)


# ── Full model ────────────────────────────────────────────────────────────────

class TropiCycloneModel(nn.Module):
    """
    End-to-end TC forecasting model for basin generalization experiments.

    Methods call backbone.forward() then heads.forward().
    The model exposes both z_phys and z_env for DG algorithms.
    """

    def __init__(self, backbone: MultimodalBackbone, heads: TaskHeads):
        super().__init__()
        self.backbone = backbone
        self.heads    = heads

    def forward(self, batch: dict) -> dict:
        feat = self.backbone(batch)
        logits_int, logits_dir = self.heads(feat["z"])
        return {
            **feat,
            "logits_intensity": logits_int,
            "logits_direction":  logits_dir,
        }

    @classmethod
    def build(
        cls,
        spatial_embed: int = 256,
        track_embed:   int = 64,
        env_embed:     int = 128,
        phys_dim:      int = 64,
        final_dim:     int = 256,
        dropout:       float = 0.1,
        use_3d:        bool = True,
        use_env:       bool = True,
    ) -> "TropiCycloneModel":
        backbone = MultimodalBackbone(
            spatial_embed=spatial_embed, track_embed=track_embed,
            env_embed=env_embed, phys_dim=phys_dim,
            final_dim=final_dim, dropout=dropout,
            use_3d=use_3d, use_env=use_env,
        )
        heads = TaskHeads(in_dim=final_dim, dropout=dropout)
        return cls(backbone, heads)