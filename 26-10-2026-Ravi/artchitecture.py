import torch
import torch.nn as nn
import numpy as np
import pandas as pd


class PatchEmbedding(nn.Module):
    """
    Custom Patch Embedding layer for [1, 10, 10] input using 2x2 patches.
    Results in 25 patches of dimension 4, projected to hidden_dim=64.
    """
    def __init__(self, in_channels=1, patch_size=2, hidden_dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.patch_dim = in_channels * patch_size * patch_size  # 1 * 2 * 2 = 4
        self.projection = nn.Linear(self.patch_dim, hidden_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, (H // p) * (W // p), C * p * p)
        return self.projection(x)


class TransformerEncoderBlock(nn.Module):
    """
    Transformer Encoder Block with LayerNorm, Multi-Head Self-Attention,
    GELU MLP, and Residual Additions.
    """
    def __init__(self, hidden_dim=64, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)

        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class HydroTransformerNet(nn.Module):
    """
    Dual-Input Audio Spectrogram Transformer (ViT + Tabular Fusion)
    with automatic batch dimension alignment.
    """
    def __init__(self, hidden_dim=64, num_layers=3, num_heads=4):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels=1, patch_size=2, hidden_dim=hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 26, hidden_dim))

        self.transformer_encoder = nn.Sequential(*[
            TransformerEncoderBlock(hidden_dim=hidden_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])

        # Tabular Branch matching checkpoint structure (BatchNorm1d)
        self.tabular_mlp = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.4)
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, acoustic, tabular):
        B = acoustic.size(0)
        x = self.patch_embed(acoustic)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        x = self.transformer_encoder(x)
        cls_repr = x[:, 0]

        if tabular.ndim == 1:
            tabular = tabular.unsqueeze(0)
        if tabular.size(0) != B:
            tabular = tabular.repeat(B, 1) if tabular.size(0) == 1 else tabular[:B]

        tab_repr = self.tabular_mlp(tabular)
        fused = torch.cat((cls_repr, tab_repr), dim=1)
        logits = self.fusion_mlp(fused)
        return logits