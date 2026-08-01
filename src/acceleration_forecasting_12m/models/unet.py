from __future__ import annotations

import math
import torch
from torch import nn


class TimeEmbedding(nn.Module):
    def __init__(self, dimension=128):
        super().__init__(); self.dimension = int(dimension)
        self.network = nn.Sequential(nn.Linear(dimension, dimension), nn.SiLU(), nn.Linear(dimension, dimension))

    def forward(self, timesteps):
        half = self.dimension // 2
        frequencies = torch.exp(
            -math.log(10000) * torch.arange(half, device=timesteps.device) / max(half - 1, 1)
        )
        angles = timesteps.float()[:, None] * frequencies[None]
        values = torch.cat([angles.sin(), angles.cos()], dim=1)
        return self.network(values)


class ResidualBlock(nn.Module):
    def __init__(self, channels, condition_dim=256, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels); self.norm2 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.condition = nn.Linear(condition_dim, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values, condition):
        hidden = self.conv1(torch.nn.functional.silu(self.norm1(values)))
        hidden = hidden + self.condition(condition)[:, :, None]
        hidden = self.conv2(self.dropout(torch.nn.functional.silu(self.norm2(hidden))))
        return values + hidden


class ResidualUNet12(nn.Module):
    """Attention-free denoiser with temporal lengths 12 -> 6 -> 3 -> 6 -> 12."""

    def __init__(self, dropout=0.1):
        super().__init__()
        self.condition_encoder = nn.Sequential(nn.Linear(11, 128), nn.SiLU(), nn.Linear(128, 128))
        self.time_encoder = TimeEmbedding(128)
        self.input = nn.Conv1d(1, 64, 3, padding=1)
        self.enc12 = nn.ModuleList([ResidualBlock(64, dropout=dropout) for _ in range(2)])
        self.down6 = nn.Conv1d(64, 128, 3, stride=2, padding=1)
        self.enc6 = nn.ModuleList([ResidualBlock(128, dropout=dropout) for _ in range(2)])
        self.down3 = nn.Conv1d(128, 256, 3, stride=2, padding=1)
        self.mid3 = nn.ModuleList([ResidualBlock(256, dropout=dropout) for _ in range(2)])
        self.up6 = nn.ConvTranspose1d(256, 128, 4, stride=2, padding=1)
        self.merge6 = nn.Conv1d(256, 128, 1)
        self.dec6 = nn.ModuleList([ResidualBlock(128, dropout=dropout) for _ in range(2)])
        self.up12 = nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1)
        self.merge12 = nn.Conv1d(128, 64, 1)
        self.dec12 = nn.ModuleList([ResidualBlock(64, dropout=dropout) for _ in range(2)])
        self.output = nn.Conv1d(64, 1, 3, padding=1)

    @staticmethod
    def _apply_blocks(blocks, values, condition):
        for block in blocks:
            values = block(values, condition)
        return values

    def forward(self, noisy, timesteps, batch):
        condition_input = torch.cat([batch["current"], batch["history"], batch["history_mask"]], dim=-1)
        condition = torch.cat([self.condition_encoder(condition_input), self.time_encoder(timesteps)], dim=-1)
        level12 = self._apply_blocks(self.enc12, self.input(noisy[:, None]), condition)
        level6 = self._apply_blocks(self.enc6, self.down6(level12), condition)
        level3 = self._apply_blocks(self.mid3, self.down3(level6), condition)
        decoded6 = self.up6(level3)
        decoded6 = self._apply_blocks(self.dec6, self.merge6(torch.cat([decoded6, level6], dim=1)), condition)
        decoded12 = self.up12(decoded6)
        decoded12 = self._apply_blocks(self.dec12, self.merge12(torch.cat([decoded12, level12], dim=1)), condition)
        return self.output(decoded12).squeeze(1)
