from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F

from .unet import TimeEmbedding


class GuideEncoder12(nn.Module):
    def __init__(self, guide_dim=64, month_dim=16, rank_dim=8):
        super().__init__()
        self.month_embedding = nn.Embedding(12, month_dim)
        self.rank_embedding = nn.Embedding(3, rank_dim)
        self.continuous = nn.Sequential(nn.Linear(3, 32), nn.SiLU())
        self.network = nn.Sequential(
            nn.Linear(32 + month_dim + rank_dim, guide_dim), nn.SiLU(),
            nn.Linear(guide_dim, guide_dim),
        )

    def forward(self, values, deltas, similarities):
        batch, guides, months = values.shape
        continuous = self.continuous(torch.stack([
            values, deltas, similarities.unsqueeze(-1).expand(-1, -1, months)
        ], dim=-1))
        month_ids = torch.arange(months, device=values.device).view(1, 1, months)
        rank_ids = torch.arange(guides, device=values.device).view(1, guides, 1)
        month = self.month_embedding(month_ids).expand(batch, guides, -1, -1)
        rank = self.rank_embedding(rank_ids).expand(batch, -1, months, -1)
        return self.network(torch.cat([continuous, month, rank], dim=-1))


class MaskedCrossAttention(nn.Module):
    def __init__(self, query_dim, guide_dim=64, heads=8):
        super().__init__()
        self.heads, self.head_dim = heads, guide_dim // heads
        self.to_query = nn.Linear(query_dim, guide_dim, bias=False)
        self.to_key = nn.Linear(guide_dim, guide_dim, bias=False)
        self.to_value = nn.Linear(guide_dim, guide_dim, bias=False)
        self.to_output = nn.Linear(guide_dim, query_dim)
        self.similarity_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, query, guide_features, guide_mask, retrieval_mask, similarities):
        batch, query_length, _ = query.shape
        guides, months = guide_features.shape[1:3]
        tokens = guide_features.reshape(batch, guides * months, -1)
        valid = (guide_mask * retrieval_mask.unsqueeze(-1)).reshape(batch, guides * months) > 0
        q = self.to_query(query).reshape(batch, query_length, self.heads, self.head_dim).transpose(1, 2)
        k = self.to_key(tokens).reshape(batch, guides * months, self.heads, self.head_dim).transpose(1, 2)
        v = self.to_value(tokens).reshape(batch, guides * months, self.heads, self.head_dim).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        bias = similarities.unsqueeze(-1).expand(-1, -1, months).reshape(batch, 1, 1, -1)
        scores = (scores + self.similarity_scale * bias).masked_fill(~valid[:, None, None, :], -1e4)
        weights = torch.softmax(scores, dim=-1) * valid[:, None, None, :].to(scores.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        output = (weights @ v).transpose(1, 2).reshape(batch, query_length, -1)
        return self.to_output(output)


class ConditionalBlock(nn.Module):
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.norm1, self.norm2 = nn.GroupNorm(8, channels), nn.GroupNorm(8, channels)
        self.conv1, self.conv2 = nn.Conv1d(channels, channels, 3, padding=1), nn.Conv1d(channels, channels, 3, padding=1)
        self.film = nn.Linear(256, channels * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values, condition):
        scale, shift = self.film(condition).chunk(2, dim=-1)
        hidden = self.conv1(F.silu(self.norm1(values)))
        hidden = self.norm2(hidden) * (1 + scale[:, :, None]) + shift[:, :, None]
        return values + self.conv2(self.dropout(F.silu(hidden)))


class AttentionStage(nn.Module):
    def __init__(self, channels):
        super().__init__(); self.attention = MaskedCrossAttention(channels)

    def forward(self, values, guides, batch):
        attended = self.attention(
            values.transpose(1, 2), guides, batch["guide_mask"],
            batch["retrieval_mask"], batch["guide_similarities"],
        )
        return values + attended.transpose(1, 2)


class AbsoluteAttentionUNet12(nn.Module):
    """Cross-attention denoiser with lengths 12 -> 6 -> 3 -> 6 -> 12."""

    def __init__(self, dropout=0.1):
        super().__init__()
        self.condition_encoder = nn.Sequential(nn.Linear(11, 128), nn.SiLU(), nn.Linear(128, 128))
        self.time_encoder = TimeEmbedding(128)
        self.guide_encoder = GuideEncoder12()
        self.input = nn.Conv1d(1, 64, 3, padding=1)
        self.enc12 = nn.ModuleList([ConditionalBlock(64, dropout) for _ in range(2)])
        self.attn12 = AttentionStage(64)
        self.down6 = nn.Conv1d(64, 128, 3, stride=2, padding=1)
        self.enc6 = nn.ModuleList([ConditionalBlock(128, dropout) for _ in range(2)])
        self.attn6 = AttentionStage(128)
        self.down3 = nn.Conv1d(128, 256, 3, stride=2, padding=1)
        self.mid3 = nn.ModuleList([ConditionalBlock(256, dropout) for _ in range(2)])
        self.attn3 = AttentionStage(256)
        self.up6 = nn.ConvTranspose1d(256, 128, 4, stride=2, padding=1)
        self.merge6 = nn.Conv1d(256, 128, 1)
        self.dec6 = nn.ModuleList([ConditionalBlock(128, dropout) for _ in range(2)])
        self.up12 = nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1)
        self.merge12 = nn.Conv1d(128, 64, 1)
        self.dec12 = nn.ModuleList([ConditionalBlock(64, dropout) for _ in range(2)])
        self.output = nn.Sequential(nn.GroupNorm(8, 64), nn.SiLU(), nn.Conv1d(64, 1, 3, padding=1))

    @staticmethod
    def _blocks(values, blocks, condition):
        for block in blocks: values = block(values, condition)
        return values

    def forward(self, noisy, timesteps, batch):
        condition = torch.cat([
            self.condition_encoder(torch.cat([batch["current"], batch["history"], batch["history_mask"]], dim=-1)),
            self.time_encoder(timesteps),
        ], dim=-1)
        guides = self.guide_encoder(batch["guide_values"], batch["guide_deltas"], batch["guide_similarities"])
        x12 = self.attn12(self._blocks(self.input(noisy[:, None]), self.enc12, condition), guides, batch)
        x6 = self.attn6(self._blocks(self.down6(x12), self.enc6, condition), guides, batch)
        x3 = self.attn3(self._blocks(self.down3(x6), self.mid3, condition), guides, batch)
        y6 = self._blocks(self.merge6(torch.cat([self.up6(x3), x6], dim=1)), self.dec6, condition)
        y12 = self._blocks(self.merge12(torch.cat([self.up12(y6), x12], dim=1)), self.dec12, condition)
        return self.output(y12).squeeze(1)
