from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F

from .absolute_attention_unet import GuideEncoder12, ConditionalBlock
from .unet import TimeEmbedding


class ReferenceModulatedAttention(nn.Module):
    """Fuse denoising features, global conditions and all 36 reference tokens."""

    def __init__(self, channels, condition_dim=256, reference_dim=64, heads=8, head_dim=8):
        super().__init__()
        self.heads, self.head_dim = int(heads), int(head_dim)
        inner = self.heads * self.head_dim
        self.query = nn.Linear(channels + condition_dim, inner, bias=False)
        self.key = nn.Linear(reference_dim + condition_dim, inner, bias=False)
        self.value = nn.Linear(reference_dim, inner, bias=False)
        self.feature_output = nn.Linear(inner, channels, bias=False)
        self.condition_output = nn.Linear(inner, condition_dim, bias=False)
        self.similarity_scale = nn.Parameter(torch.tensor(1.0))
        self.last_attention = None
        self.last_context_norm = None

    def forward(self, values, condition, references, reference_mask, similarities):
        batch, channels, length = values.shape
        tokens = references.reshape(batch, 36, -1)
        valid = reference_mask.reshape(batch, 36) > 0
        query_input = torch.cat([
            values.transpose(1, 2), condition[:, None, :].expand(-1, length, -1)
        ], dim=-1)
        key_input = torch.cat([
            tokens, condition[:, None, :].expand(-1, tokens.shape[1], -1)
        ], dim=-1)
        q = self.query(query_input).reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k = self.key(key_input).reshape(batch, 36, self.heads, self.head_dim).transpose(1, 2)
        v = self.value(tokens).reshape(batch, 36, self.heads, self.head_dim).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        bias = similarities[:, :, None].expand(-1, -1, 12).reshape(batch, 1, 1, 36)
        scores = (scores + self.similarity_scale * bias).masked_fill(~valid[:, None, None, :], -1e4)
        weights = torch.softmax(scores, dim=-1) * valid[:, None, None, :].to(scores.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        context = (weights @ v).transpose(1, 2).reshape(batch, length, -1)
        feature_delta = self.feature_output(context).transpose(1, 2)
        condition_delta = self.condition_output(context.mean(dim=1))
        self.last_attention = weights.detach()
        self.last_context_norm = context.detach().norm(dim=-1).mean()
        return feature_delta, condition_delta


class ReferenceModulatedBlock(nn.Module):
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.reference_attention = ReferenceModulatedAttention(channels)
        self.residual = ConditionalBlock(channels, dropout)

    def forward(self, values, condition, references, reference_mask, similarities):
        feature_delta, condition_delta = self.reference_attention(
            values, condition, references, reference_mask, similarities
        )
        return self.residual(values + feature_delta, condition + condition_delta)


class ReferenceModulatedUNet12(nn.Module):
    """Absolute-value U-Net with reference modulation in all ten residual blocks."""

    def __init__(self, dropout=0.1):
        super().__init__()
        self.condition_indicator = False
        self.condition_encoder = nn.Sequential(nn.Linear(11, 128), nn.SiLU(), nn.Linear(128, 128))
        self.time_encoder = TimeEmbedding(128)
        self.reference_encoder = GuideEncoder12()
        self.input = nn.Conv1d(1, 64, 3, padding=1)
        self.enc12 = nn.ModuleList([ReferenceModulatedBlock(64, dropout) for _ in range(2)])
        self.down6 = nn.Conv1d(64, 128, 3, stride=2, padding=1)
        self.enc6 = nn.ModuleList([ReferenceModulatedBlock(128, dropout) for _ in range(2)])
        self.down3 = nn.Conv1d(128, 256, 3, stride=2, padding=1)
        self.mid3 = nn.ModuleList([ReferenceModulatedBlock(256, dropout) for _ in range(2)])
        self.up6 = nn.ConvTranspose1d(256, 128, 4, stride=2, padding=1)
        self.merge6 = nn.Conv1d(256, 128, 1)
        self.dec6 = nn.ModuleList([ReferenceModulatedBlock(128, dropout) for _ in range(2)])
        self.up12 = nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1)
        self.merge12 = nn.Conv1d(128, 64, 1)
        self.dec12 = nn.ModuleList([ReferenceModulatedBlock(64, dropout) for _ in range(2)])
        self.output = nn.Sequential(nn.GroupNorm(8, 64), nn.SiLU(), nn.Conv1d(64, 1, 3, padding=1))

    @property
    def reference_blocks(self):
        return [*self.enc12, *self.enc6, *self.mid3, *self.dec6, *self.dec12]

    @staticmethod
    def _run(values, blocks, condition, references, reference_mask, similarities):
        for block in blocks:
            values = block(values, condition, references, reference_mask, similarities)
        return values

    def diagnostic_stats(self):
        norms, rank_weights = [], []
        for block in self.reference_blocks:
            attention = block.reference_attention
            if attention.last_context_norm is not None:
                norms.append(float(attention.last_context_norm))
            if attention.last_attention is not None:
                weights = attention.last_attention.reshape(
                    attention.last_attention.shape[0], attention.last_attention.shape[1],
                    attention.last_attention.shape[2], 3, 12
                ).sum(dim=-1).mean(dim=(0, 1, 2))
                rank_weights.append(weights.cpu())
        rank = torch.stack(rank_weights).mean(dim=0).tolist() if rank_weights else [0.0] * 3
        return {"reference_context_norm": sum(norms) / max(len(norms), 1),
                **{f"attention_rank_{index + 1}": float(value) for index, value in enumerate(rank)}}

    def forward(self, noisy, timesteps, batch):
        condition = torch.cat([
            self.condition_encoder(torch.cat([
                batch["current"], batch["history"], batch["history_mask"]
            ], dim=-1)),
            self.time_encoder(timesteps),
        ], dim=-1)
        references = self.reference_encoder(
            batch["guide_values"], batch["guide_deltas"], batch["guide_similarities"]
        )
        reference_mask = batch["guide_mask"] * batch["retrieval_mask"].unsqueeze(-1)
        similarities = batch["guide_similarities"]
        x12 = self._run(self.input(noisy[:, None]), self.enc12, condition, references, reference_mask, similarities)
        x6 = self._run(self.down6(x12), self.enc6, condition, references, reference_mask, similarities)
        x3 = self._run(self.down3(x6), self.mid3, condition, references, reference_mask, similarities)
        y6 = self._run(self.merge6(torch.cat([self.up6(x3), x6], dim=1)), self.dec6,
                       condition, references, reference_mask, similarities)
        y12 = self._run(self.merge12(torch.cat([self.up12(y6), x12], dim=1)), self.dec12,
                        condition, references, reference_mask, similarities)
        return self.output(y12).squeeze(1)
