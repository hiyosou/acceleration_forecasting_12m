from __future__ import annotations

import torch
from torch import nn


class WaveformAutoencoder(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.encoder_convs = nn.Sequential(
            nn.Conv1d(1, 16, 15, stride=2, padding=7), nn.LeakyReLU(inplace=True),
            nn.Conv1d(16, 32, 9, stride=2, padding=4), nn.LeakyReLU(inplace=True),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.LeakyReLU(inplace=True),
        )
        self.to_embedding = nn.Linear(64 * 63, int(embedding_dim))
        self.from_embedding = nn.Linear(int(embedding_dim), 64 * 63)
        self.decoder_convs = nn.Sequential(
            nn.ConvTranspose1d(64, 32, 5, stride=2, padding=2), nn.LeakyReLU(inplace=True),
            nn.ConvTranspose1d(32, 16, 9, stride=2, padding=4, output_padding=1), nn.LeakyReLU(inplace=True),
            nn.ConvTranspose1d(16, 1, 15, stride=2, padding=7, output_padding=1),
        )

    def encode(self, waveform):
        return self.to_embedding(self.encoder_convs(waveform).flatten(start_dim=1))

    def forward(self, waveform):
        embedding = self.encode(waveform)
        decoded = self.decoder_convs(self.from_embedding(embedding).reshape(-1, 64, 63))
        return decoded, embedding


def normalize_embeddings(values, eps=1e-12):
    return values / values.norm(dim=1, keepdim=True).clamp_min(eps)

