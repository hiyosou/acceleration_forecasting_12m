from __future__ import annotations

import math
import numpy as np
import torch


def cosine_alpha_bars(steps=1000, offset=0.008):
    times = torch.linspace(0, steps, steps + 1, dtype=torch.float64) / steps
    values = torch.cos((times + offset) / (1 + offset) * math.pi / 2).square()
    values = values / values[0]
    betas = 1 - values[1:] / values[:-1]
    betas = torch.clamp(betas, 1e-5, 0.999)
    return torch.cumprod(1 - betas, dim=0).float()


class DiffusionProcess:
    def __init__(self, model, steps=1000):
        self.model, self.steps = model, int(steps)
        self.alpha_bars = cosine_alpha_bars(steps)

    def to(self, device):
        self.model.to(device); self.alpha_bars = self.alpha_bars.to(device); return self

    def training_loss(self, batch, noise=None, timesteps=None):
        target, mask = batch["target"], batch["target_mask"]
        batch_size = target.shape[0]
        timesteps = timesteps if timesteps is not None else torch.randint(
            0, self.steps, (batch_size,), device=target.device
        )
        noise = noise if noise is not None else torch.randn_like(target)
        alpha = self.alpha_bars[timesteps][:, None]
        noisy = alpha.sqrt() * target + (1 - alpha).sqrt() * noise
        predicted = self.model(noisy, timesteps, batch)
        squared = (predicted - noise).square() * mask
        per_record = squared.sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return per_record.mean()

    @torch.inference_mode()
    def ddim(self, batch, *, shape, sampling_steps=50, eta=0.0,
             normalized_clip=None, generator=None, initial_noise_scale=1.0):
        device = batch["current"].device
        values = float(initial_noise_scale) * torch.randn(shape, device=device, generator=generator)
        sequence = np.linspace(self.steps - 1, 0, int(sampling_steps), dtype=int)
        for position, timestep in enumerate(sequence):
            next_timestep = sequence[position + 1] if position + 1 < len(sequence) else -1
            t = torch.full((shape[0],), int(timestep), device=device, dtype=torch.long)
            alpha = self.alpha_bars[int(timestep)]
            next_alpha = self.alpha_bars[int(next_timestep)] if next_timestep >= 0 else torch.tensor(1.0, device=device)
            epsilon = self.model(values, t, batch)
            predicted_clean = (values - (1 - alpha).sqrt() * epsilon) / alpha.sqrt()
            if normalized_clip is not None:
                predicted_clean = predicted_clean.clamp(float(normalized_clip[0]), float(normalized_clip[1]))
            sigma = float(eta) * torch.sqrt((1 - next_alpha) / (1 - alpha) * (1 - alpha / next_alpha))
            direction = torch.sqrt(torch.clamp(1 - next_alpha - sigma.square(), min=0)) * epsilon
            random = torch.randn(values.shape, device=device, generator=generator) if next_timestep >= 0 else 0
            values = next_alpha.sqrt() * predicted_clean + direction + sigma * random
        return values
