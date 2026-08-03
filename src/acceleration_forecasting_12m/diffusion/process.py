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
    def __init__(self, model, steps=1000, min_snr_gamma=None, prediction_type="epsilon"):
        self.model, self.steps = model, int(steps)
        if prediction_type not in {"epsilon", "v_prediction"}:
            raise ValueError("prediction_type must be epsilon or v_prediction")
        if prediction_type == "v_prediction" and min_snr_gamma is not None:
            raise ValueError("v_prediction uses plain masked v-MSE; Min-SNR must be disabled")
        self.prediction_type = prediction_type
        self.min_snr_gamma = None if min_snr_gamma is None else float(min_snr_gamma)
        self.alpha_bars = cosine_alpha_bars(steps)

    def to(self, device):
        self.model.to(device); self.alpha_bars = self.alpha_bars.to(device); return self

    def loss_details(self, batch, noise=None, timesteps=None):
        target, mask = batch["target"], batch["target_mask"]
        batch_size = target.shape[0]
        timesteps = timesteps if timesteps is not None else torch.randint(
            0, self.steps, (batch_size,), device=target.device
        )
        noise = noise if noise is not None else torch.randn_like(target)
        alpha = self.alpha_bars[timesteps][:, None]
        noisy = alpha.sqrt() * target + (1 - alpha).sqrt() * noise
        expected = noise if self.prediction_type == "epsilon" else (
            alpha.sqrt() * noise - (1 - alpha).sqrt() * target
        )
        predicted = self.model(noisy, timesteps, batch)
        squared = (predicted - expected).square() * mask
        per_record = squared.sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        snr = alpha.squeeze(1) / (1 - alpha.squeeze(1)).clamp_min(1e-12)
        if self.min_snr_gamma is None:
            weights = torch.ones_like(snr)
        else:
            weights = torch.minimum(snr, torch.full_like(snr, self.min_snr_gamma)) / snr.clamp_min(1e-12)
        predicted_clean, _ = self.model_output_to_x0_epsilon(noisy, predicted, alpha)
        clean_error = (predicted_clean - target) * mask
        details = {
            "loss": (weights * per_record).mean(),
            "unweighted_prediction_mse": per_record.mean(),
            "x0_mae_normalized": clean_error.abs().sum() / mask.sum().clamp_min(1),
            "x0_rmse_normalized": (clean_error.square().sum() / mask.sum().clamp_min(1)).sqrt(),
            "mean_snr": snr.mean(),
            "mean_min_snr_weight": weights.mean(),
        }
        details[f"unweighted_{'epsilon' if self.prediction_type == 'epsilon' else 'v'}_mse"] = per_record.mean()
        return details

    def training_loss(self, batch, noise=None, timesteps=None):
        return self.loss_details(batch, noise=noise, timesteps=timesteps)["loss"]

    @staticmethod
    def unconditional_batch(batch):
        output = dict(batch)
        for key in ("current", "history", "history_mask", "guide_values", "guide_deltas",
                    "guide_mask", "guide_similarities", "retrieval_mask"):
            if key in output:
                output[key] = torch.zeros_like(output[key])
        reference = output["current"]
        output["condition_present"] = torch.zeros((reference.shape[0], 1), device=reference.device, dtype=reference.dtype)
        return output

    def predict_model_output(self, values, timesteps, batch, cfg_scale=1.0):
        conditional = self.model(values, timesteps, batch)
        if float(cfg_scale) == 1.0 or not getattr(self.model, "condition_indicator", False):
            return conditional
        unconditional = self.model(values, timesteps, self.unconditional_batch(batch))
        return unconditional + float(cfg_scale) * (conditional - unconditional)

    def predict_epsilon(self, values, timesteps, batch, cfg_scale=1.0):
        output = self.predict_model_output(values, timesteps, batch, cfg_scale=cfg_scale)
        alpha = self.alpha_bars[timesteps][:, None]
        return self.model_output_to_x0_epsilon(values, output, alpha)[1]

    def model_output_to_x0_epsilon(self, values, output, alpha):
        if self.prediction_type == "epsilon":
            epsilon = output
            predicted_clean = (values - (1 - alpha).sqrt() * epsilon) / alpha.sqrt()
        else:
            predicted_clean = alpha.sqrt() * values - (1 - alpha).sqrt() * output
            epsilon = (1 - alpha).sqrt() * values + alpha.sqrt() * output
        return predicted_clean, epsilon

    @torch.inference_mode()
    def ddim(self, batch, *, shape, sampling_steps=50, eta=0.0,
             normalized_clip=None, generator=None, initial_noise_scale=1.0, cfg_scale=1.0):
        device = batch["current"].device
        values = float(initial_noise_scale) * torch.randn(shape, device=device, generator=generator)
        sequence = np.linspace(self.steps - 1, 0, int(sampling_steps), dtype=int)
        for position, timestep in enumerate(sequence):
            next_timestep = sequence[position + 1] if position + 1 < len(sequence) else -1
            t = torch.full((shape[0],), int(timestep), device=device, dtype=torch.long)
            alpha = self.alpha_bars[int(timestep)]
            next_alpha = self.alpha_bars[int(next_timestep)] if next_timestep >= 0 else torch.tensor(1.0, device=device)
            output = self.predict_model_output(values, t, batch, cfg_scale=cfg_scale)
            predicted_clean, epsilon = self.model_output_to_x0_epsilon(values, output, alpha)
            if normalized_clip is not None:
                predicted_clean = predicted_clean.clamp(float(normalized_clip[0]), float(normalized_clip[1]))
            sigma = float(eta) * torch.sqrt((1 - next_alpha) / (1 - alpha) * (1 - alpha / next_alpha))
            direction = torch.sqrt(torch.clamp(1 - next_alpha - sigma.square(), min=0)) * epsilon
            random = torch.randn(values.shape, device=device, generator=generator) if next_timestep >= 0 else 0
            values = next_alpha.sqrt() * predicted_clean + direction + sigma * random
        return values
