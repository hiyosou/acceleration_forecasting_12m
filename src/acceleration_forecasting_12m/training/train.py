from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from acceleration_forecasting_12m.common.io import write_json
from acceleration_forecasting_12m.common.progress import progress_bar, progress_message
from acceleration_forecasting_12m.datasets.torch_dataset import ForecastDataset
from acceleration_forecasting_12m.diffusion.process import DiffusionProcess
from acceleration_forecasting_12m.models.unet import ResidualUNet12
from acceleration_forecasting_12m.models.absolute_attention_unet import AbsoluteAttentionUNet12
from .ema import EMA


def _device(value):
    return torch.device(value or ("cuda" if torch.cuda.is_available() else "cpu"))


def _move(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _learning_rate_factor(epoch, epochs, warmup=5, minimum_ratio=0.01):
    if epoch < warmup:
        return (epoch + 1) / max(warmup, 1)
    progress = (epoch - warmup) / max(epochs - warmup - 1, 1)
    return minimum_ratio + (1 - minimum_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


@torch.inference_mode()
def _validation_loss(process, loader, device, seed=42):
    process.model.eval(); total, count = 0.0, 0
    generator = torch.Generator(device=device).manual_seed(seed)
    for batch in loader:
        batch = _move(batch, device)
        timesteps = torch.randint(0, process.steps, (batch["target"].shape[0],), device=device, generator=generator)
        noise = torch.randn(batch["target"].shape, device=device, generator=generator)
        loss = process.training_loss(batch, noise=noise, timesteps=timesteps)
        total += float(loss) * batch["target"].shape[0]; count += batch["target"].shape[0]
    return total / max(count, 1)


def train(dataset_dir, output_dir, *, device=None, epochs=200, batch_size=128,
          accumulation_steps=2, learning_rate=1e-4, weight_decay=1e-4,
          dropout=0.1, patience=20, min_delta=1e-4, ema_decay=0.999,
          seed=42, resume=True, progress=True):
    torch.manual_seed(seed)
    dataset_dir, output_dir = Path(dataset_dir).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configuration = json.loads((dataset_dir / "guide_search_config.json").read_text(encoding="utf-8"))
    build_id = configuration["dataset_build_id"]
    train_data = ForecastDataset(dataset_dir / "model_train", dataset_dir)
    valid_data = ForecastDataset(dataset_dir / "model_validation", dataset_dir)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=0)
    device = _device(device)
    target_mode = configuration.get("target_mode", "residual")
    absolute = target_mode == "absolute"
    model_config = {"dropout": float(dropout), "forecast_months": 12,
                    "cross_attention": bool(absolute), "target_mode": target_mode}
    model = (AbsoluteAttentionUNet12(dropout) if absolute else ResidualUNet12(dropout)).to(device)
    process = DiffusionProcess(model, 1000).to(device)
    ema = EMA(model, ema_decay); optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: _learning_rate_factor(epoch, epochs)
    )
    scaler_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    autocast_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    start_epoch, best, stale, history = 0, float("inf"), 0, []
    last_path = output_dir / "last_model.pt"
    if resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint.get("dataset_build_id") != build_id or checkpoint.get("model_config") != model_config:
            raise ValueError("Checkpoint dataset/model configuration does not match")
        model.load_state_dict(checkpoint["model_state_dict"]); ema.load_state_dict(checkpoint["ema_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"]); scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1; best = float(checkpoint["best_validation_loss"])
        stale = int(checkpoint.get("stale_epochs", 0))
        if (output_dir / "training_history.csv").is_file():
            history = pd.read_csv(output_dir / "training_history.csv").to_dict("records")
        progress_message(f"checkpoint再開: epoch={start_epoch}, best={best:.6f}", enabled=progress)
    outer = progress_bar(range(start_epoch, epochs), enabled=progress, desc="学習epoch", unit="epoch")
    started = time.perf_counter()
    for epoch in outer:
        model.train(); optimizer.zero_grad(set_to_none=True); running, batches = 0.0, 0
        inner = progress_bar(train_loader, enabled=progress, desc=f"epoch {epoch + 1}", unit="batch", leave=False)
        for batch_index, batch in enumerate(inner):
            batch = _move(batch, device)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=scaler_enabled):
                loss = process.training_loss(batch) / accumulation_steps
            scaler.scale(loss).backward()
            if (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(train_loader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); ema.update(model)
            running += float(loss.detach()) * accumulation_steps; batches += 1
        validation_process = DiffusionProcess(ema.model, 1000).to(device)
        validation = _validation_loss(validation_process, valid_loader, device, seed)
        train_loss = running / max(batches, 1)
        improved = validation < best - min_delta
        if improved: best, stale = validation, 0
        else: stale += 1
        row = {
            "epoch": epoch + 1, "train_loss": train_loss, "validation_loss": validation,
            "best_validation_loss": best, "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(row); pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False, encoding="utf-8-sig")
        checkpoint = {
            "epoch": epoch, "model_state_dict": model.state_dict(), "ema_state_dict": ema.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
            "best_validation_loss": best, "stale_epochs": stale, "dataset_build_id": build_id,
            "model_config": model_config, "seed": seed,
        }
        torch.save(checkpoint, last_path)
        if improved: torch.save(checkpoint, output_dir / "best_model.pt")
        scheduler.step()
        outer.set_postfix(train=f"{train_loss:.5f}", valid=f"{validation:.5f}", best=f"{best:.5f}")
        if stale >= patience: break
    resolved = {
        "dataset_dir": str(dataset_dir), "dataset_build_id": build_id, "device": str(device),
        "epochs_completed": len(history), "best_validation_loss": best, "model_config": model_config,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    write_json(output_dir / "resolved_config.json", resolved)
    return resolved
