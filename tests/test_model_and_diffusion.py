import torch

from acceleration_forecasting_12m.diffusion.process import DiffusionProcess
from acceleration_forecasting_12m.models.unet import ResidualUNet12


def batch(size=2):
    return {
        "current": torch.randn(size, 1), "history": torch.randn(size, 5),
        "history_mask": torch.tensor([[1, 1, 1, 0, 0]] * size, dtype=torch.float32),
        "target": torch.randn(size, 12), "target_mask": torch.ones(size, 12),
    }


def test_unet_is_twelve_month_attention_free_model():
    model = ResidualUNet12()
    data = batch()
    output = model(torch.randn(2, 12), torch.tensor([1, 2]), data)
    assert output.shape == (2, 12)
    assert torch.isfinite(output).all()
    assert not any("attention" in name.lower() or "guide" in name.lower() for name, _ in model.named_parameters())


def test_unet_can_be_moved_without_module_apply_collision():
    model = ResidualUNet12().to("cpu")
    assert next(model.parameters()).device.type == "cpu"


def test_masked_diffusion_loss_is_finite_and_backpropagates():
    model = ResidualUNet12(dropout=0)
    process = DiffusionProcess(model, steps=20)
    data = batch(); data["target_mask"][:, :2] = 0
    loss = process.training_loss(data)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_ddim_is_reproducible_and_clipped():
    model = ResidualUNet12(dropout=0).eval(); process = DiffusionProcess(model, steps=20)
    data = batch(1)
    condition = {key: data[key] for key in ("current", "history", "history_mask")}
    first = process.ddim(condition, shape=(1, 12), sampling_steps=5, normalized_clip=(-2, 2),
                         generator=torch.Generator().manual_seed(42))
    second = process.ddim(condition, shape=(1, 12), sampling_steps=5, normalized_clip=(-2, 2),
                          generator=torch.Generator().manual_seed(42))
    assert torch.allclose(first, second)
    assert torch.isfinite(first).all()
    assert first.min() >= -2 and first.max() <= 2
