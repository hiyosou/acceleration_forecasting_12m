import torch

from acceleration_forecasting_12m.diffusion.process import DiffusionProcess
from acceleration_forecasting_12m.models.unet import ResidualUNet12
from acceleration_forecasting_12m.common.constants import PHYSICAL_MAX, PHYSICAL_MIN
from acceleration_forecasting_12m.models.absolute_attention_unet import AbsoluteAttentionUNet12, GuideEncoder12


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


def test_final_physical_bounds_are_point_one_to_six():
    assert (PHYSICAL_MIN, PHYSICAL_MAX) == (0.1, 6.0)


def attention_batch(size=2):
    data = batch(size)
    data.update({
        "guide_values": torch.randn(size, 3, 12), "guide_deltas": torch.randn(size, 3, 12),
        "guide_mask": torch.ones(size, 3, 12), "guide_similarities": torch.ones(size, 3),
        "retrieval_mask": torch.ones(size, 3),
    })
    return data


def test_absolute_attention_unet_and_guide_encoder_shapes():
    data = attention_batch(); model = AbsoluteAttentionUNet12(dropout=0)
    guides = GuideEncoder12()(data["guide_values"], data["guide_deltas"], data["guide_similarities"])
    output = model(torch.randn(2, 12), torch.tensor([1, 2]), data)
    assert guides.shape == (2, 3, 12, 64)
    assert output.shape == (2, 12) and torch.isfinite(output).all()


def test_absolute_attention_all_invalid_guides_is_finite_and_backpropagates():
    data = attention_batch(); data["guide_mask"].zero_(); data["retrieval_mask"].zero_()
    model = AbsoluteAttentionUNet12(dropout=0); process = DiffusionProcess(model, steps=20)
    loss = process.training_loss(data); loss.backward()
    assert torch.isfinite(loss)
    assert model.guide_encoder.network[-1].weight.grad is not None


def test_initial_noise_scale_changes_deterministic_ddim_result():
    model = AbsoluteAttentionUNet12(dropout=0).eval(); process = DiffusionProcess(model, steps=20)
    data = attention_batch(1); condition = {k: v for k, v in data.items() if k not in ("target", "target_mask")}
    one = process.ddim(condition, shape=(1, 12), sampling_steps=5, normalized_clip=(-2, 2),
                       initial_noise_scale=1.0, generator=torch.Generator().manual_seed(42))
    half = process.ddim(condition, shape=(1, 12), sampling_steps=5, normalized_clip=(-2, 2),
                        initial_noise_scale=0.5, generator=torch.Generator().manual_seed(42))
    assert not torch.allclose(one, half)
    assert torch.isfinite(one).all() and torch.isfinite(half).all()


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
