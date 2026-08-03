import torch
from torch import nn

from acceleration_forecasting_12m.diffusion.process import DiffusionProcess
from acceleration_forecasting_12m.models.unet import ResidualUNet12
from acceleration_forecasting_12m.common.constants import PHYSICAL_MAX, PHYSICAL_MIN
from acceleration_forecasting_12m.models.absolute_attention_unet import AbsoluteAttentionUNet12, GuideEncoder12
from acceleration_forecasting_12m.training.train import _drop_conditions
from acceleration_forecasting_12m.inference.select_variance import calibrate_samples
from acceleration_forecasting_12m.models.reference_modulated_unet import (
    ReferenceModulatedAttention, ReferenceModulatedUNet12,
)


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


def test_min_snr_epsilon_weights_match_definition():
    class DummyModel(nn.Module):
        def forward(self, noisy, timesteps, batch):
            return torch.zeros_like(noisy)
    model = DummyModel()
    process = DiffusionProcess(model, steps=10, min_snr_gamma=5.0)
    target = torch.zeros(2, 12); mask = torch.ones_like(target); noise = torch.ones_like(target)
    batch = {"target": target, "target_mask": mask}
    timesteps = torch.tensor([0, 9])
    details = process.loss_details(batch, noise=noise, timesteps=timesteps)
    alpha = process.alpha_bars[timesteps]
    snr = alpha / (1 - alpha)
    expected = torch.minimum(snr, torch.full_like(snr, 5.0)) / snr
    assert torch.allclose(details["mean_min_snr_weight"], expected.mean())


def test_cfg_scale_one_matches_conditional_and_unconditional_is_finite():
    model = AbsoluteAttentionUNet12(dropout=0.0, condition_indicator=True).eval()
    process = DiffusionProcess(model, steps=10)
    batch = attention_batch(2)
    batch["condition_present"] = torch.ones(2, 1)
    values, timesteps = torch.randn(2, 12), torch.tensor([2, 5])
    conditional = model(values, timesteps, batch)
    assert torch.allclose(process.predict_epsilon(values, timesteps, batch, cfg_scale=1.0), conditional)
    assert torch.isfinite(process.predict_epsilon(values, timesteps, batch, cfg_scale=1.5)).all()


def test_condition_dropout_one_removes_all_conditions_and_marks_absent():
    data = attention_batch(2)
    dropped = _drop_conditions(data, 1.0)
    for key in ("current", "history", "history_mask", "guide_values", "guide_deltas",
                "guide_mask", "guide_similarities", "retrieval_mask"):
        assert torch.count_nonzero(dropped[key]) == 0
    assert torch.count_nonzero(dropped["condition_present"]) == 0


def test_variance_calibration_identity_and_monotonic_width():
    samples = torch.linspace(0.5, 2.5, 100).numpy()[:, None].repeat(12, axis=1)
    identity = calibrate_samples(samples, 1.0)
    narrow = calibrate_samples(samples, 0.1)
    assert torch.allclose(torch.from_numpy(identity), torch.from_numpy(samples))
    assert float(narrow.max() - narrow.min()) < float(identity.max() - identity.min())


def test_reference_modulated_model_has_ten_injections_and_finite_output():
    data = attention_batch(2)
    model = ReferenceModulatedUNet12(dropout=0.0)
    output = model(torch.randn(2, 12), torch.tensor([2, 5]), data)
    assert output.shape == (2, 12) and torch.isfinite(output).all()
    assert len(model.reference_blocks) == 10
    assert sum(isinstance(module, ReferenceModulatedAttention) for module in model.modules()) == 10
    assert not any(module.__class__.__name__ == "MaskedCrossAttention" for module in model.modules())


def test_reference_modulation_masks_invalid_tokens_and_changes_with_reference():
    data = attention_batch(2)
    model = ReferenceModulatedUNet12(dropout=0.0).eval()
    noisy, timesteps = torch.randn(2, 12), torch.tensor([2, 5])
    normal = model(noisy, timesteps, data)
    changed_data = dict(data); changed_data["guide_values"] = data["guide_values"] + 2.0
    changed = model(noisy, timesteps, changed_data)
    assert not torch.allclose(normal, changed)
    disabled = dict(data)
    disabled["guide_mask"] = torch.zeros_like(data["guide_mask"])
    disabled["retrieval_mask"] = torch.zeros_like(data["retrieval_mask"])
    output = model(noisy, timesteps, disabled)
    assert torch.isfinite(output).all()
    for block in model.reference_blocks:
        assert torch.count_nonzero(block.reference_attention.last_attention) == 0


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


def test_v_target_inverse_recovers_clean_and_noise():
    class Zero(nn.Module):
        def forward(self, noisy, timesteps, batch):
            return torch.zeros_like(noisy)

    process = DiffusionProcess(Zero(), steps=20, prediction_type="v_prediction")
    target, noise = torch.randn(2, 12), torch.randn(2, 12)
    timesteps = torch.tensor([3, 15])
    alpha = process.alpha_bars[timesteps][:, None]
    noisy = alpha.sqrt() * target + (1 - alpha).sqrt() * noise
    velocity = alpha.sqrt() * noise - (1 - alpha).sqrt() * target
    clean, recovered_noise = process.model_output_to_x0_epsilon(noisy, velocity, alpha)
    assert torch.allclose(clean, target, atol=1e-5)
    assert torch.allclose(recovered_noise, noise, atol=1e-5)


def test_masked_plain_v_mse_has_no_snr_weighting():
    class Zero(nn.Module):
        def forward(self, noisy, timesteps, batch):
            return torch.zeros_like(noisy)

    process = DiffusionProcess(Zero(), steps=20, prediction_type="v_prediction")
    data = batch(); data["target_mask"][:, :4] = 0
    details = process.loss_details(
        data, noise=torch.ones_like(data["target"]), timesteps=torch.tensor([2, 10])
    )
    assert torch.isfinite(details["unweighted_v_mse"])
    assert torch.allclose(details["loss"], details["unweighted_v_mse"])
    import pytest
    with pytest.raises(ValueError, match="Min-SNR"):
        DiffusionProcess(Zero(), steps=20, prediction_type="v_prediction", min_snr_gamma=5.0)


def test_v_ddim_is_reproducible_and_finite():
    model = ResidualUNet12(dropout=0).eval()
    process = DiffusionProcess(model, steps=20, prediction_type="v_prediction")
    data = batch(1); condition = {key: data[key] for key in ("current", "history", "history_mask")}
    first = process.ddim(condition, shape=(1, 12), sampling_steps=5, normalized_clip=(-2, 2),
                         generator=torch.Generator().manual_seed(42))
    second = process.ddim(condition, shape=(1, 12), sampling_steps=5, normalized_clip=(-2, 2),
                          generator=torch.Generator().manual_seed(42))
    assert torch.allclose(first, second)
    assert torch.isfinite(first).all()
