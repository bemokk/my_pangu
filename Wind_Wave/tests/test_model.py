import torch

from wind_wave.model import ConvLSTMWindWaveModel, WindWaveV2Model


def test_convlstm_model_outputs_multi_lead_wave_tensor():
    model = ConvLSTMWindWaveModel(
        input_channels=2,
        hidden_channels=4,
        lead_count=5,
        target_channels=4,
    )
    x = torch.randn(2, 24, 2, 8, 10)

    y = model(x, output_size=(3, 4))

    assert y.shape == (2, 5, 4, 3, 4)


def test_v2_model_outputs_direct_wave_tensor():
    model = WindWaveV2Model(
        hidden_channels=4,
        lead_count=5,
        target_channels=4,
        use_wave0=True,
        residual=False,
    )
    past_wind = torch.randn(2, 24, 2, 8, 10)
    future_wind = torch.randn(2, 5, 2, 8, 10)
    wave0 = torch.randn(2, 4, 3, 4)

    y = model(past_wind, future_wind=future_wind, wave0=wave0, output_size=(3, 4))

    assert y.shape == (2, 5, 4, 3, 4)


def test_v2_model_uses_continuous_future_wind_offsets():
    model = WindWaveV2Model(
        hidden_channels=4,
        lead_count=3,
        target_channels=4,
        use_wave0=False,
        residual=False,
        future_wind_mode="continuous72",
    )
    past_wind = torch.randn(2, 24, 2, 8, 10)
    future_wind = torch.randn(2, 72, 2, 8, 10)
    offsets = torch.tensor([5, 11, 71])

    y = model(
        past_wind,
        future_wind=future_wind,
        future_wind_offsets=offsets,
        output_size=(3, 4),
    )

    assert y.shape == (2, 3, 4, 3, 4)


def test_v2_residual_model_adds_swh_mwp_delta_and_identity_rotates_wave0_direction():
    model = WindWaveV2Model(
        hidden_channels=4,
        lead_count=5,
        target_channels=4,
        use_wave0=True,
        residual=True,
    )
    for head in model.heads:
        for module in head.modules():
            if hasattr(module, "weight") and module.weight is not None:
                torch.nn.init.zeros_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    past_wind = torch.randn(2, 24, 2, 8, 10)
    future_wind = torch.randn(2, 5, 2, 8, 10)
    wave0 = torch.randn(2, 4, 3, 4)

    y = model(past_wind, future_wind=future_wind, wave0=wave0, output_size=(3, 4))

    assert torch.allclose(y[:, :, :2], wave0[:, None, :2].expand_as(y[:, :, :2]))
    direction_norm = torch.sqrt((wave0[:, 2].square() + wave0[:, 3].square()).clamp_min(1e-12))
    expected_cos = wave0[:, 2] / direction_norm
    expected_sin = wave0[:, 3] / direction_norm
    assert torch.allclose(y[:, :, 2], expected_cos[:, None], atol=1e-5)
    assert torch.allclose(y[:, :, 3], expected_sin[:, None], atol=1e-5)


def test_v2_residual_model_rotates_wave_direction_residual_on_unit_circle():
    model = WindWaveV2Model(
        hidden_channels=4,
        lead_count=1,
        target_channels=4,
        use_wave0=True,
        residual=True,
    )
    for head in model.heads:
        for module in head.modules():
            if hasattr(module, "weight") and module.weight is not None:
                torch.nn.init.zeros_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        head[-1].bias.data.copy_(torch.tensor([0.5, -0.25, 0.0, 1.0]))

    past_wind = torch.zeros(1, 24, 2, 4, 4)
    future_wind = torch.zeros(1, 1, 2, 4, 4)
    wave0 = torch.zeros(1, 4, 2, 3)
    wave0[:, 0] = 2.0
    wave0[:, 1] = 3.0
    wave0[:, 2] = 1.0
    wave0[:, 3] = 0.0

    y = model(past_wind, future_wind=future_wind, wave0=wave0, output_size=(2, 3))

    assert torch.allclose(y[:, :, 0], torch.full_like(y[:, :, 0], 2.5))
    assert torch.allclose(y[:, :, 1], torch.full_like(y[:, :, 1], 2.75))
    direction_norm = torch.sqrt(y[:, :, 2].square() + y[:, :, 3].square())
    assert torch.allclose(direction_norm, torch.ones_like(direction_norm), atol=1e-5)
    assert torch.allclose(y[:, :, 2], torch.zeros_like(y[:, :, 2]), atol=1e-5)
    assert torch.allclose(y[:, :, 3], torch.ones_like(y[:, :, 3]), atol=1e-5)


def test_v2_residual_model_keeps_unit_direction_for_tiny_rotation_vector():
    model = WindWaveV2Model(
        hidden_channels=4,
        lead_count=1,
        target_channels=4,
        use_wave0=True,
        residual=True,
    )
    for head in model.heads:
        for module in head.modules():
            if hasattr(module, "weight") and module.weight is not None:
                torch.nn.init.zeros_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        head[-1].bias.data.copy_(torch.tensor([0.0, 0.0, 1e-5, 0.0]))

    past_wind = torch.zeros(1, 24, 2, 4, 4)
    future_wind = torch.zeros(1, 1, 2, 4, 4)
    wave0 = torch.zeros(1, 4, 2, 3)
    wave0[:, 2] = 1.0

    y = model(past_wind, future_wind=future_wind, wave0=wave0, output_size=(2, 3))

    direction_norm = torch.sqrt(y[:, :, 2].square() + y[:, :, 3].square())
    assert torch.allclose(direction_norm, torch.ones_like(direction_norm), atol=1e-5)
    assert torch.allclose(y[:, :, 2], torch.ones_like(y[:, :, 2]), atol=1e-5)
    assert torch.allclose(y[:, :, 3], torch.zeros_like(y[:, :, 3]), atol=1e-5)


def test_v2_direction_normalization_has_finite_gradients_for_zero_vector():
    raw_cos = torch.zeros(2, 3, requires_grad=True)
    raw_sin = torch.zeros(2, 3, requires_grad=True)

    cos_value, sin_value = WindWaveV2Model._normalize_direction(
        raw_cos,
        raw_sin,
        default_cos=1.0,
        default_sin=0.0,
    )
    loss = (cos_value + sin_value).sum()
    loss.backward()

    assert torch.isfinite(cos_value).all()
    assert torch.isfinite(sin_value).all()
    assert torch.isfinite(raw_cos.grad).all()
    assert torch.isfinite(raw_sin.grad).all()


def test_v2_residual_model_rotates_direction_in_physical_target_space():
    target_mean = torch.tensor([10.0, 20.0, 0.2, -0.3])
    target_std = torch.tensor([2.0, 4.0, 0.5, 0.25])
    model = WindWaveV2Model(
        hidden_channels=4,
        lead_count=1,
        target_channels=4,
        use_wave0=True,
        residual=True,
        target_mean=target_mean,
        target_std=target_std,
    )
    for head in model.heads:
        for module in head.modules():
            if hasattr(module, "weight") and module.weight is not None:
                torch.nn.init.zeros_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        head[-1].bias.data.copy_(torch.tensor([0.5, -0.25, 0.0, 1.0]))

    past_wind = torch.zeros(1, 24, 2, 4, 4)
    future_wind = torch.zeros(1, 1, 2, 4, 4)
    wave0_raw = torch.zeros(1, 4, 2, 3)
    wave0_raw[:, 0] = 12.0
    wave0_raw[:, 1] = 28.0
    wave0_raw[:, 2] = 1.0
    wave0_raw[:, 3] = 0.0
    wave0 = (wave0_raw - target_mean.view(1, 4, 1, 1)) / target_std.view(1, 4, 1, 1)

    y = model(past_wind, future_wind=future_wind, wave0=wave0, output_size=(2, 3))
    y_raw = y * target_std.view(1, 1, 4, 1, 1) + target_mean.view(1, 1, 4, 1, 1)

    assert torch.allclose(y_raw[:, :, 0], torch.full_like(y_raw[:, :, 0], 13.0))
    assert torch.allclose(y_raw[:, :, 1], torch.full_like(y_raw[:, :, 1], 27.0))
    direction_norm = torch.sqrt(y_raw[:, :, 2].square() + y_raw[:, :, 3].square())
    assert torch.allclose(direction_norm, torch.ones_like(direction_norm), atol=1e-5)
    assert torch.allclose(y_raw[:, :, 2], torch.zeros_like(y_raw[:, :, 2]), atol=1e-5)
    assert torch.allclose(y_raw[:, :, 3], torch.ones_like(y_raw[:, :, 3]), atol=1e-5)


def test_v2_residual_model_replaces_nan_wave0_before_residual_add():
    model = WindWaveV2Model(
        hidden_channels=4,
        lead_count=5,
        target_channels=4,
        use_wave0=True,
        residual=True,
    )
    for head in model.heads:
        for module in head.modules():
            if hasattr(module, "weight") and module.weight is not None:
                torch.nn.init.zeros_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    past_wind = torch.randn(2, 24, 2, 8, 10)
    future_wind = torch.randn(2, 5, 2, 8, 10)
    wave0 = torch.randn(2, 4, 3, 4)
    wave0[:, :, 0, 0] = float("nan")

    y = model(past_wind, future_wind=future_wind, wave0=wave0, output_size=(3, 4))

    assert torch.isfinite(y).all()
    assert torch.allclose(y[:, :, :2, 0, 0], torch.zeros_like(y[:, :, :2, 0, 0]))
    assert torch.allclose(y[:, :, 2, 0, 0], torch.ones_like(y[:, :, 2, 0, 0]))
    assert torch.allclose(y[:, :, 3, 0, 0], torch.zeros_like(y[:, :, 3, 0, 0]))
