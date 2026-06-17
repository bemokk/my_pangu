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


def test_v2_residual_model_adds_delta_to_wave0():
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

    expected = wave0[:, None].expand_as(y)
    assert torch.allclose(y, expected)


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
    assert torch.allclose(y[:, :, :, 0, 0], torch.zeros_like(y[:, :, :, 0, 0]))
