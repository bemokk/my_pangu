import torch

from wind_wave.model import ConvLSTMWindWaveModel


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
