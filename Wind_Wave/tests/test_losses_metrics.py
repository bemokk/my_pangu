import math

import torch

from wind_wave.losses import masked_mse_loss
from wind_wave.metrics import circular_mae_degrees


def test_masked_mse_ignores_nan_targets():
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.0, float("nan"), 5.0])

    loss = masked_mse_loss(pred, target)

    assert torch.isclose(loss, torch.tensor(2.0))


def test_masked_mse_rejects_all_nan_targets():
    pred = torch.tensor([1.0])
    target = torch.tensor([float("nan")])

    try:
        masked_mse_loss(pred, target)
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_circular_mae_uses_shortest_angle():
    pred_sin = torch.tensor([math.sin(math.radians(359.0))])
    pred_cos = torch.tensor([math.cos(math.radians(359.0))])
    target_sin = torch.tensor([math.sin(math.radians(1.0))])
    target_cos = torch.tensor([math.cos(math.radians(1.0))])

    assert circular_mae_degrees(pred_sin, pred_cos, target_sin, target_cos).item() < 3.0
