from __future__ import annotations

import torch


def masked_mse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(target)
    if not bool(mask.any()):
        raise ValueError("Cannot compute masked MSE without finite target values")

    diff = prediction[mask] - target[mask]
    return torch.mean(diff * diff)
