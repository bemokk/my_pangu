from __future__ import annotations

import torch


def rmse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(target)
    if not bool(mask.any()):
        raise ValueError("Cannot compute RMSE without finite target values")
    diff = prediction[mask] - target[mask]
    return torch.sqrt(torch.mean(diff * diff))


def circular_mae_degrees(
    pred_sin: torch.Tensor,
    pred_cos: torch.Tensor,
    target_sin: torch.Tensor,
    target_cos: torch.Tensor,
) -> torch.Tensor:
    mask = (
        torch.isfinite(pred_sin)
        & torch.isfinite(pred_cos)
        & torch.isfinite(target_sin)
        & torch.isfinite(target_cos)
    )
    if not bool(mask.any()):
        raise ValueError("Cannot compute circular MAE without finite values")

    pred_angle = torch.atan2(pred_sin[mask], pred_cos[mask])
    target_angle = torch.atan2(target_sin[mask], target_cos[mask])
    diff = torch.atan2(torch.sin(pred_angle - target_angle), torch.cos(pred_angle - target_angle))
    return torch.rad2deg(torch.mean(torch.abs(diff)))
