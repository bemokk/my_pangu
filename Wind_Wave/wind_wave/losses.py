from __future__ import annotations

import torch


LOSS_MODES = ("mse", "scalar-circular")


def masked_mse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(target)
    if not bool(mask.any()):
        raise ValueError("Cannot compute masked MSE without finite target values")

    diff = prediction[mask] - target[mask]
    return torch.mean(diff * diff)


def masked_mse_sum_and_count(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    require_finite_prediction: bool = False,
) -> tuple[torch.Tensor, int]:
    mask = torch.isfinite(target)
    if require_finite_prediction:
        mask = mask & torch.isfinite(prediction)
    count = int(mask.sum().item())
    if count == 0:
        return prediction.new_zeros(()), 0
    diff = prediction[mask] - target[mask]
    return torch.sum(diff * diff), count


def circular_cosine_sum_and_count(
    pred_cos: torch.Tensor,
    pred_sin: torch.Tensor,
    target_cos: torch.Tensor,
    target_sin: torch.Tensor,
    *,
    require_finite_prediction: bool = False,
) -> tuple[torch.Tensor, int]:
    mask = torch.isfinite(target_cos) & torch.isfinite(target_sin)
    if require_finite_prediction:
        mask = mask & torch.isfinite(pred_cos) & torch.isfinite(pred_sin)
    count = int(mask.sum().item())
    if count == 0:
        return pred_cos.new_zeros(()), 0

    pred_cos_valid = pred_cos[mask]
    pred_sin_valid = pred_sin[mask]
    target_cos_valid = target_cos[mask]
    target_sin_valid = target_sin[mask]
    eps = 1e-6
    pred_norm = torch.sqrt(
        (pred_cos_valid.square() + pred_sin_valid.square()).clamp_min(eps * eps)
    )
    target_norm = torch.sqrt(
        (target_cos_valid.square() + target_sin_valid.square()).clamp_min(eps * eps)
    )
    dot = (
        pred_cos_valid / pred_norm * target_cos_valid / target_norm
        + pred_sin_valid / pred_norm * target_sin_valid / target_norm
    )
    return torch.sum(1.0 - dot.clamp(-1.0, 1.0)), count


def wind_wave_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    mode: str = "mse",
    target_mean: torch.Tensor | list[float] | tuple[float, ...] | None = None,
    target_std: torch.Tensor | list[float] | tuple[float, ...] | None = None,
    direction_weight: float = 1.0,
) -> torch.Tensor:
    if mode not in LOSS_MODES:
        raise ValueError(f"loss mode must be one of {', '.join(LOSS_MODES)}")
    if mode == "mse":
        return masked_mse_loss(prediction, target)
    if prediction.shape[2] < 4 or target.shape[2] < 4:
        raise ValueError("scalar-circular loss requires SWH, MWP, cos(MWD), and sin(MWD)")
    if target_mean is None or target_std is None:
        raise ValueError("scalar-circular loss requires target normalization statistics")
    if direction_weight < 0:
        raise ValueError("direction_weight must be non-negative")

    scalar_loss = masked_mse_loss(prediction[:, :, :2], target[:, :, :2])
    mean = torch.as_tensor(target_mean, dtype=prediction.dtype, device=prediction.device)
    std = torch.as_tensor(target_std, dtype=prediction.dtype, device=prediction.device)
    mean = mean.view(1, 1, -1, 1, 1)
    std = std.view(1, 1, -1, 1, 1)
    prediction_raw = prediction * std + mean
    target_raw = target * std + mean
    direction_sum, direction_count = circular_cosine_sum_and_count(
        prediction_raw[:, :, 2],
        prediction_raw[:, :, 3],
        target_raw[:, :, 2],
        target_raw[:, :, 3],
    )
    if direction_count == 0:
        raise ValueError("Cannot compute circular loss without finite target directions")
    return scalar_loss + float(direction_weight) * direction_sum / direction_count
