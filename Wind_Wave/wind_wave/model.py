from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            batch, _, height, width = x.shape
            h = x.new_zeros(batch, self.hidden_channels, height, width)
            c = x.new_zeros(batch, self.hidden_channels, height, width)
        else:
            h, c = state

        gates = self.gates(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(gates, chunks=4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class ConvLSTMEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        layers: int = 1,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be >= 1")

        cells = []
        for layer_index in range(layers):
            in_channels = input_channels if layer_index == 0 else hidden_channels
            cells.append(ConvLSTMCell(in_channels, hidden_channels, kernel_size=kernel_size))
        self.cells = nn.ModuleList(cells)

    def forward(self, x: torch.Tensor, return_sequence: bool = False) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("Expected input tensor shape [B, T, C, H, W]")

        states: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(self.cells)
        output = None
        outputs = []
        for time_index in range(x.shape[1]):
            layer_input = x[:, time_index]
            for layer_index, cell in enumerate(self.cells):
                h, c = cell(layer_input, states[layer_index])
                states[layer_index] = (h, c)
                layer_input = h
            output = layer_input
            if return_sequence:
                outputs.append(layer_input)

        if output is None:
            raise ValueError("Input sequence has zero time steps")
        if return_sequence:
            return torch.stack(outputs, dim=1)
        return output


class ConvLSTMWindWaveModel(nn.Module):
    def __init__(
        self,
        input_channels: int = 2,
        hidden_channels: int = 32,
        lead_count: int = 5,
        target_channels: int = 4,
        layers: int = 1,
    ) -> None:
        super().__init__()
        self.lead_count = lead_count
        self.target_channels = target_channels
        self.encoder = ConvLSTMEncoder(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            layers=layers,
        )
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden_channels, target_channels, kernel_size=1),
                )
                for _ in range(lead_count)
            ]
        )

    def forward(self, x: torch.Tensor, output_size: tuple[int, int] | None = None) -> torch.Tensor:
        encoded = self.encoder(x)
        if output_size is not None and encoded.shape[-2:] != tuple(output_size):
            encoded = F.interpolate(encoded, size=output_size, mode="bilinear", align_corners=False)
        outputs = [head(encoded) for head in self.heads]
        return torch.stack(outputs, dim=1)


def _dropout_layer(dropout: float) -> nn.Module:
    return nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()


def _conv_block(input_channels: int, output_channels: int, dropout: float = 0.0) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        _dropout_layer(dropout),
        nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        _dropout_layer(dropout),
    )


class WindWaveV2Model(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 32,
        lead_count: int = 5,
        target_channels: int = 4,
        use_wave0: bool = True,
        residual: bool = False,
        layers: int = 1,
        dropout: float = 0.0,
        future_wind_mode: str = "target",
        target_mean: torch.Tensor | list[float] | tuple[float, ...] | None = None,
        target_std: torch.Tensor | list[float] | tuple[float, ...] | None = None,
    ) -> None:
        super().__init__()
        if future_wind_mode not in {"target", "continuous72"}:
            raise ValueError("future_wind_mode must be target or continuous72")
        if residual and target_channels < 4:
            raise ValueError("residual wave direction rotation requires at least 4 target channels")
        self.lead_count = lead_count
        self.target_channels = target_channels
        self.use_wave0 = use_wave0
        self.residual = residual
        self.future_wind_mode = future_wind_mode
        self.past_encoder = ConvLSTMEncoder(
            input_channels=2,
            hidden_channels=hidden_channels,
            layers=layers,
        )
        if future_wind_mode == "continuous72":
            self.future_encoder = ConvLSTMEncoder(
                input_channels=2,
                hidden_channels=hidden_channels,
                layers=layers,
            )
        else:
            self.future_encoder = _conv_block(2, hidden_channels, dropout=dropout)
        self.wave0_encoder = _conv_block(target_channels, hidden_channels, dropout=dropout) if use_wave0 else None
        fused_channels = hidden_channels * (3 if use_wave0 else 2)
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(fused_channels, hidden_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    _dropout_layer(dropout),
                    nn.Conv2d(hidden_channels, target_channels, kernel_size=1),
                )
                for _ in range(lead_count)
            ]
        )
        self.register_buffer(
            "target_mean",
            self._target_stat_tensor(target_mean, target_channels, default=0.0),
        )
        self.register_buffer(
            "target_std",
            self._target_stat_tensor(target_std, target_channels, default=1.0),
        )

    @staticmethod
    def _target_stat_tensor(
        value: torch.Tensor | list[float] | tuple[float, ...] | None,
        target_channels: int,
        default: float,
    ) -> torch.Tensor:
        if value is None:
            tensor = torch.full((target_channels,), float(default), dtype=torch.float32)
        else:
            tensor = torch.as_tensor(value, dtype=torch.float32).flatten()
        if tensor.numel() != target_channels:
            raise ValueError("target normalization stats must match target_channels")
        return tensor.view(1, target_channels, 1, 1)

    @staticmethod
    def _normalize_direction(
        cos_value: torch.Tensor,
        sin_value: torch.Tensor,
        *,
        default_cos: float,
        default_sin: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eps = 1e-6
        norm_sq = cos_value.square() + sin_value.square()
        norm = torch.sqrt(norm_sq.clamp_min(eps * eps))
        unit_cos = cos_value / norm
        unit_sin = sin_value / norm
        valid = norm_sq > eps * eps
        unit_cos = torch.where(valid, unit_cos, torch.full_like(unit_cos, default_cos))
        unit_sin = torch.where(valid, unit_sin, torch.full_like(unit_sin, default_sin))
        return unit_cos, unit_sin

    def _apply_wave0_residual(
        self,
        prediction: torch.Tensor,
        residual_base: torch.Tensor,
    ) -> torch.Tensor:
        residual_prediction = prediction + residual_base
        if self.target_channels < 4:
            return residual_prediction

        mean = self.target_mean.to(dtype=prediction.dtype, device=prediction.device)
        std = self.target_std.to(dtype=prediction.dtype, device=prediction.device)
        base_raw = residual_base * std + mean

        cos0, sin0 = self._normalize_direction(
            base_raw[:, 2:3],
            base_raw[:, 3:4],
            default_cos=1.0,
            default_sin=0.0,
        )
        raw_cos_dtheta = prediction[:, 2:3]
        raw_sin_dtheta = prediction[:, 3:4]
        cos_dtheta, sin_dtheta = self._normalize_direction(
            raw_cos_dtheta,
            raw_sin_dtheta,
            default_cos=1.0,
            default_sin=0.0,
        )

        cos_pred = cos0 * cos_dtheta - sin0 * sin_dtheta
        sin_pred = sin0 * cos_dtheta + cos0 * sin_dtheta
        residual_prediction = residual_prediction.clone()
        residual_prediction[:, 2:3] = (cos_pred - mean[:, 2:3]) / std[:, 2:3]
        residual_prediction[:, 3:4] = (sin_pred - mean[:, 3:4]) / std[:, 3:4]
        return residual_prediction

    def forward(
        self,
        past_wind: torch.Tensor,
        future_wind: torch.Tensor,
        wave0: torch.Tensor | None = None,
        future_wind_offsets: torch.Tensor | None = None,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if future_wind.ndim != 5:
            raise ValueError("Expected future_wind tensor shape [B, lead, C, H, W]")
        if self.future_wind_mode == "target" and future_wind.shape[1] != self.lead_count:
            raise ValueError("future_wind lead dimension does not match model lead_count")
        if self.use_wave0 and wave0 is None:
            raise ValueError("wave0 is required when use_wave0=True")

        past_wind = torch.nan_to_num(past_wind, nan=0.0, posinf=0.0, neginf=0.0)
        future_wind = torch.nan_to_num(future_wind, nan=0.0, posinf=0.0, neginf=0.0)
        wave0_clean = None
        if wave0 is not None:
            wave0_clean = torch.nan_to_num(wave0, nan=0.0, posinf=0.0, neginf=0.0)

        past_feature = self.past_encoder(past_wind)
        if output_size is not None and past_feature.shape[-2:] != tuple(output_size):
            past_feature = F.interpolate(
                past_feature,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )

        wave0_feature = None
        if self.use_wave0 and self.wave0_encoder is not None and wave0_clean is not None:
            wave0_feature = self.wave0_encoder(wave0_clean)
            if output_size is not None and wave0_feature.shape[-2:] != tuple(output_size):
                wave0_feature = F.interpolate(
                    wave0_feature,
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                )

        future_sequence_features = None
        future_offsets = None
        if self.future_wind_mode == "continuous72":
            if future_wind_offsets is None:
                raise ValueError("future_wind_offsets is required for continuous72 future wind mode")
            future_offsets = torch.as_tensor(future_wind_offsets, device=future_wind.device, dtype=torch.long)
            if future_offsets.ndim == 2:
                future_offsets = future_offsets[0]
            if future_offsets.numel() != self.lead_count:
                raise ValueError("future_wind_offsets must match lead_count")
            if int(future_offsets.max()) >= future_wind.shape[1]:
                raise ValueError("future_wind_offsets exceed future_wind sequence length")
            future_sequence_features = self.future_encoder(future_wind, return_sequence=True)

        outputs = []
        for lead_index, head in enumerate(self.heads):
            if self.future_wind_mode == "continuous72":
                future_feature = future_sequence_features[:, int(future_offsets[lead_index])]
            else:
                future_feature = self.future_encoder(future_wind[:, lead_index])
            if output_size is not None and future_feature.shape[-2:] != tuple(output_size):
                future_feature = F.interpolate(
                    future_feature,
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                )
            features = [past_feature, future_feature]
            if wave0_feature is not None:
                features.append(wave0_feature)
            prediction = head(torch.cat(features, dim=1))
            if self.residual:
                if wave0_clean is None:
                    raise ValueError("wave0 is required for residual prediction")
                residual_base = wave0_clean
                if output_size is not None and residual_base.shape[-2:] != tuple(output_size):
                    residual_base = F.interpolate(
                        residual_base,
                        size=output_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                prediction = self._apply_wave0_residual(prediction, residual_base)
            outputs.append(prediction)
        return torch.stack(outputs, dim=1)
