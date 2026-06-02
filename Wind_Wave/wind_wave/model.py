from __future__ import annotations

import torch
from torch import nn


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("Expected input tensor shape [B, T, C, H, W]")

        states: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(self.cells)
        output = None
        for time_index in range(x.shape[1]):
            layer_input = x[:, time_index]
            for layer_index, cell in enumerate(self.cells):
                h, c = cell(layer_input, states[layer_index])
                states[layer_index] = (h, c)
                layer_input = h
            output = layer_input

        if output is None:
            raise ValueError("Input sequence has zero time steps")
        return output


class ConvLSTMWindWaveModel(nn.Module):
    def __init__(
        self,
        input_channels: int = 2,
        hidden_channels: int = 32,
        lead_count: int = 5,
        target_channels: int = 5,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        outputs = [head(encoded) for head in self.heads]
        return torch.stack(outputs, dim=1)
