from __future__ import annotations

import math

import torch
import torch.nn as nn

from nas_project.models.cell import NASCell
from nas_project.search.search_space import Architecture


class NASNetwork(nn.Module):
    def __init__(
        self,
        architecture: Architecture,
        init_channels: int,
        num_classes: int,
        stem_multiplier: int,
        dropout: float,
        op_names: list[str],
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.drop_rate = dropout

        base_channels = max(4, int(math.ceil(init_channels * architecture.width_multiplier)))
        if base_channels % 2 != 0:
            base_channels += 1
        stem_channels = stem_multiplier * base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
        )

        c_prev_prev = stem_channels
        c_prev = stem_channels
        c_curr = base_channels
        self.cells = nn.ModuleList()
        reduction_prev = False
        reduction_layers = {architecture.depth // 3, 2 * architecture.depth // 3}

        for layer_idx in range(architecture.depth):
            reduction = layer_idx in reduction_layers
            blocks = architecture.reduction_blocks if reduction else architecture.normal_blocks
            if reduction:
                c_curr *= 2

            cell = NASCell(
                blocks=blocks,
                op_names=op_names,
                c_prev_prev=c_prev_prev,
                c_prev=c_prev,
                c=c_curr,
                reduction=reduction,
                reduction_prev=reduction_prev,
            )
            self.cells.append(cell)
            reduction_prev = reduction
            c_prev_prev, c_prev = c_prev, len(blocks) * c_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(c_prev, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = s1 = self.stem(x)
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1)
        out = self.global_pooling(s1).view(s1.size(0), -1)
        out = self.dropout(out)
        return self.classifier(out)
