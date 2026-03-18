from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from nas_project.models.operations import FactorizedReduce, OPS, ReLUConvBN


@dataclass(frozen=True)
class BlockGene:
    i1: int
    i2: int
    o1: int
    o2: int


class NASCell(nn.Module):
    def __init__(
        self,
        blocks: Sequence[BlockGene],
        op_names: Sequence[str],
        c_prev_prev: int,
        c_prev: int,
        c: int,
        reduction: bool,
        reduction_prev: bool,
    ) -> None:
        super().__init__()
        self.blocks = list(blocks)
        self.op_names = list(op_names)
        self.multiplier = len(self.blocks)
        self.preprocess0 = (
            FactorizedReduce(c_prev_prev, c)
            if reduction_prev
            else ReLUConvBN(c_prev_prev, c, 1, 1, 0)
        )
        self.preprocess1 = ReLUConvBN(c_prev, c, 1, 1, 0)
        self.block_ops = nn.ModuleList()
        self.block_inputs: list[tuple[int, int]] = []

        for block in self.blocks:
            stride1 = 2 if reduction and block.i1 < 2 else 1
            stride2 = 2 if reduction and block.i2 < 2 else 1
            op1 = OPS[self.op_names[block.o1]](c, stride1, True)
            op2 = OPS[self.op_names[block.o2]](c, stride2, True)
            self.block_ops.append(nn.ModuleList([op1, op2]))
            self.block_inputs.append((block.i1, block.i2))

    def forward(self, s0: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]

        for (i1, i2), ops in zip(self.block_inputs, self.block_ops):
            h1 = ops[0](states[i1])
            h2 = ops[1](states[i2])
            states.append(h1 + h2)

        return torch.cat(states[-self.multiplier :], dim=1)
