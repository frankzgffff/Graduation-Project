from __future__ import annotations

import torch
import torch.nn as nn


class ReLUConvBN(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: int, stride: int, padding: int) -> None:
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(c_in, c_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(c_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class DilConv(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                c_in,
                c_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=c_in,
                bias=False,
            ),
            nn.Conv2d(c_in, c_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(c_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class SepConv(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: int, stride: int, padding: int) -> None:
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                c_in,
                c_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=c_in,
                bias=False,
            ),
            nn.Conv2d(c_in, c_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(c_in),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                c_in,
                c_in,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                groups=c_in,
                bias=False,
            ),
            nn.Conv2d(c_in, c_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(c_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class Zero(nn.Module):
    def __init__(self, stride: int) -> None:
        super().__init__()
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            return x.mul(0.0)
        return x[:, :, :: self.stride, :: self.stride].mul(0.0)


class FactorizedReduce(nn.Module):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        if c_out % 2 != 0:
            raise ValueError("c_out must be divisible by 2 for FactorizedReduce.")
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(c_in, c_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(c_in, c_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        return self.bn(out)


def _pooling(pool_cls):
    def builder(c_in: int, stride: int, affine: bool = True) -> nn.Module:
        del affine
        return nn.Sequential(pool_cls(3, stride=stride, padding=1), nn.BatchNorm2d(c_in))

    return builder


OPS = {
    "none": lambda c, stride, affine=True: Zero(stride),
    "avg_pool_3x3": _pooling(nn.AvgPool2d),
    "max_pool_3x3": _pooling(nn.MaxPool2d),
    "skip_connect": lambda c, stride, affine=True: Identity() if stride == 1 else FactorizedReduce(c, c),
    "sep_conv_3x3": lambda c, stride, affine=True: SepConv(c, c, 3, stride, 1),
    "sep_conv_5x5": lambda c, stride, affine=True: SepConv(c, c, 5, stride, 2),
    "dil_conv_3x3": lambda c, stride, affine=True: DilConv(c, c, 3, stride, 2, 2),
    "dil_conv_5x5": lambda c, stride, affine=True: DilConv(c, c, 5, stride, 4, 2),
}

