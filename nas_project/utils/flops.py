from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


def estimate_flops(model: nn.Module, input_size: Sequence[int] = (1, 3, 32, 32), device: str = "cpu") -> float:
    hooks = []
    total_flops = {"value": 0.0}

    def conv_hook(module: nn.Conv2d, inputs, output) -> None:
        batch_size = output.shape[0]
        out_h, out_w = output.shape[2], output.shape[3]
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels / module.groups)
        total_flops["value"] += batch_size * out_h * out_w * module.out_channels * kernel_ops

    def linear_hook(module: nn.Linear, inputs, output) -> None:
        batch_size = inputs[0].shape[0]
        total_flops["value"] += batch_size * module.in_features * module.out_features

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()
    dummy = torch.randn(*input_size, device=device)
    with torch.no_grad():
        model(dummy)

    for hook in hooks:
        hook.remove()
    if was_training:
        model.train()
    return total_flops["value"] / 1e6

