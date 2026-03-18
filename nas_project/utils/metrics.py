from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import torch


@dataclass
class AverageMeter:
    name: str
    value: float = 0.0
    avg: float = 0.0
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.value = value
        self.total += value * n
        self.count += n
        self.avg = self.total / max(self.count, 1)


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, topk: Iterable[int] = (1,)) -> list[float]:
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    results: list[float] = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        results.append((correct_k * 100.0 / targets.size(0)).item())
    return results


def count_parameters_in_mb(model: torch.nn.Module) -> float:
    return sum(param.numel() for param in model.parameters() if param.requires_grad) / 1e6


def compute_fitness(
    val_acc: float,
    flops_m: float,
    params_m: float,
    flops_penalty: float,
    params_penalty: float,
) -> float:
    return val_acc - flops_penalty * flops_m - params_penalty * params_m


def compute_reward(
    parent_metrics: dict[str, float],
    child_metrics: dict[str, float],
    beta: float,
    cost_metric: str = "flops",
    cost_scale: float = 1.0,
) -> float:
    delta_accuracy = child_metrics["val_acc"] - parent_metrics["val_acc"]
    if cost_metric == "params":
        delta_cost = child_metrics["params_m"] - parent_metrics["params_m"]
    elif cost_metric == "combined":
        delta_cost = (child_metrics["flops_m"] - parent_metrics["flops_m"]) + (
            child_metrics["params_m"] - parent_metrics["params_m"]
        )
    else:
        delta_cost = child_metrics["flops_m"] - parent_metrics["flops_m"]
    return delta_accuracy - beta * cost_scale * delta_cost


def architecture_distance(lhs, rhs) -> float:
    total = 0
    different = 0
    for lhs_blocks, rhs_blocks in (
        (lhs.architecture.normal_blocks, rhs.architecture.normal_blocks),
        (lhs.architecture.reduction_blocks, rhs.architecture.reduction_blocks),
    ):
        for left_block, right_block in zip(lhs_blocks, rhs_blocks):
            for field_name in ("i1", "i2", "o1", "o2"):
                total += 1
                different += int(getattr(left_block, field_name) != getattr(right_block, field_name))
    total += 2
    different += int(lhs.architecture.depth != rhs.architecture.depth)
    different += int(lhs.architecture.width_multiplier != rhs.architecture.width_multiplier)
    return different / max(total, 1)


def population_diversity(individuals: list) -> float:
    if len(individuals) < 2:
        return 0.0
    distances = [architecture_distance(lhs, rhs) for lhs, rhs in combinations(individuals, 2)]
    return sum(distances) / len(distances)
