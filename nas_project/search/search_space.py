from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Iterable

import torch

from nas_project.models.cell import BlockGene


@dataclass
class Architecture:
    normal_blocks: list[BlockGene]
    reduction_blocks: list[BlockGene]
    depth: int
    width_multiplier: float

    def clone(self) -> "Architecture":
        return copy.deepcopy(self)

    @property
    def num_blocks(self) -> int:
        return len(self.normal_blocks)

    def adjacency_matrix(self, cell_type: str) -> list[list[int]]:
        blocks = self.normal_blocks if cell_type == "normal" else self.reduction_blocks
        total_nodes = len(blocks) + 2
        matrix = [[0 for _ in range(total_nodes)] for _ in range(total_nodes)]
        for block_idx, block in enumerate(blocks):
            target = block_idx + 2
            matrix[block.i1][target] = 1
            matrix[block.i2][target] = 1
        return matrix

    def operation_indices(self) -> dict[str, list[int]]:
        return {
            "normal": [value for block in self.normal_blocks for value in (block.o1, block.o2)],
            "reduction": [value for block in self.reduction_blocks for value in (block.o1, block.o2)],
        }

    def to_dict(self) -> dict:
        return {
            "normal_blocks": [block.__dict__ for block in self.normal_blocks],
            "reduction_blocks": [block.__dict__ for block in self.reduction_blocks],
            "adjacency_matrices": {
                "normal": self.adjacency_matrix("normal"),
                "reduction": self.adjacency_matrix("reduction"),
            },
            "operation_indices": self.operation_indices(),
            "depth": self.depth,
            "width_multiplier": self.width_multiplier,
        }


@dataclass(frozen=True)
class MutationAction:
    cell_type: str
    block_idx: int
    component: str
    value_idx: int


class SearchSpace:
    def __init__(
        self,
        num_blocks: int,
        op_names: Iterable[str],
        depth_choices: Iterable[int],
        width_choices: Iterable[float],
    ) -> None:
        self.num_blocks = num_blocks
        self.op_names = list(op_names)
        self.depth_choices = list(depth_choices)
        self.width_choices = list(width_choices)
        self.actions = self._build_actions()

    def _build_actions(self) -> list[MutationAction]:
        actions: list[MutationAction] = []
        for cell_type in ("normal", "reduction"):
            for block_idx in range(self.num_blocks):
                for component in ("i1", "i2"):
                    for input_idx in range(self.num_blocks + 1):
                        actions.append(MutationAction(cell_type, block_idx, component, input_idx))
                for component in ("o1", "o2"):
                    for op_idx in range(len(self.op_names)):
                        actions.append(MutationAction(cell_type, block_idx, component, op_idx))
        for depth_idx in range(len(self.depth_choices)):
            actions.append(MutationAction("global", -1, "depth", depth_idx))
        for width_idx in range(len(self.width_choices)):
            actions.append(MutationAction("global", -1, "width", width_idx))
        return actions

    @property
    def action_dim(self) -> int:
        return len(self.actions)

    @property
    def encoding_dim(self) -> int:
        reference = Architecture(
            normal_blocks=[
                BlockGene(i1=0, i2=min(1, block_idx + 1), o1=0, o2=0) for block_idx in range(self.num_blocks)
            ],
            reduction_blocks=[
                BlockGene(i1=0, i2=min(1, block_idx + 1), o1=0, o2=0) for block_idx in range(self.num_blocks)
            ],
            depth=self.depth_choices[0],
            width_multiplier=self.width_choices[0],
        )
        return int(self.encode(reference).numel())

    def valid_inputs_for_block(self, block_idx: int) -> list[int]:
        return list(range(block_idx + 2))

    def sample_block(self, block_idx: int) -> BlockGene:
        valid_inputs = self.valid_inputs_for_block(block_idx)
        return BlockGene(
            i1=random.choice(valid_inputs),
            i2=random.choice(valid_inputs),
            o1=random.randrange(len(self.op_names)),
            o2=random.randrange(len(self.op_names)),
        )

    def sample_architecture(self) -> Architecture:
        return Architecture(
            normal_blocks=[self.sample_block(block_idx) for block_idx in range(self.num_blocks)],
            reduction_blocks=[self.sample_block(block_idx) for block_idx in range(self.num_blocks)],
            depth=random.choice(self.depth_choices),
            width_multiplier=random.choice(self.width_choices),
        )

    def _encode_cell(self, blocks: list[BlockGene]) -> list[float]:
        total_nodes = self.num_blocks + 2
        features: list[float] = []
        matrix = [[0 for _ in range(total_nodes)] for _ in range(total_nodes)]
        for block_idx, block in enumerate(blocks):
            node_idx = block_idx + 2
            matrix[block.i1][node_idx] = 1
            matrix[block.i2][node_idx] = 1

        for row in matrix:
            features.extend(float(value) for value in row)

        for block_idx, block in enumerate(blocks):
            valid_inputs = self.valid_inputs_for_block(block_idx)
            for input_value in (block.i1, block.i2):
                input_one_hot = [0.0] * (self.num_blocks + 1)
                input_one_hot[input_value] = 1.0
                features.extend(input_one_hot)
            for op_idx in (block.o1, block.o2):
                op_one_hot = [0.0] * len(self.op_names)
                op_one_hot[op_idx] = 1.0
                features.extend(op_one_hot)
            features.append(block_idx / max(self.num_blocks - 1, 1))
            features.append(sum(1 for idx in (block.i1, block.i2) if idx in valid_inputs) / 2.0)
        return features

    def encode(self, architecture: Architecture) -> torch.Tensor:
        features = self._encode_cell(architecture.normal_blocks)
        features.extend(self._encode_cell(architecture.reduction_blocks))

        depth_one_hot = [0.0] * len(self.depth_choices)
        depth_one_hot[self.depth_choices.index(architecture.depth)] = 1.0
        width_one_hot = [0.0] * len(self.width_choices)
        width_one_hot[self.width_choices.index(architecture.width_multiplier)] = 1.0

        features.extend(depth_one_hot)
        features.extend(width_one_hot)
        features.append(architecture.depth / max(self.depth_choices))
        features.append(architecture.width_multiplier / max(self.width_choices))
        return torch.tensor(features, dtype=torch.float32)

    def get_action_mask(self, architecture: Architecture) -> torch.Tensor:
        mask = torch.zeros(self.action_dim, dtype=torch.bool)
        for action_idx, action in enumerate(self.actions):
            if action.component == "depth":
                mask[action_idx] = architecture.depth != self.depth_choices[action.value_idx]
                continue
            if action.component == "width":
                mask[action_idx] = architecture.width_multiplier != self.width_choices[action.value_idx]
                continue

            blocks = architecture.normal_blocks if action.cell_type == "normal" else architecture.reduction_blocks
            block = blocks[action.block_idx]
            if action.component in {"i1", "i2"}:
                valid_inputs = self.valid_inputs_for_block(action.block_idx)
                current_value = getattr(block, action.component)
                mask[action_idx] = action.value_idx in valid_inputs and current_value != action.value_idx
            else:
                current_value = getattr(block, action.component)
                mask[action_idx] = current_value != action.value_idx
        return mask

    def decode_action(self, action_idx: int) -> MutationAction:
        return self.actions[action_idx]

    def apply_action(self, architecture: Architecture, action_idx: int) -> Architecture:
        action = self.decode_action(action_idx)
        mutated = architecture.clone()

        if action.component == "depth":
            mutated.depth = self.depth_choices[action.value_idx]
            return mutated
        if action.component == "width":
            mutated.width_multiplier = self.width_choices[action.value_idx]
            return mutated

        blocks = mutated.normal_blocks if action.cell_type == "normal" else mutated.reduction_blocks
        block = blocks[action.block_idx]
        updates = block.__dict__.copy()
        if action.component in {"i1", "i2"}:
            if action.value_idx not in self.valid_inputs_for_block(action.block_idx):
                raise ValueError(f"Invalid input index {action.value_idx} for block {action.block_idx}.")
        updates[action.component] = action.value_idx
        blocks[action.block_idx] = BlockGene(**updates)
        return mutated
