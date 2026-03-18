from __future__ import annotations

import random

import torch

from nas_project.search.search_space import Architecture, SearchSpace


class Mutator:
    def __init__(self, search_space: SearchSpace) -> None:
        self.search_space = search_space

    def random_action(self, architecture: Architecture) -> int:
        mask = self.search_space.get_action_mask(architecture)
        valid_actions = torch.nonzero(mask, as_tuple=False).view(-1).tolist()
        if not valid_actions:
            raise RuntimeError("No valid mutation actions are available.")
        return random.choice(valid_actions)

    def mutate(self, architecture: Architecture, action_idx: int) -> Architecture:
        return self.search_space.apply_action(architecture, action_idx)

