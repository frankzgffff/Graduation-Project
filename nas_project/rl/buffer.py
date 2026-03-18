from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class RolloutBuffer:
    gamma: float
    gae_lambda: float
    device: str
    states: list[torch.Tensor] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    action_masks: list[torch.Tensor] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.actions)

    def add(
        self,
        state: torch.Tensor,
        action: int,
        log_prob,
        reward: float,
        done: bool,
        value,
        action_mask: torch.Tensor,
    ) -> None:
        if log_prob is None or value is None:
            return
        self.states.append(state.detach().cpu())
        self.actions.append(int(action))
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(float(value))
        self.action_masks.append(action_mask.detach().cpu())

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.action_masks.clear()

    def compute_returns_and_advantages(self) -> tuple[torch.Tensor, torch.Tensor]:
        advantages = []
        gae = 0.0
        next_value = 0.0
        next_non_terminal = 0.0

        for step in reversed(range(len(self.rewards))):
            non_terminal = 1.0 - float(self.dones[step])
            delta = self.rewards[step] + self.gamma * next_value * next_non_terminal - self.values[step]
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            advantages.insert(0, gae)
            next_value = self.values[step]
            next_non_terminal = non_terminal

        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_tensor = advantages_tensor + torch.tensor(self.values, dtype=torch.float32, device=self.device)
        return returns_tensor, advantages_tensor

