from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical
from torch.utils.data import BatchSampler, SubsetRandomSampler


class PPOAgent:
    def __init__(self, policy, config, device: str) -> None:
        self.policy = policy
        self.config = config
        self.device = device
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.policy_lr)

    @staticmethod
    def _masked_categorical(logits: torch.Tensor, action_mask: torch.Tensor) -> Categorical:
        masked_logits = logits.masked_fill(~action_mask, -1e9)
        return Categorical(logits=masked_logits)

    def select_action(self, state: torch.Tensor, action_mask: torch.Tensor) -> tuple[int, float, float]:
        state = state.to(self.device).unsqueeze(0)
        action_mask = action_mask.to(self.device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.policy(state)
            dist = self._masked_categorical(logits, action_mask)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    def update(self, buffer) -> dict[str, float]:
        if len(buffer) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        states = torch.stack(buffer.states).to(self.device)
        actions = torch.tensor(buffer.actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(buffer.log_probs, dtype=torch.float32, device=self.device)
        action_masks = torch.stack(buffer.action_masks).to(self.device)
        returns, advantages = buffer.compute_returns_and_advantages()
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        policy_loss_meter = 0.0
        value_loss_meter = 0.0
        entropy_meter = 0.0
        num_updates = 0

        for _ in range(self.config.epochs):
            sampler = BatchSampler(
                SubsetRandomSampler(range(len(buffer))),
                batch_size=min(self.config.minibatch_size, len(buffer)),
                drop_last=False,
            )
            for batch_indices in sampler:
                batch_indices = torch.tensor(batch_indices, dtype=torch.long, device=self.device)
                logits, values = self.policy(states[batch_indices])
                dist = self._masked_categorical(logits, action_masks[batch_indices])
                new_log_probs = dist.log_prob(actions[batch_indices])
                entropy = dist.entropy().mean()
                ratios = torch.exp(new_log_probs - old_log_probs[batch_indices])
                surr1 = ratios * advantages[batch_indices]
                surr2 = torch.clamp(ratios, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * advantages[
                    batch_indices
                ]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, returns[batch_indices])
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0)
                self.optimizer.step()

                policy_loss_meter += policy_loss.item()
                value_loss_meter += value_loss.item()
                entropy_meter += entropy.item()
                num_updates += 1

        return {
            "policy_loss": policy_loss_meter / max(num_updates, 1),
            "value_loss": value_loss_meter / max(num_updates, 1),
            "entropy": entropy_meter / max(num_updates, 1),
        }

