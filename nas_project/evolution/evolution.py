from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch

from nas_project.config import ExperimentConfig
from nas_project.evolution.aging_evolution import AgingEvolution
from nas_project.evolution.mutation import Mutator
from nas_project.predictor.surrogate_model import SurrogateModel
from nas_project.rl.buffer import RolloutBuffer
from nas_project.rl.policy_network import PolicyNetwork
from nas_project.rl.ppo import PPOAgent
from nas_project.search.individual import Individual
from nas_project.search.population import Population
from nas_project.search.search_space import MutationAction, SearchSpace
from nas_project.trainer.evaluator import ArchitectureEvaluator
from nas_project.utils.logger import ExperimentLogger
from nas_project.utils.metrics import compute_reward, population_diversity


class HybridRLEvolutionSearch:
    def __init__(
        self,
        config: ExperimentConfig,
        search_space: SearchSpace,
        evaluator: ArchitectureEvaluator,
        logger,
        exp_logger: ExperimentLogger,
        output_dir: str | Path,
    ) -> None:
        self.config = config
        self.search_space = search_space
        self.evaluator = evaluator
        self.logger = logger
        self.exp_logger = exp_logger
        self.output_dir = Path(output_dir)
        self.population = Population(capacity=config.search.population_size)
        self.evolution = AgingEvolution(self.population, config.search.sample_size)
        self.mutator = Mutator(search_space)
        self.archive: list[Individual] = []
        self.uid_counter = 0

        self.policy_state_dim = self.search_space.encoding_dim + 4
        self.policy = PolicyNetwork(
            input_dim=self.policy_state_dim,
            action_dim=self.search_space.action_dim,
            hidden_dim=config.ppo.hidden_dim,
        ).to(config.train.device)
        self.buffer = RolloutBuffer(config.ppo.gamma, config.ppo.gae_lambda, config.train.device)
        self.ppo = PPOAgent(self.policy, config.ppo, config.train.device)
        self.surrogate = SurrogateModel(
            input_dim=self.search_space.encoding_dim,
            hidden_dim=config.predictor.hidden_dim,
            lr=config.predictor.lr,
            weight_decay=config.predictor.weight_decay,
            device=config.train.device,
        )

    def _next_uid(self) -> int:
        self.uid_counter += 1
        return self.uid_counter

    def _state_tensor(self, individual: Individual, generation: int) -> torch.Tensor:
        arch_features = self.search_space.encode(individual.architecture)
        extra = torch.tensor(
            [
                individual.metrics["val_acc"] / 100.0,
                individual.metrics["params_m"],
                individual.metrics["flops_m"],
                generation / max(self.config.search.generations, 1),
            ],
            dtype=torch.float32,
        )
        return torch.cat([arch_features, extra], dim=0)

    def _action_to_record(self, action_idx: int) -> dict[str, object]:
        action: MutationAction = self.search_space.decode_action(action_idx)
        record = {
            "cell_type": action.cell_type,
            "block_idx": action.block_idx,
            "component": action.component,
            "value_idx": action.value_idx,
        }
        if action.component in {"o1", "o2"}:
            record["value_name"] = self.config.search.op_names[action.value_idx]
        elif action.component == "depth":
            record["value_name"] = self.config.model.depth_choices[action.value_idx]
        elif action.component == "width":
            record["value_name"] = self.config.model.width_choices[action.value_idx]
        else:
            record["value_name"] = action.value_idx
        return record

    def _evaluate_architecture(
        self,
        architecture,
        generation: int,
        parent_uid: Optional[int] = None,
        mutation_action: Optional[int] = None,
        predicted_accuracy: Optional[float] = None,
    ) -> Individual:
        metrics = self.evaluator.evaluate(architecture)
        individual = Individual(
            uid=self._next_uid(),
            architecture=architecture,
            generation=generation,
            fitness=metrics["fitness"],
            metrics=metrics,
            parent_uid=parent_uid,
            mutation_action=mutation_action,
            predicted_accuracy=predicted_accuracy,
        )
        self.archive.append(individual)
        self.exp_logger.log_metrics({"event": "evaluation", **individual.to_record()})
        return individual

    def _surrogate_ready(self) -> bool:
        return self.config.search.use_surrogate and len(self.archive) >= self.config.search.surrogate_warmup

    def _fit_surrogate(self) -> None:
        if not self._surrogate_ready():
            return
        features = torch.stack([self.search_space.encode(item.architecture) for item in self.archive])
        targets = torch.tensor([item.metrics["val_acc"] for item in self.archive], dtype=torch.float32)
        fit_metrics = self.surrogate.fit(
            features,
            targets,
            epochs=self.config.predictor.epochs,
            batch_size=self.config.predictor.batch_size,
        )
        self.logger.info("Surrogate fit mse=%.6f", fit_metrics["mse"])
        self.exp_logger.log_metrics({"event": "surrogate_fit", **fit_metrics})

    def initialize(self) -> None:
        self.logger.info("Initializing population with %d individuals", self.config.search.init_population_size)
        for _ in range(self.config.search.init_population_size):
            architecture = self.search_space.sample_architecture()
            individual = self._evaluate_architecture(architecture, generation=0)
            self.population.add(individual)
            self.logger.info(
                "Init uid=%d fitness=%.4f acc=%.2f depth=%d width=%.2f",
                individual.uid,
                individual.fitness,
                individual.metrics["val_acc"],
                individual.architecture.depth,
                individual.architecture.width_multiplier,
            )
        self.exp_logger.log_metrics(
            {
                "event": "population_init",
                "population_size": len(self.population),
                "population_diversity": population_diversity(self.population.individuals),
            }
        )

    def _propose_mutations(self, parent: Individual, generation: int) -> list[dict]:
        proposals = []
        state_tensor = self._state_tensor(parent, generation)
        action_mask = self.search_space.get_action_mask(parent.architecture)

        for _ in range(self.config.search.mutation_candidates):
            if self.config.search.use_rl:
                action_idx, log_prob, value = self.ppo.select_action(state_tensor, action_mask)
            else:
                action_idx = self.mutator.random_action(parent.architecture)
                log_prob = None
                value = None

            child_architecture = self.mutator.mutate(parent.architecture, action_idx)
            proposals.append(
                {
                    "action_idx": action_idx,
                    "action": self._action_to_record(action_idx),
                    "log_prob": log_prob,
                    "value": value,
                    "state_tensor": state_tensor,
                    "action_mask": action_mask,
                    "architecture": child_architecture,
                    "predicted_accuracy": None,
                }
            )

        if self._surrogate_ready():
            features = torch.stack([self.search_space.encode(item["architecture"]) for item in proposals])
            predictions = self.surrogate.predict(features)
            for proposal, prediction in zip(proposals, predictions):
                proposal["predicted_accuracy"] = prediction
            proposals.sort(key=lambda item: item["predicted_accuracy"], reverse=True)
        return proposals

    def _evaluate_candidates(self, parent: Individual, proposals: list[dict], generation: int) -> list[dict]:
        if self._surrogate_ready():
            top_k = min(self.config.search.candidate_eval_topk, len(proposals))
            proposals = proposals[:top_k]

        evaluated = []
        for proposal in proposals:
            child = self._evaluate_architecture(
                architecture=proposal["architecture"],
                generation=generation,
                parent_uid=parent.uid,
                mutation_action=proposal["action_idx"],
                predicted_accuracy=proposal["predicted_accuracy"],
            )
            reward = compute_reward(
                parent_metrics=parent.metrics,
                child_metrics=child.metrics,
                beta=self.config.evolution.reward_beta,
                cost_metric=self.config.evolution.reward_cost_metric,
                cost_scale=self.config.evolution.reward_cost_scale,
            )
            proposal["child"] = child
            proposal["reward"] = reward
            evaluated.append(proposal)

            self.exp_logger.log_metrics(
                {
                    "event": "candidate",
                    "generation": generation,
                    "parent_uid": parent.uid,
                    "child_uid": child.uid,
                    "predicted_accuracy": proposal["predicted_accuracy"],
                    "reward": reward,
                    "action": proposal["action"],
                    "child_val_acc": child.metrics["val_acc"],
                    "child_flops_m": child.metrics["flops_m"],
                    "child_params_m": child.metrics["params_m"],
                }
            )
            if self.config.search.use_rl:
                self.buffer.add(
                    state=proposal["state_tensor"],
                    action=proposal["action_idx"],
                    log_prob=proposal["log_prob"],
                    reward=reward,
                    done=False,
                    value=proposal["value"],
                    action_mask=proposal["action_mask"],
                )
        return evaluated

    def _maybe_update_policy(self, generation: int) -> dict | None:
        if not self.config.search.use_rl or len(self.buffer) == 0:
            return None
        if generation % self.config.ppo.update_interval != 0:
            return None
        metrics = self.ppo.update(self.buffer)
        self.buffer.clear()
        self.exp_logger.log_metrics({"event": "ppo_update", **metrics})
        self.logger.info(
            "PPO update policy_loss=%.4f value_loss=%.4f entropy=%.4f",
            metrics["policy_loss"],
            metrics["value_loss"],
            metrics["entropy"],
        )
        return metrics

    def save_best(self, best: Individual) -> None:
        best_path = self.output_dir / "best_architecture.json"
        best_path.write_text(json.dumps(best.to_record(), indent=2, ensure_ascii=False), encoding="utf-8")

    def run(self) -> Individual:
        self.initialize()
        self._fit_surrogate()

        for generation in range(1, self.config.search.generations + 1):
            parent = self.evolution.select_parent()
            proposals = self._propose_mutations(parent, generation)
            evaluated = self._evaluate_candidates(parent, proposals, generation)
            selected = max(evaluated, key=lambda item: item["child"].fitness)
            inserted_child = selected["child"]
            self.evolution.insert(inserted_child)

            diversity = population_diversity(self.population.individuals)
            self.exp_logger.log_metrics(
                {
                    "event": "generation",
                    "generation": generation,
                    "parent_uid": parent.uid,
                    "selected_child_uid": inserted_child.uid,
                    "selected_reward": selected["reward"],
                    "selected_action": selected["action"],
                    "best_fitness": self.population.best().fitness,
                    "best_val_acc": self.population.best().metrics["val_acc"],
                    "population_mean_fitness": self.population.fitness_stats()["mean"],
                    "population_diversity": diversity,
                }
            )
            self.logger.info(
                "Gen %d | parent=%d child=%d reward=%.4f acc=%.2f best=%.4f diversity=%.4f",
                generation,
                parent.uid,
                inserted_child.uid,
                selected["reward"],
                inserted_child.metrics["val_acc"],
                self.population.best().fitness,
                diversity,
            )

            if generation % self.config.search.surrogate_fit_interval == 0:
                self._fit_surrogate()
            self._maybe_update_policy(generation)

        best = self.population.best()
        self.save_best(best)
        self.logger.info("Search complete | best uid=%d fitness=%.4f", best.uid, best.fitness)
        return best
