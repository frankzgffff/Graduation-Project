from __future__ import annotations

import random
from dataclasses import dataclass, field

from nas_project.search.individual import Individual


@dataclass
class Population:
    capacity: int
    individuals: list[Individual] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.individuals)

    def add(self, individual: Individual) -> None:
        self.individuals.append(individual)

    def sample(self, k: int) -> list[Individual]:
        return random.sample(self.individuals, k=min(k, len(self.individuals)))

    def best(self) -> Individual:
        return max(self.individuals, key=lambda item: item.fitness)

    def oldest(self) -> Individual:
        return min(self.individuals, key=lambda item: (item.generation, item.uid))

    def remove_oldest(self) -> Individual:
        oldest = self.oldest()
        self.individuals.remove(oldest)
        return oldest

    def fitness_stats(self) -> dict[str, float]:
        if not self.individuals:
            return {"best": 0.0, "mean": 0.0, "worst": 0.0}
        fitness_values = [item.fitness for item in self.individuals]
        return {
            "best": max(fitness_values),
            "mean": sum(fitness_values) / len(fitness_values),
            "worst": min(fitness_values),
        }

