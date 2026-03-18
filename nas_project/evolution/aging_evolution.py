from __future__ import annotations

from nas_project.search.individual import Individual
from nas_project.search.population import Population


class AgingEvolution:
    def __init__(self, population: Population, sample_size: int) -> None:
        self.population = population
        self.sample_size = sample_size

    def select_parent(self) -> Individual:
        sample = self.population.sample(self.sample_size)
        return max(sample, key=lambda item: item.fitness)

    def insert(self, offspring: Individual) -> Individual | None:
        self.population.add(offspring)
        if len(self.population) > self.population.capacity:
            return self.population.remove_oldest()
        return None

