from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from nas_project.search.search_space import Architecture


@dataclass
class Individual:
    uid: int
    architecture: Architecture
    generation: int
    fitness: float = float("-inf")
    metrics: Dict[str, Any] = field(default_factory=dict)
    parent_uid: Optional[int] = None
    mutation_action: Optional[int] = None
    predicted_accuracy: Optional[float] = None

    def to_record(self) -> Dict[str, Any]:
        return {
            "uid": self.uid,
            "generation": self.generation,
            "fitness": self.fitness,
            "metrics": self.metrics,
            "parent_uid": self.parent_uid,
            "mutation_action": self.mutation_action,
            "predicted_accuracy": self.predicted_accuracy,
            "architecture": self.architecture.to_dict(),
        }
