from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, TypeVar

import numpy as np
import torch


@dataclass
class DatasetConfig:
    name: str = "cifar10"
    root: str = "./data"
    download: bool = True
    image_size: int = 32
    num_classes: int = 10
    train_batch_size: int = 64
    eval_batch_size: int = 128
    num_workers: int = 0
    fake_train_size: int = 512
    fake_val_size: int = 256


@dataclass
class ModelConfig:
    init_channels: int = 16
    num_blocks: int = 4
    stem_multiplier: int = 3
    dropout: float = 0.0
    depth_choices: list[int] = field(default_factory=lambda: [6, 8, 10])
    width_choices: list[float] = field(default_factory=lambda: [0.75, 1.0, 1.25])


@dataclass
class SearchConfig:
    population_size: int = 16
    sample_size: int = 4
    init_population_size: int = 8
    generations: int = 20
    mutation_candidates: int = 6
    candidate_eval_topk: int = 2
    use_rl: bool = True
    use_evolution: bool = True
    use_surrogate: bool = True
    surrogate_warmup: int = 12
    surrogate_fit_interval: int = 5
    op_names: list[str] = field(
        default_factory=lambda: [
            "sep_conv_3x3",
            "sep_conv_5x5",
            "dil_conv_3x3",
            "max_pool_3x3",
            "avg_pool_3x3",
            "skip_connect",
            "none",
        ]
    )


@dataclass
class EvolutionConfig:
    complexity_penalty_flops: float = 0.015
    complexity_penalty_params: float = 0.01
    reward_beta: float = 0.05
    reward_cost_metric: str = "flops"
    reward_cost_scale: float = 1.0


@dataclass
class TrainConfig:
    epochs_per_eval: int = 5
    final_retrain_epochs: int = 150
    lr: float = 0.025
    final_lr: float = 0.025
    momentum: float = 0.9
    weight_decay: float = 3e-4
    grad_clip: float = 5.0
    label_smoothing: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    policy_lr: float = 3e-4
    epochs: int = 4
    minibatch_size: int = 32
    update_interval: int = 4
    hidden_dim: int = 256


@dataclass
class PredictorConfig:
    hidden_dim: int = 128
    lr: float = 1e-3
    epochs: int = 30
    batch_size: int = 32
    weight_decay: float = 1e-4


@dataclass
class ExperimentConfig:
    seed: int = 42
    exp_name: str = "rle_nas"
    output_dir: str = "./outputs"
    log_interval: int = 1
    smoke_test: bool = False
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


T = TypeVar("T")


def _update_dataclass(instance: T, values: Dict[str, Any]) -> T:
    for key, value in values.items():
        if not hasattr(instance, key):
            continue
        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(value, dict):
            _update_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def load_config(path: str | Path) -> ExperimentConfig:
    config = ExperimentConfig()
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return _update_dataclass(config, raw)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_default_config(smoke_test: bool = False) -> ExperimentConfig:
    config = ExperimentConfig(smoke_test=smoke_test)
    if smoke_test:
        config.dataset.name = "fake"
        config.dataset.fake_train_size = 192
        config.dataset.fake_val_size = 96
        config.search.init_population_size = 4
        config.search.population_size = 6
        config.search.generations = 4
        config.search.mutation_candidates = 4
        config.search.candidate_eval_topk = 2
        config.search.surrogate_warmup = 6
        config.search.surrogate_fit_interval = 2
        config.train.epochs_per_eval = 1
        config.train.final_retrain_epochs = 2
        config.model.depth_choices = [4, 6]
        config.model.width_choices = [0.75, 1.0]
        config.model.init_channels = 8
    return config
