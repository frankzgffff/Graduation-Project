from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from nas_project.models.network import NASNetwork
from nas_project.trainer.train import build_dataloaders, fit_model
from nas_project.utils.flops import estimate_flops
from nas_project.utils.metrics import compute_fitness, count_parameters_in_mb


class ArchitectureEvaluator:
    def __init__(self, config) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader, self.val_loader = build_dataloaders(config.dataset)

    def build_model(self, architecture) -> NASNetwork:
        return NASNetwork(
            architecture=architecture,
            init_channels=self.config.model.init_channels,
            num_classes=self.config.dataset.num_classes,
            stem_multiplier=self.config.model.stem_multiplier,
            dropout=self.config.model.dropout,
            op_names=self.config.search.op_names,
        )

    def _build_optimizer(self, model, final_stage: bool = False):
        lr = self.config.train.final_lr if final_stage else self.config.train.lr
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=self.config.train.momentum,
            weight_decay=self.config.train.weight_decay,
        )
        epochs = self.config.train.final_retrain_epochs if final_stage else self.config.train.epochs_per_eval
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
        return optimizer, scheduler

    def _summarize_model(self, model, history: dict[str, float]) -> dict[str, float]:
        history = dict(history)
        for key in ("epoch", "train_loss", "val_loss", "val_acc"):
            if isinstance(history.get(key), list):
                history[key] = history[key][-1] if history[key] else 0.0
        history["params_m"] = count_parameters_in_mb(model)
        history["flops_m"] = estimate_flops(
            model,
            input_size=(1, 3, self.config.dataset.image_size, self.config.dataset.image_size),
            device=self.device,
        )
        history["fitness"] = compute_fitness(
            val_acc=history["val_acc"],
            flops_m=history["flops_m"],
            params_m=history["params_m"],
            flops_penalty=self.config.evolution.complexity_penalty_flops,
            params_penalty=self.config.evolution.complexity_penalty_params,
        )
        return history

    def evaluate(self, architecture) -> dict[str, float]:
        model = self.build_model(architecture).to(self.device)
        criterion = nn.CrossEntropyLoss(label_smoothing=self.config.train.label_smoothing)
        optimizer, scheduler = self._build_optimizer(model, final_stage=False)
        history = fit_model(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=self.device,
            grad_clip=self.config.train.grad_clip,
            epochs=self.config.train.epochs_per_eval,
            scheduler=scheduler,
        )
        return self._summarize_model(model, history)

    def retrain_best(self, architecture, output_dir: str | Path) -> dict[str, float]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model = self.build_model(architecture).to(self.device)
        criterion = nn.CrossEntropyLoss(label_smoothing=self.config.train.label_smoothing)
        optimizer, scheduler = self._build_optimizer(model, final_stage=True)

        history = fit_model(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=self.device,
            grad_clip=self.config.train.grad_clip,
            epochs=self.config.train.final_retrain_epochs,
            scheduler=scheduler,
        )
        history = self._summarize_model(model, history)

        checkpoint_path = output_dir / "best_retrained_model.pt"
        torch.save(
            {
                "architecture": architecture.to_dict(),
                "model_state_dict": model.state_dict(),
                "metrics": history,
            },
            checkpoint_path,
        )
        (output_dir / "final_retrain_metrics.json").write_text(
            json.dumps(history, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return history
