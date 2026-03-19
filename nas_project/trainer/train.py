from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from torchvision import datasets, transforms
except ImportError:  # pragma: no cover
    datasets = None
    transforms = None

from nas_project.utils.metrics import AverageMeter, topk_accuracy


class SyntheticClassificationDataset(Dataset):
    def __init__(self, size: int, image_size: int, num_classes: int, seed: int = 0) -> None:
        self.size = size
        self.image_size = image_size
        self.num_classes = num_classes
        self.seed = seed

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        generator = torch.Generator().manual_seed(self.seed + index)
        image = torch.randn((3, self.image_size, self.image_size), generator=generator)
        label = torch.randint(0, self.num_classes, size=(1,), generator=generator).item()
        return image, label


def build_dataloaders(dataset_config):
    if dataset_config.name.lower() == "fake":
        if datasets is not None:
            normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            eval_transform = transforms.Compose([transforms.ToTensor(), normalize])
            train_set = datasets.FakeData(
                size=dataset_config.fake_train_size,
                image_size=(3, dataset_config.image_size, dataset_config.image_size),
                num_classes=dataset_config.num_classes,
                transform=eval_transform,
            )
            val_set = datasets.FakeData(
                size=dataset_config.fake_val_size,
                image_size=(3, dataset_config.image_size, dataset_config.image_size),
                num_classes=dataset_config.num_classes,
                transform=eval_transform,
            )
        else:
            train_set = SyntheticClassificationDataset(
                size=dataset_config.fake_train_size,
                image_size=dataset_config.image_size,
                num_classes=dataset_config.num_classes,
                seed=11,
            )
            val_set = SyntheticClassificationDataset(
                size=dataset_config.fake_val_size,
                image_size=dataset_config.image_size,
                num_classes=dataset_config.num_classes,
                seed=29,
            )
    else:
        if transforms is None or datasets is None:
            raise ImportError("torchvision is required for real image datasets such as CIFAR-10.")
        if dataset_config.name.lower() != "cifar10":
            raise ValueError(f"Unsupported dataset: {dataset_config.name}")

        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(dataset_config.image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        eval_transform = transforms.Compose([transforms.ToTensor(), normalize])
        root = Path(dataset_config.root)
        train_set = datasets.CIFAR10(root=root, train=True, download=dataset_config.download, transform=train_transform)
        val_set = datasets.CIFAR10(root=root, train=False, download=dataset_config.download, transform=eval_transform)

    train_loader = DataLoader(
        train_set,
        batch_size=dataset_config.train_batch_size,
        shuffle=True,
        num_workers=dataset_config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=dataset_config.eval_batch_size,
        shuffle=False,
        num_workers=dataset_config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device: str, grad_clip: float) -> dict[str, float]:
    model.train()
    loss_meter = AverageMeter("train_loss")
    acc_meter = AverageMeter("train_acc1")

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        acc1 = topk_accuracy(logits, targets, topk=(1,))[0]
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc1, images.size(0))

    return {"train_loss": loss_meter.avg, "train_acc1": acc_meter.avg}


@torch.no_grad()
def evaluate(model, loader, criterion, device: str) -> dict[str, float]:
    model.eval()
    loss_meter = AverageMeter("val_loss")
    acc_meter = AverageMeter("val_acc1")

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        acc1 = topk_accuracy(logits, targets, topk=(1,))[0]
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc1, images.size(0))

    return {"val_loss": loss_meter.avg, "val_acc": acc_meter.avg}


def fit_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device: str,
    grad_clip: float,
    epochs: int,
    scheduler=None,
    epoch_callback: Callable[[dict[str, float]], None] | None = None,
) -> dict[str, list[float]]:
    history: dict[str, list[float]] = {
        "epoch": [],
        "train_loss": [],
        "train_acc1": [],
        "val_loss": [],
        "val_acc": [],
    }
    for epoch in range(epochs):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip=grad_clip,
        )
        val_metrics = evaluate(model=model, loader=val_loader, criterion=criterion, device=device)
        if scheduler is not None:
            scheduler.step()
        epoch_metrics = {"epoch": epoch + 1, **train_metrics, **val_metrics}
        for key, value in epoch_metrics.items():
            history.setdefault(key, []).append(value)
        if epoch_callback is not None:
            epoch_callback(epoch_metrics)
    return history
