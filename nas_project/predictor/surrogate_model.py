from __future__ import annotations

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset


class SurrogateRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class SurrogateModel:
    def __init__(self, input_dim: int, hidden_dim: int, lr: float, weight_decay: float, device: str) -> None:
        self.model = SurrogateRegressor(input_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        self.device = device
        self.last_fit_mse = 0.0

    def fit(self, features: torch.Tensor, targets: torch.Tensor, epochs: int, batch_size: int) -> dict[str, float]:
        dataset = TensorDataset(features.float(), targets.float())
        loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)
        self.model.train()
        mse_meter = 0.0
        for _ in range(epochs):
            for batch_features, batch_targets in loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                preds = self.model(batch_features)
                loss = self.criterion(preds, batch_targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                mse_meter += loss.item()
        self.last_fit_mse = mse_meter / max(epochs * len(loader), 1)
        return {"mse": self.last_fit_mse}

    def predict(self, features: torch.Tensor) -> list[float]:
        self.model.eval()
        with torch.no_grad():
            preds = self.model(features.to(self.device).float())
        return preds.detach().cpu().tolist()

