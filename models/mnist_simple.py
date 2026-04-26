from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float = 1.0e-3
    train_epochs: int = 1
    artifact_path: str = "artifacts/mnist_cnn.pt"


class MnistCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def train_model(model: nn.Module, train_loader, cfg: TrainConfig, device: torch.device) -> nn.Module:
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(cfg.train_epochs):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

    return model


def load_or_train_model(train_loader, cfg: TrainConfig, device: torch.device) -> nn.Module:
    model = MnistCNN()
    os.makedirs(os.path.dirname(cfg.artifact_path), exist_ok=True)

    if os.path.exists(cfg.artifact_path):
        state_dict = torch.load(cfg.artifact_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model.to(device)

    model = train_model(model, train_loader, cfg, device)
    torch.save(model.state_dict(), cfg.artifact_path)
    model.eval()
    return model.to(device)
