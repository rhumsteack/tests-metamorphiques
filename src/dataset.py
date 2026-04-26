from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


@dataclass(frozen=True)
class MnistDataConfig:
    data_dir: str = "data"
    subset_size: int = 1024
    eval_size: int = 128
    batch_size: int = 64
    seed: int = 42


def _deterministic_indices(size: int, subset_size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(size, generator=generator)
    return indices[:subset_size]


def build_mnist_subsets(cfg: MnistDataConfig) -> Tuple[Subset, Subset]:
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST(
        root=cfg.data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=cfg.data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_indices = _deterministic_indices(len(train_dataset), cfg.subset_size, cfg.seed)
    test_indices = _deterministic_indices(len(test_dataset), cfg.eval_size, cfg.seed + 1)

    return Subset(train_dataset, train_indices.tolist()), Subset(test_dataset, test_indices.tolist())


def build_dataloaders(cfg: MnistDataConfig) -> Tuple[DataLoader, DataLoader]:
    train_subset, eval_subset = build_mnist_subsets(cfg)

    train_loader = DataLoader(
        train_subset,
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    eval_loader = DataLoader(
        eval_subset,
        batch_size=cfg.batch_size,
        shuffle=False,
    )
    return train_loader, eval_loader
