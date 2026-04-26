from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from models.mnist_simple import TrainConfig, load_or_train_model
from src.dataset import MnistDataConfig, build_dataloaders


@pytest.fixture(scope="session")
def config() -> dict:
    config_path = Path("configs/mnist.yaml")
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session", autouse=True)
def set_determinism(config: dict):
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture(scope="session")
def dataloaders(config: dict):
    data_cfg = MnistDataConfig(
        subset_size=int(config["subset_size"]),
        eval_size=int(config["eval_size"]),
        batch_size=int(config["batch_size"]),
        seed=int(config["seed"]),
    )
    return build_dataloaders(data_cfg)


@pytest.fixture(scope="session")
def model(config: dict, dataloaders, device: torch.device):
    train_loader, _ = dataloaders
    train_cfg = TrainConfig(
        learning_rate=float(config["learning_rate"]),
        train_epochs=int(config["train_epochs"]),
        artifact_path=str(config["artifact_path"]),
    )
    return load_or_train_model(train_loader, train_cfg, device)


@pytest.fixture(scope="session")
def eval_batch(config: dict, dataloaders):
    _, eval_loader = dataloaders
    images, labels = next(iter(eval_loader))
    take = min(images.shape[0], int(config["eval_size"]))
    return images[:take], labels[:take]
