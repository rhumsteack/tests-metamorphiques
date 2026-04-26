from __future__ import annotations

import pytest

from src.dataset import MnistDataConfig, build_dataloaders


pytestmark = pytest.mark.unit


def test_mnist_dataloader_shapes():
    cfg = MnistDataConfig(subset_size=64, eval_size=32, batch_size=16, seed=42)
    train_loader, eval_loader = build_dataloaders(cfg)

    train_images, train_labels = next(iter(train_loader))
    eval_images, eval_labels = next(iter(eval_loader))

    assert train_images.shape[1:] == (1, 28, 28)
    assert eval_images.shape[1:] == (1, 28, 28)
    assert train_labels.ndim == 1
    assert eval_labels.ndim == 1
