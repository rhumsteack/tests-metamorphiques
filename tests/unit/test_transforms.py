from __future__ import annotations

import pytest
import torch

from src.transforms import adjust_brightness, gaussian_blur, horizontal_flip, rotate_tensor


pytestmark = pytest.mark.unit


def test_double_flip_returns_original_shape():
    batch = torch.rand(8, 1, 28, 28)
    flipped = horizontal_flip(horizontal_flip(batch))

    assert flipped.shape == batch.shape


def test_rotate_preserves_shape():
    batch = torch.rand(4, 1, 28, 28)
    rotated = rotate_tensor(batch, 10)

    assert rotated.shape == batch.shape


def test_brightness_stays_non_negative():
    batch = torch.rand(4, 1, 28, 28)
    bright = adjust_brightness(batch, 1.2)

    assert torch.all(bright >= 0)


def test_blur_preserves_shape():
    batch = torch.rand(4, 1, 28, 28)
    blurred = gaussian_blur(batch, kernel_size=3, sigma=0.8)

    assert blurred.shape == batch.shape
