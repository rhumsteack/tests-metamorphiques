from __future__ import annotations

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


def rotate_tensor(batch: torch.Tensor, degrees: float) -> torch.Tensor:
    return F.rotate(batch, angle=degrees, interpolation=InterpolationMode.BILINEAR)


def horizontal_flip(batch: torch.Tensor) -> torch.Tensor:
    return F.hflip(batch)


def adjust_brightness(batch: torch.Tensor, factor: float) -> torch.Tensor:
    return F.adjust_brightness(batch, brightness_factor=factor)


def gaussian_blur(batch: torch.Tensor, kernel_size: int = 3, sigma: float = 0.8) -> torch.Tensor:
    return F.gaussian_blur(batch, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
