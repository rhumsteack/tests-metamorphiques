from __future__ import annotations

import pytest
import torch

from src.inference import predict_logits, predict_top1
from src.transforms import adjust_brightness, gaussian_blur, horizontal_flip, rotate_tensor


pytestmark = pytest.mark.metamorphic


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.max(torch.abs(a - b)).item()


def _top1_match_rate(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a == b).float().mean().item()


def test_mr1_identity_logits_stable(model, eval_batch, device, config):
    images, _ = eval_batch
    base = predict_logits(model, images, device)
    same = predict_logits(model, images.clone(), device)

    diff = _max_abs_diff(base, same)
    assert diff <= float(config["thresholds"]["identity_max_logit_diff"])


def test_mr2_double_flip_preserves_logits(model, eval_batch, device, config):
    images, _ = eval_batch
    transformed = horizontal_flip(horizontal_flip(images))

    base = predict_logits(model, images, device)
    follow_up = predict_logits(model, transformed, device)

    diff = _max_abs_diff(base, follow_up)
    assert diff <= float(config["thresholds"]["double_flip_max_logit_diff"])


def test_mr3_rotate_then_inverse_is_close(model, eval_batch, device, config):
    images, _ = eval_batch
    transformed = rotate_tensor(rotate_tensor(images, 12.0), -12.0)

    base = predict_logits(model, images, device)
    follow_up = predict_logits(model, transformed, device)

    diff = _max_abs_diff(base, follow_up)
    assert diff <= float(config["thresholds"]["rotate_inverse_max_logit_diff"])


def test_mr4_brightness_keeps_majority_top1(model, eval_batch, device, config):
    images, _ = eval_batch
    transformed = adjust_brightness(images, 1.15)

    base_top1 = predict_top1(model, images, device)
    follow_up_top1 = predict_top1(model, transformed, device)

    match_rate = _top1_match_rate(base_top1, follow_up_top1)
    assert match_rate >= float(config["thresholds"]["brightness_top1_match_rate"])


def test_mr5_blur_keeps_majority_top1(model, eval_batch, device, config):
    images, _ = eval_batch
    transformed = gaussian_blur(images, kernel_size=3, sigma=0.8)

    base_top1 = predict_top1(model, images, device)
    follow_up_top1 = predict_top1(model, transformed, device)

    match_rate = _top1_match_rate(base_top1, follow_up_top1)
    assert match_rate >= float(config["thresholds"]["blur_top1_match_rate"])


def test_mr6_single_vs_batch_consistency(model, eval_batch, device, config):
    images, _ = eval_batch
    batch_logits = predict_logits(model, images, device)

    first = images[0:1]
    single_logits = predict_logits(model, first, device)

    diff = _max_abs_diff(batch_logits[0:1], single_logits)
    assert diff <= float(config["thresholds"]["batch_consistency_max_logit_diff"])
