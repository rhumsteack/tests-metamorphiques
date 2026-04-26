"""Interactive demo showing metamorphic relations visually with pass/fail cases."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml

from models.mnist_simple import TrainConfig, load_or_train_model
from src.dataset import MnistDataConfig, build_dataloaders
from src.inference import predict_logits, predict_top1
from src.transforms import (
    adjust_brightness,
    gaussian_blur,
    horizontal_flip,
    rotate_tensor,
)


def plot_images_side_by_side(
    source: torch.Tensor,
    follow_up: torch.Tensor,
    source_label: str,
    follow_up_label: str,
    source_pred: int,
    follow_up_pred: int,
    source_logits: torch.Tensor,
    follow_up_logits: torch.Tensor,
    title: str,
    passed: bool = True,
) -> None:
    """Plot source and follow-up images side-by-side with predictions and status."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(source.squeeze().cpu().numpy(), cmap="gray")
    max_conf_0 = source_logits.max().item()
    axes[0].set_title(f"{source_label}\nPredicted: {source_pred} (conf: {max_conf_0:.2f})")
    axes[0].axis("off")

    axes[1].imshow(follow_up.squeeze().cpu().numpy(), cmap="gray")
    max_conf_1 = follow_up_logits.max().item()
    axes[1].set_title(f"{follow_up_label}\nPredicted: {follow_up_pred} (conf: {max_conf_1:.2f})")
    axes[1].axis("off")

    status = "✓ PASS" if passed else "✗ FAIL"
    color = "green" if passed else "red"
    fig.patch.set_facecolor("mistyrose" if not passed else "white")
    fig.suptitle(f"{title} [{status}]", fontsize=14, fontweight="bold", color=color)
    plt.tight_layout()
    plt.show()


def get_model_and_data():
    """Load model and evaluation data."""
    config_path = Path("configs/mnist.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    data_cfg = MnistDataConfig(seed=int(config["seed"]))
    train_loader, eval_loader = build_dataloaders(data_cfg)

    train_cfg = TrainConfig(
        learning_rate=float(config["learning_rate"]),
        train_epochs=int(config["train_epochs"]),
        artifact_path=str(config["artifact_path"]),
    )
    device = torch.device("cpu")
    model = load_or_train_model(train_loader, train_cfg, device)

    images, labels = next(iter(eval_loader))
    return model, device, images, labels


def demo_mr1_identity():
    """MR1: Identity - same image should always give same prediction."""
    print("\n" + "=" * 70)
    print("MR1: IDENTITY - Same image should give same prediction")
    print("=" * 70)

    model, device, images, labels = get_model_and_data()
    idx = 0

    source = images[idx : idx + 1]
    follow_up = source.clone()

    source_logits = predict_logits(model, source, device)[0]
    follow_up_logits = predict_logits(model, follow_up, device)[0]
    source_pred = torch.argmax(source_logits).item()
    follow_up_pred = torch.argmax(follow_up_logits).item()

    passed = source_pred == follow_up_pred
    print(f"\nSource: digit {source_pred}")
    print(f"Cloned: digit {follow_up_pred}")
    print(f"Status: {'PASS ✓' if passed else 'FAIL ✗'}")

    plot_images_side_by_side(
        source[0], follow_up[0], "Original", "Cloned (same)",
        source_pred, follow_up_pred, source_logits, follow_up_logits,
        "MR1: Identity - Expected: predictions match", passed,
    )


def demo_mr2_double_flip():
    """MR2: Double flip should be reversible."""
    print("\n" + "=" * 70)
    print("MR2: DOUBLE FLIP - Flip twice should restore original")
    print("=" * 70)

    model, device, images, labels = get_model_and_data()
    idx = 0

    source = images[idx : idx + 1]
    follow_up = horizontal_flip(horizontal_flip(source))

    source_logits = predict_logits(model, source, device)[0]
    follow_up_logits = predict_logits(model, follow_up, device)[0]
    source_pred = torch.argmax(source_logits).item()
    follow_up_pred = torch.argmax(follow_up_logits).item()

    passed = source_pred == follow_up_pred
    print(f"\nSource: digit {source_pred}")
    print(f"Double-flipped: digit {follow_up_pred}")
    print(f"Status: {'PASS ✓' if passed else 'FAIL ✗'}")

    plot_images_side_by_side(
        source[0], follow_up[0], "Original", "After 2× horizontal flip",
        source_pred, follow_up_pred, source_logits, follow_up_logits,
        "MR2: Double Flip - Expected: predictions match", passed,
    )


def demo_mr3_rotate_inverse():
    """MR3: Small rotation then inverse should be stable."""
    print("\n" + "=" * 70)
    print("MR3: ROTATE + INVERSE - Small rotation then rotate back")
    print("=" * 70)

    model, device, images, labels = get_model_and_data()
    idx = 0

    source = images[idx : idx + 1]
    follow_up = rotate_tensor(rotate_tensor(source, 12.0), -12.0)

    source_logits = predict_logits(model, source, device)[0]
    follow_up_logits = predict_logits(model, follow_up, device)[0]
    source_pred = torch.argmax(source_logits).item()
    follow_up_pred = torch.argmax(follow_up_logits).item()

    passed = source_pred == follow_up_pred
    print(f"\nSource: digit {source_pred}")
    print(f"Rotate ±12°: digit {follow_up_pred}")
    print(f"Status: {'PASS ✓' if passed else 'FAIL ✗'}")

    plot_images_side_by_side(
        source[0], follow_up[0], "Original", "Rotate +12°, then -12°",
        source_pred, follow_up_pred, source_logits, follow_up_logits,
        "MR3: Rotate Inverse - Expected: predictions match", passed,
    )


def demo_mr4_brightness():
    """MR4: Slight brightness change should keep prediction."""
    print("\n" + "=" * 70)
    print("MR4: BRIGHTNESS - Slight brightness change should keep prediction")
    print("=" * 70)

    model, device, images, labels = get_model_and_data()
    idx = 0

    source = images[idx : idx + 1]
    follow_up = adjust_brightness(source, 1.15)

    source_logits = predict_logits(model, source, device)[0]
    follow_up_logits = predict_logits(model, follow_up, device)[0]
    source_pred = torch.argmax(source_logits).item()
    follow_up_pred = torch.argmax(follow_up_logits).item()

    passed = source_pred == follow_up_pred
    print(f"\nSource: digit {source_pred}")
    print(f"15% brighter: digit {follow_up_pred}")
    print(f"Status: {'PASS ✓' if passed else 'FAIL ✗'}")

    plot_images_side_by_side(
        source[0], follow_up[0], "Original", "15% brighter",
        source_pred, follow_up_pred, source_logits, follow_up_logits,
        "MR4: Brightness - Expected: same prediction", passed,
    )


def demo_mr5_blur():
    """MR5: Slight blur should keep prediction."""
    print("\n" + "=" * 70)
    print("MR5: BLUR - Slight blur should keep prediction")
    print("=" * 70)

    model, device, images, labels = get_model_and_data()
    idx = 0

    source = images[idx : idx + 1]
    follow_up = gaussian_blur(source, kernel_size=3, sigma=0.8)

    source_logits = predict_logits(model, source, device)[0]
    follow_up_logits = predict_logits(model, follow_up, device)[0]
    source_pred = torch.argmax(source_logits).item()
    follow_up_pred = torch.argmax(follow_up_logits).item()

    passed = source_pred == follow_up_pred
    print(f"\nSource: digit {source_pred}")
    print(f"Blurred: digit {follow_up_pred}")
    print(f"Status: {'PASS ✓' if passed else 'FAIL ✗'}")

    plot_images_side_by_side(
        source[0], follow_up[0], "Original", "Gaussian blur (σ=0.8)",
        source_pred, follow_up_pred, source_logits, follow_up_logits,
        "MR5: Blur - Expected: same prediction", passed,
    )


def demo_extreme_rotation_failure():
    """Challenge case: Extreme rotation (45°) - likely to fail."""
    print("\n" + "=" * 70)
    print("CHALLENGE 1: Extreme Rotation (±45°) - Expected to FAIL")
    print("=" * 70)
    print("Large transformations can break predictions - this MR should detect it\n")

    model, device, images, labels = get_model_and_data()
    idx = 0

    source = images[idx : idx + 1]
    follow_up = rotate_tensor(rotate_tensor(source, 45.0), -45.0)

    source_logits = predict_logits(model, source, device)[0]
    follow_up_logits = predict_logits(model, follow_up, device)[0]
    source_pred = torch.argmax(source_logits).item()
    follow_up_pred = torch.argmax(follow_up_logits).item()

    passed = source_pred == follow_up_pred
    print(f"Source: digit {source_pred}")
    print(f"Rotate ±45°: digit {follow_up_pred}")
    print(f"Status: {'PASS ✓' if passed else 'FAIL ✗ (detected anomaly)'}")

    plot_images_side_by_side(
        source[0], follow_up[0], "Original", "Rotate ±45° (extreme)",
        source_pred, follow_up_pred, source_logits, follow_up_logits,
        "Challenge: Extreme Rotation - EXPECTED FAILURE", passed,
    )


def demo_aggressive_blur_failure():
    """Challenge case: Aggressive blur (large kernel, high sigma)."""
    print("\n" + "=" * 70)
    print("CHALLENGE 2: Aggressive Blur (k=7, σ=2.0) - Likely to FAIL")
    print("=" * 70)
    print("Shows robustness boundaries - blur can destroy digit features\n")

    model, device, images, labels = get_model_and_data()
    idx = 0

    source = images[idx : idx + 1]
    follow_up = gaussian_blur(source, kernel_size=7, sigma=2.0)

    source_logits = predict_logits(model, source, device)[0]
    follow_up_logits = predict_logits(model, follow_up, device)[0]
    source_pred = torch.argmax(source_logits).item()
    follow_up_pred = torch.argmax(follow_up_logits).item()

    passed = source_pred == follow_up_pred
    print(f"Source: digit {source_pred}")
    print(f"Aggressive blur: digit {follow_up_pred}")
    print(f"Status: {'PASS ✓' if passed else 'FAIL ✗ (detected anomaly)'}")

    plot_images_side_by_side(
        source[0], follow_up[0], "Original", "Aggressive blur (k=7, σ=2.0)",
        source_pred, follow_up_pred, source_logits, follow_up_logits,
        "Challenge: Aggressive Blur", passed,
    )


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("METAMORPHIC TESTING VISUAL DEMO - MNIST (Enhanced)")
    print("=" * 70)
    print("\nShowing PASSING cases + CHALLENGING cases (expected failures)\n")

    # Passing metamorphic relations
    demo_mr1_identity()
    demo_mr2_double_flip()
    demo_mr3_rotate_inverse()
    demo_mr4_brightness()
    demo_mr5_blur()

    # Challenging/stress-test cases
    demo_extreme_rotation_failure()
    demo_aggressive_blur_failure()

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)
    print("\nKey insights:")
    print("✓ MRs PASS: model handles normal transformations well")
    print("✗ MRs FAIL: detects when transformations are too extreme")
    print("→ Find the right thresholds = sweet spot for robustness automation\n")
