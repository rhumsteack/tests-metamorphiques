"""Demo showing how metamorphic tests DETECT bugs when they're introduced."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml

from models.mnist_simple import TrainConfig, load_or_train_model
from src.dataset import MnistDataConfig, build_dataloaders
from src.inference import predict_logits
from src.transforms import adjust_brightness, gaussian_blur, horizontal_flip, rotate_tensor


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


def demo_normal_mr2():
    """Normal MR2: Double flip should be reversible."""
    print("\n" + "=" * 70)
    print("BASELINE: MR2 with CORRECT pipeline")
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
    print(f"Source: digit {source_pred}")
    print(f"Double-flipped: digit {follow_up_pred}")
    print(f"Status: {'PASS ✓' if passed else 'FAIL ✗'}")

    plot_images_side_by_side(
        source[0], follow_up[0], "Original", "After 2× horizontal flip",
        source_pred, follow_up_pred, source_logits, follow_up_logits,
        "MR2 (NORMAL): Double Flip - Should PASS", passed,
    )


def demo_buggy_mr2_invert():
    """Buggy MR2: Accidentally invert follow-up image colors."""
    print("\n" + "=" * 70)
    print("BUG INTRODUCED: Inverting colors in follow-up image")
    print("=" * 70)
    print("(Simulating a preprocessing bug: RGB vs BGR, or channel swap)\n")

    model, device, images, labels = get_model_and_data()
    idx = 0

    source = images[idx : idx + 1]
    follow_up = horizontal_flip(horizontal_flip(source))
    
    # BUG: Invert the colors (1 - pixel)
    follow_up_buggy = 1.0 - follow_up

    source_logits = predict_logits(model, source, device)[0]
    follow_up_logits = predict_logits(model, follow_up_buggy, device)[0]
    source_pred = torch.argmax(source_logits).item()
    follow_up_pred = torch.argmax(follow_up_logits).item()

    passed = source_pred == follow_up_pred
    print(f"Source: digit {source_pred}")
    print(f"Double-flipped + INVERTED: digit {follow_up_pred}")
    print(f"Status: {'PASS ✓' if passed else 'FAIL ✗ (BUG DETECTED!)'}")

    plot_images_side_by_side(
        source[0], follow_up_buggy[0], "Original", "Double-flip + color invert (BUG!)",
        source_pred, follow_up_pred, source_logits, follow_up_logits,
        "MR2 (BUGGY): Color Inversion - SHOULD FAIL to detect bug", passed,
    )


def demo_buggy_brightness_extreme():
    """Buggy brightness: Apply extreme darkening instead of slight brightening."""
    print("\n" + "=" * 70)
    print("BUG INTRODUCED: Extreme darkening instead of brightening")
    print("=" * 70)
    print("(Simulating a hyperparameter bug: wrong brightness factor)\n")

    model, device, images, labels = get_model_and_data()
    idx = 0

    source = images[idx : idx + 1]
    
    # INTENDED: brightness factor 1.15 (slight brightening)
    # BUG: applied 0.2 instead (extreme darkening)
    follow_up_buggy = adjust_brightness(source, 0.2)

    source_logits = predict_logits(model, source, device)[0]
    follow_up_logits = predict_logits(model, follow_up_buggy, device)[0]
    source_pred = torch.argmax(source_logits).item()
    follow_up_pred = torch.argmax(follow_up_logits).item()

    passed = source_pred == follow_up_pred
    print(f"Source: digit {source_pred}")
    print(f"Extreme darkening (0.2×): digit {follow_up_pred}")
    print(f"Status: {'PASS ✓' if passed else 'FAIL ✗ (BUG DETECTED!)'}")

    plot_images_side_by_side(
        source[0], follow_up_buggy[0], "Original", "Extreme darkening (BUG!)",
        source_pred, follow_up_pred, source_logits, follow_up_logits,
        "MR4 (BUGGY): Wrong Brightness - SHOULD FAIL to detect bug", passed,
    )


def demo_buggy_blur_extreme():
    """Buggy blur: Apply extreme blur instead of slight blur."""
    print("\n" + "=" * 70)
    print("BUG INTRODUCED: Extreme blur kernel size")
    print("=" * 70)
    print("(Simulating a preprocessing bug: wrong kernel or sigma)\n")

    model, device, images, labels = get_model_and_data()
    idx = 0

    source = images[idx : idx + 1]
    
    # INTENDED: kernel=3, sigma=0.8 (light blur)
    # BUG: kernel=13, sigma=3.0 (destroys image)
    follow_up_buggy = gaussian_blur(source, kernel_size=13, sigma=3.0)

    source_logits = predict_logits(model, source, device)[0]
    follow_up_logits = predict_logits(model, follow_up_buggy, device)[0]
    source_pred = torch.argmax(source_logits).item()
    follow_up_pred = torch.argmax(follow_up_logits).item()

    passed = source_pred == follow_up_pred
    print(f"Source: digit {source_pred}")
    print(f"Extreme blur (k=13, σ=3): digit {follow_up_pred}")
    print(f"Status: {'PASS ✓' if passed else 'FAIL ✗ (BUG DETECTED!)'}")

    plot_images_side_by_side(
        source[0], follow_up_buggy[0], "Original", "Extreme blur (BUG!)",
        source_pred, follow_up_pred, source_logits, follow_up_logits,
        "MR5 (BUGGY): Wrong Blur Params - SHOULD FAIL to detect bug", passed,
    )


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("METAMORPHIC TESTING - BUG DETECTION DEMO")
    print("=" * 70)
    print("\nShowing how MRs can DETECT bugs introduced in pipeline\n")

    # Normal case first (baseline)
    demo_normal_mr2()

    # Then show bugs being detected
    demo_buggy_mr2_invert()
    demo_buggy_brightness_extreme()
    demo_buggy_blur_extreme()

    print("\n" + "=" * 70)
    print("Bug Detection Demo complete!")
    print("=" * 70)
    print("\nKey learning:")
    print("✓ MRs PASS with correct pipeline")
    print("✗ MRs FAIL when bugs are introduced")
    print("→ Automated early detection of regressions!")
    print("→ No need for oracle truth value - just detect relation breakage\n")
