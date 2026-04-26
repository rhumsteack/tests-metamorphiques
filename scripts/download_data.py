from __future__ import annotations

from src.dataset import MnistDataConfig, build_mnist_subsets


if __name__ == "__main__":
    cfg = MnistDataConfig()
    train_subset, eval_subset = build_mnist_subsets(cfg)
    print(f"MNIST subset ready: train={len(train_subset)}, eval={len(eval_subset)}")
