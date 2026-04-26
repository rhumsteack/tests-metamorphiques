from __future__ import annotations

import torch


def predict_logits(model: torch.nn.Module, batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        logits = model(batch.to(device))
    return logits.cpu()


def predict_top1(model: torch.nn.Module, batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    logits = predict_logits(model, batch, device)
    return torch.argmax(logits, dim=1)
