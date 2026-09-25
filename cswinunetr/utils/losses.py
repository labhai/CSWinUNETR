"""Foreground batch Dice and cross-entropy loss."""

import torch
from torch.nn import functional as F


def dice_ce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Average Dice loss over present foreground classes, combined with cross entropy."""
    probabilities = logits.float().softmax(1)[:, 1:]
    labels = F.one_hot(target, logits.shape[1]).movedim(-1, 1)[:, 1:].float()
    dims = (0, *range(2, logits.ndim))
    intersection = (probabilities * labels).sum(dims)
    target_sum = labels.sum(dims)
    denominator = probabilities.sum(dims) + target_sum
    present = target_sum > 0
    dice = (2 * intersection + 1e-6) / (denominator + 1e-6)
    dice_loss = 1 - dice[present].mean() if present.any() else logits.sum() * 0
    return 0.5 * dice_loss + 0.5 * F.cross_entropy(logits.float(), target)
