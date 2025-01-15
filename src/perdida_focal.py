import torch
import torch.nn as nn
from typing import Self


class FocalLoss(nn.Module):
    def __init__(self: Self, gamma: float, alpha: float, reduction: str) -> None:
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self: Self, inputs, targets) -> torch.Tensor | float | complex:
        bce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-bce_loss)
        f_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return torch.mean(f_loss)

        elif self.reduction == 'sum':
            return torch.sum(f_loss)

        else:
            return f_loss