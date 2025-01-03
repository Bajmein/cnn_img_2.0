import torch
import torch.nn as nn
from typing import Self


class ModeloCNN(nn.Module):
    def __init__(self: Self) -> None:
        super(ModeloCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(32 * 56 * 56, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=2)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout(0.5)

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        x = x.view(-1, 32 * 56 * 56)
        x = torch.relu(self.fc1(x))

        x = self.dropout(x)
        x = self.fc2(x)

        return x