"""模型定义（任务 3，以及“全连接层对比”）。

- `SmallCNN` : 卷积层 + 激活 + 池化 组成的卷积网络；
- `MLP`      : 纯全连接网络，用来对比“卷积 vs 巨大全连接层”。
"""
from __future__ import annotations

import torch
import torch.nn as nn


def conv_block(in_ch: int, out_ch: int) -> nn.Module:
    """Conv -> BatchNorm -> ReLU -> MaxPool（每个空间维度减半）。"""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


class SmallCNN(nn.Module):
    """3 个卷积块的 CNN，输入 (N,3,S,S)，输出 (N,num_classes)。"""

    def __init__(self, num_classes: int = 2, in_ch: int = 3, width: int = 32,
                 dropout: float = 0.3):
        super().__init__()
        self.block1 = conv_block(in_ch, width)        # S   -> S/2
        self.block2 = conv_block(width, width * 2)    # S/2 -> S/4
        self.block3 = conv_block(width * 2, width * 4)  # S/4 -> S/8
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.head(x)


class MLP(nn.Module):
    """纯全连接基线：先把图片拉平，再接若干全连接层。"""

    def __init__(self, num_classes: int = 2, in_ch: int = 3, size: int = 64,
                 hidden: int = 512, dropout: float = 0.3):
        super().__init__()
        self.size = size
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch * size * size, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    cnn = SmallCNN()
    mlp = MLP()
    x = torch.randn(2, 3, 128, 128)
    print("SmallCNN 输出:", tuple(cnn(x).shape), " 参数量:", f"{count_parameters(cnn):,}")
    print("MLP(64) 输出:", tuple(mlp(torch.randn(2, 3, 64, 64)).shape),
          " 参数量:", f"{count_parameters(mlp):,}")
