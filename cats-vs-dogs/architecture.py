"""任务 9：绘制模型结构图。

用法：python architecture.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import viz  # noqa: E402

LAYERS = [
    ("输入\n3 x 128 x 128", "#DCE9F7"),
    ("Conv 3x3, 32\nBN + ReLU", "#CFE3F7"),
    ("MaxPool 2x2\n32 x 64 x 64", "#E8F0DC"),
    ("Conv 3x3, 64\nBN + ReLU", "#CFE3F7"),
    ("MaxPool 2x2\n64 x 32 x 32", "#E8F0DC"),
    ("Conv 3x3, 128\nBN + ReLU", "#CFE3F7"),
    ("MaxPool 2x2\n128 x 16 x 16", "#E8F0DC"),
    ("Global AvgPool\n128 x 1 x 1", "#F3E1C7"),
    ("Dropout 0.3\nFlatten -> 128", "#F3E1C7"),
    ("Linear 128 -> 2\nSoftmax", "#F7D9D9"),
    ("输出\n猫 / 狗", "#DCE9F7"),
]


def draw(path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 13))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(LAYERS) * 1.15 + 0.6)
    ax.axis("off")
    height, width = 0.82, 6.4
    x = (10 - width) / 2

    for i, (text, color) in enumerate(LAYERS):
        y = (len(LAYERS) - 1 - i) * 1.15 + 0.4
        box = FancyBboxPatch((x, y), width, height,
                             boxstyle="round,pad=0.02,rounding_size=0.12",
                             linewidth=1.4, edgecolor="#3C4858", facecolor=color)
        ax.add_patch(box)
        ax.text(x + width / 2, y + height / 2, text, ha="center", va="center",
                fontsize=11)
        if i < len(LAYERS) - 1:
            ax.add_patch(FancyArrowPatch((x + width / 2, y),
                                         (x + width / 2, y - 0.33),
                                         arrowstyle="-|>", mutation_scale=14,
                                         color="#3C4858", linewidth=1.4))
    ax.set_title("SmallCNN 结构图（输入 3x128x128）\n"
                 "每个卷积块：Conv3x3 -> BatchNorm -> ReLU -> MaxPool2（宽高减半）",
                 fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    out = draw(viz.OUT_DIR / "11_architecture.png")
    print("已保存:", out)
