"""绘图与展示工具：统一中文字体、保存图片网格与训练曲线。"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def setup_cjk_font() -> None:
    for p in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
        if Path(p).exists():
            try:
                font_manager.fontManager.addfont(p)
            except Exception:
                pass
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


setup_cjk_font()


def save_grid(panels, path, ncols=4, figsize=None, suptitle=None, titlesize=11):
    """panels: [(title, image(HWC or HW, 0-1 or 0-255)), ...]"""
    n = len(panels)
    nrows = int(np.ceil(n / ncols))
    if figsize is None:
        figsize = (3.2 * ncols, 3.4 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, image) in zip(axes, panels):
        arr = np.asarray(image)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 1) if arr.max() <= 1.001 else np.clip(arr / 255.0, 0, 1)
        ax.imshow(arr, cmap="gray" if arr.ndim == 2 else None,
                  vmin=0 if arr.dtype == np.uint8 else None,
                  vmax=255 if arr.dtype == np.uint8 else None)
        ax.set_title(title, fontsize=titlesize)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return Path(path)


def save_curves(history, path, title="训练曲线"):
    """history 含 train_loss/train_acc/val_loss/val_acc。"""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(epochs, history["train_loss"], "-o", label="train")
    axes[0].plot(epochs, history["val_loss"], "-s", label="val")
    axes[0].set_title("loss")
    axes[0].set_xlabel("epoch")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].plot(epochs, history["train_acc"], "-o", label="train")
    axes[1].plot(epochs, history["val_acc"], "-s", label="val")
    axes[1].set_title("accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylim(0.4, 1.0)
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return Path(path)


def save_bars(values: dict, path, title="", ylabel=""):
    fig, ax = plt.subplots(figsize=(max(5, 1.5 * len(values)), 4))
    keys = list(values.keys())
    vals = [values[k] for k in keys]
    bars = ax.bar(keys, vals, color="#4C78A8")
    ax.set_title(title, fontsize=13)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10)
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return Path(path)


def save_confusion(cm, path, labels=("Cat", "Dog")):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("预测")
    ax.set_ylabel("真实")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return Path(path)
