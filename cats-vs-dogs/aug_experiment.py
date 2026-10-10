"""任务 6：数据增强对照实验。

对同一网络、同样的数据划分与随机种子，只改变训练集增强策略：
    none / flip / flip+crop / flip+crop+color
比较验证集准确率，并可视化增强后的样本。

用法：python aug_experiment.py --epochs 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import data_prep as dp  # noqa: E402
import dataset as ds  # noqa: E402
import models as M  # noqa: E402
import train as T  # noqa: E402
import viz  # noqa: E402

CONFIGS = [
    ("none", "无增强", dict(flip=False, crop=False, color=False)),
    ("flip", "随机水平翻转", dict(flip=True, crop=False, color=False)),
    ("flip+crop", "翻转+随机裁剪", dict(flip=True, crop=True, color=False)),
    ("flip+crop+color", "翻转+裁剪+颜色扰动", dict(flip=True, crop=True, color=True)),
]


def show_augmented(samples: int = 4, size: int = ds.IMG_SIZE) -> None:
    rows = [r for r in dp.load_split("train") if r["class"] == "Dog"][:1]
    img = Image.open(dp.DATA_ROOT / rows[0]["path"]).convert("RGB")
    panels = [("原图", np.asarray(img.resize((size, size))))]
    for _key, label, cfg in CONFIGS:
        tf = ds.tf_train_aug(size, **cfg)
        for _ in range(samples):
            panels.append((label, ds.denormalize(tf(img))))
    out = viz.save_grid(panels, viz.OUT_DIR / "08_augmented_samples.png", ncols=5,
                        figsize=(16, 3.4 * ((len(panels) + 4) // 5)),
                        suptitle="同一张图在不同增强策略下的样本")
    print("已保存:", out)


def run(epochs: int, size: int, batch_size: int, lr: float, workers: int) -> dict:
    results = {}
    histories = {}
    for key, label, cfg in CONFIGS:
        T.set_seed(42)
        aug = {**cfg, "none": not any(cfg.values())}
        train_loader = ds.build_loader("train", ds.tf_train_aug(size, **cfg),
                                       batch_size, num_workers=workers)
        val_loader = ds.build_loader("val", ds.tf_resize_crop(size),
                                     batch_size, num_workers=workers)
        model = M.SmallCNN(width=32)
        history, best = T.fit(model, train_loader, val_loader, epochs, lr,
                              f"aug_{key}", size, verbose=True,
                              meta={"arch": "cnn", "width": 32, "aug_key": key})
        results[label] = best["val_acc"]
        histories[label] = history
        print(f"== {label}: 最佳 val acc = {best['val_acc']:.4f}")

    viz.save_bars(results, viz.OUT_DIR / "09_aug_comparison.png",
                  title="数据增强对照实验（验证集准确率）", ylabel="val acc")
    # 各配置的训练曲线叠在一张图上
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, h in histories.items():
        ax.plot(range(1, len(h["val_acc"]) + 1), h["val_acc"], "-o", label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("val acc")
    ax.set_title("不同增强策略的验证准确率曲线")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(viz.OUT_DIR / "10_aug_curves.png", dpi=110)
    plt.close(fig)

    (viz.OUT_DIR / "aug_results.json").write_text(
        json.dumps({"val_acc": results, "histories": histories},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("结果:", {k: round(v, 4) for k, v in results.items()})
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--size", type=int, default=ds.IMG_SIZE)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--samples-only", action="store_true")
    args = ap.parse_args(argv)
    show_augmented(size=args.size)
    if not args.samples_only:
        run(args.epochs, args.size, args.batch_size, args.lr, args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
