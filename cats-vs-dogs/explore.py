"""任务 1、2：观察数据集、统计数量、展示样例与预处理前后的对比。

用法：
    python explore.py --overview      # 数量 + 尺寸/清晰度统计图
    python explore.py --samples 8     # 随机展示样例
    python explore.py --preprocess    # 预处理前后对比
    python explore.py --all
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import data_prep as dp  # noqa: E402
import dataset as ds  # noqa: E402
import viz  # noqa: E402


def overview() -> None:
    rows = dp.load_manifest()
    counts = dp.count_by_class(rows)
    w = np.array([int(r["width"]) for r in rows])
    h = np.array([int(r["height"]) for r in rows])
    ar = w / h
    sharp = np.array([float(r["sharpness"]) for r in rows])

    print(f"有效图片总数: {len(rows)}   Cat: {counts['Cat']}   Dog: {counts['Dog']}")
    print(f"宽  : min {w.min():4d}  中位 {int(np.median(w)):4d}  max {w.max():5d}  均值 {w.mean():6.1f}")
    print(f"高  : min {h.min():4d}  中位 {int(np.median(h)):4d}  max {h.max():5d}  均值 {h.mean():6.1f}")
    print(f"宽高比: min {ar.min():.2f}  中位 {np.median(ar):.2f}  max {ar.max():.2f}")
    near_square = float(np.mean((ar > 0.9) & (ar < 1.1)))
    print(f"近似正方形(0.9<宽高比<1.1)的占比: {near_square:.1%}  —— 说明尺寸/比例并不统一")
    p10, p50, p90 = np.percentile(sharp, [10, 50, 90])
    print(f"清晰度(拉普拉斯方差, 64x64 缩略图): p10 {p10:.1f}  中位 {p50:.1f}  p90 {p90:.1f}  "
          f"max {sharp.max():.1f}")
    print(f"清晰度差异：p90/p10 ≈ {p90 / max(p10, 1e-6):.1f} 倍 —— 清晰度差异明显")

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].hist(w, bins=40, color="#4C78A8"); axes[0, 0].set_title("宽度分布")
    axes[0, 1].hist(h, bins=40, color="#72B7B2"); axes[0, 1].set_title("高度分布")
    axes[1, 0].hist(ar, bins=40, color="#F58518"); axes[1, 0].set_title("宽高比分布")
    axes[1, 1].hist(sharp, bins=40, range=(0, 400), color="#E45756")
    axes[1, 1].set_title("清晰度(拉普拉斯方差)分布")
    for ax in axes.ravel():
        ax.grid(alpha=0.3)
    fig.suptitle("Dogs vs Cats 数据集统计", fontsize=15)
    fig.tight_layout()
    fig.savefig(viz.OUT_DIR / "01_dataset_stats.png", dpi=110)
    plt.close(fig)
    print("已保存:", viz.OUT_DIR / "01_dataset_stats.png")


def samples(n: int = 8, seed: int = 0) -> None:
    rows = dp.load_manifest()
    rng = random.Random(seed)
    picks = rng.sample(rows, n)
    panels = []
    for r in picks:
        img = Image.open(dp.DATA_ROOT / r["path"]).convert("RGB")
        img.thumbnail((320, 320))
        panels.append((f"{r['class']}  {r['width']}x{r['height']}", np.asarray(img)))
    out = viz.save_grid(panels, viz.OUT_DIR / "02_random_samples.png", ncols=4,
                        figsize=(14, 7), suptitle="随机样例（可见尺寸/背景/清晰度各不相同）")
    print("已保存:", out)


def preprocess(k: int = 3, size: int = ds.IMG_SIZE) -> None:
    rows = [r for r in dp.load_split("train")][:k]
    panels = []
    for r in rows:
        img = Image.open(dp.DATA_ROOT / r["path"]).convert("RGB")
        orig = np.asarray(img.copy())
        stretch = ds.denormalize(ds.tf_stretch(size)(img))
        crop = ds.denormalize(ds.tf_resize_crop(size)(img))
        aug = ds.denormalize(ds.tf_train_aug(size, flip=True, crop=True, color=True)(img))
        panels += [
            (f"原图 {orig.shape[1]}x{orig.shape[0]}", orig),
            ("拉伸到 128x128（变形）", stretch),
            ("等比缩放+裁剪", crop),
            ("增强(裁剪+翻转+颜色)", aug),
        ]
    out = viz.save_grid(panels, viz.OUT_DIR / "03_preprocess_compare.png", ncols=4,
                        figsize=(15, 3.6 * k),
                        suptitle="预处理前后对比：拉伸 vs 等比缩放+裁剪 vs 数据增强")
    print("已保存:", out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overview", action="store_true")
    ap.add_argument("--samples", type=int, default=0)
    ap.add_argument("--preprocess", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args(argv)
    if args.all or args.overview:
        overview()
    if args.all or args.samples:
        samples(args.samples or 8)
    if args.all or args.preprocess:
        preprocess()
    return 0


if __name__ == "__main__":
    sys.exit(main())
